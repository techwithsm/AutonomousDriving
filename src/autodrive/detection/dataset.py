"""Part 1 - Task 2: prepare the vehicle dataset for training.

Steps
-----
1. ``ensure_images``      - fast, parallel, resumable unzip of ``Images.zip`` when the images
                            are not yet extracted (the PDF warns unzipping is slow on a VM).
2. ``load_annotations``   - read ``labels.csv`` (id, class, xmin, ymin, xmax, ymax; no header),
                            keep only boxes of images that exist, attach image sizes.
3. ``clean_annotations``  - drop zero/negative/degenerate boxes, clip to image bounds, dedupe.
4. ``split_images``       - image-level train/val/test split, stratified on the *rarest* class
                            present in each image so rare classes appear in every split.
5. ``write_yolo_dataset`` - link/copy images and write normalised YOLO label files + data.yaml.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from autodrive.common import get_logger, resolve_path

from .workspace import SPLITS, Workspace

log = get_logger("part1.dataset")
LABEL_COLUMNS = ["image_id", "cls", "xmin", "ymin", "xmax", "ymax"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- 1. images
def _extract_member(zf_path: Path, name: str, dest: Path) -> None:
    target = dest / name
    if target.exists():
        return  # resumable: skip already-extracted files
    with zipfile.ZipFile(zf_path) as zf:   # one handle per call -> thread-safe
        zf.extract(name, dest)


def extract_zip(zip_path: Path, dest: Path, workers: int = 8) -> Path:
    """Parallel, resumable extraction of image members only (skips __MACOSX junk)."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if Path(m).suffix.lower() in IMG_EXTS and "__MACOSX" not in m]
    log.info("Extracting %d images from %s with %d threads", len(members), zip_path.name, workers)
    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        list(tqdm(pool.map(lambda m: _extract_member(zip_path, m, dest), members),
                  total=len(members), desc="unzip"))
    # return the folder that actually holds the images
    first = dest / members[0]
    return first.parent


def ensure_images(cfg: dict, ws: Workspace) -> Path:
    """Return the folder with raw images, extracting the zip only when needed."""
    src = cfg["source"]
    img_dir = resolve_path(src["images_dir"])
    if img_dir.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in img_dir.iterdir()):
        log.info("Using already-extracted images in %s", img_dir)
        return img_dir
    zip_path = resolve_path(src["images_zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"No images in {img_dir} and no zip at {zip_path}")
    return extract_zip(zip_path, ws.raw, src.get("extract_workers", 8))


def list_images(img_dir: Path) -> pd.DataFrame:
    """image_id, file, width, height for every image (reads headers only - fast)."""
    rows = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:  # corrupt file
            log.warning("Skipping unreadable image %s (%s)", p.name, e)
            continue
        rows.append({"image_id": int(p.stem), "file": str(p), "width": w, "height": h})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2-3. annotations
def load_annotations(labels_csv: Path, images: pd.DataFrame) -> pd.DataFrame:
    ann = pd.read_csv(labels_csv, header=None, names=LABEL_COLUMNS)
    total = len(ann)
    ann = ann.merge(images[["image_id", "width", "height"]], on="image_id", how="inner")
    log.info("labels.csv: %d boxes total, %d belong to the %d available images",
             total, len(ann), len(images))
    return ann


def clean_annotations(ann: pd.DataFrame, classes: list[str], min_box_px: int = 2) -> pd.DataFrame:
    n0 = len(ann)
    ann = ann[ann["cls"].isin(classes)].copy()
    ann["xmin"] = ann["xmin"].clip(lower=0)
    ann["ymin"] = ann["ymin"].clip(lower=0)
    ann["xmax"] = ann[["xmax", "width"]].min(axis=1)
    ann["ymax"] = ann[["ymax", "height"]].min(axis=1)
    ok = ((ann["xmax"] - ann["xmin"]) >= min_box_px) & ((ann["ymax"] - ann["ymin"]) >= min_box_px)
    ann = ann[ok].drop_duplicates(subset=LABEL_COLUMNS)
    ann["class_id"] = ann["cls"].map({c: i for i, c in enumerate(classes)})
    log.info("Clean annotations: kept %d / %d boxes", len(ann), n0)
    return ann.reset_index(drop=True)


# --------------------------------------------------------------------------- 4. split
def split_images(ann: pd.DataFrame, images: pd.DataFrame, ratios: dict, seed: int = 42) -> pd.DataFrame:
    """Stratify on the rarest class in each image (keeps rare classes in val/test)."""
    freq = ann["cls"].value_counts()
    rarest = (ann.assign(f=ann["cls"].map(freq)).sort_values("f")
                 .drop_duplicates("image_id").set_index("image_id")["cls"])
    imgs = images[images["image_id"].isin(rarest.index)].copy()
    imgs["stratum"] = imgs["image_id"].map(rarest)
    counts = imgs["stratum"].value_counts()
    imgs.loc[imgs["stratum"].isin(counts[counts < 5].index), "stratum"] = "_rare"

    test_size = ratios["test"]
    val_size = ratios["val"] / (1 - test_size)
    trval, test = train_test_split(imgs, test_size=test_size, stratify=imgs["stratum"], random_state=seed)
    train, val = train_test_split(trval, test_size=val_size, stratify=trval["stratum"], random_state=seed)
    imgs["split"] = "train"
    imgs.loc[val.index, "split"] = "val"
    imgs.loc[test.index, "split"] = "test"
    log.info("Split: %s", imgs["split"].value_counts().to_dict())
    return imgs


# --------------------------------------------------------------------------- 5. YOLO export
def to_yolo_lines(boxes: pd.DataFrame) -> list[str]:
    """Pixel corners -> 'class cx cy w h' normalised to [0, 1]."""
    W, H = boxes["width"], boxes["height"]
    cx = ((boxes["xmin"] + boxes["xmax"]) / 2) / W
    cy = ((boxes["ymin"] + boxes["ymax"]) / 2) / H
    bw = (boxes["xmax"] - boxes["xmin"]) / W
    bh = (boxes["ymax"] - boxes["ymin"]) / H
    return [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for c, x, y, w, h in zip(boxes["class_id"], cx, cy, bw, bh)]


def _place(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:  # different drive / unsupported FS -> fall back to copy
            pass
    elif mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_yolo_dataset(ws: Workspace, imgs: pd.DataFrame, ann: pd.DataFrame,
                       classes: list[str], link_mode: str = "hardlink") -> Path:
    for s in SPLITS:  # start from a clean slate so re-runs with a new split are consistent
        for d in (ws.images(s), ws.labels(s)):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    by_img = dict(tuple(ann.groupby("image_id")))
    for r in tqdm(imgs.itertuples(), total=len(imgs), desc="write YOLO dataset"):
        src = Path(r.file)
        _place(src, ws.images(r.split) / src.name, link_mode)
        lines = to_yolo_lines(by_img[r.image_id]) if r.image_id in by_img else []
        (ws.labels(r.split) / f"{src.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data = {
        "path": ws.dataset.resolve().as_posix(),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": len(classes),
        "names": {i: c for i, c in enumerate(classes)},
    }
    ws.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    imgs.drop(columns=["file"]).to_csv(ws.split_csv, index=False)
    log.info("YOLO dataset written -> %s", ws.data_yaml)
    return ws.data_yaml


def split_summary(imgs: pd.DataFrame, ann: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Boxes per class per split - check every class is represented everywhere."""
    m = ann.merge(imgs[["image_id", "split"]], on="image_id")
    t = m.pivot_table(index="cls", columns="split", values="image_id", aggfunc="count", fill_value=0)
    t = t.reindex(index=classes, columns=list(SPLITS), fill_value=0)
    t.loc["TOTAL boxes"] = t.sum()
    t.loc["TOTAL images"] = imgs["split"].value_counts().reindex(list(SPLITS)).values
    return t


def prepare_dataset(cfg: dict, ws: Workspace) -> dict:
    """Run steps 1-5. Returns the frames used, for EDA / notebooks."""
    classes = cfg["dataset"]["classes"]
    img_dir = ensure_images(cfg, ws)
    images = list_images(img_dir)
    ann = load_annotations(resolve_path(cfg["source"]["labels_csv"]), images)
    ann = clean_annotations(ann, classes, cfg["dataset"].get("min_box_px", 2))
    imgs = split_images(ann, images, cfg["dataset"]["split"], cfg.get("seed", 42))
    data_yaml = write_yolo_dataset(ws, imgs, ann, classes, cfg["dataset"].get("link_mode", "hardlink"))
    summary = split_summary(imgs, ann, classes)
    summary.to_csv(ws.reports / "split_summary.csv")
    return {"images": imgs, "annotations": ann, "data_yaml": data_yaml, "split_summary": summary}
