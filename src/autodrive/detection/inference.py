"""Part 1 - Task 5: run inference on sample images and check vehicles are detected accurately.

Produces, per image: annotated prediction, per-class vehicle counts (the "vehicle counting"
use case), and - for test images - a side-by-side ground truth vs prediction comparison with
a simple matching score (TP / FP / FN at IoU >= 0.5).
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from autodrive.common import get_logger, resolve_path
from autodrive.common.plotting import save

from .eda import class_colors, draw_boxes
from .model import best_weights, resolve_device
from .workspace import Workspace

log = get_logger("part1.inference")


def load_model(cfg: dict, ws: Workspace, weights=None):
    from ultralytics import YOLO

    w = Path(weights or cfg["evaluate"].get("weights") or best_weights(cfg, ws))
    w = w if w.is_absolute() else resolve_path(w)
    if not w.exists():
        raise FileNotFoundError(f"Weights not found: {w}. Train first.")
    return YOLO(str(w))


def sample_sources(cfg: dict, ws: Workspace) -> list[Path]:
    p = cfg["predict"]
    if p.get("source"):
        src = resolve_path(p["source"])
        return sorted(src.glob("*.jpg")) if src.is_dir() else [src]
    files = sorted(ws.images("test").glob("*.jpg"))
    random.Random(cfg.get("seed", 42)).shuffle(files)
    return files[: p.get("n_samples", 12)]


def predictions_frame(results) -> pd.DataFrame:
    rows = []
    for r in results:
        b = r.boxes
        for xyxy, conf, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
            rows.append({"image": Path(r.path).name, "cls": r.names[c], "conf": float(conf),
                         "xmin": xyxy[0], "ymin": xyxy[1], "xmax": xyxy[2], "ymax": xyxy[3]})
    return pd.DataFrame(rows, columns=["image", "cls", "conf", "xmin", "ymin", "xmax", "ymax"])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt: pd.DataFrame, pred: pd.DataFrame, iou_thr: float = 0.5) -> dict:
    """Greedy class-aware matching -> TP/FP/FN for one image."""
    cols = ["xmin", "ymin", "xmax", "ymax"]
    tp = 0
    used = set()
    pred = pred.sort_values("conf", ascending=False)
    ious = iou_matrix(pred[cols].to_numpy(float), gt[cols].to_numpy(float))
    for i, pc in enumerate(pred["cls"].to_numpy()):
        cand = [(ious[i, j], j) for j, gc in enumerate(gt["cls"].to_numpy())
                if j not in used and gc == pc and ious[i, j] >= iou_thr]
        if cand:
            used.add(max(cand)[1]); tp += 1
    return {"gt": len(gt), "pred": len(pred), "tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}


def ground_truth_for(ws: Workspace, image: Path, names: dict) -> pd.DataFrame:
    """Read a YOLO label file back into pixel boxes."""
    lbl = ws.labels("test") / f"{image.stem}.txt"
    if not lbl.exists():
        return pd.DataFrame(columns=["cls", "xmin", "ymin", "xmax", "ymax"])
    W, H = Image.open(image).size
    rows = []
    for line in lbl.read_text().split("\n"):
        if line.strip():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            rows.append({"cls": names[int(c)], "xmin": cx - w / 2, "ymin": cy - h / 2,
                         "xmax": cx + w / 2, "ymax": cy + h / 2})
    return pd.DataFrame(rows, columns=["cls", "xmin", "ymin", "xmax", "ymax"])


def predict(cfg: dict, ws: Workspace, weights=None, sources: list[Path] | None = None) -> dict:
    p = cfg["predict"]
    model = load_model(cfg, ws, weights)
    sources = sources or sample_sources(cfg, ws)
    out_dir = ws.inference
    out_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict([str(s) for s in sources], conf=p["conf"], iou=p["iou"], max_det=p["max_det"],
                            imgsz=cfg["train"]["imgsz"], device=resolve_device(cfg["train"].get("device")),
                            save=True, project=str(out_dir), name="annotated", exist_ok=True, verbose=False)
    preds = predictions_frame(results)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    counts = (preds.pivot_table(index="image", columns="cls", values="conf", aggfunc="count", fill_value=0)
              if len(preds) else pd.DataFrame())
    counts["total_vehicles"] = counts.sum(axis=1) if len(counts) else 0
    counts.to_csv(out_dir / "vehicle_counts.csv")

    names = model.names
    scores = []
    for s in sources:
        gt = ground_truth_for(ws, Path(s), names)
        if len(gt):
            scores.append({"image": Path(s).name, **match(gt, preds[preds["image"] == Path(s).name])})
    scores = pd.DataFrame(scores)
    if len(scores):
        tot = scores[["gt", "pred", "tp", "fp", "fn"]].sum()
        scores.loc["TOTAL"] = ["TOTAL", *tot.values]
        scores.to_csv(out_dir / "sample_accuracy.csv", index=False)
        log.info("Sample images: %d GT, %d predicted, %d correct (precision %.2f, recall %.2f)",
                 tot["gt"], tot["pred"], tot["tp"], tot["tp"] / max(tot["pred"], 1), tot["tp"] / max(tot["gt"], 1))

    comp = save(fig_gt_vs_pred(ws, sources[:6], preds, names), out_dir / "gt_vs_pred.png")
    return {"predictions": preds, "counts": counts, "sample_accuracy": scores,
            "comparison_figure": comp, "annotated_dir": out_dir / "annotated"}


def fig_gt_vs_pred(ws: Workspace, sources: list[Path], preds: pd.DataFrame, names: dict):
    colors = class_colors(list(names.values()))
    fig, axes = plt.subplots(len(sources), 2, figsize=(11, 3.6 * len(sources)), squeeze=False)
    for row, s in zip(axes, sources):
        s = Path(s)
        img = Image.open(s)
        gt = ground_truth_for(ws, s, names)
        pr = preds[preds["image"] == s.name]
        row[0].imshow(draw_boxes(img, gt, colors)); row[0].set_title(f"{s.stem} - ground truth ({len(gt)})", fontsize=9)
        row[1].imshow(draw_boxes(img, pr, colors)); row[1].set_title(f"{s.stem} - prediction ({len(pr)})", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    return fig
