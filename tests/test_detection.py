import numpy as np
import pandas as pd
import yaml
from PIL import Image

from autodrive.detection.dataset import (clean_annotations, split_images, to_yolo_lines,
                                         write_yolo_dataset)
from autodrive.detection.inference import iou_matrix, match
from autodrive.detection.workspace import create_workspace

CLASSES = ["car", "bus", "bicycle"]


def test_to_yolo_lines():
    b = pd.DataFrame({"class_id": [1], "xmin": [100], "ymin": [50], "xmax": [300], "ymax": [150],
                      "width": [400], "height": [200]})
    assert to_yolo_lines(b) == ["1 0.500000 0.500000 0.500000 0.500000"]


def test_clean_annotations_drops_and_clips():
    ann = pd.DataFrame({
        "image_id": [0, 0, 0, 0], "cls": ["car", "car", "truck", "bus"],
        "xmin": [10, 50, 0, -5], "ymin": [10, 50, 0, 0], "xmax": [40, 50, 20, 800], "ymax": [40, 90, 20, 30],
        "width": [720] * 4, "height": [480] * 4,
    })
    out = clean_annotations(ann, CLASSES)
    assert len(out) == 2                         # zero-width box + unknown class dropped
    bus = out[out.cls == "bus"].iloc[0]
    assert bus.xmin == 0 and bus.xmax == 720     # clipped to image
    assert set(out.class_id) == {0, 1}


def _toy(n=60):
    rng = np.random.default_rng(0)
    images = pd.DataFrame({"image_id": range(n), "file": [f"{i:08d}.jpg" for i in range(n)],
                           "width": 64, "height": 48})
    rows = []
    for i in range(n):
        rows.append([i, "car", 1, 1, 20, 20])
        if i % 6 == 0:
            rows.append([i, "bicycle", 30, 10, 40, 30])
        if i % 4 == 1:
            rows.append([i, "bus", 5, 5, 60, 40])
    ann = pd.DataFrame(rows, columns=["image_id", "cls", "xmin", "ymin", "xmax", "ymax"])
    ann = ann.merge(images[["image_id", "width", "height"]], on="image_id")
    return images, clean_annotations(ann, CLASSES), rng


def test_split_is_stratified_and_disjoint():
    images, ann, _ = _toy()
    s = split_images(ann, images, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert s["image_id"].is_unique and len(s) == 60
    bikes = s[s.image_id % 6 == 0]
    assert set(bikes.split) == {"train", "val", "test"}   # rare class in every split


def test_write_yolo_dataset(tmp_path):
    images, ann, _ = _toy(20)
    src = tmp_path / "src"; src.mkdir()
    for f in images.file:
        Image.new("RGB", (64, 48)).save(src / f)
    images["file"] = [str(src / f) for f in images.file]
    cfg = {"workspace": {"root": str(tmp_path / "ws"), "children": {
        "raw": "raw", "dataset": "dataset", "runs": "runs", "inference": "inf", "reports": "rep"}}}
    ws = create_workspace(cfg)
    s = split_images(ann, images, {"train": 0.6, "val": 0.2, "test": 0.2})
    data_yaml = write_yolo_dataset(ws, s, ann, CLASSES, "copy")
    d = yaml.safe_load(data_yaml.read_text())
    assert d["nc"] == 3 and d["names"][2] == "bicycle"
    n_img = sum(len(list(ws.images(x).glob("*.jpg"))) for x in ("train", "val", "test"))
    n_lbl = sum(len(list(ws.labels(x).glob("*.txt"))) for x in ("train", "val", "test"))
    assert n_img == n_lbl == 20


def test_iou_and_match():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] > 0.99
    gt = pd.DataFrame({"cls": ["car", "bus"], "xmin": [0, 50], "ymin": [0, 50], "xmax": [10, 80], "ymax": [10, 80]})
    pr = pd.DataFrame({"cls": ["car", "car"], "conf": [0.9, 0.8],
                       "xmin": [1, 50], "ymin": [1, 50], "xmax": [10, 80], "ymax": [10, 80]})
    assert match(gt, pr) == {"gt": 2, "pred": 2, "tp": 1, "fp": 1, "fn": 1}
