"""Part 1 - Task 4: evaluate the trained model on the held-out test split."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from autodrive.common import get_logger, resolve_path
from autodrive.common.plotting import PALETTE, label_bars, save

from .model import best_weights, resolve_device
from .workspace import Workspace

log = get_logger("part1.evaluate")


def evaluate(cfg: dict, ws: Workspace, weights: str | Path | None = None) -> dict:
    """Run validation on ``evaluate.split``; save overall + per-class metrics."""
    from ultralytics import YOLO

    e = cfg["evaluate"]
    weights = Path(weights or e.get("weights") or best_weights(cfg, ws))
    weights = resolve_path(weights) if not weights.is_absolute() else weights
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}. Train first (run_part1.py train).")
    model = YOLO(str(weights))
    split = e.get("split", "test")
    m = model.val(data=str(ws.data_yaml), split=split, imgsz=cfg["train"]["imgsz"],
                  conf=e.get("conf", 0.001), iou=e.get("iou", 0.6),
                  device=resolve_device(cfg["train"].get("device")), batch=cfg["train"]["batch"],
                  project=str(ws.runs), name=f"{cfg['train']['run_name']}_{split}", exist_ok=True, plots=True)

    box = m.box
    overall = {
        "split": split,
        "weights": str(weights),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "f1": round(float(2 * box.mp * box.mr / (box.mp + box.mr + 1e-9)), 4),
        "speed_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
    }
    names = m.names
    rows = []
    for i, cid in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows.append({"class": names[int(cid)], "precision": p, "recall": r, "AP50": ap50, "AP50-95": ap})
    per_class = pd.DataFrame(rows).set_index("class").round(4)

    # attach support (boxes in this split) to read AP in context of class frequency
    split_csv = ws.reports / "split_summary.csv"
    if split_csv.exists():
        sup = pd.read_csv(split_csv, index_col=0)
        if split in sup.columns:
            per_class["test_boxes"] = sup[split].reindex(per_class.index)
    per_class = per_class.sort_values("AP50", ascending=False)

    out = ws.reports
    (out / f"metrics_{split}.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    per_class.to_csv(out / f"per_class_metrics_{split}.csv")
    save(fig_per_class(per_class), out / f"per_class_ap_{split}.png")
    log.info("Test metrics: %s", {k: overall[k] for k in ("precision", "recall", "mAP50", "mAP50-95")})
    return {"overall": overall, "per_class": per_class, "save_dir": Path(m.save_dir)}


def fig_per_class(per_class: pd.DataFrame):
    s = per_class["AP50"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(s.index, s.values, color=PALETTE[0], height=0.65)
    ax.set_xlim(0, 1.05); ax.set_title("AP@0.5 per class (test split)")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    label_bars(ax, fmt="{:.2f}", horizontal=True)
    fig.tight_layout()
    return fig


def training_curves(run_dir: Path):
    """Loss and mAP curves from Ultralytics' results.csv."""
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = df.columns.str.strip()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (k, lbl) in enumerate([("box_loss", "box"), ("cls_loss", "cls"), ("dfl_loss", "dfl")]):
        axes[0].plot(df["epoch"], df[f"train/{k}"], color=PALETTE[i], label=f"train {lbl}")
        axes[0].plot(df["epoch"], df[f"val/{k}"], color=PALETTE[i], ls="--", label=f"val {lbl}")
    axes[0].set_title("Losses"); axes[0].legend(ncols=3, fontsize=8)
    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], color=PALETTE[0], label="mAP50")
    axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], color=PALETTE[1], label="mAP50-95")
    axes[1].set_title("Validation mAP"); axes[1].legend()
    axes[2].plot(df["epoch"], df["metrics/precision(B)"], color=PALETTE[0], label="precision")
    axes[2].plot(df["epoch"], df["metrics/recall(B)"], color=PALETTE[1], label="recall")
    axes[2].set_title("Validation precision / recall"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    return fig
