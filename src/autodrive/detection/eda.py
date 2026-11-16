"""Part 1 - dataset EDA: class balance, objects per image, box geometry, ground-truth samples.

These plots justify the modelling choices (imbalance, small objects, two resolutions).
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from autodrive.common.plotting import PALETTE, PRIMARY, label_bars, save

# fixed class -> colour mapping (identity never depends on rank)
def class_colors(classes: list[str]) -> dict[str, str]:
    extra = ["#9a9893", "#5b4636", "#0f6b70"]
    pal = PALETTE + extra
    return {c: pal[i % len(pal)] for i, c in enumerate(classes)}


def fig_class_distribution(ann: pd.DataFrame, classes: list[str]):
    boxes = ann["cls"].value_counts().reindex(classes, fill_value=0)
    imgs = ann.groupby("cls")["image_id"].nunique().reindex(classes, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, s, t in [(axes[0], boxes, "Boxes per class"), (axes[1], imgs, "Images containing class")]:
        s = s.iloc[::-1]
        ax.barh(s.index, s.values, color=PRIMARY, height=0.65)
        ax.set_xscale("log"); ax.set_title(t + " (log scale)")
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        label_bars(ax, horizontal=True)
    fig.tight_layout()
    return fig


def fig_box_stats(ann: pd.DataFrame):
    w = ann["xmax"] - ann["xmin"]
    h = ann["ymax"] - ann["ymin"]
    rel_area = 100 * w * h / (ann["width"] * ann["height"])
    per_img = ann.groupby("image_id").size()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(per_img, bins=range(1, per_img.max() + 2), color=PRIMARY, rwidth=0.85)
    axes[0].set_title("Objects per image"); axes[0].set_xlabel("Boxes")
    axes[1].hist(rel_area.clip(upper=20), bins=50, color=PRIMARY, rwidth=0.85)
    axes[1].set_title("Box area (% of image, clipped at 20)"); axes[1].set_xlabel("% of image area")
    axes[2].scatter(w / ann["width"], h / ann["height"], s=3, alpha=0.25, color=PRIMARY)
    axes[2].set_title("Normalised box width vs height"); axes[2].set_xlabel("w / W"); axes[2].set_ylabel("h / H")
    axes[2].grid(axis="x", visible=True)
    fig.tight_layout()
    return fig


def fig_split_balance(summary: pd.DataFrame):
    t = summary.drop(index=["TOTAL boxes", "TOTAL images"], errors="ignore")
    pct = 100 * t.div(t.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    left = pd.Series(0.0, index=pct.index)
    for i, split in enumerate(pct.columns):
        ax.barh(pct.index[::-1], pct[split][::-1], left=left[::-1], color=PALETTE[i], height=0.65,
                label=split, edgecolor="white", linewidth=2)
        left += pct[split]
    ax.set_title("Share of each class's boxes per split (%)")
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    fig.tight_layout()
    return fig


def draw_boxes(img: Image.Image, boxes: pd.DataFrame, colors: dict, scale=(1.0, 1.0), width=2) -> Image.Image:
    from PIL import ImageDraw

    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    sx, sy = scale
    for r in boxes.itertuples():
        c = colors.get(r.cls, "#ffffff")
        xy = [r.xmin * sx, r.ymin * sy, r.xmax * sx, r.ymax * sy]
        d.rectangle(xy, outline=c, width=width)
        d.text((xy[0] + 2, max(xy[1] - 11, 0)), r.cls, fill=c)
    return img


def fig_samples(ann: pd.DataFrame, images: pd.DataFrame, classes: list[str], n: int = 6, seed: int = 0):
    colors = class_colors(classes)
    rng = random.Random(seed)
    # prefer images with several classes so the grid is informative
    rich = ann.groupby("image_id")["cls"].nunique().sort_values(ascending=False)
    picks = rng.sample(list(rich.index[: max(n * 10, n)]), n)
    files = images.set_index("image_id")["file"]
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows))
    for ax, iid in zip(axes.ravel(), picks):
        img = draw_boxes(Image.open(files[iid]), ann[ann["image_id"] == iid], colors)
        ax.imshow(img); ax.set_title(f"{iid:08d}", fontsize=9); ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Ground-truth annotations", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def run_eda(prep: dict, classes: list[str], out_dir: Path) -> dict[str, Path]:
    ann, imgs = prep["annotations"], prep["images"]
    return {
        "class_distribution": save(fig_class_distribution(ann, classes), out_dir / "eda_class_distribution.png"),
        "box_stats": save(fig_box_stats(ann), out_dir / "eda_box_stats.png"),
        "split_balance": save(fig_split_balance(prep["split_summary"]), out_dir / "eda_split_balance.png"),
        "samples": save(fig_samples(ann, imgs, classes), out_dir / "eda_ground_truth_samples.png"),
    }
