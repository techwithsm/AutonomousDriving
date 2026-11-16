"""Shared matplotlib style: validated categorical palette, thin marks, recessive axes."""
from __future__ import annotations

from pathlib import Path

import sys

import matplotlib

if "ipykernel" not in sys.modules:   # scripts: headless backend; notebooks keep inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light mode), assigned in fixed order - never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRIMARY = PALETTE[0]          # single-series marks
OTHER = "#9a9893"             # folded "Other" / de-emphasised series
SEQUENTIAL_CMAP = "Blues"     # one hue, light -> dark, for magnitude (heatmaps)
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    })


def label_bars(ax, fmt: str = "{:.0f}", horizontal: bool = False, max_labels: int = 25) -> None:
    """Direct value labels at the end of each bar (skipped when there are too many)."""
    patches = [p for p in ax.patches if p.get_width() and p.get_height()]
    if len(patches) > max_labels:
        return
    for p in patches:
        if horizontal:
            val, x, y, ha, va = p.get_width(), p.get_width(), p.get_y() + p.get_height() / 2, "left", "center"
        else:
            val, x, y, ha, va = p.get_height(), p.get_x() + p.get_width() / 2, p.get_height(), "center", "bottom"
        ax.annotate(fmt.format(val), (x, y), xytext=(3, 0) if horizontal else (0, 2),
                    textcoords="offset points", ha=ha, va=va, fontsize=8, color=TEXT_MUTED)


def show(*figs) -> None:
    """Notebook helper: display each figure once, then free it."""
    from IPython.display import display

    for f in figs:
        display(f)
        plt.close(f)


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


apply_style()
