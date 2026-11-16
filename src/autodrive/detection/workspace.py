"""Part 1 - Task 1: parent folder for custom model training + child folders for data.

Layout created (names configurable in ``configs/part1_detection.yaml``)::

    workspace/vehicle_detection/          <- parent folder
    |-- raw/                              <- zip extraction target (if images come zipped)
    |-- dataset/                          <- YOLO-format training data
    |   |-- images/{train,val,test}/
    |   |-- labels/{train,val,test}/
    |   `-- data.yaml
    |-- runs/                             <- training/validation runs, weights
    |-- inference/                        <- annotated predictions
    `-- reports/                          <- EDA plots, metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autodrive.common import get_logger, resolve_path

log = get_logger("part1.workspace")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Workspace:
    root: Path
    raw: Path
    dataset: Path
    runs: Path
    inference: Path
    reports: Path

    def images(self, split: str) -> Path:
        return self.dataset / "images" / split

    def labels(self, split: str) -> Path:
        return self.dataset / "labels" / split

    @property
    def data_yaml(self) -> Path:
        return self.dataset / "data.yaml"

    @property
    def split_csv(self) -> Path:
        return self.dataset / "split_manifest.csv"


def get_workspace(cfg: dict) -> Workspace:
    ws = cfg["workspace"]
    root = resolve_path(ws["root"])
    ch = ws["children"]
    return Workspace(root=root, raw=root / ch["raw"], dataset=root / ch["dataset"],
                     runs=root / ch["runs"], inference=root / ch["inference"], reports=root / ch["reports"])


def create_workspace(cfg: dict) -> Workspace:
    """Create the parent folder and all child folders (idempotent)."""
    w = get_workspace(cfg)
    dirs = [w.root, w.raw, w.dataset, w.runs, w.inference, w.reports]
    dirs += [w.images(s) for s in SPLITS] + [w.labels(s) for s in SPLITS]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    log.info("Workspace ready at %s", w.root)
    return w


def tree(path: Path, max_depth: int = 3) -> str:
    """Folder tree (directories only + file counts) for display."""
    lines = [path.name + "/"]

    def walk(p: Path, prefix: str, depth: int):
        if depth >= max_depth:
            return
        subs = sorted(d for d in p.iterdir() if d.is_dir())
        for i, d in enumerate(subs):
            last = i == len(subs) - 1
            n_files = sum(1 for f in d.iterdir() if f.is_file())
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}{d.name}/" + (f"  ({n_files} files)" if n_files else ""))
            walk(d, prefix + ("    " if last else "|   "), depth + 1)

    walk(path, "", 0)
    return "\n".join(lines)
