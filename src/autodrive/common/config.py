"""YAML configuration loading with dotted-key overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_root() -> Path:
    """Repository root (the folder that contains ``src/``)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    """Resolve ``p`` relative to ``base`` (default: project root) unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else (base or project_root()) / p


def _coerce(value: str) -> Any:
    """Turn a CLI string into int/float/bool/None/list where it obviously is one."""
    return yaml.safe_load(value)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``key.sub=value`` overrides (e.g. ``train.epochs=5``)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    """Load a YAML config file and apply optional overrides."""
    with open(resolve_path(path), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return apply_overrides(cfg, overrides)
