"""Part 1 - Task 3: CNN architecture for object detection + training.

Architecture choice: **YOLO11 (Ultralytics), COCO-pretrained, fine-tuned**
-------------------------------------------------------------------------
Why it fits *this* dataset (findings from data verification):

* **Real-time requirement** - the business case is live vehicle tracking/counting from road
  cameras. A one-stage detector (single forward pass) runs at real-time speed even on modest
  hardware; two-stage detectors (Faster R-CNN) are far slower.
* **Small objects** - half the boxes cover <0.8% of the image. YOLO11's multi-scale
  feature pyramid (PAN neck, strides 8/16/32) detects small, distant vehicles well.
* **Limited data** - 5.6k images / 18k boxes is small for training from scratch. COCO
  pretraining already knows car/bus/truck/person/bicycle/motorcycle, so transfer learning
  converges fast and generalises better.
* **Class imbalance** - 67% of boxes are cars. Mosaic/mixup augmentation increases rare-class
  exposure, and the anchor-free decoupled head with DFL/CIoU losses copes better with skewed
  classes than anchor-based SSD.
* **Day/night cameras** - HSV augmentation simulates the lighting changes seen in the data.

Network (all convolutional):
    Backbone : Conv-BN-SiLU stem -> C3k2 CSP blocks -> SPPF -> C2PSA (spatial attention)
    Neck     : PAN-FPN (top-down + bottom-up feature fusion), 3 scales
    Head     : anchor-free, decoupled classification / box-regression branches (DFL)

Variants are swapped through ``model.weights`` (n/s/m/l/x) without code changes.
"""
from __future__ import annotations

from pathlib import Path

from autodrive.common import get_logger, resolve_path, set_seed

from .workspace import Workspace

log = get_logger("part1.model")


def resolve_device(device: str | int | None = "auto") -> str:
    if device not in (None, "auto"):
        return str(device)
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_model(cfg: dict):
    """Instantiate the YOLO detector (pretrained weights are downloaded on first use)."""
    from ultralytics import YOLO

    m = cfg["model"]
    weights = m["weights"]
    if not m.get("pretrained", True) and weights.endswith(".pt"):
        weights = weights.replace(".pt", ".yaml")  # same architecture, random init
    else:
        # Prefer a locally downloaded copy (workspace/weights/<name>) - needed where Python
        # cannot reach GitHub (proxy / SSL inspection). Otherwise Ultralytics downloads it.
        local = resolve_path(cfg["workspace"]["root"]).parent / "weights" / Path(weights).name
        if local.exists():
            weights = str(local)
    model = YOLO(weights)
    log.info("Built model %s", weights)
    return model


def describe_model(model) -> str:
    """Layer/parameter summary for the report/notebook."""
    n_params = sum(p.numel() for p in model.model.parameters())
    n_layers = len(list(model.model.modules()))
    return f"{model.model.__class__.__name__}: {n_params / 1e6:.2f}M parameters, {n_layers} modules"


def train(cfg: dict, ws: Workspace):
    """Fine-tune the detector on the prepared dataset. Returns (model, results)."""
    set_seed(cfg.get("seed", 42))
    t = cfg["train"]
    device = resolve_device(t.get("device", "auto"))
    if device == "cpu":
        log.warning("Training on CPU - expect long epochs. Consider Colab/GPU or train.fraction<1.")
    model = build_model(cfg)
    results = model.train(
        data=str(ws.data_yaml),
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        device=device,
        workers=t.get("workers", 4),
        patience=t.get("patience", 15),
        optimizer=t.get("optimizer", "auto"),
        lr0=t.get("lr0", 0.01),
        momentum=t.get("momentum", 0.937),
        weight_decay=t.get("weight_decay", 0.0005),
        cos_lr=t.get("cos_lr", True),
        close_mosaic=t.get("close_mosaic", 10),
        fraction=t.get("fraction", 1.0),
        cache=t.get("cache", False),
        seed=cfg.get("seed", 42),
        deterministic=True,
        project=str(ws.runs),
        name=t["run_name"],
        exist_ok=True,
        plots=True,
        **t.get("augment", {}),
    )
    log.info("Training done. Best weights: %s", best_weights(cfg, ws))
    return model, results


def best_weights(cfg: dict, ws: Workspace) -> Path:
    return ws.runs / cfg["train"]["run_name"] / "weights" / "best.pt"
