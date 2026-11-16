"""Part 1 - vehicle detection CLI.

Stages (run individually or all at once):
    setup     Task 1  create parent workspace folder + child folders
    prepare   Task 2  unzip (if needed), clean labels, split, write YOLO dataset
    eda               dataset plots (class balance, box sizes, samples)
    train     Task 3  fine-tune the YOLO11 CNN detector
    evaluate  Task 4  metrics on the held-out test split
    predict   Task 5  inference on sample images + GT comparison + vehicle counts
    all               every stage in order

Examples:
    python scripts/run_part1.py all
    python scripts/run_part1.py train --set train.epochs=5 train.fraction=0.25
    python scripts/run_part1.py predict --set predict.source="path/to/images"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autodrive.common import get_logger, load_config  # noqa: E402
from autodrive.common.plotting import save  # noqa: E402
from autodrive.detection import create_workspace  # noqa: E402
from autodrive.detection.workspace import tree  # noqa: E402

log = get_logger("part1")
STAGES = ["setup", "prepare", "eda", "train", "evaluate", "predict"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default="configs/part1_detection.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--weights", default=None, help="weights for evaluate/predict (default: best.pt)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    stages = STAGES if args.stage == "all" else [args.stage]
    ws = create_workspace(cfg)
    prep = None

    for stage in stages:
        log.info("===== %s =====", stage.upper())
        if stage == "setup":
            print(tree(ws.root))
        elif stage in ("prepare", "eda"):
            if prep is None:
                from autodrive.detection.dataset import prepare_dataset
                prep = prepare_dataset(cfg, ws)
                print(prep["split_summary"].to_string())
            if stage == "eda":
                from autodrive.detection.eda import run_eda
                for name, p in run_eda(prep, cfg["dataset"]["classes"], ws.reports).items():
                    log.info("EDA figure %s -> %s", name, p)
        elif stage == "train":
            from autodrive.detection.evaluate import training_curves
            from autodrive.detection.model import describe_model, train
            model, _ = train(cfg, ws)
            log.info(describe_model(model))
            run_dir = ws.runs / cfg["train"]["run_name"]
            save(training_curves(run_dir), ws.reports / "training_curves.png")
        elif stage == "evaluate":
            from autodrive.detection.evaluate import evaluate
            res = evaluate(cfg, ws, args.weights)
            print(json.dumps(res["overall"], indent=2))
            print(res["per_class"].to_string())
        elif stage == "predict":
            from autodrive.detection.inference import predict
            res = predict(cfg, ws, args.weights)
            print(res["counts"].to_string())
            if len(res["sample_accuracy"]):
                print(res["sample_accuracy"].to_string(index=False))
            log.info("Annotated images -> %s", res["annotated_dir"])


if __name__ == "__main__":
    main()
