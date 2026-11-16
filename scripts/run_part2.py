"""Part 2 - Tesla autopilot deaths: inspection, cleaning, EDA, figures and report.

    python scripts/run_part2.py
    python scripts/run_part2.py --set analysis.top_n_states=15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autodrive.common import load_config  # noqa: E402
from autodrive.tesla.pipeline import run  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/part2_tesla.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    res = run(load_config(args.config, args.set))
    print("\nCleaning log:\n" + res["cleaning_log"].to_markdown())
    print(f"\nReport: {res['report']}")


if __name__ == "__main__":
    main()
