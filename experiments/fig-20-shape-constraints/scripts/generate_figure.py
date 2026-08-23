from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = BUNDLE_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from merge_shape_fix_results import (  # noqa: E402
    PAPER_PANEL_ORDER,
    plot_convergence_grid,
    register_libertinus_sans,
)


WIDTH = 16


def build_rows(runs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for func in PAPER_PANEL_ORDER:
        exp = json.loads((runs_dir / f"{func}_{WIDTH}_exp" / "summary.json").read_text())
        none = json.loads((runs_dir / f"{func}_{WIDTH}_none" / "summary.json").read_text())
        exp_best_mse = float(exp["best_mse"])
        none_best_mse = float(none["best_mse"])
        rows.append(
            {
                "func": func,
                "num_units": WIDTH,
                "none_over_exp_best_mse": none_best_mse / exp_best_mse,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the paper-style Figure 20 shape-constraint comparison."
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=5_000)
    parser.add_argument("--subplot-box-aspect", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if args.subplot_box_aspect <= 0.0:
        raise ValueError("--subplot-box-aspect must be positive")

    register_libertinus_sans()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence_grid(
        args.runs_dir,
        build_rows(args.runs_dir),
        args.output_dir / "figure20.png",
        "",
        args.max_epochs,
        box_aspect=args.subplot_box_aspect,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
