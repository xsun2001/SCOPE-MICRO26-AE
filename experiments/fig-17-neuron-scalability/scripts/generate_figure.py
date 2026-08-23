from __future__ import annotations

import argparse
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = BUNDLE_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from merge_shape_fix_results import (  # noqa: E402
    PAPER_PANEL_ORDER,
    build_scna_width_rows,
    plot_scna_width_grid,
    register_libertinus_sans,
)


WIDTHS = [4, 8, 16, 32]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the paper-style Figure 17 SCNA width comparison."
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=10_000)
    parser.add_argument("--subplot-box-aspect", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if args.subplot_box_aspect <= 0.0:
        raise ValueError("--subplot-box-aspect must be positive")

    register_libertinus_sans()
    rows = build_scna_width_rows(args.runs_dir, PAPER_PANEL_ORDER, WIDTHS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_scna_width_grid(
        args.runs_dir,
        rows,
        args.output_dir / "figure17.png",
        args.max_epochs,
        box_aspect=args.subplot_box_aspect,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
