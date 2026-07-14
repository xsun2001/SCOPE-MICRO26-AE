from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.experiment_root / "expected/paper_table4.csv").open()))
    payload = {
        "status": "not-reproducible",
        "published_rows": len(rows),
        "reason": "Taylor/Frac-T/Interp/Frac-I/LinearLUT/NN-LUT/T-LUT shared-grid harness and raw results are absent from the workspace.",
        "included_code": ["SCNA trainer", "GQA-LUT", "NLI", "NN-LUT", "local Taylor diagnostic"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
