from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads((args.actual / "validation.json").read_text())
    rows = list(csv.DictReader((args.actual / "comparison.csv").open()))
    ok = payload.get("status") == "pass" and len(rows) == 80 and all(row["pass"] == "True" for row in rows)
    print(json.dumps({"status": "pass" if ok else "fail", "comparisons": len(rows)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__": raise SystemExit(main())
