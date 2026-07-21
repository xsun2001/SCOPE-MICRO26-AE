from __future__ import annotations

import argparse
import json
from pathlib import Path

from table5_metrics import FOUR_TASKS, officialize_final_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive official Table 5 four-task averages from retained per-task metrics."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    updated: list[str] = []
    for path in sorted(args.run_root.glob("*/metrics.json")):
        payload = json.loads(path.read_text())
        lm_eval = payload.get("lm_eval", {})
        final = lm_eval.get("final")
        if not isinstance(final, dict):
            continue
        metrics = final.get("metrics", {})
        if not all(task in metrics for task in FOUR_TASKS):
            continue
        lm_eval["final"] = officialize_final_metrics(final)
        payload["lm_eval"] = lm_eval
        path.write_text(json.dumps(payload, indent=2) + "\n")
        updated.append(path.parent.name)

    print(json.dumps({"status": "pass", "updated": len(updated), "result_dirs": updated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
