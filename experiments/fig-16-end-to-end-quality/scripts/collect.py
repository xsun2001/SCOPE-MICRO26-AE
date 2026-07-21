from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TASK_METRICS = {
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}
METRICS = ["ppl", "group1_mean", *TASK_METRICS]
PUBLISHED_METRICS = {"ppl", "group1_mean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--ppl-tolerance", type=float, default=0.05)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.005)
    return parser.parse_args()


def model_slug(value: str) -> str:
    return value.lower().replace(".", "_").replace("-", "_")


def load_metric(path: Path, kind: str) -> dict[str, float]:
    payload = json.loads(path.read_text())
    if kind == "wikitext":
        return {"ppl": float(payload["ppl"])}
    results = payload["results"]["results"]
    values = {task: float(results[task][metric]) for task, metric in TASK_METRICS.items()}
    values["group1_mean"] = sum(values.values()) / len(values)
    return values


def read_expected(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    expected: dict[tuple[str, str], dict[str, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            expected[(row["model"], row["metric"])] = {
                key: float(value) for key, value in row.items() if key not in {"model", "metric"}
            }
    return expected


def main() -> int:
    args = parse_args()
    manifest = args.experiment_root / "data" / "manifest.tsv"
    expected_path = args.experiment_root / "expected-results" / "paper_figure16.csv"
    rows: list[dict[str, str | float]] = []
    missing: list[str] = []
    with manifest.open() as handle:
        for spec in csv.DictReader(handle, delimiter="\t"):
            metrics_path = (
                args.result_dir
                / spec["kind"]
                / model_slug(spec["model"])
                / spec["variant"]
                / "metrics.json"
            )
            if not metrics_path.is_file():
                missing.append(str(metrics_path.relative_to(args.result_dir)))
                continue
            for metric, value in load_metric(metrics_path, spec["kind"]).items():
                rows.append(
                    {
                        "model": spec["model"],
                        "kind": spec["kind"],
                        "variant": spec["variant"],
                        "metric": metric,
                        "value": value,
                        "metrics_path": str(metrics_path.relative_to(args.result_dir)),
                    }
                )

    out_dir = args.result_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "kind", "variant", "metric", "value", "metrics_path"])
        writer.writeheader()
        writer.writerows(rows)

    index = {(str(row["model"]), str(row["metric"]), str(row["variant"])): float(row["value"]) for row in rows}
    expected = read_expected(expected_path)
    comparisons: list[dict[str, str | float | bool]] = []
    source_diagnostics: list[dict[str, str | float | bool]] = []
    for (model, metric), variants in expected.items():
        for variant, target in variants.items():
            actual = index.get((model, metric, variant))
            tolerance = args.ppl_tolerance if metric == "ppl" else args.accuracy_tolerance
            error = math.nan if actual is None else actual - target
            comparison = {
                "model": model,
                "metric": metric,
                "variant": variant,
                "paper": target,
                "reproduced": "" if actual is None else actual,
                "error": "" if actual is None else error,
                "tolerance": tolerance,
                "pass": actual is not None and abs(error) <= tolerance,
            }
            if metric in PUBLISHED_METRICS:
                comparisons.append(comparison)
            else:
                source_diagnostics.append(comparison)
    comparison_fields = ["model", "metric", "variant", "paper", "reproduced", "error", "tolerance", "pass"]
    with (out_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)
    with (out_dir / "source_task_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(source_diagnostics)

    passed = sum(bool(row["pass"]) for row in comparisons)
    diagnostic_passed = sum(bool(row["pass"]) for row in source_diagnostics)
    summary = {
        "status": "pass" if not missing and passed == len(comparisons) else "fail",
        "comparisons": len(comparisons),
        "passed": passed,
        "missing_metrics": missing,
        "ppl_tolerance": args.ppl_tolerance,
        "accuracy_tolerance": args.accuracy_tolerance,
        "validation_scope": "Figure 16 plotted values: perplexity and four-task mean zero-shot accuracy",
        "source_task_diagnostics": {
            "comparisons": len(source_diagnostics),
            "within_tolerance": diagnostic_passed,
            "note": "Per-task source values are retained for diagnostics but are not individually plotted in Figure 16.",
        },
    }
    (out_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "validation.md").write_text(
        "# Figure 16 reproduction\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"Passed {passed}/{len(comparisons)} plotted paper-value comparisons.\n\n"
        f"Per-task source diagnostics within tolerance: {diagnostic_passed}/{len(source_diagnostics)}.\n\n"
        f"Missing metric files: {len(missing)}.\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
