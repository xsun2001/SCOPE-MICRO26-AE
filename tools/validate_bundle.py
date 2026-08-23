from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


VALIDATED_EXPERIMENTS = {
    "fig-16-end-to-end-quality": 80,
    "tbl-5-ostquant-quality": 20,
    "fig-17-neuron-scalability": 36,
    "fig-20-shape-constraints": 18,
}
EXPERIMENTS = ("tbl-4-function-approximation-accuracy", *VALIDATED_EXPERIMENTS)
REQUIRED = ("README.md", "Makefile", "data", "expected-results", "actual-results", "scripts")
REPOSITORIES = ("end2endacc", "OSTQuant", "train")


def validator_command(directory: Path, temporary_root: Path) -> list[str]:
    if directory.name == "fig-16-end-to-end-quality":
        return [
            sys.executable,
            str(directory / "scripts" / "validate_reference.py"),
            "--expected",
            str(directory / "expected-results" / "paper_figure16.csv"),
        ]
    if directory.name == "tbl-5-ostquant-quality":
        return [
            sys.executable,
            str(directory / "scripts" / "validate.py"),
            "--summary",
            str(directory / "expected-results" / "maskfix_summary.csv"),
            "--output-dir",
            str(temporary_root / directory.name),
            "--expected",
            str(directory / "expected-results" / "four_task_table5.csv"),
            "--bf16-source",
            str(
                directory.parent
                / "fig-16-end-to-end-quality"
                / "expected-results"
                / "paper_figure16.csv"
            ),
        ]
    figure_number = "17" if directory.name == "fig-17-neuron-scalability" else "20"
    mode = "width" if figure_number == "17" else "shape"
    return [
        sys.executable,
        str(directory.parents[1] / "tools" / "validate_function_reference.py"),
        "--manifest",
        str(directory / "data" / "manifest.tsv"),
        "--expected",
        str(directory / "expected-results" / "paired_summary.csv"),
        "--analysis-dir",
        str(temporary_root / directory.name),
        "--mode",
        mode,
        "--figure",
        str(directory / "expected-results" / f"figure{figure_number}.png"),
        "--figure",
        str(directory / "expected-results" / f"figure{figure_number}.pdf"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    failures: list[str] = []
    reports: list[dict[str, object]] = []
    exp_root = root / "experiments"
    repositories = [root / name for name in REPOSITORIES]

    for repository in repositories:
        if not repository.is_dir():
            failures.append(f"missing first-level repository: {repository.name}")
    if (root / "code").exists():
        failures.append("obsolete code wrapper directory exists; repositories must be directly under bundle root")

    names = sorted(path.name for path in exp_root.iterdir() if path.is_dir())
    missing_experiments = sorted(set(EXPERIMENTS) - set(names))
    if missing_experiments:
        failures.append(f"missing GPU experiment directories: {missing_experiments}")

    for name in EXPERIMENTS:
        directory = exp_root / name
        if not re.fullmatch(r"(?:fig|tbl)-\d+-[a-z0-9-]+", name):
            failures.append(f"invalid experiment name: {name}")
        missing = [entry for entry in REQUIRED if not (directory / entry).exists()]
        if missing:
            failures.append(f"{name}: missing {missing}")
        if (directory / "code").exists():
            failures.append(f"{name}: must not contain a private code directory")

    with tempfile.TemporaryDirectory(prefix="scope-ae-validation-") as temporary:
        temporary_root = Path(temporary)
        table4_dir = exp_root / "tbl-4-function-approximation-accuracy"
        table4_parameters = table4_dir / "data" / "scna_parameters.json"
        table4_variants = (
            ("scna16", "paper_table4.csv", "361", "14.9"),
            ("scna32", "scna32_reference_table4.csv", "836", "31.5"),
        )
        table4_results: dict[str, dict[str, object]] = {}
        for variant, expected_name, nnlut_geomean, tlut_geomean in table4_variants:
            table4_output = temporary_root / "table4" / variant
            reproduce_command = [
                sys.executable,
                str(table4_dir / "scripts" / "reproduce_scna.py"),
                "--parameters",
                str(table4_parameters),
                "--output-dir",
                str(table4_output),
            ]
            if variant != "scna16":
                reproduce_command.extend(["--variant", variant])
            reproduce = subprocess.run(
                reproduce_command,
                cwd=table4_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            table4_validation = table4_output / "validation.json"
            audit = subprocess.run(
                [
                    sys.executable,
                    str(table4_dir / "scripts" / "audit.py"),
                    "--generated",
                    str(table4_output),
                    "--expected",
                    str(table4_dir / "expected-results" / expected_name),
                    "--parameters",
                    str(table4_parameters),
                    "--variant",
                    variant,
                    "--expected-nnlut-geomean",
                    nnlut_geomean,
                    "--expected-tlut-geomean",
                    tlut_geomean,
                    "--output",
                    str(table4_validation),
                ],
                cwd=table4_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            try:
                table4 = json.loads(audit.stdout)
            except json.JSONDecodeError:
                table4 = {"status": "invalid-output", "published_rows": None}
            table4_results[variant] = table4
            if reproduce.returncode != 0 or audit.returncode != 0:
                failures.append(
                    f"tbl-4-function-approximation-accuracy ({variant}): "
                    "independent execution/validation failed: "
                    + str(table4.get("failures", []))
                )
            if (
                table4.get("status") != "pass"
                or table4.get("published_rows") != 11
                or table4.get("comparisons") != 22
            ):
                failures.append(
                    f"tbl-4-function-approximation-accuracy ({variant}): "
                    f"unexpected validation status {table4}"
                )
        reports.append(
            {
                "experiment": "tbl-4-function-approximation-accuracy",
                "status": (
                    "pass"
                    if all(result.get("status") == "pass" for result in table4_results.values())
                    else "fail"
                ),
                "variants": {
                    variant: {
                        "status": result.get("status"),
                        "published_rows": result.get("published_rows"),
                        "comparisons": result.get("comparisons"),
                    }
                    for variant, result in table4_results.items()
                },
            }
        )

        for name, expected_count in VALIDATED_EXPERIMENTS.items():
            directory = exp_root / name
            completed = subprocess.run(
                validator_command(directory, temporary_root),
                cwd=directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                payload = {"status": "invalid-output", "comparisons": None}
                failures.append(
                    f"{name}: validator emitted invalid JSON: "
                    f"{completed.stdout[-400:] or completed.stderr[-400:]}"
                )
            reports.append(
                {
                    "experiment": name,
                    "status": payload.get("status"),
                    "comparisons": payload.get("comparisons"),
                }
            )
            if completed.returncode != 0:
                failures.append(f"{name}: evidence validator failed: {payload.get('failures', [])}")
            if payload.get("status") != "pass" or payload.get("comparisons") != expected_count:
                failures.append(f"{name}: unexpected recomputed validation {payload}")

    trainer = root / "train/train.py"
    if not trainer.is_file():
        failures.append("missing shared function trainer: train/train.py")

    for tree in (*repositories, exp_root):
        for path in tree.rglob("*"):
            if path.is_symlink():
                resolved = path.resolve()
                if root not in (resolved, *resolved.parents):
                    failures.append(f"escaping symlink: {path} -> {resolved}")

    payload = {"status": "fail" if failures else "pass", "experiments": reports, "failures": failures}
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
