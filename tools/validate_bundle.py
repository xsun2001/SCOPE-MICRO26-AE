from __future__ import annotations

import argparse
import hashlib
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
TRAINER_SHA256 = "8aa2816d76343b3ae294fbc80da03f51e617c8e78da13c0c9b90ab8237a5010a"
REPOSITORIES = ("end2endacc", "OSTQuant", "train")
PACKAGED_RUN = "2026-07-13_ae-validation"


def validator_command(directory: Path, temporary_root: Path) -> list[str]:
    packaged = directory / "actual-results" / PACKAGED_RUN
    if directory.name == "fig-16-end-to-end-quality":
        return [
            sys.executable,
            str(directory / "scripts" / "validate_actual.py"),
            "--actual",
            str(packaged / "analysis"),
        ]
    if directory.name == "tbl-5-ostquant-quality":
        return [
            sys.executable,
            str(directory / "scripts" / "validate_actual.py"),
            "--actual",
            str(packaged / "analysis"),
            "--expected",
            str(directory / "expected-results" / "four_task_table5.csv"),
            "--fp16-source",
            str(
                directory.parent
                / "fig-16-end-to-end-quality"
                / "actual-results"
                / PACKAGED_RUN
                / "generated"
                / "actual_plot.csv"
            ),
        ]
    return [
        sys.executable,
        str(directory / "scripts" / "collect.py"),
        "--runs-dir",
        str(packaged / "runs"),
        "--analysis-dir",
        str(temporary_root / directory.name),
        "--expected",
        str(directory / "expected-results" / "paired_summary.csv"),
        "--relative-tolerance",
        "0.15",
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

    table4_path = exp_root / "tbl-4-function-approximation-accuracy" / "actual-results" / PACKAGED_RUN / "validation.json"
    table4 = json.loads(table4_path.read_text())
    reports.append(
        {
            "experiment": "tbl-4-function-approximation-accuracy",
            "status": table4.get("status"),
            "published_rows": table4.get("published_rows"),
        }
    )
    if (
        table4.get("status") != "pass"
        or table4.get("published_rows") != 11
        or table4.get("reproduced_method") != "SCNA"
    ):
        failures.append(f"tbl-4-function-approximation-accuracy: unexpected validation status {table4}")

    with tempfile.TemporaryDirectory(prefix="scope-ae-validation-") as temporary:
        temporary_root = Path(temporary)
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
                failures.append(f"{name}: raw-evidence validator failed: {payload.get('failures', [])}")
            if payload.get("status") != "pass" or payload.get("comparisons") != expected_count:
                failures.append(f"{name}: unexpected recomputed validation {payload}")

    trainer = root / "train/train.py"
    if trainer.is_file():
        digest = hashlib.sha256(trainer.read_bytes()).hexdigest()
        if digest != TRAINER_SHA256:
            failures.append(f"shared function trainer hash {digest}")
    else:
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
