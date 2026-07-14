from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPERIMENTS = {
    "tbl-4-function-approximation-accuracy": None,
    "fig-16-end-to-end-quality": ("actual/analysis/validation.json", "pass", 80),
    "tbl-5-ostquant-quality": ("actual/analysis/validation.json", "pass", 16),
    "fig-17-neuron-scalability": ("actual/analysis/validation.json", "pass", 36),
    "fig-20-shape-constraints": ("actual/analysis/validation.json", "pass", 18),
}
REQUIRED = ("README.md", "Makefile", "data", "expected", "actual", "scripts", "generated")
TRAINER_SHA256 = "8aa2816d76343b3ae294fbc80da03f51e617c8e78da13c0c9b90ab8237a5010a"
REPOSITORIES = ("end2endacc", "OSTQuant", "train")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    failures = []
    reports = []
    exp_root = root / "experiments"
    repositories = [root / name for name in REPOSITORIES]
    for repository in repositories:
        if not repository.is_dir():
            failures.append(f"missing first-level repository: {repository.name}")
    if (root / "code").exists():
        failures.append("obsolete code wrapper directory exists; repositories must be directly under bundle root")
    names = sorted(path.name for path in exp_root.iterdir() if path.is_dir())
    if names != sorted(EXPERIMENTS):
        failures.append(f"experiment directories differ: {names}")
    for name, validation in EXPERIMENTS.items():
        directory = exp_root / name
        if not re.fullmatch(r"(?:fig|tbl)-\d+-[a-z0-9-]+", name):
            failures.append(f"invalid experiment name: {name}")
        missing = [entry for entry in REQUIRED if not (directory / entry).exists()]
        if missing:
            failures.append(f"{name}: missing {missing}")
        if (directory / "code").exists():
            failures.append(f"{name}: must not contain a private code directory")
        if validation:
            relative, expected_status, expected_count = validation
            payload = json.loads((directory / relative).read_text())
            count = payload.get("comparisons")
            reports.append({"experiment": name, "status": payload.get("status"), "comparisons": count})
            if payload.get("status") != expected_status or count != expected_count:
                failures.append(f"{name}: unexpected validation {payload}")
        else:
            payload = json.loads((directory / "actual/validation.json").read_text())
            reports.append({"experiment": name, "status": payload.get("status"), "published_rows": payload.get("published_rows")})
            if payload.get("status") != "not-reproducible" or payload.get("published_rows") != 11:
                failures.append(f"{name}: unexpected evidence status {payload}")
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


if __name__ == "__main__": raise SystemExit(main())
