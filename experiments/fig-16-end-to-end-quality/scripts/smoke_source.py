from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = EXPERIMENT_ROOT.parents[1]
END2ENDACC_ROOT = BUNDLE_ROOT / "end2endacc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test one Figure 16 row from an unpacked source tree.")
    parser.add_argument("--require-unknown-provenance", action="store_true")
    return parser.parse_args()


def smoke_quant_core() -> None:
    sys.path.insert(0, str(END2ENDACC_ROOT))
    import torch
    from torch import nn

    from PINNacle.quantization import model_quant_wrapper  # noqa: F401
    from PINNacle.quantization.quant_linear import BackboneQuantLinear
    from quant.core.observers import VectorAbsMaxObserver

    observer = VectorAbsMaxObserver()
    observer.update(torch.tensor([1.0, -2.0]))
    observer.update(torch.tensor([-3.0, 1.0]))
    if observer.to_list() != [3.0, 2.0]:
        raise AssertionError(f"unexpected observer state: {observer.to_list()}")

    torch.manual_seed(0)
    source = nn.Linear(4, 3, bias=True)
    quantized = BackboneQuantLinear.from_linear(
        source,
        weight_bits=8,
        activation_bits=8,
        weight_scheme="per_channel",
        activation_scheme="per_token",
        activation_quant_mode="dynamic",
    )
    output = quantized(torch.randn(2, 4))
    if output.shape != (2, 3) or not torch.isfinite(output).all():
        raise AssertionError("archived quant/core smoke calculation failed")


def smoke_row(require_unknown_provenance: bool) -> dict[str, object]:
    config = EXPERIMENT_ROOT / "data/configs/wikitext/facebook_opt_6_7b_fp16_exact.json"
    with tempfile.TemporaryDirectory(prefix="scope-fig16-smoke-") as temporary:
        output_dir = Path(temporary) / "row-output"
        env = os.environ.copy()
        env.update(
            {
                "BUNDLE_ROOT": str(BUNDLE_ROOT),
                "END2ENDACC_DIRECT": "1",
                # Exercise config and wrapper plumbing without downloading model weights.
                "PYTHON_BIN": "/bin/true",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(EXPERIMENT_ROOT / "scripts/end2endacc_runner.py"),
                "--kind",
                "wikitext",
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
            ],
            cwd=EXPERIMENT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Figure 16 row smoke failed (rc={completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        required = ["experiment_config.json", "command.txt", "git_commit.txt"]
        missing = [name for name in required if not (output_dir / name).is_file()]
        if missing:
            raise AssertionError(f"Figure 16 row smoke omitted outputs: {missing}")
        provenance = (output_dir / "git_commit.txt").read_text().strip()
        if require_unknown_provenance and not provenance.startswith("unknown"):
            raise AssertionError(f"source archive provenance should be unknown, got {provenance!r}")
        return {
            "config": str(config.relative_to(BUNDLE_ROOT)),
            "outputs": required,
            "provenance": provenance,
        }


def main() -> int:
    args = parse_args()
    smoke_quant_core()
    row = smoke_row(args.require_unknown_provenance)
    print(json.dumps({"status": "pass", "quant_core": "pass", "row": row}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
