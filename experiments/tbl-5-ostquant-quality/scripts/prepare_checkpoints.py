from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def hardlink_or_copy(source: str, target: str) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    for model in ("llama2_7b", "llama3_8b"):
        for quant in ("w4a4kv4", "w6a6kv6"):
            for prefix, artifact in (("exact", "model.bin"), ("qmodel", "qmodel.pt")):
                name = f"{prefix}_{model}_{quant}_sdpa"
                source = args.source / name
                if not (source / artifact).is_file():
                    raise SystemExit(f"Missing {source / artifact}")
                target = args.result_dir / name
                if not target.exists():
                    shutil.copytree(source, target, copy_function=hardlink_or_copy)
    print(f"Prepared cached checkpoints in {args.result_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
