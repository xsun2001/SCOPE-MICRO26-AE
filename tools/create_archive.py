from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


EXCLUDED_TOP_LEVEL = {".git", ".venv", "cache", "runs", "models", "figure16", "table5"}
EXCLUDED_ANYWHERE = {"__pycache__", "target", ".pytest_cache", ".mypy_cache"}


def excluded(relative: Path) -> bool:
    if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    if relative == Path("config/local.env"):
        return True
    return any(part in EXCLUDED_ANYWHERE for part in relative.parts) or relative.suffix == ".pyc"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if excluded(relative):
                continue
            archive.add(path, arcname=Path(root.name) / relative, recursive=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n")
    print(f"archive={output}")
    print(f"sha256={digest}")
    print(f"sidecar={sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
