from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


ABSOLUTE_AUTHOR_PATH = re.compile(r"(?:/home/[^/\s\"'`,;)]+|/data/user/[^/\s\"'`,;)]+|/Users/[^/\s\"'`,;)]+|[A-Za-z]:\\\\Users\\\\[^\\\s\"'`,;)]+)")
TEXT_SUFFIXES = {".csv", ".env", ".json", ".log", ".md", ".mk", ".py", ".scala", ".sh", ".tsv", ".txt"}
SKIP_PARTS = {".git", ".venv", "target", "__pycache__", "actual-results"}
SKIP_ROOT_DIRECTORIES = {"models", "runs"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject absolute user-home paths in packaged text provenance.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    findings: list[dict[str, object]] = []
    scanned = 0

    for directory, directory_names, file_names in os.walk(root):
        current = Path(directory)
        directory_names[:] = [
            name
            for name in directory_names
            if name not in SKIP_PARTS
            and not (current == root and name in SKIP_ROOT_DIRECTORIES)
        ]
        for file_name in file_names:
            path = Path(directory) / file_name
            if not path.is_file() or path.resolve() == Path(__file__).resolve():
                continue
            if path.name != "Makefile" and path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.stat().st_size > 20_000_000:
                continue
            scanned += 1
            try:
                lines = path.read_text(errors="strict").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                match = ABSOLUTE_AUTHOR_PATH.search(line)
                if match:
                    findings.append(
                        {
                            "file": str(path.relative_to(root)),
                            "line": line_number,
                            "path_prefix": match.group(0),
                        }
                    )

    payload = {
        "status": "fail" if findings else "pass",
        "scanned_text_files": scanned,
        "absolute_author_paths": len(findings),
        "findings": findings[:50],
    }
    print(json.dumps(payload, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
