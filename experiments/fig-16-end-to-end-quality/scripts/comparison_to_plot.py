from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.comparison.open() as handle:
        values = {(row["model"], row["metric"], row["variant"]): row["reproduced"] for row in csv.DictReader(handle)}
    with args.template.open() as handle:
        template = list(csv.DictReader(handle))
        fields = list(template[0])
    template = [
        row for row in template
        if (row["model"], row["metric"], fields[2]) in values
    ]
    for row in template:
        for variant in fields[2:]:
            row[variant] = values[(row["model"], row["metric"], variant)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(template)
    print(args.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
