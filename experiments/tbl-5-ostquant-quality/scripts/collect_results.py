from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = ("llama2_7b", "llama3_8b")
QUANTS = ("w4a4kv4", "w6a6kv6")
EVAL_MODES = ("exact_eager", "scna_d8", "scna_d16", "scna_d32")
QMODEL_MODE = "qmodel_sdpa"
TASKS = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "lambada_openai",
    "openbookqa",
    "piqa",
    "social_iqa",
    "winogrande",
)
README_TARGETS = {
    ("llama2_7b", "w4a4kv4"): {"ppl": 5.91, "acc_avg": 0.6318},
    ("llama3_8b", "w4a4kv4"): {"ppl": 7.29, "acc_avg": 0.6537},
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}"


def delta(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:+.{digits}f}"


def pct_delta(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:+.2f}"


def row_from_dir(run_root: Path, model: str, quant: str, mode: str, result_dir: Path) -> dict[str, Any]:
    metrics = read_json(result_dir / "metrics.json") or {}
    status = read_json(result_dir / "status.json") or {}
    slurm_env = read_json(result_dir / "slurm_env.json") or {}
    lm_eval = metrics.get("lm_eval", {}).get("final", {})
    task_metrics = lm_eval.get("metrics", {})
    scna = metrics.get("scna", {})
    use_sdpa = slurm_env.get("use_sdpa")
    if isinstance(use_sdpa, str):
        use_sdpa = use_sdpa.lower() == "true"
    attention_path = "sdpa" if mode == "exact_sdpa" or use_sdpa is True else "eager"

    row: dict[str, Any] = {
        "model_key": model,
        "quant_key": quant,
        "mode": mode,
        "attention_path": attention_path,
        "scna": bool(scna.get("enabled", False)),
        "scna_dim": scna.get("dim"),
        "scna_artifact_root": scna.get("artifact_root"),
        "scna_input_quant_bits": scna.get("input_quant_bits"),
        "scna_input_clip_min": scna.get("input_clip_min"),
        "scna_input_scale": scna.get("input_scale"),
        "scna_output_floor_log": scna.get("output_floor_log"),
        "ppl": metrics.get("ppl"),
        "acc_avg": lm_eval.get("acc_avg"),
        "returncode": status.get("returncode"),
        "result_dir": str(result_dir.relative_to(run_root)),
    }
    for task in TASKS:
        row[task] = task_metrics.get(task)
    return row


def build_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for quant in QUANTS:
            seen: set[Path] = set()
            prefix = f"exact_{model}_{quant}_"
            for result_dir in sorted(run_root.glob(prefix + "*")):
                suffix = result_dir.name[len(prefix) :]
                rows.append(row_from_dir(run_root, model, quant, f"exact_{suffix}", result_dir))
                seen.add(result_dir)

            prefix = f"eval_{model}_{quant}_"
            for result_dir in sorted(run_root.glob(prefix + "*")):
                suffix = result_dir.name[len(prefix) :]
                rows.append(row_from_dir(run_root, model, quant, suffix, result_dir))
                seen.add(result_dir)

            prefix = f"train_{model}_{quant}_"
            for result_dir in sorted(run_root.glob(prefix + "*")):
                suffix = result_dir.name[len(prefix) :]
                rows.append(
                    row_from_dir(run_root, model, quant, f"train_{suffix}", result_dir)
                )
                seen.add(result_dir)

    exact_eager = {
        (row["model_key"], row["quant_key"]): row
        for row in rows
        if row["mode"] == "exact_eager"
    }
    exact_sdpa = {
        (row["model_key"], row["quant_key"]): row
        for row in rows
        if row["mode"] == "exact_sdpa"
    }
    for row in rows:
        key = (row["model_key"], row["quant_key"])
        eager_base = exact_eager.get(key, {})
        sdpa_base = exact_sdpa.get(key, {})
        row["ppl_delta_vs_exact_eager"] = (
            row["ppl"] - eager_base["ppl"]
            if row.get("ppl") is not None and eager_base.get("ppl") is not None
            else None
        )
        row["acc_delta_vs_exact_eager"] = (
            row["acc_avg"] - eager_base["acc_avg"]
            if row.get("acc_avg") is not None and eager_base.get("acc_avg") is not None
            else None
        )
        row["ppl_delta_vs_exact_sdpa"] = (
            row["ppl"] - sdpa_base["ppl"]
            if row.get("ppl") is not None and sdpa_base.get("ppl") is not None
            else None
        )
        row["acc_delta_vs_exact_sdpa"] = (
            row["acc_avg"] - sdpa_base["acc_avg"]
            if row.get("acc_avg") is not None and sdpa_base.get("acc_avg") is not None
            else None
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_key",
        "quant_key",
        "mode",
        "attention_path",
        "scna",
        "scna_dim",
        "scna_input_quant_bits",
        "scna_input_clip_min",
        "scna_input_scale",
        "scna_output_floor_log",
        "ppl",
        "ppl_delta_vs_exact_eager",
        "ppl_delta_vs_exact_sdpa",
        "acc_avg",
        "acc_delta_vs_exact_eager",
        "acc_delta_vs_exact_sdpa",
        *TASKS,
        "returncode",
        "scna_artifact_root",
        "result_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def find_row(rows: list[dict[str, Any]], model: str, quant: str, mode: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["model_key"] == model and row["quant_key"] == quant and row["mode"] == mode
        ),
        None,
    )


def write_maskfix_report(run_root: Path, rows: list[dict[str, Any]]) -> bool:
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    corrected_modes = ("exact_eager_maskfix_acc", "scna_d8_maskfix_acc", "scna_d16_maskfix_acc", "scna_d32_maskfix_acc")
    missing = []
    for model in MODELS:
        for quant in QUANTS:
            for mode in ("exact_sdpa", *corrected_modes):
                row = find_row(rows, model, quant, mode)
                if row is None or row.get("returncode") not in (None, 0) or row.get("ppl") is None:
                    missing.append((model, quant, mode, None if row is None else row.get("returncode")))
                elif mode.endswith("_acc") and row.get("acc_avg") is None:
                    missing.append((model, quant, mode, row.get("returncode")))
    if missing:
        return False

    readme_rows = []
    for key, target in README_TARGETS.items():
        row = find_row(rows, key[0], key[1], "exact_sdpa") or {}
        readme_rows.append(
            [
                key[0],
                key[1],
                fmt(row.get("ppl")),
                fmt(target["ppl"]),
                delta(row.get("ppl") - target["ppl"] if row.get("ppl") is not None else None),
                pct(row.get("acc_avg")),
                pct(target["acc_avg"]),
                pct_delta(row.get("acc_avg") - target["acc_avg"] if row.get("acc_avg") is not None else None),
            ]
        )

    baseline_rows = []
    for model in MODELS:
        for quant in QUANTS:
            sdpa = find_row(rows, model, quant, "exact_sdpa") or {}
            eager = find_row(rows, model, quant, "exact_eager_maskfix_acc") or {}
            baseline_rows.append(
                [
                    model,
                    quant,
                    fmt(sdpa.get("ppl")),
                    pct(sdpa.get("acc_avg")),
                    fmt(eager.get("ppl")),
                    delta(
                        eager.get("ppl") - sdpa.get("ppl")
                        if eager.get("ppl") is not None and sdpa.get("ppl") is not None
                        else None
                    ),
                    pct(eager.get("acc_avg")),
                    pct_delta(
                        eager.get("acc_avg") - sdpa.get("acc_avg")
                        if eager.get("acc_avg") is not None and sdpa.get("acc_avg") is not None
                        else None
                    ),
                ]
            )

    scna_rows = []
    for model in MODELS:
        for quant in QUANTS:
            sdpa = find_row(rows, model, quant, "exact_sdpa") or {}
            eager = find_row(rows, model, quant, "exact_eager_maskfix_acc") or {}
            for dim in (8, 16, 32):
                row = find_row(rows, model, quant, f"scna_d{dim}_maskfix_acc") or {}
                scna_rows.append(
                    [
                        model,
                        quant,
                        f"SCNA-{dim}",
                        fmt(row.get("ppl")),
                        delta(
                            row.get("ppl") - eager.get("ppl")
                            if row.get("ppl") is not None and eager.get("ppl") is not None
                            else None
                        ),
                        delta(
                            row.get("ppl") - sdpa.get("ppl")
                            if row.get("ppl") is not None and sdpa.get("ppl") is not None
                            else None
                        ),
                        pct(row.get("acc_avg")),
                        pct_delta(
                            row.get("acc_avg") - eager.get("acc_avg")
                            if row.get("acc_avg") is not None and eager.get("acc_avg") is not None
                            else None
                        ),
                        pct_delta(
                            row.get("acc_avg") - sdpa.get("acc_avg")
                            if row.get("acc_avg") is not None and sdpa.get("acc_avg") is not None
                            else None
                        ),
                        str(row.get("scna_input_quant_bits")),
                        str(row.get("returncode")),
                    ]
                )

    best_rows = []
    for model in MODELS:
        for quant in QUANTS:
            eager = find_row(rows, model, quant, "exact_eager_maskfix_acc") or {}
            candidates = [
                find_row(rows, model, quant, f"scna_d{dim}_maskfix_acc")
                for dim in (8, 16, 32)
            ]
            complete = [row for row in candidates if row and row.get("ppl") is not None]
            best = min(complete, key=lambda row: row["ppl"])
            best_rows.append(
                [
                    model,
                    quant,
                    best["mode"].replace("_maskfix_acc", "").replace("scna_d", "SCNA-"),
                    fmt(best.get("ppl")),
                    delta(
                        best.get("ppl") - eager.get("ppl")
                        if best.get("ppl") is not None and eager.get("ppl") is not None
                        else None
                    ),
                    pct(best.get("acc_avg")),
                    pct_delta(
                        best.get("acc_avg") - eager.get("acc_avg")
                        if best.get("acc_avg") is not None and eager.get("acc_avg") is not None
                        else None
                    ),
                ]
            )

    maskfix_csv_rows = []
    for model in MODELS:
        for quant in QUANTS:
            sdpa = find_row(rows, model, quant, "exact_sdpa") or {}
            eager = find_row(rows, model, quant, "exact_eager_maskfix_acc") or {}
            for mode in ("exact_sdpa", "exact_eager_maskfix_acc", "scna_d8_maskfix_acc", "scna_d16_maskfix_acc", "scna_d32_maskfix_acc"):
                row = find_row(rows, model, quant, mode) or {}
                maskfix_csv_rows.append(
                    {
                        "model_key": model,
                        "quant_key": quant,
                        "mode": mode,
                        "ppl": row.get("ppl"),
                        "ppl_delta_vs_fixed_eager": (
                            row.get("ppl") - eager.get("ppl")
                            if row.get("ppl") is not None and eager.get("ppl") is not None
                            else None
                        ),
                        "ppl_delta_vs_sdpa": (
                            row.get("ppl") - sdpa.get("ppl")
                            if row.get("ppl") is not None and sdpa.get("ppl") is not None
                            else None
                        ),
                        "acc_avg": row.get("acc_avg"),
                        "acc_delta_vs_fixed_eager": (
                            row.get("acc_avg") - eager.get("acc_avg")
                            if row.get("acc_avg") is not None and eager.get("acc_avg") is not None
                            else None
                        ),
                        "acc_delta_vs_sdpa": (
                            row.get("acc_avg") - sdpa.get("acc_avg")
                            if row.get("acc_avg") is not None and sdpa.get("acc_avg") is not None
                            else None
                        ),
                        "scna_input_quant_bits": row.get("scna_input_quant_bits"),
                        "returncode": row.get("returncode"),
                        "result_dir": row.get("result_dir"),
                    }
                )
    with (analysis_dir / "maskfix_summary.csv").open("w", newline="") as f:
        fieldnames = [
            "model_key",
            "quant_key",
            "mode",
            "ppl",
            "ppl_delta_vs_fixed_eager",
            "ppl_delta_vs_sdpa",
            "acc_avg",
            "acc_delta_vs_fixed_eager",
            "acc_delta_vs_sdpa",
            "scna_input_quant_bits",
            "returncode",
            "result_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(maskfix_csv_rows)

    diagnostics_dir = run_root / "diagnostics"
    aq_diag = read_json(diagnostics_dir / "attention_path_compare_llama2_7b_w4a4kv4_seq256_nomask_after_maskfix.json") or {}
    noaq_diag = read_json(diagnostics_dir / "attention_path_compare_llama2_7b_w4a4kv4_seq256_nomask_noaq_after_maskfix.json") or {}
    aq_diff = aq_diag.get("diffs", {}).get("explicit_eager_mask_policy_vs_sdpa", {})
    noaq_diff = noaq_diag.get("diffs", {}).get("explicit_eager_mask_policy_vs_sdpa", {})

    report_parts = [
        "# OSTQuant SCNA Mask-Fix Final Report",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Root Cause",
        "",
        "- The large eager-attention PPL regression was caused by `QuantSoftmax._align_attn_mask` trimming oversized mask dimensions from the end.",
        "- For the LLaMA eager no-user-mask path, the key dimension can be one column longer than the attention scores. LLaMA attention keeps the first `key_len` columns; the old helper dropped column 0 and kept the extra masked column, shifting the causal mask.",
        "- SCNA was evaluated through the explicit attention path, so the same mask bug made SCNA look much worse than it was.",
        "",
        "## README Check",
        "",
        markdown_table(
            ["Model", "Quant", "SDPA PPL", "README PPL", "Delta", "SDPA Acc %", "README Acc %", "Delta %"],
            readme_rows,
        ),
        "",
        "## Exact Attention Sanity",
        "",
        markdown_table(
            ["Model", "Quant", "SDPA PPL", "SDPA Acc %", "Fixed Eager PPL", "Delta PPL", "Fixed Eager Acc %", "Delta Acc %"],
            baseline_rows,
        ),
        "",
        "## Final SCNA Results",
        "",
        "SCNA is full precision in these rows: `scna_input_quant_bits=0`, `scna_input_scale=1.0`. It is not forced to 4-bit or 6-bit.",
        "",
        markdown_table(
            [
                "Model",
                "Quant",
                "Mode",
                "PPL",
                "Delta PPL vs Fixed Eager",
                "Delta PPL vs SDPA",
                "Acc %",
                "Delta Acc % vs Fixed Eager",
                "Delta Acc % vs SDPA",
                "SCNA Input Quant Bits",
                "RC",
            ],
            scna_rows,
        ),
        "",
        "## Best SCNA By PPL",
        "",
        markdown_table(
            ["Model", "Quant", "Best Mode", "PPL", "Delta PPL vs Fixed Eager", "Acc %", "Delta Acc % vs Fixed Eager"],
            best_rows,
        ),
        "",
        "## Diagnostics",
        "",
        f"- Prefix LLaMA2 W4 exact eager was 9.4410 PPL; fixed eager is {fmt((find_row(rows, 'llama2_7b', 'w4a4kv4', 'exact_eager_maskfix_acc') or {}).get('ppl'))} PPL.",
        f"- Prefix LLaMA3 W4 exact eager was 10.2964 PPL; fixed eager is {fmt((find_row(rows, 'llama3_8b', 'w4a4kv4', 'exact_eager_maskfix_acc') or {}).get('ppl'))} PPL.",
        f"- LLaMA2 W4 direct full-forward logits after the mask fix, qmodel loaded, no user mask: activation quant on gives mean diff {fmt(aq_diff.get('mean'))}, max diff {fmt(aq_diff.get('max'))} versus SDPA.",
        f"- Disabling activation quant for that diagnostic reduces the mean diff to {fmt(noaq_diff.get('mean'))}, max diff {fmt(noaq_diff.get('max'))}; dynamic activation quant amplifies small SDPA/manual arithmetic differences.",
        "- The metric path is now the decisive signal: fixed eager matches SDPA PPL closely, and SCNA-16/32 stay close to fixed eager across both models and bit-widths.",
    ]

    report = "\n".join(report_parts) + "\n"
    (analysis_dir / "maskfix_report.md").write_text(report)
    (analysis_dir / "final_report.md").write_text(report)
    return True


def write_reports(run_root: Path, rows: list[dict[str, Any]]) -> None:
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    readme_rows = []
    for key, target in README_TARGETS.items():
        row = next((r for r in rows if (r["model_key"], r["quant_key"]) == key and r["mode"] == "exact_sdpa"), {})
        readme_rows.append(
            [
                key[0],
                key[1],
                fmt(row.get("ppl")),
                fmt(target["ppl"]),
                fmt(row.get("ppl") - target["ppl"] if row.get("ppl") is not None else None),
                pct(row.get("acc_avg")),
                pct(target["acc_avg"]),
                pct(row.get("acc_avg") - target["acc_avg"] if row.get("acc_avg") is not None else None),
            ]
        )

    scna_rows = []
    for row in rows:
        if row["mode"] == "exact_sdpa":
            continue
        scna_rows.append(
            [
                row["model_key"],
                row["quant_key"],
                row["mode"],
                fmt(row.get("ppl")),
                fmt(row.get("ppl_delta_vs_exact_sdpa")),
                fmt(row.get("ppl_delta_vs_exact_eager")),
                pct(row.get("acc_avg")),
                pct(row.get("acc_delta_vs_exact_sdpa")),
                pct(row.get("acc_delta_vs_exact_eager")),
                str(row.get("returncode")),
            ]
        )

    qmodel_rows = []
    for model in MODELS:
        for quant in QUANTS:
            prefix = f"qmodel_{model}_{quant}_"
            for result_dir in sorted(run_root.glob(prefix + "*")):
                suffix = result_dir.name[len(prefix) :]
                metrics = read_json(result_dir / "metrics.json") or {}
                status = read_json(result_dir / "status.json") or {}
                exact = next(
                    (
                        r
                        for r in rows
                        if r["model_key"] == model and r["quant_key"] == quant and r["mode"] == f"exact_{suffix}"
                    ),
                    None,
                )
                if exact is None:
                    exact = next(
                        (
                            r
                            for r in rows
                            if r["model_key"] == model and r["quant_key"] == quant and r["mode"] == "exact_sdpa"
                        ),
                        {},
                    )
                qmodel_rows.append(
                    [
                        model,
                        quant,
                        suffix,
                        fmt(metrics.get("ppl")),
                        fmt(
                            metrics.get("ppl") - exact.get("ppl")
                            if metrics.get("ppl") is not None and exact.get("ppl") is not None
                            else None
                        ),
                        str(status.get("returncode")),
                        str((result_dir / "qmodel.pt").exists()),
                    ]
                )

    missing = [
        row
        for row in rows
        if row.get("returncode") != 0 or row.get("ppl") is None
    ]
    best_scna = {}
    for model in MODELS:
        for quant in QUANTS:
            candidates = [
                row
                for row in rows
                if row["model_key"] == model and row["quant_key"] == quant and row["mode"].startswith("scna") and row.get("ppl") is not None
            ]
            if candidates:
                best_scna[(model, quant)] = min(candidates, key=lambda r: r["ppl"])

    summary_parts = [
        "# OSTQuant SCNA Corrected Protocol Results",
        "",
        f"Run root: `{run_root}`",
        "",
        "## README W4A4KV4 Check",
        "",
        markdown_table(
            ["Model", "Quant", "Measured PPL", "README PPL", "Delta", "Measured Acc %", "README Acc %", "Delta %"],
            readme_rows,
        ),
        "",
        "## Exact Eager and SCNA Resume Evaluation",
        "",
        markdown_table(
            [
                "Model",
                "Quant",
                "Mode",
                "PPL",
                "Delta PPL vs Exact SDPA",
                "Delta PPL vs Exact Eager",
                "Acc %",
                "Delta Acc % vs Exact SDPA",
                "Delta Acc % vs Exact Eager",
                "RC",
            ],
            scna_rows,
        ),
        "",
        "## Qmodel GPTQ Verification",
        "",
        markdown_table(
            ["Model", "Quant", "Suffix", "Qmodel PPL", "Delta PPL vs Matching Exact", "RC", "Qmodel File"],
            qmodel_rows,
        ),
        "",
        "## Best SCNA by PPL",
        "",
    ]
    if best_scna:
        summary_parts.append(
            markdown_table(
                [
                    "Model",
                    "Quant",
                    "Best Mode",
                    "PPL",
                    "Delta PPL vs Exact SDPA",
                    "Delta PPL vs Exact Eager",
                    "Acc %",
                    "Delta Acc % vs Exact SDPA",
                    "Delta Acc % vs Exact Eager",
                ],
                [
                    [
                        model,
                        quant,
                        row["mode"],
                        fmt(row.get("ppl")),
                        fmt(row.get("ppl_delta_vs_exact_sdpa")),
                        fmt(row.get("ppl_delta_vs_exact_eager")),
                        pct(row.get("acc_avg")),
                        pct(row.get("acc_delta_vs_exact_sdpa")),
                        pct(row.get("acc_delta_vs_exact_eager")),
                    ]
                    for (model, quant), row in best_scna.items()
                ],
            )
        )
    else:
        summary_parts.append("No complete SCNA rows yet.")

    summary_parts.extend(
        [
            "",
            "## Notes",
            "",
            "- `exact_sdpa` uses SDPA attention and `gradient_accumulation_steps=8` to match the README script's effective training batch size; direct `torchrun` was not usable in this environment.",
            "- `qmodel_*_sdpa` regenerates GPTQ weights under the SDPA exact path from the learned OST transform and saves `qmodel.pt`.",
            "- `exact_eager` and all SCNA rows load both the `exact_sdpa` learned OST transform and the SDPA-generated `qmodel.pt`, then use explicit attention for nonlinear calculation.",
            "- Additional suffixed rows such as `exact_eager20` or `*_eager20` are investigation probes and are compared against their matching qmodel suffix when present.",
            "- Rows with blank accuracy were intentionally run as PPL-only probes.",
            "- Deltas against `exact_sdpa` are the requested README-aligned baseline deltas; deltas against `exact_eager` isolate SCNA from the explicit-attention fallback.",
        ]
    )
    if missing:
        summary_parts.extend(["", "## Incomplete Or Failed Rows", ""])
        summary_parts.append(
            markdown_table(
                ["Model", "Quant", "Mode", "Return Code", "Result Dir"],
                [
                    [
                        row["model_key"],
                        row["quant_key"],
                        row["mode"],
                        str(row.get("returncode")),
                        row["result_dir"],
                    ]
                    for row in missing
                ],
            )
        )

    report = "\n".join(summary_parts) + "\n"
    (analysis_dir / "summary.md").write_text(report)
    if not write_maskfix_report(run_root, rows):
        (analysis_dir / "final_report.md").write_text(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect corrected OSTQuant SCNA experiment results.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.run_root.resolve())
    analysis_dir = args.run_root / "analysis"
    write_csv(analysis_dir / "summary.csv", rows)
    write_reports(args.run_root.resolve(), rows)
    print(analysis_dir / "summary.csv")
    print(analysis_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
