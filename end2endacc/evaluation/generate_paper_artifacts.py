from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
PPL_ROOT = EXPERIMENTS / "end2endacc_llama2_7b_int8_ppl" / "results"
LMEVAL_ROOT = EXPERIMENTS / "end2endacc_llama2_7b_int8_lm_eval" / "results"

PPL_RUNS = {
    "exact_wikitext2": PPL_ROOT / "2026-04-02_033500_exact_wikitext2_raw",
    "exact_wikitext103": PPL_ROOT / "2026-04-02_033800_exact_wikitext103_raw",
    "raw_w8a16": PPL_ROOT / "2026-04-02_034300_w8a16_exact_nonlinear",
    "raw_w16a8": PPL_ROOT / "2026-04-02_034450_w16a8_exact_nonlinear",
    "raw_w8a8": PPL_ROOT / "2026-04-02_034600_w8a8_exact_nonlinear",
    "raw_w8a8_per_tensor": PPL_ROOT / "2026-04-02_034750_w8a8_per_tensor_weight",
    "raw_w8a8_calib512": PPL_ROOT / "2026-04-02_035000_w8a8_calib512",
    "raw_w8a8_calib1024": PPL_ROOT / "2026-04-02_035200_w8a8_calib1024",
    "pinn_exact_dim8": PPL_ROOT / "2026-04-02_035755_pinn_exact_dim8",
    "pinn_exact_dim16": PPL_ROOT / "2026-04-02_035755_pinn_exact_dim16",
    "pinn_exact_dim32": PPL_ROOT / "2026-04-02_035755_pinn_exact_dim32",
    "pinn_wq_dim8_w8": PPL_ROOT / "2026-04-02_040648_pinn_wq_dim8_w8",
    "pinn_wq_dim16_w8": PPL_ROOT / "2026-04-02_040648_pinn_wq_dim16_w8",
    "pinn_wq_dim32_w8": PPL_ROOT / "2026-04-02_040648_pinn_wq_dim32_w8",
    "pinn_wq_dim8_w6": PPL_ROOT / "2026-04-02_040754_pinn_wq_dim8_w6",
    "pinn_wq_dim16_w6": PPL_ROOT / "2026-04-02_040754_pinn_wq_dim16_w6",
    "pinn_wq_dim32_w6": PPL_ROOT / "2026-04-02_040754_pinn_wq_dim32_w6",
    "pinn_wq_dim8_w4": PPL_ROOT / "2026-04-02_040811_pinn_wq_dim8_w4",
    "pinn_wq_dim16_w4": PPL_ROOT / "2026-04-02_040811_pinn_wq_dim16_w4",
    "pinn_wq_dim32_w4": PPL_ROOT / "2026-04-02_040811_pinn_wq_dim32_w4",
    "pinn_aq_a8_128": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a8_calib128",
    "pinn_aq_a8_512": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a8_calib512",
    "pinn_aq_a8_1024": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a8_calib1024",
    "pinn_aq_a6_128": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a6_calib128",
    "pinn_aq_a6_512": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a6_calib512",
    "pinn_aq_a6_1024": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a6_calib1024",
    "pinn_aq_a4_128": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a4_calib128",
    "pinn_aq_a4_512": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a4_calib512",
    "pinn_aq_a4_1024": PPL_ROOT / "2026-04-02_042406_pinn_aq_dim32_a4_calib1024",
    "pinn_waq_a8_128": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a8_calib128",
    "pinn_waq_a8_512": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a8_calib512",
    "pinn_waq_a8_1024": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a8_calib1024",
    "pinn_waq_a6_128": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a6_calib128",
    "pinn_waq_a6_512": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a6_calib512",
    "pinn_waq_a6_1024": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a6_calib1024",
    "pinn_waq_a4_128": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a4_calib128",
    "pinn_waq_a4_512": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a4_calib512",
    "pinn_waq_a4_1024": PPL_ROOT / "2026-04-02_042406_pinn_waq_dim32_a4_calib1024",
    "w8a8_pinn_exact": PPL_ROOT / "2026-04-02_050623_w8a8_pinn_exact_dim32",
    "w8a8_pinn_wq": PPL_ROOT / "2026-04-02_050623_w8a8_pinn_wq_dim32_w8",
    "w8a8_pinn_waq": PPL_ROOT / "2026-04-02_050623_w8a8_pinn_waq_dim32_w8_a8_calib128",
}

LMEVAL_RUNS = {
    "exact_group1": LMEVAL_ROOT / "2026-04-02_051615_exact_group1",
    "raw_w8a8_group1": LMEVAL_ROOT / "2026-04-02_051615_raw_w8a8_group1",
    "pinn_exact_group1": LMEVAL_ROOT / "2026-04-02_051615_pinn_exact_dim32_group1",
}


def load_ppl(name: str) -> float:
    with open(PPL_RUNS[name] / "metrics.json") as f:
        return float(json.load(f)["ppl"])


def load_lmeval(name: str) -> dict:
    with open(LMEVAL_RUNS[name] / "metrics.json") as f:
        return json.load(f)["results"]["results"]


def fmt(value: float) -> str:
    return f"{value:.6f}"


def make_markdown() -> str:
    exact = load_ppl("exact_wikitext2")
    raw_w8a8 = load_ppl("raw_w8a8")
    pinn_exact = load_ppl("pinn_exact_dim32")
    pinn_wq = load_ppl("pinn_wq_dim32_w8")
    pinn_aq = load_ppl("pinn_aq_a8_128")
    pinn_waq = load_ppl("pinn_waq_a8_128")
    w8a8_pinn_exact = load_ppl("w8a8_pinn_exact")
    w8a8_pinn_wq = load_ppl("w8a8_pinn_wq")
    w8a8_pinn_waq = load_ppl("w8a8_pinn_waq")

    exact_group1 = load_lmeval("exact_group1")
    raw_group1 = load_lmeval("raw_w8a8_group1")
    pinn_group1 = load_lmeval("pinn_exact_group1")

    lines = [
        "# End2EndAcc Paper Artifacts",
        "",
        "Generated from committed experiment directories on 2026-04-02.",
        "",
        "## Trustworthy Exact Baseline PPL Table",
        "",
        "| Configuration | Dataset | PPL |",
        "| --- | --- | ---: |",
        f"| Exact BF16 | WikiText2 raw | {fmt(load_ppl('exact_wikitext2'))} |",
        f"| Exact BF16 | WikiText103 raw | {fmt(load_ppl('exact_wikitext103'))} |",
        "",
        "## Raw Backbone Quantization PPL Table",
        "",
        "| Configuration | PPL | Delta vs exact |",
        "| --- | ---: | ---: |",
        f"| W8A16 exact nonlinear | {fmt(load_ppl('raw_w8a16'))} | {fmt(load_ppl('raw_w8a16') - exact)} |",
        f"| W16A8 exact nonlinear | {fmt(load_ppl('raw_w16a8'))} | {fmt(load_ppl('raw_w16a8') - exact)} |",
        f"| W8A8 exact nonlinear | {fmt(raw_w8a8)} | {fmt(raw_w8a8 - exact)} |",
        f"| W8A8 per-tensor weights | {fmt(load_ppl('raw_w8a8_per_tensor'))} | {fmt(load_ppl('raw_w8a8_per_tensor') - exact)} |",
        f"| W8A8 calib512 | {fmt(load_ppl('raw_w8a8_calib512'))} | {fmt(load_ppl('raw_w8a8_calib512') - exact)} |",
        f"| W8A8 calib1024 | {fmt(load_ppl('raw_w8a8_calib1024'))} | {fmt(load_ppl('raw_w8a8_calib1024') - exact)} |",
        "",
        "## Protocol-Matched Reference Comparison Table",
        "",
        "See [experiments/end2endacc_reference_table.md](./end2endacc_reference_table.md).",
        "",
        "## Approximation-Only Ablation Table",
        "",
        "| PINN dim | PPL | Delta vs exact |",
        "| --- | ---: | ---: |",
        f"| 8 | {fmt(load_ppl('pinn_exact_dim8'))} | {fmt(load_ppl('pinn_exact_dim8') - exact)} |",
        f"| 16 | {fmt(load_ppl('pinn_exact_dim16'))} | {fmt(load_ppl('pinn_exact_dim16') - exact)} |",
        f"| 32 | {fmt(pinn_exact)} | {fmt(pinn_exact - exact)} |",
        "",
        "## PINN Weight Quantization Ablation Table",
        "",
        "| PINN dim | w_bits | PPL | Delta vs exact-PINN baseline |",
        "| --- | ---: | ---: | ---: |",
    ]

    phase_c = {8: load_ppl("pinn_exact_dim8"), 16: load_ppl("pinn_exact_dim16"), 32: load_ppl("pinn_exact_dim32")}
    for dim, bits, key in [
        (8, 8, "pinn_wq_dim8_w8"),
        (16, 8, "pinn_wq_dim16_w8"),
        (32, 8, "pinn_wq_dim32_w8"),
        (8, 6, "pinn_wq_dim8_w6"),
        (16, 6, "pinn_wq_dim16_w6"),
        (32, 6, "pinn_wq_dim32_w6"),
        (8, 4, "pinn_wq_dim8_w4"),
        (16, 4, "pinn_wq_dim16_w4"),
        (32, 4, "pinn_wq_dim32_w4"),
    ]:
        ppl = load_ppl(key)
        lines.append(f"| {dim} | {bits} | {fmt(ppl)} | {fmt(ppl - phase_c[dim])} |")

    lines.extend(
        [
            "",
            "## PINN Activation Quantization Ablation Table",
            "",
            "| Branch | a_bits | calib samples | PPL | Delta vs seed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for branch, seed, entries in [
        (
            "activation-only",
            pinn_exact,
            [
                (8, 128, "pinn_aq_a8_128"),
                (8, 512, "pinn_aq_a8_512"),
                (8, 1024, "pinn_aq_a8_1024"),
                (6, 128, "pinn_aq_a6_128"),
                (6, 512, "pinn_aq_a6_512"),
                (6, 1024, "pinn_aq_a6_1024"),
                (4, 128, "pinn_aq_a4_128"),
                (4, 512, "pinn_aq_a4_512"),
                (4, 1024, "pinn_aq_a4_1024"),
            ],
        ),
        (
            "weight+activation",
            pinn_wq,
            [
                (8, 128, "pinn_waq_a8_128"),
                (8, 512, "pinn_waq_a8_512"),
                (8, 1024, "pinn_waq_a8_1024"),
                (6, 128, "pinn_waq_a6_128"),
                (6, 512, "pinn_waq_a6_512"),
                (6, 1024, "pinn_waq_a6_1024"),
                (4, 128, "pinn_waq_a4_128"),
                (4, 512, "pinn_waq_a4_512"),
                (4, 1024, "pinn_waq_a4_1024"),
            ],
        ),
    ]:
        for a_bits, calib, key in entries:
            ppl = load_ppl(key)
            lines.append(f"| {branch} | {a_bits} | {calib} | {fmt(ppl)} | {fmt(ppl - seed)} |")

    lines.extend(
        [
            "",
            "## LM-Eval Group1 Table",
            "",
            "| Configuration | arc_easy acc_norm | hellaswag acc_norm | piqa acc_norm | winogrande acc |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Exact | {fmt(exact_group1['arc_easy']['acc_norm,none'])} | {fmt(exact_group1['hellaswag']['acc_norm,none'])} | {fmt(exact_group1['piqa']['acc_norm,none'])} | {fmt(exact_group1['winogrande']['acc,none'])} |",
            f"| Raw W8A8 | {fmt(raw_group1['arc_easy']['acc_norm,none'])} | {fmt(raw_group1['hellaswag']['acc_norm,none'])} | {fmt(raw_group1['piqa']['acc_norm,none'])} | {fmt(raw_group1['winogrande']['acc,none'])} |",
            f"| PINN exact dim32 | {fmt(pinn_group1['arc_easy']['acc_norm,none'])} | {fmt(pinn_group1['hellaswag']['acc_norm,none'])} | {fmt(pinn_group1['piqa']['acc_norm,none'])} | {fmt(pinn_group1['winogrande']['acc,none'])} |",
            "",
            "## Combined Decomposition Data",
            "",
            f"- Exact baseline PPL: `{fmt(exact)}`",
            f"- Raw W8A8 PPL: `{fmt(raw_w8a8)}`",
            f"- W8A8 + PINN exact PPL: `{fmt(w8a8_pinn_exact)}`",
            f"- W8A8 + PINN w8 PPL: `{fmt(w8a8_pinn_wq)}`",
            f"- W8A8 + PINN w8+a8 PPL: `{fmt(w8a8_pinn_waq)}`",
            "",
            f"Decomposition figure: [experiments/end2endacc_combined_decomposition.png](./end2endacc_combined_decomposition.png)",
            "",
        ]
    )
    return "\n".join(lines)


def write_decomposition_csv() -> Path:
    exact = load_ppl("exact_wikitext2")
    raw_w8a8 = load_ppl("raw_w8a8")
    pinn_exact = load_ppl("w8a8_pinn_exact")
    pinn_wq = load_ppl("w8a8_pinn_wq")
    pinn_waq = load_ppl("w8a8_pinn_waq")
    csv_path = EXPERIMENTS / "end2endacc_combined_decomposition.csv"
    rows = [
        ("Exact BF16", exact, 0.0),
        ("Raw W8A8", raw_w8a8, raw_w8a8 - exact),
        ("W8A8 + PINN exact", pinn_exact, pinn_exact - exact),
        ("W8A8 + PINN w8", pinn_wq, pinn_wq - exact),
        ("W8A8 + PINN w8+a8", pinn_waq, pinn_waq - exact),
    ]
    csv_lines = ["label,ppl,delta_vs_exact"]
    csv_lines.extend(f"{label},{ppl:.6f},{delta:.6f}" for label, ppl, delta in rows)
    csv_path.write_text("\n".join(csv_lines) + "\n")
    return csv_path


def write_decomposition_figure() -> Path:
    exact = load_ppl("exact_wikitext2")
    raw_w8a8 = load_ppl("raw_w8a8")
    pinn_exact = load_ppl("w8a8_pinn_exact")
    pinn_wq = load_ppl("w8a8_pinn_wq")
    pinn_waq = load_ppl("w8a8_pinn_waq")

    labels = ["Exact", "Raw W8A8", "+ PINN exact", "+ PINN w8", "+ PINN w8+a8"]
    absolute = [exact, raw_w8a8, pinn_exact, pinn_wq, pinn_waq]
    deltas = [
        raw_w8a8 - exact,
        pinn_exact - raw_w8a8,
        pinn_wq - pinn_exact,
        pinn_waq - pinn_wq,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(labels, absolute, color=["#4C78A8", "#E45756", "#72B7B2", "#54A24B", "#EECA3B"])
    axes[0].set_ylabel("Perplexity")
    axes[0].set_title("Absolute PPL")
    axes[0].tick_params(axis="x", rotation=25)

    delta_labels = ["Backbone W8A8", "PINN exact", "PINN w8", "PINN a8"]
    axes[1].bar(delta_labels, deltas, color=["#E45756", "#72B7B2", "#54A24B", "#EECA3B"])
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("Incremental delta")
    axes[1].set_title("Decomposition vs previous stage")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out_path = EXPERIMENTS / "end2endacc_combined_decomposition.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    markdown_path = EXPERIMENTS / "end2endacc_paper_tables.md"
    markdown_path.write_text(make_markdown() + "\n")
    csv_path = write_decomposition_csv()
    png_path = write_decomposition_figure()
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
