import argparse
import json
import os
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def read_text(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def format_backbone_quantization(config: dict[str, Any]) -> str:
    runtime = config.get("runtime", {})
    quant = runtime.get("backbone_quantization")
    if not quant:
        return "disabled"
    summary = (
        f"enabled ({quant['weight_bits']}W/{quant['activation_bits']}A, "
        f"weight={quant['weight_scheme']}, act={quant['activation_scheme']}, "
        f"calibration={quant['activation_quant_mode']})"
    )
    smoothquant = quant.get("smoothquant")
    if isinstance(smoothquant, dict) and smoothquant.get("enabled"):
        summary = summary[:-1] + f", smoothquant_alpha={smoothquant.get('alpha', 0.85)})"
    return summary


def format_approximation(config: dict[str, Any]) -> str:
    args = config.get("args", {})
    backend = args.get("approx_backend")
    if backend is None:
        backend = "pinn" if args.get("pinn", False) else "none"
    if backend == "none":
        return "disabled"

    details = [f"backend={backend}", f"scope={args.get('approx_scope', 'all')}"]
    if backend == "pinn":
        details.append(f"dim={args.get('pinn_dim')}")
    if backend == "nnlut" and args.get("approx_exp_lut_path"):
        details.append(f"exp_lut={args.get('approx_exp_lut_path')}")
    if backend == "nli" and args.get("approx_exp_lut_path"):
        details.append(f"exp_lut={args.get('approx_exp_lut_path')}")
    if backend == "gqalut" and args.get("approx_exp_lut_path"):
        details.append(f"exp_lut={args.get('approx_exp_lut_path')}")
        details.append(f"lut_bits={args.get('approx_exp_lut_bits')}")

    if args.get("quant_approx_weights", args.get("quant_pinn_weights")):
        details.append(f"weight_quant={args.get('w_bits')}b")
    else:
        details.append("weight_quant=off")
    if args.get("quant_approx_activations", args.get("quant_pinn_activations")):
        details.append(f"act_quant={args.get('a_bits')}b")
    else:
        details.append("act_quant=off")
    return "enabled (" + ", ".join(details) + ")"


def format_approx_audit(config: dict[str, Any]) -> str:
    runtime = config.get("runtime", {})
    audit = runtime.get("approx_state_audit")
    if not audit:
        return "n/a"

    emitted_int = 0
    total_quantizers = 0
    module_tags: list[str] = []
    for quantizer in audit.get("quantizers", {}).values():
        total_quantizers += 1
        if quantizer.get("emit_int_codes"):
            emitted_int += 1
    for module in audit.get("modules", {}).values():
        metadata = module.get("metadata", {})
        runtime_domain = metadata.get("runtime_domain")
        control_encoding = metadata.get("control_encoding")
        control_source_encoding = metadata.get("control_source_encoding")
        weight_encoding = metadata.get("weight_encoding")
        tags = [tag for tag in [runtime_domain, control_encoding, control_source_encoding, weight_encoding] if tag]
        if tags:
            module_tags.append("/".join(tags))

    details = [f"int_input_quantizers={emitted_int}/{total_quantizers}"]
    if module_tags:
        details.append(f"modules={'; '.join(module_tags)}")
    return ", ".join(details)


def infer_kind(metrics: dict[str, Any]) -> str:
    if "ppl" in metrics:
        return "wikitext"
    if "results" in metrics:
        return "lm_eval"
    return "unknown"


def summarize_wikitext(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    dataset = metrics.get("dataset", {})
    args = config.get("args", {})
    lines = [
        "## Metrics",
        f"- Perplexity: `{metrics['ppl']:.6f}`",
        f"- Dataset: `{dataset.get('name')}` / `{dataset.get('config')}` / split `{dataset.get('split')}`",
        f"- Sequence length: `{metrics.get('sequence_length')}`",
        f"- Evaluated blocks: `{metrics.get('num_blocks')}`",
        f"- Evaluated tokens: `{metrics.get('evaluated_tokens')}`",
        f"- Dropped remainder tokens: `{metrics.get('dropped_tokens')}`",
        "",
        "## Experiment",
        f"- Dtype: `{args.get('dtype')}`",
        f"- Backbone quantization: {format_backbone_quantization(config)}",
        f"- Approximation: {format_approximation(config)}",
        f"- Approximation state audit: {format_approx_audit(config)}",
    ]
    return lines


def summarize_lm_eval(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    rows = []
    results = metrics.get("results", {}).get("results", {})
    for task, task_metrics in sorted(results.items()):
        preferred_metric = None
        for candidate in ["acc_norm,none", "acc,none", "exact_match,strict-match", "perplexity,none"]:
            if candidate in task_metrics:
                preferred_metric = candidate
                break
        if preferred_metric is None:
            continue
        value = task_metrics[preferred_metric]
        if isinstance(value, (int, float)):
            rows.append(f"| `{task}` | `{preferred_metric}` | `{value:.6f}` |")
        else:
            rows.append(f"| `{task}` | `{preferred_metric}` | `{value}` |")

    args = config.get("args", {})
    lines = [
        "## Metrics",
        f"- Task group: `{metrics.get('task_group')}`",
        f"- Tasks: `{', '.join(metrics.get('tasks', []))}`",
        f"- Few-shot: `{metrics.get('num_fewshot')}`",
        f"- Batch size: `{metrics.get('batch_size')}`",
        "",
        "## Key Results",
        "| Task | Metric | Value |",
        "| --- | --- | ---: |",
        *(rows or ["| `n/a` | `n/a` | `n/a` |"]),
        "",
        "## Experiment",
        f"- Dtype: `{args.get('dtype')}`",
        f"- Backbone quantization: {format_backbone_quantization(config)}",
        f"- Approximation: {format_approximation(config)}",
        f"- Approximation state audit: {format_approx_audit(config)}",
    ]
    return lines


def build_summary(result_dir: str) -> str:
    config = load_json(os.path.join(result_dir, "config.json"))
    metrics = load_json(os.path.join(result_dir, "metrics.json"))
    command = read_text(os.path.join(result_dir, "command.txt"))
    git_commit = read_text(os.path.join(result_dir, "git_commit.txt"))
    kind = infer_kind(metrics)

    lines = [
        "# Run Summary",
        "",
        "## Identity",
        f"- Kind: `{kind}`",
        f"- Timestamp: `{config.get('timestamp')}`",
        f"- Git commit: `{git_commit}`",
        f"- Model: `{config.get('args', {}).get('model')}`",
        f"- Command: `{command}`",
        "",
    ]

    if kind == "wikitext":
        lines.extend(summarize_wikitext(metrics, config))
    elif kind == "lm_eval":
        lines.extend(summarize_lm_eval(metrics, config))
    else:
        lines.extend(
            [
                "## Metrics",
                "- Unable to infer run kind from `metrics.json`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Files",
            "- `config.json`",
            "- `metrics.json`",
            "- `command.txt`",
            "- `git_commit.txt`",
            "- `stdout.log`",
        ]
    )

    if os.path.exists(os.path.join(result_dir, "backbone_calibration.json")):
        lines.append("- `backbone_calibration.json`")
    if os.path.exists(os.path.join(result_dir, "approx_activation_calibration.json")):
        lines.append("- `approx_activation_calibration.json`")
    if os.path.exists(os.path.join(result_dir, "pinn_activation_calibration.json")):
        lines.append("- `pinn_activation_calibration.json`")
    if os.path.exists(os.path.join(result_dir, "block_metrics.csv")):
        lines.append("- `block_metrics.csv`")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a short README summary for an experiment result directory.")
    parser.add_argument("--result_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    readme_path = os.path.join(args.result_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(build_summary(args.result_dir))
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
