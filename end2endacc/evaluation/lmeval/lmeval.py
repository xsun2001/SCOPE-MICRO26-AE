import argparse
import os
import sys

import torch
import torch.nn.functional as F

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
END2ENDACC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in [PROJECT_ROOT, END2ENDACC_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.runtime import (
    add_shared_runtime_args,
    build_runtime,
    default_output_dir,
    ensure_output_dir,
    normalize_args,
    runtime_config_payload,
    save_runtime_config,
    write_json,
)


TASK_GROUPS = {
    "group1": {"tasks": ["piqa", "hellaswag", "winogrande", "arc_easy"], "batch_size": 8, "num_fewshot": 0},
    "group2": {"tasks": ["gsm8k"], "batch_size": 8, "num_fewshot": 8},
    "group3": {"tasks": ["mmlu"], "batch_size": 4, "num_fewshot": 5},
    "group4": {"tasks": ["wikitext"], "batch_size": 2, "num_fewshot": 0},
}

PREFERRED_METRICS = ["acc_norm,none", "acc,none", "exact_match,strict-match", "perplexity,none"]


def forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        return model(input_ids).logits

    outputs = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    hidden_states = outputs[0]
    lm_head = model.lm_head
    lm_head_weight = getattr(lm_head, "weight", None)
    if lm_head_weight is not None and hidden_states.dtype != lm_head_weight.dtype:
        hidden_states = hidden_states.to(lm_head_weight.dtype)

    pretraining_tp = getattr(model.config, "pretraining_tp", 1)
    if pretraining_tp > 1:
        lm_head_slices = lm_head.weight.split(model.config.vocab_size // pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = lm_head(hidden_states)
    return logits


class End2EndAccHFLM(HFLM):
    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                return super()._model_call(inps, attn_mask=attn_mask, labels=labels)
            return forward_logits(self.model, inps)


def summarize_preferred_metrics(results: dict) -> dict:
    task_results = results.get("results", {})
    summary = {}
    scalar_values = []
    for task, task_metrics in task_results.items():
        preferred_metric = None
        for candidate in PREFERRED_METRICS:
            if candidate in task_metrics:
                preferred_metric = candidate
                break
        if preferred_metric is None:
            continue
        value = task_metrics[preferred_metric]
        summary[task] = {
            "metric": preferred_metric,
            "value": value,
        }
        if isinstance(value, (int, float)):
            scalar_values.append(float(value))

    avg_value = None
    if scalar_values:
        avg_value = float(sum(scalar_values) / len(scalar_values))

    return {
        "tasks": summary,
        "avg_value": avg_value,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lm-eval on end2endacc exact, backbone-INT8, and PINN variants."
    )
    add_shared_runtime_args(parser)
    parser.add_argument("--task_group", type=str, default="group1", choices=sorted(TASK_GROUPS))
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = normalize_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found. Run this script under `srun ... --gres=gpu:1`.")

    output_dir = ensure_output_dir(args.output_dir or default_output_dir("lm_eval"))
    model, tokenizer, extras, metadata = build_runtime(args, output_dir=output_dir)
    save_runtime_config(output_dir, runtime_config_payload(args, command="lm_eval"), metadata)

    task_group = TASK_GROUPS[args.task_group]
    lm = End2EndAccHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=task_group["batch_size"],
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_group["tasks"],
        num_fewshot=task_group["num_fewshot"],
        batch_size=task_group["batch_size"],
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
    )

    payload = {
        "task_group": args.task_group,
        "tasks": task_group["tasks"],
        "num_fewshot": task_group["num_fewshot"],
        "batch_size": task_group["batch_size"],
        "limit": args.limit,
        "bootstrap_iters": args.bootstrap_iters,
        "runtime": {
            "model_path": metadata.model_path,
            "dtype": metadata.dtype,
            "quant_backbone": metadata.quant_backbone,
            "approx_enabled": metadata.approx_enabled,
            "approx_backend": metadata.approx_backend,
            "approx_scope": metadata.approx_scope,
        },
        "backbone_quantization": extras["backbone_quantization"],
        "approximation": extras["approximation"],
        "approx_calibration": extras["approx_calibration"],
        "preferred_metrics_summary": summarize_preferred_metrics(results),
        "results": results,
    }
    write_json(os.path.join(output_dir, "metrics.json"), payload)
    print("Output directory:", output_dir)
    print(make_table(results))


if __name__ == "__main__":
    main()
