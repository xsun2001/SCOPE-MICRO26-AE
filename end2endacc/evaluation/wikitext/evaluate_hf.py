import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

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
    return logits.float()


def load_corpus_input_ids(
    *,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    text_column: str,
    joiner: str,
    num_samples: int | None,
    tokenizer,
) -> torch.Tensor:
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    texts = dataset[text_column]
    return tokenizer(joiner.join(texts), return_tensors="pt").input_ids


def build_block_batches(
    input_ids: torch.Tensor,
    *,
    sequence_length: int,
    batch_size: int,
    max_blocks: int | None,
) -> List[torch.Tensor]:
    total_tokens = input_ids.numel()
    total_blocks = total_tokens // sequence_length
    if max_blocks is not None:
        total_blocks = min(total_blocks, int(max_blocks))
    batches: List[torch.Tensor] = []
    for start_block in range(0, total_blocks, batch_size):
        block_count = min(batch_size, total_blocks - start_block)
        blocks = []
        for offset in range(block_count):
            block_index = start_block + offset
            start = block_index * sequence_length
            end = (block_index + 1) * sequence_length
            blocks.append(input_ids[:, start:end])
        batches.append(torch.cat(blocks, dim=0))
    return batches


def evaluate_token_perplexity(
    model,
    *,
    input_ids: torch.Tensor,
    device: str,
    sequence_length: int,
    batch_size: int,
    max_blocks: int | None,
) -> Dict[str, Any]:
    total_tokens = input_ids.numel()
    total_blocks = total_tokens // sequence_length
    if max_blocks is not None:
        total_blocks = min(total_blocks, int(max_blocks))
    dropped_tokens = total_tokens - total_blocks * sequence_length
    if total_blocks <= 0:
        raise ValueError(
            f"Corpus produced zero full blocks for sequence_length={sequence_length}. Total tokens={total_tokens}."
        )

    nll_sum = 0.0
    block_rows = []
    previous_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    try:
        with torch.inference_mode():
            for batch_offset, batch in enumerate(
                tqdm(
                    build_block_batches(
                        input_ids,
                        sequence_length=sequence_length,
                        batch_size=batch_size,
                        max_blocks=max_blocks,
                    ),
                    desc="wikitext-eval",
                    unit="batch",
                )
            ):
                batch = batch.to(device)
                logits = forward_logits(model, batch)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                token_losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(batch.size(0), -1)
                mean_losses = token_losses.mean(dim=1)
                nlls = mean_losses * (sequence_length - 1)
                for sample_offset in range(batch.size(0)):
                    block_index = batch_offset * batch_size + sample_offset
                    start_token = block_index * sequence_length
                    end_token = start_token + sequence_length
                    mean_loss = float(mean_losses[sample_offset].item())
                    nll = float(nlls[sample_offset].item())
                    nll_sum += nll
                    block_rows.append(
                        {
                            "block_index": block_index,
                            "start_token": start_token,
                            "end_token": end_token,
                            "token_count": sequence_length - 1,
                            "mean_loss": mean_loss,
                            "nll": nll,
                        }
                    )
    finally:
        model.config.use_cache = previous_use_cache

    ppl = math.exp(nll_sum / (total_blocks * (sequence_length - 1)))
    return {
        "ppl": ppl,
        "total_tokens": total_tokens,
        "evaluated_tokens": total_blocks * sequence_length,
        "loss_tokens": total_blocks * (sequence_length - 1),
        "dropped_tokens": dropped_tokens,
        "num_blocks": total_blocks,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "blocks": block_rows,
    }


def write_block_metrics(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["block_index", "start_token", "end_token", "token_count", "mean_loss", "nll"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run trustworthy WikiText token-level perplexity for end2endacc exact, backbone-INT8, and PINN variants."
    )
    add_shared_runtime_args(parser)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_text_column", type=str, default="text")
    parser.add_argument("--dataset_joiner", type=str, default="\n\n")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = normalize_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found. Run this script under `srun ... --gres=gpu:1`.")

    output_dir = ensure_output_dir(args.output_dir or default_output_dir("wikitext"))
    model, tokenizer, extras, metadata = build_runtime(args, output_dir=output_dir)
    save_runtime_config(output_dir, runtime_config_payload(args, command="wikitext"), metadata)
    max_blocks = args.max_blocks if args.max_blocks is not None else args.max_chunks

    input_ids = load_corpus_input_ids(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.dataset_text_column,
        joiner=args.dataset_joiner,
        num_samples=args.num_samples,
        tokenizer=tokenizer,
    )
    if args.max_tokens is not None:
        input_ids = input_ids[:, : args.max_tokens]

    metrics = evaluate_token_perplexity(
        model,
        input_ids=input_ids,
        device=args.device,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_blocks=max_blocks,
    )
    metrics.update(
        {
            "dataset": {
                "name": args.dataset_name,
                "config": args.dataset_config,
                "split": args.dataset_split,
                "text_column": args.dataset_text_column,
                "joiner": args.dataset_joiner,
                "num_samples": args.num_samples,
                "max_chunks": args.max_chunks,
                "max_tokens": args.max_tokens,
            },
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
            "protocol": {
                "style": "gptq_awq_wikitext_token_ppl",
                "non_overlapping_blocks": True,
                "drop_remainder": True,
                "use_cache": False,
            },
        }
    )

    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    write_block_metrics(os.path.join(output_dir, "block_metrics.csv"), metrics["blocks"])
    print(f"Output directory: {output_dir}")
    print(f"Perplexity: {metrics['ppl']:.6f}")
    print(f"Blocks evaluated: {metrics['num_blocks']}")


if __name__ == "__main__":
    main()
