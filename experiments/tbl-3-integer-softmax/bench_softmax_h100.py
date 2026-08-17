#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


EXPERIMENT_DIR = Path(__file__).resolve().parent


@triton.jit
def _fp_stream_softmax_u8_kernel(
    x_ptr,
    y_ptr,
    N_COLS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_start = row * N_COLS

    m = tl.full((), -float("inf"), tl.float32)
    d = tl.full((), 0.0, tl.float32)

    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x_i8 = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.float32)
        x = tl.where(mask, x_i8 * SCALE, -float("inf"))

        block_m = tl.max(x, axis=0)
        new_m = tl.maximum(m, block_m)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

    inv_d = 255.0 / tl.maximum(d, 1.0e-20)
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x_i8 = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.float32)
        x = tl.where(mask, x_i8 * SCALE, -float("inf"))
        probs = tl.exp(x - m) * inv_d
        probs_u8 = tl.minimum(tl.maximum(probs + 0.5, 0.0), 255.0).to(tl.uint8)
        tl.store(y_ptr + row_start + cols, probs_u8, mask=mask)


@triton.jit
def _intattention_stream_softmax_u8_kernel(
    x_ptr,
    lut_ptr,
    y_ptr,
    N_COLS: tl.constexpr,
    ZERO_THR_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_start = row * N_COLS

    row_max = tl.full((), -2147483648, tl.int32)
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.int32)
        x = tl.where(mask, x, -2147483648)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = tl.full((), 0, tl.int32)
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.int32)
        delta = row_max - x
        delta = tl.minimum(tl.maximum(delta, 0), ZERO_THR_Q)
        lut_idx = (delta * 31) // ZERO_THR_Q
        exp_vals = tl.load(lut_ptr + lut_idx, mask=mask, other=0).to(tl.int32)
        exp_vals = tl.where(mask, exp_vals, 0)
        row_sum += tl.sum(exp_vals, axis=0)

    row_sum = tl.maximum(row_sum, 1)
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.int32)
        delta = row_max - x
        delta = tl.minimum(tl.maximum(delta, 0), ZERO_THR_Q)
        lut_idx = (delta * 31) // ZERO_THR_Q
        exp_vals = tl.load(lut_ptr + lut_idx, mask=mask, other=0).to(tl.int32)
        exp_vals = tl.where(mask, exp_vals, 0)
        probs = (exp_vals * 255 + (row_sum // 2)) // row_sum
        probs = tl.minimum(tl.maximum(probs, 0), 255).to(tl.uint8)
        tl.store(y_ptr + row_start + cols, probs, mask=mask)


@triton.jit
def _di_exp_i32_norm16(
    x_delta_i32,
    MX: tl.constexpr,
    KX: tl.constexpr,
    M_E: tl.constexpr,
    K_E: tl.constexpr,
    EXP_BITS: tl.constexpr,
):
    n_i32: tl.constexpr = KX + K_E
    z_i32 = -(x_delta_i32 * MX * M_E)
    p_i32 = z_i32 >> n_i32
    r_i32 = z_i32 - (p_i32 << n_i32)

    y_i32 = (1 << n_i32) - (r_i32 >> 1)
    shift_i32 = p_i32 - EXP_BITS
    shr = tl.maximum(shift_i32, 0)
    shl = tl.maximum(-shift_i32, 0)

    y_right = y_i32 >> tl.minimum(shr, 31)
    y_left = y_i32 << tl.minimum(shl, 20)
    y_i32 = tl.where(shift_i32 >= 0, y_right, y_left)
    y_i32 = tl.maximum(y_i32, 0)

    # The user-provided DI-Exp expression contains a row-common 2^(kx + k_e)
    # factor. Dropping it before normalization keeps the 16-bit exp scale and
    # avoids immediate overflow on long rows without changing real ratios.
    y_i32 = y_i32 >> n_i32
    y_i32 = tl.minimum(tl.maximum(y_i32, 0), (1 << EXP_BITS) - 1)
    return y_i32


@triton.jit
def _illm_di_stream_softmax_u8_kernel(
    x_ptr,
    y_ptr,
    N_COLS: tl.constexpr,
    MX: tl.constexpr,
    KX: tl.constexpr,
    CLIP_Q: tl.constexpr,
    M_E: tl.constexpr,
    K_E: tl.constexpr,
    EXP_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_start = row * N_COLS

    row_max = tl.full((), -2147483648, tl.int32)
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.int32)
        x = tl.where(mask, x, -2147483648)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.int32)
        x_delta = x - row_max
        x_delta = tl.maximum(x_delta, -CLIP_Q)
        e = _di_exp_i32_norm16(
            x_delta,
            MX=MX,
            KX=KX,
            M_E=M_E,
            K_E=K_E,
            EXP_BITS=EXP_BITS,
        )
        e_f = tl.where(mask, e, 0).to(tl.float32)
        row_sum += tl.sum(e_f, axis=0)

    inv_sum = 255.0 / tl.maximum(row_sum, 1.0)
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-128).to(tl.int32)
        x_delta = x - row_max
        x_delta = tl.maximum(x_delta, -CLIP_Q)
        e = _di_exp_i32_norm16(
            x_delta,
            MX=MX,
            KX=KX,
            M_E=M_E,
            K_E=K_E,
            EXP_BITS=EXP_BITS,
        ).to(tl.float32)
        probs = e * inv_sum
        probs_u8 = tl.minimum(tl.maximum(probs + 0.5, 0.0), 255.0).to(tl.uint8)
        tl.store(y_ptr + row_start + cols, probs_u8, mask=mask)


@triton.jit
def _memory_stream_u8_kernel(
    x_ptr,
    y_ptr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_start = row * N_COLS
    for start in tl.range(0, N_COLS, BLOCK_N):
        cols = start + offs
        mask = cols < N_COLS
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.int16)
        y = tl.minimum(tl.maximum(x + 128, 0), 255).to(tl.uint8)
        tl.store(y_ptr + row_start + cols, y, mask=mask)


def synchronize() -> None:
    torch.cuda.synchronize()


def timed_cuda(fn: Callable[[], None], warmup: int, runs: int) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    synchronize()

    samples: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        synchronize()
        samples.append(float(start.elapsed_time(end)))

    return {
        "latency_ms_median": statistics.median(samples),
        "latency_ms_mean": statistics.fmean(samples),
        "latency_ms_min": min(samples),
        "latency_ms_max": max(samples),
        "latency_ms_stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def make_scores(n_rows: int, n_cols: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randint(-128, 128, (n_rows, n_cols), device=device, dtype=torch.int8, generator=gen)


def make_intattention_lut(device: torch.device, quant_bits: int, zero_thr: float) -> torch.Tensor:
    bins = 1 << quant_bits
    steps = torch.arange(bins, device=device, dtype=torch.float32)
    table = (torch.exp(steps * (-zero_thr / (bins - 1))) * 255.0).round().to(torch.uint8)
    table[0] = 255
    table[-1] = 0
    return table


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_clip_q(mx: int, kx: int, clip_c: float) -> int:
    return int(math.ceil(float(clip_c) * float(1 << int(kx)) / float(mx)))


def rows_for_seq_len(seq_len: int, target_elems: int, min_rows: int, max_rows: int | None) -> int:
    rows = max(min_rows, target_elems // seq_len)
    if max_rows is not None:
        rows = min(rows, max_rows)
    return max(1, rows)


def launch_fp_stream(
    scores_i8: torch.Tensor,
    out_u8: torch.Tensor,
    scale: float,
    block_n: int,
    num_warps: int,
) -> None:
    _fp_stream_softmax_u8_kernel[(scores_i8.shape[0],)](
        scores_i8,
        out_u8,
        N_COLS=scores_i8.shape[1],
        SCALE=scale,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def launch_intattention_stream(
    scores_i8: torch.Tensor,
    lut_u8: torch.Tensor,
    out_u8: torch.Tensor,
    zero_thr_q: int,
    block_n: int,
    num_warps: int,
) -> None:
    _intattention_stream_softmax_u8_kernel[(scores_i8.shape[0],)](
        scores_i8,
        lut_u8,
        out_u8,
        N_COLS=scores_i8.shape[1],
        ZERO_THR_Q=zero_thr_q,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def launch_illm_di_stream(
    scores_i8: torch.Tensor,
    out_u8: torch.Tensor,
    mx: int,
    kx: int,
    clip_q: int,
    m_e: int,
    k_e: int,
    exp_bits: int,
    block_n: int,
    num_warps: int,
) -> None:
    _illm_di_stream_softmax_u8_kernel[(scores_i8.shape[0],)](
        scores_i8,
        out_u8,
        N_COLS=scores_i8.shape[1],
        MX=mx,
        KX=kx,
        CLIP_Q=clip_q,
        M_E=m_e,
        K_E=k_e,
        EXP_BITS=exp_bits,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def launch_memory_stream(
    scores_i8: torch.Tensor,
    out_u8: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> None:
    _memory_stream_u8_kernel[(scores_i8.shape[0],)](
        scores_i8,
        out_u8,
        N_COLS=scores_i8.shape[1],
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.to(torch.int16) - b.to(torch.int16)).abs().max().item())


def benchmark_seq_len(
    seq_len: int, args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, object], list[dict[str, object]]]:
    n_rows = rows_for_seq_len(seq_len, args.target_elems, args.min_rows, args.max_rows)
    scale = args.mx / float(1 << args.kx)
    zero_thr_q = make_clip_q(args.mx, args.kx, args.intattention_zero_thr)
    clip_q = make_clip_q(args.mx, args.kx, args.illm_clip_c)
    num_warps = args.num_warps

    scores_i8 = make_scores(n_rows, seq_len, device, args.seed + seq_len)
    out_fp = torch.empty_like(scores_i8, dtype=torch.uint8)
    out_intattention = torch.empty_like(scores_i8, dtype=torch.uint8)
    out_illm = torch.empty_like(scores_i8, dtype=torch.uint8)
    out_mem = torch.empty_like(scores_i8, dtype=torch.uint8)
    lut_u8 = make_intattention_lut(device, args.intattention_quant_bits, args.intattention_zero_thr)

    launch_fp_stream(scores_i8, out_fp, scale, args.block_n, num_warps)
    launch_intattention_stream(scores_i8, lut_u8, out_intattention, zero_thr_q, args.block_n, num_warps)
    launch_illm_di_stream(
        scores_i8,
        out_illm,
        args.mx,
        args.kx,
        clip_q,
        args.illm_m_e,
        args.illm_k_e,
        args.illm_exp_bits,
        args.block_n,
        num_warps,
    )
    launch_memory_stream(scores_i8, out_mem, args.block_n, num_warps)
    synchronize()

    diff_fp_torch = None
    if args.check and n_rows * seq_len <= args.check_max_elems:
        torch_out = torch.softmax(scores_i8.float() * scale, dim=-1).mul(255.0).round().clamp(0, 255).to(torch.uint8)
        diff_fp_torch = max_abs_diff(out_fp, torch_out)

    baseline = timed_cuda(
        lambda: launch_fp_stream(scores_i8, out_fp, scale, args.block_n, num_warps),
        warmup=args.warmup,
        runs=args.runs,
    )
    intattention = timed_cuda(
        lambda: launch_intattention_stream(scores_i8, lut_u8, out_intattention, zero_thr_q, args.block_n, num_warps),
        warmup=args.warmup,
        runs=args.runs,
    )
    illm = timed_cuda(
        lambda: launch_illm_di_stream(
            scores_i8,
            out_illm,
            args.mx,
            args.kx,
            clip_q,
            args.illm_m_e,
            args.illm_k_e,
            args.illm_exp_bits,
            args.block_n,
            num_warps,
        ),
        warmup=args.warmup,
        runs=args.runs,
    )
    memory = timed_cuda(
        lambda: launch_memory_stream(scores_i8, out_mem, args.block_n, num_warps),
        warmup=args.warmup,
        runs=args.runs,
    )

    raw_samples: list[dict[str, object]] = []
    for method, measurement in (
        ("fp_stream_softmax_u8", baseline),
        ("illm_di_softmax_u8", illm),
        ("intattention_idx_softmax_u8", intattention),
        ("memory_floor_1read_1write", memory),
    ):
        samples = measurement.pop("samples")
        assert isinstance(samples, list)
        raw_samples.extend(
            {
                "seq_len": seq_len,
                "n_rows": n_rows,
                "method": method,
                "repetition": repetition,
                "latency_ms": sample,
            }
            for repetition, sample in enumerate(samples, start=1)
        )

    elems = n_rows * seq_len
    fp_ms = baseline["latency_ms_median"]
    intattention_ms = intattention["latency_ms_median"]
    illm_ms = illm["latency_ms_median"]
    memory_ms = memory["latency_ms_median"]
    summary = {
        "seq_len": seq_len,
        "n_rows": n_rows,
        "shape": f"{n_rows}x{seq_len}",
        "num_elements": elems,
        "block_n": args.block_n,
        "score_scale": scale,
        "fp_stream_softmax_u8_ms": fp_ms,
        "illm_di_softmax_u8_ms": illm_ms,
        "intattention_idx_softmax_u8_ms": intattention_ms,
        "memory_floor_1read_1write_ms": memory_ms,
        "fp_stream_softmax_u8_mean_ms": baseline["latency_ms_mean"],
        "fp_stream_softmax_u8_stddev_ms": baseline["latency_ms_stddev"],
        "illm_di_softmax_u8_mean_ms": illm["latency_ms_mean"],
        "illm_di_softmax_u8_stddev_ms": illm["latency_ms_stddev"],
        "intattention_idx_softmax_u8_mean_ms": intattention["latency_ms_mean"],
        "intattention_idx_softmax_u8_stddev_ms": intattention["latency_ms_stddev"],
        "memory_floor_1read_1write_mean_ms": memory["latency_ms_mean"],
        "memory_floor_1read_1write_stddev_ms": memory["latency_ms_stddev"],
        "fp_stream_gelem_s": elems / fp_ms / 1e6,
        "illm_di_gelem_s": elems / illm_ms / 1e6,
        "intattention_gelem_s": elems / intattention_ms / 1e6,
        "memory_floor_gelem_s": elems / memory_ms / 1e6,
        "illm_over_fp": illm_ms / fp_ms,
        "intattention_over_fp": intattention_ms / fp_ms,
        "illm_over_intattention": illm_ms / intattention_ms,
        "fp_speedup_vs_illm": illm_ms / fp_ms,
        "fp_speedup_vs_intattention": intattention_ms / fp_ms,
        "intattention_speedup_vs_illm": illm_ms / intattention_ms,
        "max_abs_diff_fp_vs_torch": diff_fp_torch,
        "runs": args.runs,
        "warmup": args.warmup,
        "measurement": "physical_h100_cuda",
    }
    return summary, raw_samples


def write_summary(output_dir: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    lines = [
        "# Long-Sequence Softmax H100 Microbenchmark",
        "",
        "These are actual CUDA/Triton timings on H100.",
        "The benchmark uses fixed total elements per sequence length rather than a square attention matrix, so 512k sequence length is physically runnable.",
        "All three measured kernels read the same INT8 logits and write U8 probabilities.",
        "`raw_samples.csv` contains every timed repetition; the latency table reports medians and the CSV also records means and sample standard deviations.",
        "",
        f"- Score scale: `x_real = x_i * {metadata['mx']} / 2^{metadata['kx']}`.",
        f"- IntAttention LUT zero threshold: `{metadata['intattention_zero_thr']}` real units, `{metadata['intattention_zero_thr_q']}` integer units.",
        f"- I-LLM DI clip: `{metadata['illm_clip_c']}` real units, `{metadata['illm_clip_q']}` integer units.",
        f"- I-LLM DI-Exp uses `m_e={metadata['illm_m_e']}`, `k_e={metadata['illm_k_e']}`, `exp_bits={metadata['illm_exp_bits']}`.",
        "",
        "## Latency",
        "",
        "| Seq Len | Rows | Elements | Triton FP ms | I-LLM DI ms | IntAttention ms | I-LLM / FP | IntAttention / FP | I-LLM / IntAttention |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq_len']} | {row['n_rows']} | {row['num_elements']} | "
            f"{row['fp_stream_softmax_u8_ms']:.3f} | "
            f"{row['illm_di_softmax_u8_ms']:.3f} | "
            f"{row['intattention_idx_softmax_u8_ms']:.3f} | "
            f"{row['illm_over_fp']:.3f}x | "
            f"{row['intattention_over_fp']:.3f}x | "
            f"{row['illm_over_intattention']:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Throughput",
            "",
            "| Seq Len | Triton FP Gelem/s | I-LLM DI Gelem/s | IntAttention Gelem/s | Memory Floor Gelem/s |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['seq_len']} | "
            f"{row['fp_stream_gelem_s']:.2f} | "
            f"{row['illm_di_gelem_s']:.2f} | "
            f"{row['intattention_gelem_s']:.2f} | "
            f"{row['memory_floor_gelem_s']:.2f} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- Baseline is a fused Triton online FP softmax: INT8 load, scale conversion, online max/sum, exp, normalization, U8 store.",
            "- I-LLM is a fused Triton implementation of DI-Exp plus clipped DI-Softmax. The row-common `2^(kx + k_e)` factor is removed before normalization to keep the intended 16-bit exp scale on long rows.",
            "- The I-LLM kernel uses FP32 for the final row sum and U8 normalization so the 16-bit DI-Exp path remains runnable at 512k without 64-bit division dominating the H100 measurement.",
            "- IntAttention is a fused Triton IndexSoftmax implementation using a 5-bit index and U8 exponential LUT.",
            "- The memory floor is one INT8 read plus one U8 write pass and is included only as a bandwidth diagnostic.",
            f"- Device: {metadata['device']}, torch {metadata['torch']}, Triton {metadata['triton']}, CUDA {metadata['cuda']}.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FP, I-LLM DI-Softmax, and IntAttention softmax on H100.")
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
    )
    parser.add_argument("--target-elems", type=int, default=67_108_864)
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--block-n", type=int, default=1024)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-max-elems", type=int, default=16_777_216)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mx", type=int, default=1)
    parser.add_argument("--kx", type=int, default=4)
    parser.add_argument("--intattention-quant-bits", type=int, default=5)
    parser.add_argument("--intattention-zero-thr", type=float, default=6.6)
    parser.add_argument("--illm-clip-c", type=int, default=15)
    parser.add_argument("--illm-m-e", type=int, default=185)
    parser.add_argument("--illm-k-e", type=int, default=7)
    parser.add_argument("--illm-exp-bits", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for H100 measurement.")
    if args.block_n & (args.block_n - 1):
        raise SystemExit("--block-n must be a power of two.")

    device = torch.device("cuda")
    output_dir = args.output_dir or (
        EXPERIMENT_DIR / "results" / time.strftime("%Y-%m-%d_%H%M%S_h100_softmax_longseq")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    zero_thr_q = make_clip_q(args.mx, args.kx, args.intattention_zero_thr)
    clip_q = make_clip_q(args.mx, args.kx, args.illm_clip_c)
    metadata = {
        "measurement": "physical_h100_cuda",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "seq_lens": args.seq_lens,
        "target_elems": args.target_elems,
        "min_rows": args.min_rows,
        "max_rows": args.max_rows,
        "block_n": args.block_n,
        "num_warps": args.num_warps,
        "correctness_check": args.check,
        "check_max_elems": args.check_max_elems,
        "mx": args.mx,
        "kx": args.kx,
        "score_scale": args.mx / float(1 << args.kx),
        "intattention_quant_bits": args.intattention_quant_bits,
        "intattention_zero_thr": args.intattention_zero_thr,
        "intattention_zero_thr_q": zero_thr_q,
        "illm_clip_c": args.illm_clip_c,
        "illm_clip_q": clip_q,
        "illm_m_e": args.illm_m_e,
        "illm_k_e": args.illm_k_e,
        "illm_exp_bits": args.illm_exp_bits,
        "baseline_method": "custom fused Triton streaming online FP softmax",
        "illm_method": "custom fused Triton streaming DI-ClippedSoftmax/DI-Softmax",
        "intattention_method": "custom fused Triton streaming IndexSoftmax",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for seq_len in args.seq_lens:
        n_rows = rows_for_seq_len(seq_len, args.target_elems, args.min_rows, args.max_rows)
        print(f"[softmax-longseq] seq_len={seq_len} rows={n_rows}", flush=True)
        row, samples = benchmark_seq_len(seq_len, args, device)
        rows.append(row)
        raw_rows.extend(samples)
        write_csv(output_dir / "softmax_latency.csv", rows)
        write_csv(output_dir / "raw_samples.csv", raw_rows)

    write_csv(output_dir / "softmax_latency.csv", rows)
    write_csv(output_dir / "raw_samples.csv", raw_rows)
    write_summary(output_dir, rows, metadata)
    latest = EXPERIMENT_DIR / "results" / "latest_longseq"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(output_dir.name)
    except OSError:
        pass
    print(f"[done] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
