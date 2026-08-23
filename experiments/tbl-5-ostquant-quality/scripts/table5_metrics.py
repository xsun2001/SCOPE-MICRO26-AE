from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path
from typing import Any


FOUR_TASKS = ("arc_easy", "hellaswag", "piqa", "winogrande")
PROTOCOL_DESCRIPTION = "Unweighted arithmetic mean over ARC-Easy, HellaSwag, PIQA, and WinoGrande."
MODEL_PREFIXES = {"llama2": "llama2_7b", "llama3": "llama3_8b"}
FIGURE16_MODELS = {"llama2_7b": "Llama-2-7b-hf", "llama3_8b": "Llama-3-8b"}


def four_task_average(final: dict[str, Any]) -> float:
    metrics = final.get("metrics", {})
    missing = [task for task in FOUR_TASKS if metrics.get(task) is None]
    if missing:
        raise ValueError(f"missing four-task metrics: {missing}")
    return round(sum(float(metrics[task]) for task in FOUR_TASKS) / len(FOUR_TASKS), 4)


def four_task_stderr_average(final: dict[str, Any]) -> float | None:
    stderr = final.get("stderr", {})
    if any(stderr.get(task) is None for task in FOUR_TASKS):
        return None
    return round(sum(float(stderr[task]) for task in FOUR_TASKS) / len(FOUR_TASKS), 4)


def officialize_final_metrics(final: dict[str, Any]) -> dict[str, Any]:
    """Promote the four-task subset while retaining any wider raw sweep as diagnostics."""
    result = deepcopy(final)
    diagnostics = result.get("all_task_diagnostics")
    if diagnostics is None and set(result.get("tasks", [])) != set(FOUR_TASKS):
        diagnostics = {
            "tasks": deepcopy(result.get("tasks", [])),
            "metrics": deepcopy(result.get("metrics", {})),
            "stderr": deepcopy(result.get("stderr", {})),
            "acc_avg": result.get("acc_avg"),
            "acc_avg_stderr": result.get("acc_avg_stderr"),
        }

    average = four_task_average(result)
    stderr_average = four_task_stderr_average(result)
    result["tasks"] = list(FOUR_TASKS)
    result["metrics"] = {
        "acc_avg": average,
        **{task: result["metrics"][task] for task in FOUR_TASKS},
    }
    result["stderr"] = {
        task: result.get("stderr", {}).get(task) for task in FOUR_TASKS
    }
    result["acc_avg"] = average
    result["acc_avg_stderr"] = stderr_average
    result["accuracy_protocol"] = {
        "tasks": list(FOUR_TASKS),
        "aggregation": PROTOCOL_DESCRIPTION,
    }
    if diagnostics is not None:
        result["all_task_diagnostics"] = diagnostics
    return result


def load_bf16_baselines(path: Path) -> dict[str, tuple[float, float]]:
    """Load BF16 baselines from Figure 16's legacy ``fp16_exact`` columns."""
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["model"], row["metric"]): row for row in rows}
    baselines: dict[str, tuple[float, float]] = {}
    for model_key, figure_model in FIGURE16_MODELS.items():
        baselines[model_key] = (
            float(indexed[(figure_model, "ppl")]["fp16_exact"]),
            100.0 * float(indexed[(figure_model, "group1_mean")]["fp16_exact"]),
        )
    return baselines
