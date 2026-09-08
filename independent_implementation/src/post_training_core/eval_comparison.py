"""Paired B0/S1 comparison for frozen LightEval detail artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset

from post_training_core.data import sha256_file

_DETAIL_NAME = re.compile(
    r"^details_(?P<task>.+)_(?P<date>\d{4}-\d{2}-\d{2}T.+)\.parquet$"
)
_INVARIANT_FIELDS = (
    "example",
    "instruction",
    "full_prompt",
    "num_effective_few_shots",
    "num_asked_few_shots",
    "input_tokens",
    "gold",
    "choices",
    "gold_index",
)
_MANIFEST_FIELDS = (
    "contract_sha256",
    "contract_version",
    "lighteval_commit",
    "math_verify",
    "suite",
    "tasks",
    "use_chat_template",
    "model_args",
)


def _discover_manifests(root: Path) -> dict[str, dict[str, Any]]:
    manifests = {}
    for path in root.rglob("invocation_manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        suite = manifest.get("suite")
        if not isinstance(suite, str) or not suite:
            raise ValueError(f"invalid invocation manifest: {path}")
        if suite in manifests:
            raise ValueError(f"duplicate invocation manifest for suite {suite}")
        if manifest.get("status") != "completed" or manifest.get("returncode") != 0:
            raise ValueError(f"evaluation suite {suite} did not complete successfully")
        manifests[suite] = {"path": path, "manifest": manifest}
    if not manifests:
        raise ValueError(f"no completed invocation manifests found under {root}")
    return manifests


def _discover_details(root: Path) -> dict[str, Path]:
    details = {}
    for path in root.rglob("details_*.parquet"):
        match = _DETAIL_NAME.match(path.name)
        if match is None:
            raise ValueError(f"unrecognized LightEval detail filename: {path.name}")
        task = match.group("task")
        if task in details:
            raise ValueError(f"multiple detail files found for task {task}")
        details[task] = path
    if not details:
        raise ValueError(f"no LightEval detail parquet files found under {root}")
    return details


def _as_number(value: Any, *, task: str, metric: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric metric {metric} in task {task}")
    return float(value)


def compare_lighteval_runs(
    baseline_root: str | Path,
    trained_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare identical LightEval samples; all registered metrics are higher-is-better."""
    baseline_root = Path(baseline_root).resolve()
    trained_root = Path(trained_root).resolve()
    baseline_manifests = _discover_manifests(baseline_root)
    trained_manifests = _discover_manifests(trained_root)
    if baseline_manifests.keys() != trained_manifests.keys():
        raise ValueError("B0 and S1 evaluation suite sets differ")

    manifest_evidence = {}
    for suite in sorted(baseline_manifests):
        baseline = baseline_manifests[suite]
        trained = trained_manifests[suite]
        for field in _MANIFEST_FIELDS:
            if baseline["manifest"].get(field) != trained["manifest"].get(field):
                raise ValueError(f"evaluation contract mismatch for {suite}: {field}")
        manifest_evidence[suite] = {
            "baseline_manifest": str(baseline["path"]),
            "baseline_manifest_sha256": sha256_file(baseline["path"]),
            "trained_manifest": str(trained["path"]),
            "trained_manifest_sha256": sha256_file(trained["path"]),
        }

    baseline_details = _discover_details(baseline_root)
    trained_details = _discover_details(trained_root)
    if baseline_details.keys() != trained_details.keys():
        raise ValueError("B0 and S1 LightEval task sets differ")

    task_summaries = {}
    changed_records = []
    for task in sorted(baseline_details):
        baseline_rows = list(Dataset.from_parquet(str(baseline_details[task])))
        trained_rows = list(Dataset.from_parquet(str(trained_details[task])))
        if len(baseline_rows) != len(trained_rows):
            raise ValueError(f"sample count differs for task {task}")

        metric_totals: dict[str, dict[str, Any]] = {}
        for index, (baseline_row, trained_row) in enumerate(
            zip(baseline_rows, trained_rows, strict=True)
        ):
            for field in _INVARIANT_FIELDS:
                if baseline_row.get(field) != trained_row.get(field):
                    raise ValueError(
                        f"paired sample mismatch for task {task}, row {index}: {field}"
                    )
            baseline_metrics = baseline_row.get("metrics") or {}
            trained_metrics = trained_row.get("metrics") or {}
            if baseline_metrics.keys() != trained_metrics.keys():
                raise ValueError(f"metric keys differ for task {task}, row {index}")

            deltas = {}
            for metric in sorted(baseline_metrics):
                baseline_value = _as_number(
                    baseline_metrics[metric], task=task, metric=metric
                )
                trained_value = _as_number(
                    trained_metrics[metric], task=task, metric=metric
                )
                delta = trained_value - baseline_value
                deltas[metric] = delta
                aggregate = metric_totals.setdefault(
                    metric,
                    {
                        "baseline_sum": 0.0,
                        "trained_sum": 0.0,
                        "directions": Counter(),
                    },
                )
                aggregate["baseline_sum"] += baseline_value
                aggregate["trained_sum"] += trained_value
                direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
                aggregate["directions"][direction] += 1

            predictions_changed = baseline_row.get("predictions") != trained_row.get(
                "predictions"
            )
            if predictions_changed or any(delta != 0 for delta in deltas.values()):
                changed_records.append(
                    {
                        "task": task,
                        "row_index": index,
                        "example": baseline_row.get("example"),
                        "gold": baseline_row.get("gold"),
                        "baseline_predictions": baseline_row.get("predictions"),
                        "trained_predictions": trained_row.get("predictions"),
                        "baseline_metrics": baseline_metrics,
                        "trained_metrics": trained_metrics,
                        "metric_deltas_s1_minus_b0": deltas,
                    }
                )

        metrics = {}
        for metric, aggregate in sorted(metric_totals.items()):
            count = len(baseline_rows)
            baseline_mean = aggregate["baseline_sum"] / count
            trained_mean = aggregate["trained_sum"] / count
            metrics[metric] = {
                "baseline_mean": baseline_mean,
                "trained_mean": trained_mean,
                "delta_s1_minus_b0": trained_mean - baseline_mean,
                **{
                    direction: aggregate["directions"][direction]
                    for direction in ("improved", "regressed", "unchanged")
                },
            }
        task_summaries[task] = {
            "sample_count": len(baseline_rows),
            "baseline_details": str(baseline_details[task]),
            "baseline_details_sha256": sha256_file(baseline_details[task]),
            "trained_details": str(trained_details[task]),
            "trained_details_sha256": sha256_file(trained_details[task]),
            "metrics": metrics,
        }

    return (
        {
            "baseline_root": str(baseline_root),
            "trained_root": str(trained_root),
            "manifests": manifest_evidence,
            "tasks": task_summaries,
            "changed_record_count": len(changed_records),
            "metric_direction": "higher_is_better",
        },
        changed_records,
    )
