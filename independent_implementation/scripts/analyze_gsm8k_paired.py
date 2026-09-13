"""Analyze paired GSM8K results without rerunning model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from post_training_core.experiment import atomic_write_json, utc_now
from post_training_core.gsm8k_analysis import (
    bootstrap_delta_interval,
    exact_mcnemar_pvalue,
    summarize_changed_records,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _generation_length(value: Any) -> int:
    lengths: list[int] = []

    def visit(node: Any) -> None:
        if isinstance(node, list) and node and all(isinstance(item, int) for item in node):
            lengths.append(len(node))
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    if len(lengths) != 1:
        raise ValueError(f"expected one generated token sequence, found {len(lengths)}")
    return lengths[0]


def _summarize_trained_details(
    path: Path, *, generation_limit: int
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to inspect LightEval parquet details") from exc

    rows = pq.read_table(
        path, columns=["cont_tokens", "truncated", "padded", "metrics"]
    ).to_pylist()
    lengths = [_generation_length(row["cont_tokens"]) for row in rows]
    truncated = sum(any(bool(value) for value in row["truncated"]) for row in rows)
    padded = sum(any(bool(value) for value in row["padded"]) for row in rows)
    at_limit = sum(length >= generation_limit for length in lengths)
    at_limit_wrong = sum(
        length >= generation_limit and row["metrics"]["qem"] == 0
        for row, length in zip(rows, lengths, strict=True)
    )
    return {
        "sample_count": len(rows),
        "generation_tokens": {
            "min": min(lengths),
            "mean": mean(lengths),
            "median": median(lengths),
            "max": max(lengths),
            "generation_limit": generation_limit,
            "at_generation_limit": at_limit,
            "at_generation_limit_qem_wrong": at_limit_wrong,
        },
        "lighteval_truncated_field_count": truncated,
        "lighteval_padded_field_count": padded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-summary", type=Path, required=True)
    parser.add_argument("--changed-samples", type=Path, required=True)
    parser.add_argument("--trained-details", type=Path, required=True)
    parser.add_argument("--trained-invocation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison_summary.read_text(encoding="utf-8"))
    task_name, task = next(iter(comparison["tasks"].items()))
    metric_name, metric = next(iter(task["metrics"].items()))
    interval = bootstrap_delta_interval(
        improved=metric["improved"],
        regressed=metric["regressed"],
        unchanged=metric["unchanged"],
    )
    records = _read_jsonl(args.changed_samples)
    invocation = json.loads(args.trained_invocation_manifest.read_text(encoding="utf-8"))
    generation_limit = invocation["model_args"]["generation_parameters"][
        "max_new_tokens"
    ]
    report = {
        "created_at": utc_now(),
        "contract": {
            "task": task_name,
            "metric": metric_name,
            "sample_count": task["sample_count"],
            "comparison_summary": str(args.comparison_summary.resolve()),
            "changed_samples": str(args.changed_samples.resolve()),
            "trained_details": str(args.trained_details.resolve()),
            "trained_invocation_manifest": str(
                args.trained_invocation_manifest.resolve()
            ),
        },
        "paired_effect": {
            **metric,
            "delta_95_percent_bootstrap_ci": list(interval),
            "exact_mcnemar_pvalue": exact_mcnemar_pvalue(
                metric["improved"], metric["regressed"]
            ),
        },
        "format_and_length_proxy": {
            **summarize_changed_records(records, metric=metric_name),
            "coverage_note": (
                "Only predictions that changed are present; three identical B0/S1 "
                "predictions are excluded from this proxy analysis."
            ),
        },
        "trained_generation": _summarize_trained_details(
            args.trained_details, generation_limit=generation_limit
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
