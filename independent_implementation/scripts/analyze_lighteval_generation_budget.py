"""Apply a frozen generation-budget gate to LightEval detail artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from post_training_core.experiment import atomic_write_json, utc_now
from post_training_core.generation_budget import summarize_generation_budget


def _token_lengths(value: Any) -> Iterable[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(token, int) for token in value):
            yield len(value)
            return
        for item in value:
            yield from _token_lengths(item)


def _load_policy(contract_path: Path, suite_name: str) -> tuple[dict, dict]:
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    try:
        suite = contract["suites"][suite_name]
        policy = suite["evaluation_policy"]
        generation = suite["model_args"]["generation_parameters"]
    except KeyError as error:
        raise ValueError(
            f"suite {suite_name} lacks a generation budget policy: {error}"
        ) from error
    if policy.get("role") != "budget_probe":
        raise ValueError(f"suite {suite_name} is not a budget probe")
    return policy, generation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--suite", default="math500_probe")
    parser.add_argument("--details-root", type=Path, required=True)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy, generation = _load_policy(args.contract, args.suite)
    detail_paths = sorted(args.details_root.rglob("details_*.parquet"))
    if not detail_paths:
        raise ValueError(f"no LightEval details found under {args.details_root}")
    lengths = []
    for path in detail_paths:
        for row in Dataset.from_parquet(str(path)):
            lengths.extend(_token_lengths(row.get("cont_tokens")))

    summary = summarize_generation_budget(
        lengths,
        max_new_tokens=generation["max_new_tokens"],
        elapsed_seconds=args.elapsed_seconds,
        expected_total_generations=policy["expected_total_generations"],
        minimum_probe_generations=policy["minimum_probe_generations"],
        minimum_stop_rate=policy["minimum_stop_rate"],
        maximum_truncation_rate=policy["maximum_truncation_rate"],
        projection_safety_factor=policy["projection_safety_factor"],
        maximum_projected_gpu_hours=policy["maximum_projected_gpu_hours"],
        gpu_hourly_cost_cny=policy["gpu_hourly_cost_cny"],
        maximum_projected_cost_cny=policy["maximum_projected_cost_cny"],
    )
    report = {
        **summary,
        "created_at": utc_now(),
        "contract": str(args.contract.resolve()),
        "suite": args.suite,
        "details": [str(path.resolve()) for path in detail_paths],
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
