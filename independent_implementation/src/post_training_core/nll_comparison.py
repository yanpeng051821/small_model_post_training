"""Paired validation-NLL comparison for the frozen B0/S1 contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_nll_records(path: str | Path) -> dict[str, dict]:
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank NLL record at line {line_number}")
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"invalid sample_id at line {line_number}")
            if sample_id in records:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            if not isinstance(record.get("mean_nll"), (int, float)):
                raise ValueError(f"invalid mean_nll at line {line_number}")
            records[sample_id] = record
    if not records:
        raise ValueError("NLL records must not be empty")
    return records


def compare_paired_nll(
    baseline: dict[str, dict],
    trained: dict[str, dict],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 1234,
) -> dict:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if baseline.keys() != trained.keys():
        missing = sorted(baseline.keys() - trained.keys())
        extra = sorted(trained.keys() - baseline.keys())
        raise ValueError(f"sample_id sets differ: missing={missing}, extra={extra}")

    sample_ids = sorted(baseline)
    deltas = np.asarray(
        [trained[key]["mean_nll"] - baseline[key]["mean_nll"] for key in sample_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 1_000):
        count = min(1_000, bootstrap_samples - start)
        indices = rng.integers(0, len(deltas), size=(count, len(deltas)))
        bootstrap_means[start : start + count] = deltas[indices].mean(axis=1)

    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "baseline_mean_nll": float(
            np.mean([baseline[key]["mean_nll"] for key in sample_ids])
        ),
        "bootstrap_samples": bootstrap_samples,
        "improved_records": int(np.sum(deltas < 0)),
        "mean_paired_delta_s1_minus_b0": float(deltas.mean()),
        "paired_delta_95pct_ci": [float(lower), float(upper)],
        "record_count": len(sample_ids),
        "seed": seed,
        "trained_mean_nll": float(
            np.mean([trained[key]["mean_nll"] for key in sample_ids])
        ),
        "unchanged_records": int(np.sum(deltas == 0)),
        "worsened_records": int(np.sum(deltas > 0)),
    }
