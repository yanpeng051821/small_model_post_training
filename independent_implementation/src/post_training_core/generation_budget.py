"""Generation-length and paid-compute gates for bounded evaluations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_generation_budget(
    lengths: Sequence[int],
    *,
    max_new_tokens: int,
    elapsed_seconds: float,
    expected_total_generations: int,
    minimum_probe_generations: int,
    minimum_stop_rate: float,
    maximum_truncation_rate: float,
    projection_safety_factor: float,
    maximum_projected_gpu_hours: float,
    gpu_hourly_cost_cny: float,
    maximum_projected_cost_cny: float,
) -> dict[str, Any]:
    """Summarize a probe and decide whether a larger evaluation is affordable."""
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("generation lengths must contain only positive values")
    if max_new_tokens <= 0 or elapsed_seconds <= 0:
        raise ValueError("max_new_tokens and elapsed_seconds must be positive")
    if expected_total_generations <= 0 or minimum_probe_generations <= 0:
        raise ValueError("generation counts must be positive")
    if expected_total_generations < len(lengths):
        raise ValueError("expected_total_generations is smaller than the probe")
    for name, value in (
        ("minimum_stop_rate", minimum_stop_rate),
        ("maximum_truncation_rate", maximum_truncation_rate),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name, value in (
        ("projection_safety_factor", projection_safety_factor),
        ("maximum_projected_gpu_hours", maximum_projected_gpu_hours),
        ("gpu_hourly_cost_cny", gpu_hourly_cost_cny),
        ("maximum_projected_cost_cny", maximum_projected_cost_cny),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    truncated_count = sum(length >= max_new_tokens for length in lengths)
    truncation_rate = truncated_count / len(lengths)
    stop_rate = 1 - truncation_rate
    projected_seconds = (
        elapsed_seconds
        * expected_total_generations
        / len(lengths)
        * projection_safety_factor
    )
    projected_gpu_hours = projected_seconds / 3600
    projected_cost_cny = projected_gpu_hours * gpu_hourly_cost_cny

    checks = {
        "probe_size": len(lengths) >= minimum_probe_generations,
        "stop_rate": stop_rate >= minimum_stop_rate,
        "truncation_rate": truncation_rate <= maximum_truncation_rate,
        "projected_gpu_hours": (
            projected_gpu_hours <= maximum_projected_gpu_hours
        ),
        "projected_cost_cny": projected_cost_cny <= maximum_projected_cost_cny,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_generations": len(lengths),
        "expected_total_generations": expected_total_generations,
        "max_new_tokens": max_new_tokens,
        "lengths": {
            "min": min(lengths),
            "mean": sum(lengths) / len(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "truncated_count": truncated_count,
        "truncation_rate": truncation_rate,
        "estimated_stop_rate": stop_rate,
        "elapsed_seconds": elapsed_seconds,
        "projection_safety_factor": projection_safety_factor,
        "projected_seconds": projected_seconds,
        "projected_gpu_hours": projected_gpu_hours,
        "gpu_hourly_cost_cny": gpu_hourly_cost_cny,
        "projected_cost_cny": projected_cost_cny,
        "thresholds": {
            "minimum_probe_generations": minimum_probe_generations,
            "minimum_stop_rate": minimum_stop_rate,
            "maximum_truncation_rate": maximum_truncation_rate,
            "maximum_projected_gpu_hours": maximum_projected_gpu_hours,
            "maximum_projected_cost_cny": maximum_projected_cost_cny,
        },
    }
