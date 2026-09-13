"""Deterministic helpers for paired GSM8K capability analysis."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from statistics import mean, median
from typing import Any


def prediction_text(value: Any) -> str:
    """Return the single generated text stored by LightEval."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise ValueError("expected one LightEval prediction string")


def exact_mcnemar_pvalue(improved: int, regressed: int) -> float:
    """Return the exact two-sided McNemar p-value for discordant pairs."""
    if improved < 0 or regressed < 0:
        raise ValueError("paired counts must be non-negative")
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    lower = min(improved, regressed)
    lower_tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def classify_pair(baseline_metric: float, trained_metric: float) -> str:
    """Classify a paired binary-metric outcome."""
    pair = (baseline_metric, trained_metric)
    labels = {
        (0, 1): "improved",
        (1, 0): "regressed",
        (0, 0): "both_wrong",
        (1, 1): "both_correct",
    }
    try:
        return labels[pair]
    except KeyError as exc:
        raise ValueError("expected binary paired metrics") from exc


def bootstrap_delta_interval(
    *,
    improved: int,
    regressed: int,
    unchanged: int,
    samples: int = 20_000,
    seed: int = 20260913,
) -> tuple[float, float]:
    """Bootstrap a 95% interval for the paired accuracy delta."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    import numpy as np

    deltas = np.asarray(
        [1.0] * improved + [-1.0] * regressed + [0.0] * unchanged,
        dtype=np.float64,
    )
    if deltas.size == 0:
        raise ValueError("at least one paired sample is required")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = rng.choice(deltas, size=deltas.size, replace=True).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_changed_records(
    records: Iterable[dict[str, Any]],
    *,
    metric: str = "qem",
    final_marker: str = "####",
) -> dict[str, Any]:
    """Summarize format and output-length behavior for changed predictions."""
    groups: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in ("improved", "regressed", "both_wrong", "both_correct")
    }
    transitions: Counter[str] = Counter()

    for record in records:
        baseline = prediction_text(record["baseline_predictions"])
        trained = prediction_text(record["trained_predictions"])
        group = classify_pair(
            record["baseline_metrics"][metric], record["trained_metrics"][metric]
        )
        baseline_marker = final_marker in baseline
        trained_marker = final_marker in trained
        transition = (
            f"{'present' if baseline_marker else 'missing'}_to_"
            f"{'present' if trained_marker else 'missing'}"
        )
        transitions[transition] += 1
        groups[group].append(
            {
                "baseline_marker": baseline_marker,
                "trained_marker": trained_marker,
                "baseline_chars": len(baseline),
                "trained_chars": len(trained),
                "delta_chars": len(trained) - len(baseline),
            }
        )

    group_summaries = {}
    for name, rows in groups.items():
        if not rows:
            group_summaries[name] = {"count": 0}
            continue
        group_summaries[name] = {
            "count": len(rows),
            "baseline_final_marker_rate": mean(
                row["baseline_marker"] for row in rows
            ),
            "trained_final_marker_rate": mean(row["trained_marker"] for row in rows),
            "baseline_chars_median": median(row["baseline_chars"] for row in rows),
            "trained_chars_median": median(row["trained_chars"] for row in rows),
            "delta_chars_median": median(row["delta_chars"] for row in rows),
        }

    return {
        "covered_changed_predictions": sum(len(rows) for rows in groups.values()),
        "final_marker": final_marker,
        "marker_transitions": dict(sorted(transitions.items())),
        "groups": group_summaries,
    }
