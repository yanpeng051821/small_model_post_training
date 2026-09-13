"""Deterministic helpers for paired GSM8K capability analysis."""

from __future__ import annotations

import hashlib
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


def annotate_review_record(
    record: dict[str, Any],
    *,
    metric: str = "qem",
    final_marker: str = "####",
) -> dict[str, Any]:
    """Attach stable sampling strata to one paired prediction record."""
    baseline = prediction_text(record["baseline_predictions"])
    trained = prediction_text(record["trained_predictions"])
    baseline_marker = final_marker in baseline
    trained_marker = final_marker in trained
    delta_chars = len(trained) - len(baseline)
    if delta_chars <= -50:
        length_bucket = "shorter_by_50_or_more"
    elif delta_chars >= 50:
        length_bucket = "longer_by_50_or_more"
    else:
        length_bucket = "within_49_chars"
    outcome = classify_pair(
        record["baseline_metrics"][metric], record["trained_metrics"][metric]
    )
    marker_transition = (
        f"{'present' if baseline_marker else 'missing'}_to_"
        f"{'present' if trained_marker else 'missing'}"
    )
    return {
        **record,
        "outcome": outcome,
        "marker_transition": marker_transition,
        "length_bucket": length_bucket,
        "delta_chars": delta_chars,
    }


def fixed_stratified_sample(
    records: Iterable[dict[str, Any]],
    *,
    outcomes: tuple[str, ...] = ("improved", "regressed", "both_wrong"),
    per_outcome: int = 18,
    seed: str = "gsm8k-target-capability-v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Coverage-first, hash-stable sampling within outcome/format/length strata."""
    if per_outcome <= 0:
        raise ValueError("per_outcome must be positive")
    annotated = [annotate_review_record(record) for record in records]
    selected: list[dict[str, Any]] = []
    population_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()

    def stratum(record: dict[str, Any]) -> str:
        return "|".join(
            (record["outcome"], record["marker_transition"], record["length_bucket"])
        )

    def priority(record: dict[str, Any]) -> str:
        identity = f"{seed}|{record['row_index']}|{record.get('example', '')}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    for record in annotated:
        population_counts[stratum(record)] += 1

    for outcome in outcomes:
        candidates = [record for record in annotated if record["outcome"] == outcome]
        if len(candidates) < per_outcome:
            raise ValueError(f"not enough {outcome} records for requested sample")
        buckets: dict[str, list[dict[str, Any]]] = {}
        for record in candidates:
            buckets.setdefault(stratum(record), []).append(record)
        for bucket in buckets.values():
            bucket.sort(key=priority)

        outcome_selected = [buckets[name][0] for name in sorted(buckets)]
        already_selected = {record["row_index"] for record in outcome_selected}
        remaining = sorted(
            (
                record
                for record in candidates
                if record["row_index"] not in already_selected
            ),
            key=priority,
        )
        outcome_selected.extend(remaining[: per_outcome - len(outcome_selected)])
        outcome_selected.sort(key=priority)
        selected.extend(outcome_selected)

    selected.sort(key=lambda record: (outcomes.index(record["outcome"]), priority(record)))
    for index, record in enumerate(selected, start=1):
        record["review_index"] = index
        record["sampling_priority_sha256"] = priority(record)
        selected_counts[stratum(record)] += 1
    return selected, {
        "population_by_stratum": dict(sorted(population_counts.items())),
        "selected_by_stratum": dict(sorted(selected_counts.items())),
    }
