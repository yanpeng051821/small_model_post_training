"""Checkpoint generation-health records and gate summaries."""

from __future__ import annotations

from typing import Any


def summarize_generation_health(
    records: list[dict[str, Any]],
    *,
    minimum_eos_rate: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError("generation health requires at least one record")
    if not 0 <= minimum_eos_rate <= 1:
        raise ValueError("minimum_eos_rate must be in [0, 1]")
    runtime_errors = sum(record.get("error") is not None for record in records)
    empty_generations = sum(record.get("generated_tokens", 0) <= 0 for record in records)
    eos_count = sum(record.get("generated_eos") is True for record in records)
    eos_rate = eos_count / len(records)
    return {
        "passed": (
            runtime_errors == 0
            and empty_generations == 0
            and eos_rate >= minimum_eos_rate
        ),
        "samples": len(records),
        "runtime_errors": runtime_errors,
        "empty_generations": empty_generations,
        "eos_count": eos_count,
        "eos_rate": eos_rate,
        "minimum_eos_rate": minimum_eos_rate,
    }
