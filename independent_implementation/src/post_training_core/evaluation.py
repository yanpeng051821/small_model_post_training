"""Model-output health labels layered on deterministic task evaluators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationOutcome:
    labels: tuple[str, ...]
    parsed_answer: str | None
    verifier_passed: bool | None


def classify_generation(
    *,
    parsed_answer: str | None,
    verifier_passed: bool | None,
    generated_eos: bool,
    finish_reason: str,
    runtime_error: str | None = None,
) -> EvaluationOutcome:
    """Classify health independently from a task evaluator's aggregate score."""
    labels = []
    if runtime_error is not None:
        labels.append("runtime_error")
    else:
        if parsed_answer is None:
            labels.append("unparseable")
        elif verifier_passed is True:
            labels.append("correct")
        elif verifier_passed is False:
            labels.append("wrong_answer")

    if not generated_eos:
        labels.append("no_eos")
    if finish_reason == "length":
        labels.append("length_truncated")

    if not labels:
        raise ValueError("evaluation outcome has no classifiable state")
    return EvaluationOutcome(
        labels=tuple(labels),
        parsed_answer=parsed_answer,
        verifier_passed=verifier_passed,
    )
