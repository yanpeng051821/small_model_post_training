import pytest

from post_training_core.generation import summarize_generation_health


def test_generation_health_requires_nonempty_eos_terminated_outputs():
    summary = summarize_generation_health(
        [
            {"generated_tokens": 4, "generated_eos": True, "error": None},
            {"generated_tokens": 2, "generated_eos": True, "error": None},
        ],
        minimum_eos_rate=1.0,
    )

    assert summary["passed"] is True
    assert summary["eos_rate"] == 1.0


def test_generation_health_reports_runtime_empty_and_eos_failures():
    summary = summarize_generation_health(
        [
            {"generated_tokens": 0, "generated_eos": False, "error": "OOM"},
            {"generated_tokens": 3, "generated_eos": False, "error": None},
        ],
        minimum_eos_rate=0.5,
    )

    assert summary["passed"] is False
    assert summary["runtime_errors"] == 1
    assert summary["empty_generations"] == 1
    assert summary["eos_rate"] == 0.0


@pytest.mark.parametrize("minimum", [-0.1, 1.1])
def test_generation_health_rejects_invalid_threshold(minimum):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        summarize_generation_health(
            [{"generated_tokens": 1, "generated_eos": True, "error": None}],
            minimum_eos_rate=minimum,
        )
