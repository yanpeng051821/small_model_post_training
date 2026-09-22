import pytest

from post_training_core.generation_budget import summarize_generation_budget


def test_generation_budget_passes_bounded_probe():
    summary = summarize_generation_budget(
        [400] * 190 + [2048] * 10,
        max_new_tokens=2048,
        elapsed_seconds=600,
        expected_total_generations=2000,
        minimum_probe_generations=200,
        minimum_stop_rate=0.90,
        maximum_truncation_rate=0.10,
        projection_safety_factor=1.25,
        maximum_projected_gpu_hours=3.0,
        gpu_hourly_cost_cny=9.0,
        maximum_projected_cost_cny=30.0,
    )

    assert summary["passed"] is True
    assert summary["truncation_rate"] == pytest.approx(0.05)
    assert summary["projected_gpu_hours"] == pytest.approx(2.0833333333)
    assert summary["projected_cost_cny"] == pytest.approx(18.75)


def test_generation_budget_rejects_length_and_cost_risk():
    summary = summarize_generation_budget(
        [2048] * 200,
        max_new_tokens=2048,
        elapsed_seconds=1800,
        expected_total_generations=2000,
        minimum_probe_generations=200,
        minimum_stop_rate=0.90,
        maximum_truncation_rate=0.10,
        projection_safety_factor=1.25,
        maximum_projected_gpu_hours=3.0,
        gpu_hourly_cost_cny=9.0,
        maximum_projected_cost_cny=30.0,
    )

    assert summary["passed"] is False
    assert summary["checks"]["stop_rate"] is False
    assert summary["checks"]["truncation_rate"] is False
    assert summary["checks"]["projected_gpu_hours"] is False
    assert summary["checks"]["projected_cost_cny"] is False


@pytest.mark.parametrize("lengths", [[], [1, 0]])
def test_generation_budget_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError, match="generation lengths"):
        summarize_generation_budget(
            lengths,
            max_new_tokens=32,
            elapsed_seconds=1,
            expected_total_generations=2,
            minimum_probe_generations=1,
            minimum_stop_rate=0.0,
            maximum_truncation_rate=1.0,
            projection_safety_factor=1.0,
            maximum_projected_gpu_hours=1.0,
            gpu_hourly_cost_cny=1.0,
            maximum_projected_cost_cny=1.0,
        )
