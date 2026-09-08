import pytest

from post_training_core.nll_comparison import compare_paired_nll


def test_paired_nll_reports_direction_and_deterministic_interval():
    baseline = {
        "a": {"mean_nll": 2.0},
        "b": {"mean_nll": 3.0},
        "c": {"mean_nll": 4.0},
    }
    trained = {
        "a": {"mean_nll": 1.0},
        "b": {"mean_nll": 2.5},
        "c": {"mean_nll": 4.5},
    }

    first = compare_paired_nll(baseline, trained, bootstrap_samples=2_000, seed=7)
    second = compare_paired_nll(baseline, trained, bootstrap_samples=2_000, seed=7)

    assert first == second
    assert first["mean_paired_delta_s1_minus_b0"] == pytest.approx(-1 / 3)
    assert first["improved_records"] == 2
    assert first["worsened_records"] == 1


def test_paired_nll_refuses_nonidentical_sample_sets():
    with pytest.raises(ValueError, match="sample_id sets differ"):
        compare_paired_nll(
            {"a": {"mean_nll": 1.0}},
            {"b": {"mean_nll": 1.0}},
        )
