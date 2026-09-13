import pytest

from post_training_core.gsm8k_analysis import (
    classify_pair,
    exact_mcnemar_pvalue,
    prediction_text,
    summarize_changed_records,
)


def test_exact_mcnemar_is_symmetric_and_handles_no_changes():
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(7, 2) == exact_mcnemar_pvalue(2, 7)
    assert exact_mcnemar_pvalue(10, 0) == pytest.approx(2 / 2**10)


def test_prediction_and_pair_contracts():
    assert prediction_text(["answer #### 3"]) == "answer #### 3"
    assert classify_pair(0, 1) == "improved"
    with pytest.raises(ValueError):
        prediction_text(["a", "b"])
    with pytest.raises(ValueError):
        classify_pair(0.5, 1)


def test_changed_record_summary_separates_metric_and_format():
    records = [
        {
            "baseline_predictions": ["wrong #### 2"],
            "trained_predictions": ["right #### 3"],
            "baseline_metrics": {"qem": 0},
            "trained_metrics": {"qem": 1},
        },
        {
            "baseline_predictions": ["right #### 3"],
            "trained_predictions": ["unfinished"],
            "baseline_metrics": {"qem": 1},
            "trained_metrics": {"qem": 0},
        },
    ]

    result = summarize_changed_records(records)

    assert result["groups"]["improved"]["count"] == 1
    assert result["groups"]["improved"]["baseline_final_marker_rate"] == 1
    assert result["marker_transitions"] == {
        "present_to_missing": 1,
        "present_to_present": 1,
    }

