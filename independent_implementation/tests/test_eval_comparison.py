import json
from pathlib import Path

import pytest
from datasets import Dataset

from post_training_core.eval_comparison import compare_lighteval_runs


def _write_run(
    root: Path, *, model: str, metrics: list[float], prompts=None, predictions=None
) -> None:
    suite = root / "math500"
    details = suite / "details" / model / "2026-09-04T00-00-00"
    details.mkdir(parents=True)

    manifest = {
        "contract_sha256": "contract-hash",
        "contract_version": "gate0b-eval-v1",
        "lighteval_commit": "fixed-commit",
        "math_verify": "0.5.2",
        "suite": "math500",
        "tasks": ["math_500"],
        "use_chat_template": True,
        "model_args": {"dtype": "bfloat16", "seed": 1234},
        "model": model,
        "status": "completed",
        "returncode": 0,
    }
    (suite / "invocation_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for index, score in enumerate(metrics):
        prediction = f"answer-{score}" if predictions is None else predictions[index]
        prompt = f"question-{index}" if prompts is None else prompts[index]
        rows.append(
            {
                "example": prompt,
                "instruction": "",
                "full_prompt": prompt,
                "num_effective_few_shots": 0,
                "num_asked_few_shots": 0,
                "input_tokens": [index, 1],
                "gold": ["gold"],
                "choices": [],
                "gold_index": [],
                "metrics": {"exact_match": score},
                "predictions": [prediction],
            }
        )
    Dataset.from_list(rows).to_parquet(
        details / "details_math_500_2026-09-04T00-00-00.parquet"
    )


def test_preserves_changed_predictions_when_metric_is_unchanged(tmp_path):
    baseline = tmp_path / "b0"
    trained = tmp_path / "s1"

    _write_run(
        baseline,
        model="base",
        metrics=[0.0],
        predictions=["wrong answer A"],
    )
    _write_run(
        trained,
        model="trained",
        metrics=[0.0],
        predictions=["wrong answer B"],
    )

    summary, changed = compare_lighteval_runs(
        baseline,
        trained,
    )

    metric = summary["tasks"]["math_500"]["metrics"]["exact_match"]

    assert metric["unchanged"] == 1
    assert summary["changed_record_count"] == 1
    assert len(changed) == 1
    assert changed[0]["baseline_predictions"] == ["wrong answer A"]
    assert changed[0]["trained_predictions"] == ["wrong answer B"]


def test_compares_paired_lighteval_metrics_and_changed_samples(tmp_path):
    baseline = tmp_path / "b0"
    trained = tmp_path / "s1"
    _write_run(baseline, model="base", metrics=[0.0, 1.0, 0.0])
    _write_run(trained, model="trained", metrics=[1.0, 0.0, 0.0])

    summary, changed = compare_lighteval_runs(baseline, trained)

    metric = summary["tasks"]["math_500"]["metrics"]["exact_match"]
    assert metric["baseline_mean"] == pytest.approx(1 / 3)
    assert metric["trained_mean"] == pytest.approx(1 / 3)
    assert metric["improved"] == 1
    assert metric["regressed"] == 1
    assert metric["unchanged"] == 1
    assert len(changed) == 2


def test_refuses_prompt_mismatch(tmp_path):
    baseline = tmp_path / "b0"
    trained = tmp_path / "s1"
    _write_run(baseline, model="base", metrics=[0.0])
    _write_run(trained, model="trained", metrics=[1.0], prompts=["different"])

    with pytest.raises(ValueError, match="paired sample mismatch"):
        compare_lighteval_runs(baseline, trained)


def test_refuses_evaluation_contract_mismatch(tmp_path):
    baseline = tmp_path / "b0"
    trained = tmp_path / "s1"
    _write_run(baseline, model="base", metrics=[0.0])
    _write_run(trained, model="trained", metrics=[1.0])
    manifest_path = trained / "math500" / "invocation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["use_chat_template"] = False
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="evaluation contract mismatch"):
        compare_lighteval_runs(baseline, trained)
