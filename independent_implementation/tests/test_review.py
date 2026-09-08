import json

import pytest

from post_training_core.data import sha256_file
from post_training_core.review import (
    load_review_bundle,
    run_interactive_review,
)


def _write_bundle(tmp_path):
    review_path = tmp_path / "review_samples.jsonl"
    records = [
        {
            "generation": "A valid solution.",
            "problem": "Accepted problem",
            "row_index": 1,
            "sample_id": "accepted-id",
            "source": "fixture",
            "status": "accepted",
        },
        {
            "generation": "A leaked solution.",
            "issues": ["evaluation_exact_match:math500"],
            "problem": "Rejected problem",
            "review_reason": "evaluation_exact_match:math500",
            "row_index": 2,
            "source": "fixture",
            "status": "rejected",
        },
    ]
    review_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "review": {
                    "count": len(records),
                    "sha256": sha256_file(review_path),
                }
            }
        ),
        encoding="utf-8",
    )
    return review_path, manifest_path


def _answers(*values):
    iterator = iter(values)
    return lambda _prompt: next(iterator)


def test_review_session_persists_and_resumes_without_reprompting(tmp_path):
    review_path, manifest_path = _write_bundle(tmp_path)
    bundle = load_review_bundle(review_path, manifest_path)
    decisions = tmp_path / "decisions.jsonl"

    first_output = []
    first_exit = run_interactive_review(
        bundle=bundle,
        decisions_path=decisions,
        reviewer="learner",
        input_fn=_answers("k", "looks correct", "q"),
        output_fn=first_output.append,
    )
    assert first_exit == 1

    second_output = []
    second_exit = run_interactive_review(
        bundle=bundle,
        decisions_path=decisions,
        reviewer="learner",
        input_fn=_answers("a", "clear leak"),
        output_fn=second_output.append,
    )
    assert second_exit == 0
    assert not any("Accepted problem" in text for text in second_output)
    written = [json.loads(line) for line in decisions.read_text().splitlines()]
    assert [decision["verdict"] for decision in written] == ["keep", "agree"]
    summary = json.loads(
        decisions.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"] is True
    assert summary["completed_count"] == 2
    assert summary["decision_artifact_sha256"] == sha256_file(decisions)


def test_review_session_marks_flags_as_needing_resolution(tmp_path):
    review_path, manifest_path = _write_bundle(tmp_path)
    bundle = load_review_bundle(review_path, manifest_path)
    decisions = tmp_path / "decisions.jsonl"

    exit_code = run_interactive_review(
        bundle=bundle,
        decisions_path=decisions,
        reviewer="learner",
        input_fn=_answers("f", "questionable", "a", "correct rejection"),
        output_fn=lambda _text: None,
    )

    assert exit_code == 2
    summary = json.loads(decisions.with_suffix(".summary.json").read_text())
    assert summary["status"] == "needs_resolution"
    assert summary["unresolved_count"] == 1

    corrected_exit = run_interactive_review(
        bundle=bundle,
        decisions_path=decisions,
        reviewer="learner",
        redo_indices=[1],
        input_fn=_answers("k", "resolved"),
        output_fn=lambda _text: None,
    )
    assert corrected_exit == 0
    corrected = json.loads(decisions.with_suffix(".summary.json").read_text())
    assert corrected["status"] == "passed"


def test_review_bundle_rejects_hash_mismatch(tmp_path):
    review_path, manifest_path = _write_bundle(tmp_path)
    review_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        load_review_bundle(review_path, manifest_path)


def test_existing_decisions_cannot_be_reused_for_another_reviewer(tmp_path):
    review_path, manifest_path = _write_bundle(tmp_path)
    bundle = load_review_bundle(review_path, manifest_path)
    decisions = tmp_path / "decisions.jsonl"
    run_interactive_review(
        bundle=bundle,
        decisions_path=decisions,
        reviewer="first",
        input_fn=_answers("k", "", "q"),
        output_fn=lambda _text: None,
    )

    with pytest.raises(ValueError, match="another reviewer"):
        run_interactive_review(
            bundle=bundle,
            decisions_path=decisions,
            reviewer="second",
            input_fn=_answers(),
            output_fn=lambda _text: None,
        )
