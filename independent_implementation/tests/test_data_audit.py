import json
import shutil

import pytest

from post_training_core.audit import (
    AuditSettings,
    ContaminationIndex,
    _materialize_sorted_artifacts,
    audit_openr1_rows,
    finalize_completed_audit_scan,
    normalize_problem,
)
from post_training_core.data import TokenizedSFTDataset


class AuditTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        if add_generation_prompt:
            return [10, 11, 12]
        assistant = messages[1]["content"]
        payload = [20 + index for index, _ in enumerate(assistant.split())]
        return [10, 11, 12, *payload, 99, 30]

    def convert_tokens_to_ids(self, token):
        return 99 if token == "<|im_end|>" else None


def _row(problem, generation, *, verified=True, complete=True, row_id="id"):
    return {
        "problem": problem,
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": generation},
        ],
        "generations": [generation],
        "correctness_math_verify": [verified],
        "is_reasoning_complete": [complete],
        "source": "fixture",
        "uuid": row_id,
    }


def _generation(answer):
    return f"<think> reasoning {answer} </think> final \\boxed{{{answer}}}"


def test_normalization_is_stable_across_whitespace_and_unicode():
    assert normalize_problem("  What\u3000IS  2 + 2? ") == "what is 2 + 2?"


def test_contamination_index_reports_exact_and_ngram_matches():
    index = ContaminationIndex(
        {
            "math500": [
                "find the exact value of this very specific algebra expression"
            ],
            "gsm8k": ["A different short question"],
        },
        ngram_size=4,
    )

    assert index.matches(
        "Find the exact value of this very specific algebra expression"
    ) == ["evaluation_exact_match:math500"]
    assert index.matches(
        "Please find the exact value of this very specific algebra expression now"
    ) == ["evaluation_4gram_overlap:math500"]


def test_contamination_ngrams_match_openr1_whitespace_tokenization():
    index = ContaminationIndex(
        {"suite": ["alpha beta gamma delta epsilon zeta eta theta"]},
        ngram_size=8,
    )

    assert index.matches("alpha beta gamma delta epsilon zeta eta theta") == [
        "evaluation_exact_match:suite"
    ]
    assert index.matches("alpha, beta gamma delta epsilon zeta eta theta") == []


def test_audit_writes_deterministic_train_validation_and_rejections(tmp_path):
    rows = [
        _row("problem one", _generation("1"), row_id="1"),
        _row("problem two", _generation("2"), row_id="2"),
        _row("problem three", _generation("3"), row_id="3"),
        _row("problem two", _generation("2"), row_id="duplicate"),
        _row("bad verification", _generation("4"), verified=False, row_id="4"),
        _row("evaluation leak", _generation("5"), row_id="5"),
    ]
    settings = AuditSettings(
        validation_size=1,
        max_length=64,
        split_salt="test-salt",
        ngram_size=2,
    )

    first = audit_openr1_rows(
        rows,
        tokenizer=AuditTokenizer(),
        output_dir=tmp_path / "first",
        settings=settings,
        evaluation_problems={"math500": ["evaluation leak"]},
        source_identity={"dataset": "fixture", "revision": "abc"},
    )
    second = audit_openr1_rows(
        rows,
        tokenizer=AuditTokenizer(),
        output_dir=tmp_path / "second",
        settings=settings,
        evaluation_problems={"math500": ["evaluation leak"]},
        source_identity={"dataset": "fixture", "revision": "abc"},
    )

    first_manifest = json.loads(first.manifest_path.read_text())
    second_manifest = json.loads(second.manifest_path.read_text())
    assert first.train_count == 2
    assert first.validation_count == 1
    assert first.rejected_count == 3
    assert first_manifest["train"]["sha256"] == second_manifest["train"]["sha256"]
    assert (
        first_manifest["validation"]["sha256"]
        == second_manifest["validation"]["sha256"]
    )
    assert first_manifest["reason_counts"] == {
        "duplicate_problem": 1,
        "evaluation_exact_match:math500": 1,
        "math_verify_failed": 1,
    }
    assert first_manifest["length_statistics"]["sequence_tokens"]["count"] == 3
    assert first_manifest["length_statistics"]["supervised_tokens"]["min"] > 0
    assert (
        first_manifest["review"]["sha256"]
        == second_manifest["review"]["sha256"]
    )
    review = [
        json.loads(line)
        for line in (tmp_path / "first" / "review_samples.jsonl")
        .read_text()
        .splitlines()
    ]
    accepted_review = [record for record in review if record["status"] == "accepted"]
    rejected_review = [record for record in review if record["status"] == "rejected"]
    assert len(accepted_review) == 3
    assert {record["problem"] for record in accepted_review} == {
        "problem one",
        "problem two",
        "problem three",
    }
    assert {record["review_reason"] for record in rejected_review} == {
        "duplicate_problem",
        "evaluation_exact_match:math500",
        "math_verify_failed",
    }
    assert all(record.get("generation") for record in review)
    accepted_artifacts = []
    for artifact in ("train.jsonl", "validation.jsonl"):
        accepted_artifacts.extend(
            json.loads(line)
            for line in (tmp_path / "first" / artifact).read_text().splitlines()
        )
    assert all(record["metadata"]["content_sha256"] for record in accepted_artifacts)
    assert all(
        record["metadata"]["correctness_math_verify"] is True
        and record["metadata"]["is_reasoning_complete"] is True
        for record in accepted_artifacts
    )
    assert all(
        record["metadata"]["token_count"] == len(record["input_ids"])
        for record in accepted_artifacts
    )
    assert len(TokenizedSFTDataset.from_jsonl(tmp_path / "first" / "train.jsonl")) == 2
    assert not (tmp_path / "first" / ".accepted-candidates.jsonl.tmp").exists()


def test_audit_refuses_silent_truncation(tmp_path):
    row = _row("long problem", _generation("1"), row_id="1")
    with pytest.raises(ValueError, match="larger than validation_size"):
        audit_openr1_rows(
            [row],
            tokenizer=AuditTokenizer(),
            output_dir=tmp_path / "audit",
            settings=AuditSettings(validation_size=1, max_length=4),
            evaluation_problems={},
            source_identity={"dataset": "fixture", "revision": "abc"},
        )

    rejected = [
        json.loads(line)
        for line in (tmp_path / "audit" / "rejected.jsonl").read_text().splitlines()
    ]
    assert rejected[0]["issues"] == ["over_max_length"]


def test_external_sort_materializes_hash_order_across_small_chunks(tmp_path):
    candidate = tmp_path / "candidates.jsonl"
    records = [
        ("c" * 64, {"sample_id": "third"}),
        ("a" * 64, {"sample_id": "first"}),
        ("b" * 64, {"sample_id": "second"}),
    ]
    locations = []
    with candidate.open("wb") as stream:
        for key, record in records:
            offset = stream.tell()
            stream.write((json.dumps(record) + "\n").encode())
            locations.append((key, offset))

    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _materialize_sorted_artifacts(
        candidate,
        locations,
        validation_size=1,
        train_path=train,
        validation_path=validation,
        max_chunk_bytes=1,
    )

    validation_record = json.loads(validation.read_text())
    train_records = [json.loads(line) for line in train.read_text().splitlines()]
    assert validation_record["sample_id"] == "first"
    assert [record["sample_id"] for record in train_records] == ["second", "third"]
    assert not list(tmp_path.glob(".sorted-chunk-*.tmp"))


def test_completed_scan_recovery_matches_fresh_audit_artifacts(tmp_path):
    rows = [
        _row("problem one", _generation("1"), row_id="1"),
        _row("problem two", _generation("2"), row_id="2"),
        _row("problem three", _generation("3"), row_id="3"),
        _row("problem two", _generation("2"), row_id="duplicate"),
        _row("bad verification", _generation("4"), verified=False, row_id="4"),
        _row("evaluation leak", _generation("5"), row_id="5"),
    ]
    settings = AuditSettings(
        validation_size=1,
        max_length=64,
        split_salt="recovery-test-salt",
        ngram_size=2,
    )
    source_identity = {"dataset": "fixture", "revision": "abc"}
    fresh_dir = tmp_path / "fresh"
    recovered_dir = tmp_path / "recovered"
    audit_openr1_rows(
        rows,
        tokenizer=AuditTokenizer(),
        output_dir=fresh_dir,
        settings=settings,
        evaluation_problems={"math500": ["evaluation leak"]},
        source_identity=source_identity,
    )

    recovered_dir.mkdir()
    candidate_path = recovered_dir / ".accepted-candidates.jsonl.tmp"
    with candidate_path.open("wb") as candidate_stream:
        for artifact_name in ("train.jsonl", "validation.jsonl"):
            candidate_stream.write((fresh_dir / artifact_name).read_bytes())
    shutil.copy2(fresh_dir / "rejected.jsonl", recovered_dir / "rejected.jsonl")

    recovered = finalize_completed_audit_scan(
        rows,
        output_dir=recovered_dir,
        settings=settings,
        source_identity=source_identity,
    )
    fresh_manifest = json.loads((fresh_dir / "data_manifest.json").read_text())
    recovered_manifest = json.loads(recovered.manifest_path.read_text())

    for artifact in ("train", "validation", "rejected", "review"):
        assert recovered_manifest[artifact] == fresh_manifest[artifact]
    assert (
        recovered_manifest["length_statistics"]
        == fresh_manifest["length_statistics"]
    )
    assert recovered_manifest["reason_counts"] == fresh_manifest["reason_counts"]
    assert not candidate_path.exists()
