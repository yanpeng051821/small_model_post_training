import json

import pytest

from post_training_core.observability import (
    JsonlMetricWriter,
    write_failure_snapshot,
)


def test_metric_writer_appends_durable_json_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = JsonlMetricWriter(path)

    writer.write({"event": "train", "loss": 1.25, "step": 1})
    writer.write({"event": "train", "loss": 1.0, "step": 2})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    assert all(record["recorded_at"] for record in records)


def test_metric_writer_rejects_non_finite_values(tmp_path):
    writer = JsonlMetricWriter(tmp_path / "metrics.jsonl")

    with pytest.raises(ValueError, match="must be finite"):
        writer.write({"loss": float("nan")})


def test_failure_snapshot_preserves_error_and_context(tmp_path):
    try:
        raise RuntimeError("deliberate failure")
    except RuntimeError as error:
        error.add_note("SFT sample_ids: ['sample-a']")
        path = write_failure_snapshot(tmp_path, error, optimizer_step=3)

    payload = json.loads(path.read_text())
    assert payload["error_type"] == "RuntimeError"
    assert payload["message"] == "deliberate failure"
    assert payload["notes"] == ["SFT sample_ids: ['sample-a']"]
    assert payload["context"]["optimizer_step"] == 3
    assert "RuntimeError: deliberate failure" in payload["traceback"]
