import json
from pathlib import Path

import pytest

from post_training_core.config import ExperimentConfig
from post_training_core.experiment import initialize_run, update_run_status


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="tiny-run",
        model_name_or_path="local-tiny-qwen",
        model_revision="test-revision",
        train_artifact=tmp_path / "train.jsonl",
        validation_artifact=tmp_path / "validation.jsonl",
        output_dir=tmp_path / "runs",
        dtype="float32",
    )


def test_initializes_traceable_run_directory(tmp_path):
    config = _config(tmp_path)

    run_dir = initialize_run(config)

    assert run_dir == (tmp_path / "runs" / "tiny-run").resolve()
    resolved = json.loads((run_dir / "config.resolved.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())

    assert resolved["learning_rate"] == config.learning_rate
    assert manifest["config_hash"] == config.semantic_hash()
    assert manifest["status"] == "initialized"
    assert environment["torch"]
    assert environment["transformers"]


def test_refuses_to_overwrite_existing_run(tmp_path):
    config = _config(tmp_path)
    initialize_run(config)

    with pytest.raises(FileExistsError, match="run directory is not empty"):
        initialize_run(config)


def test_updates_run_status_without_losing_identity(tmp_path):
    config = _config(tmp_path)
    run_dir = initialize_run(config)

    update_run_status(run_dir, "failed", reason="non-finite loss")

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["config_hash"] == config.semantic_hash()
    assert manifest["status"] == "failed"
    assert manifest["reason"] == "non-finite loss"
    assert manifest["updated_at"]


def test_resume_preserves_original_manifest_identity(tmp_path):
    config = _config(tmp_path)
    run_dir = initialize_run(config)
    update_run_status(run_dir, "interrupted", optimizer_step=2)
    checkpoint = run_dir / "checkpoints" / "step-00000002"
    checkpoint.mkdir(parents=True)
    resume_config = ExperimentConfig(
        **{
            **config.__dict__,
            "resume_from_checkpoint": checkpoint,
        }
    )

    resumed_dir = initialize_run(resume_config)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert resumed_dir == run_dir
    assert manifest["config_hash"] == config.semantic_hash()
    assert manifest["status"] == "resuming"
    assert manifest["optimizer_step"] == 2
