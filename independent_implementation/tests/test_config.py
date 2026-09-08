from pathlib import Path

import pytest

from post_training_core.config import ExperimentConfig


def _payload(tmp_path: Path) -> dict:
    return {
        "run_name": "tiny-qwen-smoke",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "model_revision": "fixed-revision",
        "train_artifact": tmp_path / "train.jsonl",
        "validation_artifact": tmp_path / "validation.jsonl",
        "output_dir": tmp_path / "runs",
    }


def test_loads_yaml_and_resolves_relative_paths(tmp_path):
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: tiny-qwen-smoke",
                "model_name_or_path: Qwen/Qwen3-0.6B-Base",
                "model_revision: fixed-revision",
                "train_artifact: artifacts/train.jsonl",
                "validation_artifact: artifacts/validation.jsonl",
                "output_dir: runs",
            ]
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(config_path)

    assert config.train_artifact == tmp_path / "artifacts" / "train.jsonl"
    assert config.validation_artifact == tmp_path / "artifacts" / "validation.jsonl"
    assert config.output_dir == tmp_path / "runs"
    assert config.gradient_accumulation_steps == 1


def test_rejects_unknown_yaml_fields(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: bad",
                "model_name_or_path: model",
                "model_revision: revision",
                "train_artifact: train.jsonl",
                "validation_artifact: validation.jsonl",
                "output_dir: runs",
                "learing_rate: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="learing_rate"):
        ExperimentConfig.from_yaml(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gradient_accumulation_steps", 0, "must be positive"),
        ("save_total_limit", 0, "must be positive"),
        ("warmup_ratio", 1.0, "must be in"),
        ("dtype", "float16", "dtype must be"),
        ("parameter_dtype", "float16", "parameter_dtype must be"),
        ("attn_implementation", "auto", "attn_implementation must be"),
        ("run_name", "bad/name", "path separators"),
    ],
)
def test_rejects_invalid_contract_values(tmp_path, field, value, message):
    payload = _payload(tmp_path)
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**payload)


def test_semantic_hash_changes_when_training_semantics_change(tmp_path):
    first = ExperimentConfig(**_payload(tmp_path))
    second_payload = _payload(tmp_path)
    second_payload["learning_rate"] = 1.0e-4
    second = ExperimentConfig(**second_payload)

    assert first.semantic_hash() != second.semantic_hash()
