import json
import subprocess
import sys

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


def _write_artifact(path):
    records = [
        {
            "sample_id": "a",
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
        },
        {
            "sample_id": "b",
            "input_ids": [2, 3, 4],
            "labels": [-100, 3, 4],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_cli_runs_tiny_qwen_and_writes_evidence(tmp_path):
    model_dir = tmp_path / "model"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
    ).save_pretrained(model_dir)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"<pad>": 0, "<eos>": 1, "<unk>": 2},
                unk_token="<unk>",
            )
        ),
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    ).save_pretrained(model_dir)
    _write_artifact(tmp_path / "train.jsonl")
    _write_artifact(tmp_path / "validation.jsonl")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "run_name: cli-smoke",
                f"model_name_or_path: {model_dir.as_posix()}",
                "model_revision: local-test",
                "train_artifact: train.jsonl",
                "validation_artifact: validation.jsonl",
                "output_dir: runs",
                "pad_token_id: 0",
                "device: cpu",
                "dtype: float32",
                "per_device_train_batch_size: 1",
                "gradient_accumulation_steps: 1",
                "num_train_epochs: 1",
                "max_steps: 2",
                "learning_rate: 0.01",
                "max_grad_norm: 1.0",
                "eval_every_steps: 1",
                "save_every_steps: 1",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "post_training_core.cli",
            "--config",
            str(config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "runs" / "cli-smoke"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["optimizer_step"] == 2
    assert (run_dir / "checkpoints" / "step-00000002").is_dir()
    checkpoint = run_dir / "checkpoints" / "step-00000002"
    assert (checkpoint / "tokenizer.json").is_file()
    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["use_cache"] is True
    saved_generation = json.loads(
        (checkpoint / "generation_config.json").read_text()
    )
    assert saved_generation["eos_token_id"] == 1
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 5


def test_missing_artifact_leaves_failure_evidence(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "run_name: missing-data",
                "model_name_or_path: unused",
                "model_revision: local-test",
                "train_artifact: missing-train.jsonl",
                "validation_artifact: missing-validation.jsonl",
                "output_dir: runs",
                "device: cpu",
                "dtype: float32",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "post_training_core.cli", "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    run_dir = tmp_path / "runs" / "missing-data"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    failure = json.loads((run_dir / "failure.json").read_text())
    assert manifest["status"] == "failed"
    assert failure["error_type"] == "FileNotFoundError"
