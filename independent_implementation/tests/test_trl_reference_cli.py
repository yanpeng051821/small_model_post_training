import json
import subprocess
import sys

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


def _write_fixture(tmp_path, *, model_name):
    records = [
        {
            "sample_id": str(index),
            "input_ids": [index + 1, index + 2],
            "labels": [-100, index + 2],
        }
        for index in range(4)
    ]
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    payload = "".join(json.dumps(record) + "\n" for record in records)
    train.write_text(payload, encoding="utf-8")
    validation.write_text(payload, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "run_name: reference-test",
                f"model_name_or_path: {model_name}",
                "model_revision: fixed",
                "train_artifact: train.jsonl",
                "validation_artifact: validation.jsonl",
                "output_dir: unused",
                "pad_token_id: 0",
                "device: cpu",
                "dtype: float32",
                "per_device_train_batch_size: 1",
                "gradient_accumulation_steps: 2",
                "warmup_ratio: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_trl_reference_dry_run_freezes_order_without_loading_model(tmp_path):
    config = _write_fixture(tmp_path, model_name="unavailable-model")
    output = tmp_path / "reference"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_sft_trl_reference.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--max-steps",
            "2",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["required_records"] == 4
    assert len((output / "sample_order.jsonl").read_text().splitlines()) == 4


def test_trl_reference_executes_tiny_training_and_exports_model(tmp_path):
    model_dir = tmp_path / "model"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
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
    config = _write_fixture(tmp_path, model_name=model_dir.as_posix())
    output = tmp_path / "trained-reference"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_sft_trl_reference.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--max-steps",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert (output / "final_model" / "model.safetensors").is_file()
    assert (output / "final_model" / "tokenizer.json").is_file()
