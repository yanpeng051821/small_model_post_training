import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from post_training_core.config import ExperimentConfig
from post_training_core.data import TokenizedSFTDataset, collate_sft_batch
from post_training_core.engine import (
    build_optimizer,
    build_scheduler,
    run_optimizer_step,
)
from post_training_core.trl_reference import OrderedCompletionMaskDataset
from post_training_core.trl_training import (
    CudaMemoryTelemetryCallback,
    FrozenOrderSFTTrainer,
    SaveAndStopCallback,
    build_trl_sft_args,
    compute_trl_sft_schedule,
)


def test_cuda_memory_telemetry_records_named_optimizer_phases(tmp_path, monkeypatch):
    current = [
        2 * 1024**3,
        3 * 1024**3,
        6 * 1024**3,
        7 * 1024**3,
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 1024**3, 80 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: current[1])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: current[2])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: current[3])

    callback = CudaMemoryTelemetryCallback(tmp_path / "cuda_memory.jsonl", "memory")
    args = SimpleNamespace(process_index=0, logging_steps=1)
    state = SimpleNamespace(global_step=1)
    callback.on_pre_optimizer_step(args, state, None)
    callback.on_optimizer_step(args, state, None)
    callback.on_step_end(args, state, None)

    records = [
        json.loads(line)
        for line in (tmp_path / "cuda_memory.jsonl").read_text().splitlines()
    ]
    assert [record["phase"] for record in records] == [
        "before_optimizer_step",
        "after_optimizer_step",
        "after_zero_grad",
        "optimizer_step_interval",
    ]
    assert all(record["run_id"] == "memory" for record in records)
    assert callback.summary()["peak_allocated_gib"] == 6
    assert callback.summary()["peak_phase"] == "before_optimizer_step"


def test_cuda_memory_telemetry_is_noop_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callback = CudaMemoryTelemetryCallback(tmp_path / "cuda_memory.jsonl")
    callback.on_step_end(
        SimpleNamespace(process_index=0, logging_steps=1),
        SimpleNamespace(global_step=1),
        None,
    )
    assert callback.summary() == {"enabled": False, "records": 0}
    assert not callback.output_path.exists()


def test_schedule_for_full_16k_s1_uses_479_updates_and_15_warmup_steps(tmp_path):
    config = _config(tmp_path)
    config = replace(
        config,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,
        warmup_ratio=0.03,
    )
    assert compute_trl_sft_schedule(config, 61_224) == (479, 15)


def _config(tmp_path):
    return ExperimentConfig(
        run_name="trl-test",
        model_name_or_path="tiny",
        model_revision="test",
        train_artifact=tmp_path / "train.jsonl",
        validation_artifact=tmp_path / "validation.jsonl",
        output_dir=tmp_path,
        device="cpu",
        dtype="float32",
        pad_token_id=0,
        gradient_accumulation_steps=2,
        learning_rate=0.01,
        warmup_ratio=0,
        save_every_steps=1,
        max_grad_norm=1.0,
    )


def test_formal_plan_keeps_tail_and_rounds_warmup(tmp_path):
    config = replace(
        _config(tmp_path), gradient_accumulation_steps=128, warmup_ratio=0.03
    )
    args = build_trl_sft_args(config, tmp_path, 61224)
    assert args.max_steps == 479
    assert args.warmup_steps == 15
    with pytest.raises(ValueError, match="repeat"):
        build_trl_sft_args(replace(config, max_steps=480), tmp_path, 61224)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_native_trl_tail_update_and_resume_match_independent(tmp_path, num_workers):
    torch.manual_seed(7)
    initial = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            attention_dropout=0.0,
            use_cache=False,
        )
    )
    records = [
        {"sample_id": "a", "input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"sample_id": "b", "input_ids": [2, 4], "labels": [-100, 4]},
        {"sample_id": "c", "input_ids": [3, 5, 6, 7], "labels": [-100, 5, 6, 7]},
    ]
    dataset = OrderedCompletionMaskDataset(TokenizedSFTDataset(records), [0, 1, 2])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    config = replace(_config(tmp_path), dataloader_num_workers=num_workers)

    def make_trainer(model, path, stop=None):
        return FrozenOrderSFTTrainer(
            model=model,
            args=build_trl_sft_args(config, path, len(records)),
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(0, completion_only_loss=True),
            callbacks=[SaveAndStopCallback(stop)],
        )

    independent = copy.deepcopy(initial)
    optimizer = build_optimizer(config, independent)
    scheduler = build_scheduler(config, optimizer, total_steps=2)
    for chunk in (records[:2], records[2:]):
        run_optimizer_step(
            model=independent,
            optimizer=optimizer,
            scheduler=scheduler,
            micro_batches=[collate_sft_batch([row], 0) for row in chunk],
            device=torch.device("cpu"),
            dtype="float32",
            max_grad_norm=config.max_grad_norm,
        )

    full = make_trainer(copy.deepcopy(initial), tmp_path / "full")
    full.train()
    assert full.state.global_step == 2
    for expected, actual in zip(
        independent.parameters(), full.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    partial = make_trainer(copy.deepcopy(initial), tmp_path / "resume", stop=1)
    partial.train()
    checkpoint = tmp_path / "resume" / "checkpoint-1"
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    assert (checkpoint / "rng_state.pth").is_file()
    resumed = make_trainer(
        Qwen3ForCausalLM.from_pretrained(checkpoint), tmp_path / "resume"
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    for expected, actual in zip(
        full.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        full.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    assert full.lr_scheduler.state_dict() == resumed.lr_scheduler.state_dict()


def test_formal_cli_runs_pauses_and_resumes_offline(tmp_path):
    model_dir = tmp_path / "model"
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            use_cache=False,
        )
    )
    model.save_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1, "<|im_end|>": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="<|im_end|>",
    )
    tokenizer.save_pretrained(model_dir)
    artifact = tmp_path / "train.jsonl"
    artifact.write_text(
        "".join(
            json.dumps(
                {"sample_id": str(i), "input_ids": [i + 3, 2], "labels": [-100, 2]}
            )
            + "\n"
            for i in range(3)
        ),
        encoding="utf-8",
    )
    config = _config(tmp_path).to_dict()
    config.update(
        model_name_or_path=str(model_dir),
        train_artifact=str(artifact),
        validation_artifact=str(artifact),
        save_every_steps=1,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/train_sft_trl.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
    ]
    gpu_only_dry_config = dict(config)
    gpu_only_dry_config.update(device="cuda", dtype="bfloat16")
    dry_config_path = tmp_path / "dry-config.yaml"
    dry_config_path.write_text(yaml.safe_dump(gpu_only_dry_config), encoding="utf-8")
    dry_run = subprocess.run(
        [
            command[0],
            command[1],
            "--config",
            str(dry_config_path),
            "--output-dir",
            str(tmp_path / "dry-run"),
            "--dry-run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    dry_manifest = json.loads(
        (tmp_path / "dry-run" / "run_manifest.json").read_text()
    )
    assert dry_manifest["status"] == "dry_run"
    assert dry_manifest["identity"]["max_steps"] == 2
    paused = subprocess.run(
        command + ["--stop-after-steps", "1"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert paused.returncode == 0, paused.stdout + paused.stderr
    assert json.loads((run_dir / "run_manifest.json").read_text())["status"] == "paused"
    resumed = subprocess.run(
        command + ["--resume-from-checkpoint", str(run_dir / "trainer/checkpoint-1")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (
        json.loads((run_dir / "run_manifest.json").read_text())["status"] == "completed"
    )
    exported = Qwen3ForCausalLM.from_pretrained(run_dir / "final_model")
    assert exported.generation_config.eos_token_id == 2
    assert exported.config.use_cache is True
    assert (run_dir / "final_model/tokenizer_config.json").is_file()
    assert len(list(run_dir.glob("attempt-*.json"))) == 2
