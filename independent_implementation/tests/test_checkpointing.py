import shutil

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Qwen3Config, Qwen3ForCausalLM

import post_training_core.checkpointing as checkpointing
from post_training_core.checkpointing import (
    TrainingState,
    prune_training_checkpoints,
    save_training_checkpoint,
)
from post_training_core.data import StatefulRandomSampler, TokenizedSFTDataset


def _components():
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
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    dataset = TokenizedSFTDataset(
        [
            {
                "sample_id": "a",
                "input_ids": [1, 2],
                "labels": [-100, 2],
            }
        ]
    )
    sampler = StatefulRandomSampler(dataset, seed=42)
    return model, optimizer, scheduler, sampler


def test_checkpoint_refuses_to_start_without_required_disk_space(tmp_path, monkeypatch):
    model, optimizer, scheduler, sampler = _components()
    usage = shutil._ntuple_diskusage(total=1, used=1, free=0)
    monkeypatch.setattr(checkpointing.shutil, "disk_usage", lambda _: usage)

    with pytest.raises(OSError, match="insufficient disk space"):
        save_training_checkpoint(
            tmp_path / "checkpoint",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            training_state=TrainingState(),
            config_hash="hash",
        )

    assert not (tmp_path / "checkpoint").exists()
    assert not (tmp_path / ".checkpoint.tmp").exists()


def test_cleanup_failure_does_not_hide_primary_checkpoint_error(tmp_path, monkeypatch):
    model, optimizer, scheduler, sampler = _components()

    def fail_save(*args, **kwargs):
        raise RuntimeError("primary checkpoint failure")

    def fail_cleanup(*args, **kwargs):
        raise PermissionError("cleanup failure")

    monkeypatch.setattr(checkpointing.torch, "save", fail_save)
    monkeypatch.setattr(checkpointing.shutil, "rmtree", fail_cleanup)

    with pytest.raises(RuntimeError, match="primary checkpoint failure") as caught:
        save_training_checkpoint(
            tmp_path / "checkpoint",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            training_state=TrainingState(),
            config_hash="hash",
        )

    assert any("cleanup failure" in note for note in caught.value.__notes__)


def test_checkpoint_retention_keeps_newest_complete_checkpoints(tmp_path):
    root = tmp_path / "checkpoints"
    for step in (1, 2, 10):
        (root / f"step-{step:08d}").mkdir(parents=True)
    unrelated = root / ".step-00000011.tmp"
    unrelated.mkdir()

    prune_training_checkpoints(root, limit=2)

    assert not (root / "step-00000001").exists()
    assert (root / "step-00000002").is_dir()
    assert (root / "step-00000010").is_dir()
    assert unrelated.is_dir()
