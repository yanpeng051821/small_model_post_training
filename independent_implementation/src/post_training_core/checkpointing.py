"""Atomic model and training-state checkpoints."""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from post_training_core.data import StatefulRandomSampler
from post_training_core.experiment import atomic_write_json, utc_now


@dataclass
class TrainingState:
    optimizer_step: int = 0
    valid_tokens_seen: int = 0


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _estimate_checkpoint_bytes(model: nn.Module, optimizer: Optimizer) -> int:
    model_bytes = sum(_tensor_bytes(parameter) for parameter in model.parameters())
    model_bytes += sum(_tensor_bytes(buffer) for buffer in model.buffers())
    optimizer_bytes = 0
    for state in optimizer.state.values():
        optimizer_bytes += sum(
            _tensor_bytes(value)
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
    raw_bytes = model_bytes + optimizer_bytes
    return int(raw_bytes * 1.1) + 256 * 1024 * 1024


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"no existing parent for checkpoint path: {path}")
        candidate = candidate.parent
    return candidate


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    sampler: StatefulRandomSampler,
    training_state: TrainingState,
    config_hash: str,
    processing_class=None,
) -> Path:
    target = Path(checkpoint_dir)
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")

    required_bytes = _estimate_checkpoint_bytes(model, optimizer)
    free_bytes = shutil.disk_usage(_nearest_existing_parent(target.parent)).free
    if free_bytes < required_bytes:
        raise OSError(
            "insufficient disk space for atomic checkpoint: "
            f"required approximately {required_bytes / (1024**3):.2f} GiB, "
            f"available {free_bytes / (1024**3):.2f} GiB"
        )

    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        if not hasattr(model, "save_pretrained"):
            raise TypeError("model must implement save_pretrained")
        original_use_cache = getattr(model.config, "use_cache", None)
        if original_use_cache is not None:
            model.config.use_cache = True
        try:
            model.save_pretrained(temporary, safe_serialization=True)
        finally:
            if original_use_cache is not None:
                model.config.use_cache = original_use_cache
        if processing_class is not None:
            if not hasattr(processing_class, "save_pretrained"):
                raise TypeError("processing_class must implement save_pretrained")
            processing_class.save_pretrained(temporary)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "sampler": sampler.state_dict(),
                "training_state": asdict(training_state),
                "rng": _capture_rng_state(),
            },
            temporary / "training_state.pt",
        )
        atomic_write_json(
            temporary / "checkpoint_manifest.json",
            {
                "config_hash": config_hash,
                "created_at": utc_now(),
                "optimizer_step": training_state.optimizer_step,
                "valid_tokens_seen": training_state.valid_tokens_seen,
            },
        )
        temporary.replace(target)
    except BaseException as error:
        if temporary.exists():
            try:
                shutil.rmtree(temporary)
            except OSError as cleanup_error:
                error.add_note(
                    "temporary checkpoint cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise

    return target


def prune_training_checkpoints(checkpoint_root: str | Path, *, limit: int) -> None:
    """Retain only the newest complete, conventionally named checkpoints."""
    if limit <= 0:
        raise ValueError("checkpoint retention limit must be positive")
    root = Path(checkpoint_root)
    if not root.exists():
        return
    checkpoints = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and re.fullmatch(r"step-\d{8}", path.name)
        ),
        key=lambda path: int(path.name.removeprefix("step-")),
    )
    for stale in checkpoints[:-limit]:
        shutil.rmtree(stale)


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    sampler: StatefulRandomSampler,
    expected_config_hash: str,
) -> TrainingState:
    source = Path(checkpoint_dir)
    with (source / "checkpoint_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["config_hash"] != expected_config_hash:
        raise ValueError("checkpoint config hash does not match current config")

    payload = torch.load(
        source / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    _restore_rng_state(payload["rng"])
    return TrainingState(**payload["training_state"])
