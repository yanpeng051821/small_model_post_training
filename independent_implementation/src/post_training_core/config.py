"""Validated configuration for the independent SFT runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    model_name_or_path: str
    model_revision: str
    train_artifact: Path
    validation_artifact: Path
    output_dir: Path
    pad_token_id: int = 151643
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
    parameter_dtype: str = "float32"
    attn_implementation: str = "sdpa"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    train_sample_limit: int | None = None
    validation_sample_limit: int | None = None
    num_train_epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 4.0e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    scheduler_type: str = "cosine_with_min_lr"
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 0.2
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    eval_on_start: bool = True
    log_every_steps: int = 1
    eval_every_steps: int = 100
    save_every_steps: int = 100
    save_total_limit: int = 2
    resume_from_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        nonempty = {
            "run_name": self.run_name,
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
        }
        for name, value in nonempty.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")

        if any(char in self.run_name for char in "/\\"):
            raise ValueError("run_name must not contain path separators")

        positive_ints = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "log_every_steps": self.log_every_steps,
            "eval_every_steps": self.eval_every_steps,
            "save_every_steps": self.save_every_steps,
            "save_total_limit": self.save_total_limit,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        for name, value in {
            "train_sample_limit": self.train_sample_limit,
            "validation_sample_limit": self.validation_sample_limit,
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must not be negative")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must not be negative")

        positive_floats = {
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
        }
        for name, value in positive_floats.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        for name, value in {
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
        }.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be 'float32' or 'bfloat16'")
        if self.parameter_dtype not in {"float32", "bfloat16"}:
            raise ValueError("parameter_dtype must be 'float32' or 'bfloat16'")
        if self.attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "attn_implementation must be eager, sdpa, or flash_attention_2"
            )
        if self.scheduler_type != "cosine_with_min_lr":
            raise ValueError("scheduler_type must be 'cosine_with_min_lr'")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)

        if not isinstance(payload, dict):
            raise ValueError("experiment config must be a YAML mapping")

        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(unknown)}")

        base_dir = config_path.parent
        for name in (
            "train_artifact",
            "validation_artifact",
            "output_dir",
            "resume_from_checkpoint",
        ):
            value = payload.get(name)
            if value is not None:
                candidate = Path(value)
                payload[name] = (
                    candidate if candidate.is_absolute() else base_dir / candidate
                )

        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name, value in payload.items():
            if isinstance(value, Path):
                payload[name] = str(value.resolve())
        return payload

    def semantic_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("resume_from_checkpoint", None)
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
