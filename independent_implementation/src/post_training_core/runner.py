"""Composition root for a complete single-device SFT experiment."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_training_core.config import ExperimentConfig
from post_training_core.data import (
    IndexedTokenizedSFTDataset,
    StatefulRandomSampler,
    collate_sft_batch,
    sha256_file,
)
from post_training_core.engine import TrainingSoftStop, train
from post_training_core.experiment import (
    atomic_write_json,
    initialize_run,
    set_reproducible_seed,
    update_run_status,
)
from post_training_core.observability import write_failure_snapshot


def _current_data_manifest(config: ExperimentConfig) -> dict:
    return {
        "train": {
            "path": str(config.train_artifact.resolve()),
            "sha256": sha256_file(config.train_artifact),
            "sample_limit": config.train_sample_limit,
        },
        "validation": {
            "path": str(config.validation_artifact.resolve()),
            "sha256": sha256_file(config.validation_artifact),
            "sample_limit": config.validation_sample_limit,
        },
    }


def _write_or_validate_data_manifest(
    run_dir: Path,
    config: ExperimentConfig,
) -> None:
    import json

    manifest_path = run_dir / "data_manifest.json"
    current = _current_data_manifest(config)
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            original = json.load(stream)
        if original != current:
            raise ValueError("training artifact hash changed since run initialization")
        return
    atomic_write_json(manifest_path, current)


def load_model(config: ExperimentConfig):
    source = config.resume_from_checkpoint or config.model_name_or_path
    kwargs = {}
    if config.resume_from_checkpoint is None:
        kwargs["revision"] = config.model_revision
    if config.parameter_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32
    kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def load_tokenizer(config: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError(
            "configured pad_token_id does not match the pinned tokenizer: "
            f"{config.pad_token_id} != {tokenizer.pad_token_id}"
        )
    return tokenizer


def _build_dataloaders(config: ExperimentConfig):
    train_dataset = IndexedTokenizedSFTDataset(
        config.train_artifact,
        limit=config.train_sample_limit,
    )
    validation_dataset = IndexedTokenizedSFTDataset(
        config.validation_artifact,
        limit=config.validation_sample_limit,
    )
    sampler = StatefulRandomSampler(train_dataset, seed=config.seed)
    collator = partial(
        collate_sft_batch,
        pad_token_id=config.pad_token_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=False,
    )
    return sampler, train_loader, validation_loader


def run_experiment(
    config: ExperimentConfig,
    *,
    stop_after_steps: int | None = None,
) -> Path:
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError("stop_after_steps must be positive")

    run_dir = initialize_run(config)
    try:
        for artifact in (config.train_artifact, config.validation_artifact):
            if not artifact.is_file():
                raise FileNotFoundError(f"SFT artifact does not exist: {artifact}")
        _write_or_validate_data_manifest(run_dir, config)
        update_run_status(
            run_dir,
            "running",
            stop_after_steps=stop_after_steps,
        )
        set_reproducible_seed(config.seed)
        sampler, train_loader, validation_loader = _build_dataloaders(config)
        tokenizer = load_tokenizer(config)
        model = load_model(config)
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        state = train(
            config=config,
            run_dir=run_dir,
            model=model,
            train_dataloader=train_loader,
            validation_dataloader=validation_loader,
            sampler=sampler,
            processing_class=tokenizer,
            stop_after_steps=stop_after_steps,
        )
        interrupted = (
            stop_after_steps is not None
            and state.optimizer_step >= stop_after_steps
            and (config.max_steps is None or state.optimizer_step < config.max_steps)
        )
        update_run_status(
            run_dir,
            "interrupted" if interrupted else "completed",
            optimizer_step=state.optimizer_step,
            valid_tokens_seen=state.valid_tokens_seen,
        )
    except TrainingSoftStop as error:
        write_failure_snapshot(
            run_dir,
            error,
            checkpoint=str(error.checkpoint.resolve()),
            optimizer_step=error.state.optimizer_step,
            stop_kind="soft",
        )
        update_run_status(
            run_dir,
            "soft_stopped",
            checkpoint=str(error.checkpoint.resolve()),
            optimizer_step=error.state.optimizer_step,
            reason=error.reason,
        )
        raise
    except BaseException as error:
        write_failure_snapshot(run_dir, error)
        update_run_status(
            run_dir,
            "failed",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise
    return run_dir
