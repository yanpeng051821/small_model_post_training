"""Single-epoch TRL execution over the frozen assistant-only artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from post_training_core.config import ExperimentConfig


class RemainingIndices(Sampler):
    def __init__(self, size: int, offset: int):
        self.size, self.offset = size, offset

    def __iter__(self):
        return iter(range(self.offset, self.size))

    def __len__(self):
        return self.size - self.offset


class FrozenOrderSFTTrainer(SFTTrainer):
    _resume_offset = 0

    def train(self, resume_from_checkpoint=None, **kwargs):
        self._resume_offset = 0
        if resume_from_checkpoint:
            state = json.loads(
                (Path(resume_from_checkpoint) / "trainer_state.json").read_text()
            )
            step = state["global_step"]
            if step >= self.args.max_steps:
                raise ValueError("checkpoint already reached the planned final step")
            self._resume_offset = (
                step
                * self.args.gradient_accumulation_steps
                * self.args.per_device_train_batch_size
            )
            if self._resume_offset >= len(self.train_dataset):
                raise ValueError("resume position is outside the frozen epoch")
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        source = self.train_dataset if train_dataset is None else train_dataset
        return RemainingIndices(len(source), self._resume_offset)

    def set_initial_training_values(self, args, dataloader, total_train_batch_size):
        values = list(
            super().set_initial_training_values(
                args, dataloader, total_train_batch_size
            )
        )
        # Keep the original schedule while the resumed loader exposes only the suffix.
        # Native 4.52.3 skip_first_batches leaves stale steps_in_epoch for tail sync.
        original_batches = math.ceil(
            len(self.train_dataset) / args.per_device_train_batch_size
        )
        values[0] = 1
        values[1] = math.ceil(original_batches / args.gradient_accumulation_steps)
        return tuple(values)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        loss = result[0] if return_outputs else result
        if not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite TRL loss")
        return result


class SaveAndStopCallback(TrainerCallback):
    def __init__(self, stop_after_steps: int | None = None):
        self.stop_after_steps = stop_after_steps

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite TRL gradient: {name}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                raise FloatingPointError(f"non-finite TRL parameter: {name}")
        if (
            self.stop_after_steps is not None
            and state.global_step >= self.stop_after_steps
        ):
            control.should_save = True
            control.should_training_stop = True
        if state.global_step >= state.max_steps:
            control.should_save = True
        return control


def build_trl_sft_args(
    config: ExperimentConfig, output_dir: Path, record_count: int
) -> SFTConfig:
    if config.num_train_epochs != 1:
        raise ValueError("frozen TRL runner supports exactly one epoch")
    if config.parameter_dtype != "float32":
        raise ValueError("frozen TRL runner requires FP32 master parameters")
    if record_count <= 0:
        raise ValueError("training dataset must not be empty")
    batches = math.ceil(record_count / config.per_device_train_batch_size)
    complete_steps = math.ceil(batches / config.gradient_accumulation_steps)
    if config.max_steps is not None and config.max_steps > complete_steps:
        raise ValueError("max_steps would repeat the frozen epoch")
    total_steps = config.max_steps or complete_steps
    # Freeze the complete update count, including the short final window.
    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=total_steps,
        num_train_epochs=1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        optim="adamw_torch",
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.scheduler_type,
        lr_scheduler_kwargs={"min_lr_rate": config.min_lr_ratio},
        warmup_steps=math.ceil(total_steps * config.warmup_ratio),
        max_grad_norm=config.max_grad_norm,
        bf16=config.dtype == "bfloat16",
        fp16=False,
        use_cpu=config.device == "cpu",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        ignore_data_skip=True,
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        packing=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=config.save_every_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.log_every_steps,
        logging_nan_inf_filter=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        disable_tqdm=True,
    )
