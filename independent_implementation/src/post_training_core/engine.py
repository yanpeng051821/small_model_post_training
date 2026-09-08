"""Single-device SFT engine with explicit token-level normalization."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import deque
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_scheduler

from post_training_core.checkpointing import (
    TrainingState,
    load_training_checkpoint,
    prune_training_checkpoints,
    save_training_checkpoint,
)
from post_training_core.config import ExperimentConfig
from post_training_core.data import StatefulRandomSampler
from post_training_core.observability import JsonlMetricWriter
from post_training_core.sft import masked_sft_loss_sum_and_count
from post_training_core.training import forward_sft_batch


@dataclass(frozen=True)
class StepMetrics:
    mean_loss: float
    valid_tokens: int
    grad_norm: float
    learning_rate: float
    max_cuda_memory_mb: float
    step_seconds: float
    valid_tokens_per_second: float


class TrainingSoftStop(RuntimeError):
    def __init__(self, reason: str, state: TrainingState, checkpoint: Path) -> None:
        super().__init__(reason)
        self.reason = reason
        self.state = state
        self.checkpoint = checkpoint


class TrainingGuard:
    def __init__(self) -> None:
        self.loss_history: deque[float] = deque(maxlen=20)
        self.gradient_history: deque[float] = deque(maxlen=20)
        self.high_loss_streak = 0
        self.gradient_spike_streak = 0

    def observe(self, metrics: StepMetrics) -> str | None:
        if (
            len(self.loss_history) == self.loss_history.maxlen
            and metrics.mean_loss > 5 * statistics.median(self.loss_history)
        ):
            self.high_loss_streak += 1
        else:
            self.high_loss_streak = 0
        self.loss_history.append(metrics.mean_loss)

        if (
            len(self.gradient_history) == self.gradient_history.maxlen
            and metrics.grad_norm > 10 * statistics.median(self.gradient_history)
        ):
            self.gradient_spike_streak += 1
        else:
            self.gradient_spike_streak = 0
        self.gradient_history.append(metrics.grad_norm)

        if self.high_loss_streak >= 3:
            return "loss exceeded five times the prior 20-step median for 3 steps"
        if self.gradient_spike_streak >= 3:
            return "gradient norm exceeded ten times the prior 20-step median for 3 steps"
        return None


def _restore_training_guard(run_dir: Path) -> TrainingGuard:
    guard = TrainingGuard()
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return guard
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") != "train":
            continue
        guard.observe(
            StepMetrics(
                mean_loss=record["mean_loss"],
                valid_tokens=record["valid_tokens"],
                grad_norm=record["grad_norm"],
                learning_rate=record["learning_rate"],
                max_cuda_memory_mb=record["max_cuda_memory_mb"],
                step_seconds=record.get("step_seconds", 0.0),
                valid_tokens_per_second=record.get(
                    "valid_tokens_per_second", 0.0
                ),
            )
        )
    return guard


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_batch_to_device(
    batch: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _autocast_context(device: torch.device, dtype: str):
    if dtype == "float32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("bfloat16 training requires a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _global_grad_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("no trainable parameter received a gradient")
    return torch.stack(gradients).norm(2)


def run_optimizer_step(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    micro_batches: list[Mapping[str, torch.Tensor]],
    device: torch.device,
    dtype: str,
    max_grad_norm: float,
) -> StepMetrics:
    if not micro_batches:
        raise ValueError("micro_batches must not be empty")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    detached_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    expected_valid_tokens = sum(
        int(host_batch["labels"][..., 1:].ne(-100).sum().item())
        for host_batch in micro_batches
    )
    if expected_valid_tokens <= 0:
        raise RuntimeError("optimizer step has no valid supervision tokens")
    valid_tokens = torch.zeros((), dtype=torch.long, device=device)

    for host_batch in micro_batches:
        batch = move_batch_to_device(host_batch, device)
        try:
            with _autocast_context(device, dtype):
                logits, labels = forward_sft_batch(model, batch)
                loss_sum, token_count = masked_sft_loss_sum_and_count(logits, labels)
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("non-finite SFT loss")
            (loss_sum / expected_valid_tokens).backward()
            detached_loss_sum += loss_sum.detach().double()
            valid_tokens += token_count.detach()
        except BaseException as error:
            sample_ids = batch.get("sample_ids")
            if sample_ids is not None:
                error.add_note(f"SFT sample_ids: {sample_ids}")
            raise

    if valid_tokens.item() != expected_valid_tokens:
        raise RuntimeError("forward loss token count differs from batch labels")

    grad_norm = _global_grad_norm(model.parameters())
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite gradient norm")
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
        error_if_nonfinite=True,
    )

    learning_rate = optimizer.param_groups[0]["lr"]
    optimizer.step()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.isfinite(parameter).all():
            raise FloatingPointError(
                f"non-finite parameter after optimizer step: {name}"
            )
    scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - started
    mean_loss = detached_loss_sum / valid_tokens
    max_cuda_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return StepMetrics(
        mean_loss=float(mean_loss.item()),
        valid_tokens=int(valid_tokens.item()),
        grad_norm=float(grad_norm.item()),
        learning_rate=float(learning_rate),
        max_cuda_memory_mb=float(max_cuda_memory_mb),
        step_seconds=step_seconds,
        valid_tokens_per_second=int(valid_tokens.item()) / step_seconds,
    )


@torch.no_grad()
def evaluate_sft_nll(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    dtype: str,
) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    valid_tokens = torch.zeros((), dtype=torch.long, device=device)
    for host_batch in dataloader:
        batch = move_batch_to_device(host_batch, device)
        with _autocast_context(device, dtype):
            logits, labels = forward_sft_batch(model, batch)
            batch_loss, batch_tokens = masked_sft_loss_sum_and_count(logits, labels)
        loss_sum += batch_loss.double()
        valid_tokens += batch_tokens
    if was_training:
        model.train()
    if valid_tokens.item() <= 0:
        raise RuntimeError("validation set has no supervision tokens")
    mean_nll = loss_sum / valid_tokens
    if not torch.isfinite(mean_nll):
        raise FloatingPointError("non-finite validation NLL")
    return float(mean_nll.item()), int(valid_tokens.item())


def build_optimizer(config: ExperimentConfig, model: nn.Module) -> Optimizer:
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def compute_total_optimizer_steps(
    config: ExperimentConfig,
    train_batches_per_epoch: int,
) -> int:
    steps_per_epoch = math.ceil(
        train_batches_per_epoch / config.gradient_accumulation_steps
    )
    planned = steps_per_epoch * config.num_train_epochs
    return min(planned, config.max_steps) if config.max_steps else planned


def build_scheduler(
    config: ExperimentConfig,
    optimizer: Optimizer,
    total_steps: int,
) -> LRScheduler:
    # Match TrainingArguments.get_warmup_steps used by TRL/Transformers.
    warmup_steps = math.ceil(total_steps * config.warmup_ratio)
    return get_scheduler(
        config.scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs={"min_lr_rate": config.min_lr_ratio},
    )


def train(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    sampler: StatefulRandomSampler,
    processing_class=None,
    stop_after_steps: int | None = None,
) -> TrainingState:
    device = resolve_device(config.device)
    model.to(device)
    model.train()
    optimizer = build_optimizer(config, model)
    total_steps = compute_total_optimizer_steps(config, len(train_dataloader))
    scheduler = build_scheduler(config, optimizer, total_steps)
    state = TrainingState()

    if config.resume_from_checkpoint is not None:
        state = load_training_checkpoint(
            config.resume_from_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_config_hash=config.semantic_hash(),
        )

    writer = JsonlMetricWriter(run_dir / "metrics.jsonl")
    guard = _restore_training_guard(run_dir)
    last_validation_step = None
    if config.eval_on_start and state.optimizer_step == 0:
        validation_nll, validation_tokens = evaluate_sft_nll(
            model,
            validation_dataloader,
            device=device,
            dtype=config.dtype,
        )
        writer.write(
            {
                "event": "validation",
                "optimizer_step": 0,
                "nll": validation_nll,
                "valid_tokens": validation_tokens,
            }
        )
        last_validation_step = 0
    iterator = iter(train_dataloader)
    while state.optimizer_step < total_steps:
        micro_batches = []
        for _ in range(config.gradient_accumulation_steps):
            try:
                micro_batches.append(next(iterator))
            except StopIteration:
                break

        if not micro_batches:
            if sampler.epoch + 1 >= config.num_train_epochs:
                break
            sampler.advance_epoch()
            iterator = iter(train_dataloader)
            continue

        metrics = run_optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            micro_batches=micro_batches,
            device=device,
            dtype=config.dtype,
            max_grad_norm=config.max_grad_norm,
        )
        sampler.commit(
            sum(int(batch["input_ids"].shape[0]) for batch in micro_batches)
        )
        state.optimizer_step += 1
        state.valid_tokens_seen += metrics.valid_tokens
        soft_stop_reason = guard.observe(metrics)
        step_sample_ids = [
            sample_id
            for batch in micro_batches
            for sample_id in batch["sample_ids"]
        ]

        if state.optimizer_step % config.log_every_steps == 0:
            writer.write(
                {
                    "event": "train",
                    "optimizer_step": state.optimizer_step,
                    "sample_ids": step_sample_ids,
                    "valid_tokens_seen": state.valid_tokens_seen,
                    **metrics.__dict__,
                }
            )

        if state.optimizer_step % config.eval_every_steps == 0:
            validation_nll, validation_tokens = evaluate_sft_nll(
                model,
                validation_dataloader,
                device=device,
                dtype=config.dtype,
            )
            writer.write(
                {
                    "event": "validation",
                    "optimizer_step": state.optimizer_step,
                    "nll": validation_nll,
                    "valid_tokens": validation_tokens,
                }
            )
            last_validation_step = state.optimizer_step

        should_save = state.optimizer_step % config.save_every_steps == 0
        should_stop = (
            stop_after_steps is not None and state.optimizer_step >= stop_after_steps
        )
        if (
            should_save
            or should_stop
            or soft_stop_reason is not None
            or state.optimizer_step == total_steps
        ):
            checkpoint_path = (
                run_dir / "checkpoints" / f"step-{state.optimizer_step:08d}"
            )
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                training_state=state,
                config_hash=config.semantic_hash(),
                processing_class=processing_class,
            )
            prune_training_checkpoints(
                run_dir / "checkpoints",
                limit=config.save_total_limit,
            )
        if soft_stop_reason is not None:
            writer.write(
                {
                    "event": "soft_stop",
                    "optimizer_step": state.optimizer_step,
                    "reason": soft_stop_reason,
                }
            )
            raise TrainingSoftStop(
                soft_stop_reason,
                state,
                checkpoint_path,
            )
        if should_stop:
            break

    if state.optimizer_step != last_validation_step:
        validation_nll, validation_tokens = evaluate_sft_nll(
            model,
            validation_dataloader,
            device=device,
            dtype=config.dtype,
        )
        writer.write(
            {
                "event": "validation",
                "optimizer_step": state.optimizer_step,
                "nll": validation_nll,
                "valid_tokens": validation_tokens,
            }
        )

    return state
