from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from post_training_core.sft import (
    masked_sft_loss_sum_and_count,
)


def forward_sft_batch(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an HF-style forward pass while keeping labels for the local loss."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    return outputs.logits, batch["labels"]


def sft_optimizer_step(
    model: nn.Module,
    optimizer: Optimizer,
    micro_batches: Iterable[Mapping[str, torch.Tensor]],
    max_grad_norm: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one token-normalized optimizer step over one or more micro-batches."""
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    micro_batches = list(micro_batches)
    if not micro_batches:
        raise ValueError("micro_batches must not be empty")
    expected_valid_token_count = sum(
        int(micro_batch["labels"][..., 1:].ne(-100).sum().item())
        for micro_batch in micro_batches
    )
    if expected_valid_token_count <= 0:
        raise ValueError("micro_batches must contain valid supervision tokens")

    optimizer.zero_grad(set_to_none=True)

    total_loss_sum = None
    total_valid_token_count = None

    for micro_batch in micro_batches:
        logits, labels = forward_sft_batch(model, micro_batch)

        loss_sum, valid_token_count = masked_sft_loss_sum_and_count(
            logits,
            labels,
        )

        (loss_sum / expected_valid_token_count).backward()

        if total_loss_sum is None:
            total_loss_sum = loss_sum.detach()
            total_valid_token_count = valid_token_count.detach()
        else:
            total_loss_sum = total_loss_sum + loss_sum.detach()
            total_valid_token_count = (
                total_valid_token_count + valid_token_count.detach()
            )

    if total_valid_token_count.item() != expected_valid_token_count:
        raise RuntimeError("forward loss token count differs from batch labels")

    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

    optimizer.step()

    mean_loss = total_loss_sum / total_valid_token_count
    return mean_loss, total_valid_token_count
