"""Supervised fine-tuning core computations."""

import torch
import torch.nn.functional as F


def _validate_sft_inputs(logits: torch.Tensor, labels: torch.Tensor):
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")

    if labels.ndim != 2:
        raise ValueError("labels must have shape [B, T]")

    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")

    if labels.dtype != torch.long:
        raise ValueError("labels must have dtype torch.long")

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must match on batch and sequence dimensions"
        )

    if logits.size(1) < 2:
        raise ValueError("sequence length must be at least 2")


def masked_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return mean next-token loss over non-ignored target tokens."""

    loss_sum, valid_token_count = masked_sft_loss_sum_and_count(
        logits, labels, ignore_index
    )

    return loss_sum / valid_token_count


def masked_sft_loss_sum_and_count(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
):
    _validate_sft_inputs(logits, labels)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != ignore_index

    # any:至少有一个存在就行，all：是否全部，只有全部元素是true，结果就是true
    if not valid.any():
        raise ValueError("all tokens are ignored")

    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )

    valid_token_count = valid.sum()

    return loss_sum, valid_token_count
