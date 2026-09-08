"""Per-record and token-weighted NLL evaluation for frozen SFT artifacts."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from post_training_core.engine import _autocast_context, move_batch_to_device
from post_training_core.sft import masked_sft_loss_sum_and_count
from post_training_core.training import forward_sft_batch


@torch.no_grad()
def evaluate_sft_nll_records(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    dtype: str,
) -> tuple[dict, list[dict]]:
    was_training = model.training
    model.eval()
    records = []
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0

    for host_batch in dataloader:
        sample_ids = host_batch.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) != 1:
            raise ValueError("per-record NLL evaluation requires batch_size=1")
        batch: Mapping[str, object] = move_batch_to_device(host_batch, device)
        with _autocast_context(device, dtype):
            logits, labels = forward_sft_batch(model, batch)
            loss_sum, token_count = masked_sft_loss_sum_and_count(logits, labels)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(
                f"non-finite validation NLL for sample {sample_ids[0]}"
            )
        count = int(token_count.item())
        loss = float(loss_sum.item())
        records.append(
            {
                "loss_sum": loss,
                "mean_nll": loss / count,
                "sample_id": sample_ids[0],
                "valid_tokens": count,
            }
        )
        total_loss += loss_sum.double()
        total_tokens += count

    if was_training:
        model.train()
    if not records or total_tokens <= 0:
        raise RuntimeError("validation set has no evaluable supervision tokens")
    return (
        {
            "mean_nll": float((total_loss / total_tokens).item()),
            "record_count": len(records),
            "total_loss_sum": float(total_loss.item()),
            "valid_tokens": total_tokens,
        },
        records,
    )
