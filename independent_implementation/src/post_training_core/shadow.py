"""Numerical shadow comparison against the Transformers causal-LM loss path."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from post_training_core.sft import masked_sft_loss_sum_and_count
from post_training_core.training import forward_sft_batch


@dataclass(frozen=True)
class GradientSketch:
    norm: float
    probes: tuple[float, ...]


def _gradient_sketch(model: nn.Module, probes_per_parameter: int = 4) -> GradientSketch:
    squared_norm = 0.0
    probes = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float().reshape(-1)
        squared_norm += float(torch.sum(gradient * gradient).item())
        if gradient.numel() <= probes_per_parameter:
            indices = range(gradient.numel())
        else:
            indices = torch.linspace(
                0,
                gradient.numel() - 1,
                probes_per_parameter,
                dtype=torch.long,
            ).tolist()
        probes.extend(float(gradient[index].item()) for index in indices)
    if not probes:
        raise RuntimeError("shadow comparison produced no gradients")
    return GradientSketch(norm=math.sqrt(squared_norm), probes=tuple(probes))


def compare_sft_batch_with_hf_reference(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    loss_atol: float = 1e-6,
    loss_rtol: float = 1e-5,
    gradient_atol: float = 1e-6,
    gradient_rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare our loss with the HF path invoked by TRL 0.18 SFTTrainer."""
    was_training = model.training
    model.eval()

    model.zero_grad(set_to_none=True)
    local_logits, labels = forward_sft_batch(model, batch)
    local_sum, valid_tokens = masked_sft_loss_sum_and_count(local_logits, labels)
    reference_valid_tokens = batch["labels"][..., 1:].ne(-100).sum()
    if not torch.equal(valid_tokens, reference_valid_tokens):
        raise RuntimeError("local and reference causal-shift token counts differ")
    local_loss = local_sum / valid_tokens
    local_loss.backward()
    local_gradients = _gradient_sketch(model)

    model.zero_grad(set_to_none=True)
    reference_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        num_items_in_batch=valid_tokens,
    )
    reference_loss = reference_outputs.loss
    reference_loss.backward()
    reference_gradients = _gradient_sketch(model)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    local_probe = torch.tensor(local_gradients.probes, dtype=torch.float64)
    reference_probe = torch.tensor(reference_gradients.probes, dtype=torch.float64)
    loss_matches = torch.isclose(
        local_loss.detach().double(),
        reference_loss.detach().double(),
        atol=loss_atol,
        rtol=loss_rtol,
    ).item()
    probes_match = torch.allclose(
        local_probe,
        reference_probe,
        atol=gradient_atol,
        rtol=gradient_rtol,
    )
    norm_matches = math.isclose(
        local_gradients.norm,
        reference_gradients.norm,
        abs_tol=gradient_atol,
        rel_tol=gradient_rtol,
    )
    return {
        "passed": bool(loss_matches and probes_match and norm_matches),
        "valid_tokens": int(valid_tokens.item()),
        "local_loss": float(local_loss.detach().item()),
        "reference_loss": float(reference_loss.detach().item()),
        "absolute_loss_difference": abs(
            float(local_loss.detach().item()) - float(reference_loss.detach().item())
        ),
        "local_gradients": asdict(local_gradients),
        "reference_gradients": asdict(reference_gradients),
        "maximum_probe_absolute_difference": float(
            torch.max(torch.abs(local_probe - reference_probe)).item()
        ),
        "tolerances": {
            "loss_atol": loss_atol,
            "loss_rtol": loss_rtol,
            "gradient_atol": gradient_atol,
            "gradient_rtol": gradient_rtol,
        },
    }
