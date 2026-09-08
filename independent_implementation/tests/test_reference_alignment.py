import copy
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from post_training_core.data import collate_sft_batch
from post_training_core.engine import run_optimizer_step


def _tiny_qwen():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
    )


def _samples():
    return [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
        },
        {
            "input_ids": [2, 3, 4],
            "labels": [-100, 3, 4],
        },
    ]


def test_collator_matches_trl_018_completion_mask_contract():
    samples = _samples()
    trl_samples = [
        {
            "input_ids": sample["input_ids"],
            "completion_mask": [int(label != -100) for label in sample["labels"]],
        }
        for sample in samples
    ]

    local = collate_sft_batch(samples, pad_token_id=0)
    reference = DataCollatorForLanguageModeling(
        pad_token_id=0,
        completion_only_loss=True,
    )(trl_samples)

    assert set(local) == set(reference)
    for name in local:
        torch.testing.assert_close(local[name], reference[name])


def test_accumulated_update_matches_transformers_causal_loss_contract():
    torch.manual_seed(0)
    reference_model = _tiny_qwen()
    local_model = copy.deepcopy(reference_model)
    collator = partial(collate_sft_batch, pad_token_id=0)
    micro_batches = [collator([sample]) for sample in _samples()]
    total_tokens = sum(
        int((batch["labels"][:, 1:] != -100).sum()) for batch in micro_batches
    )

    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
    reference_optimizer.zero_grad(set_to_none=True)
    for batch in micro_batches:
        outputs = reference_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            num_items_in_batch=total_tokens,
        )
        outputs.loss.backward()
    reference_optimizer.step()

    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)
    scheduler = LambdaLR(local_optimizer, lr_lambda=lambda _: 1.0)
    metrics = run_optimizer_step(
        model=local_model,
        optimizer=local_optimizer,
        scheduler=scheduler,
        micro_batches=micro_batches,
        device=torch.device("cpu"),
        dtype="float32",
        max_grad_norm=1.0e6,
    )

    assert metrics.valid_tokens == total_tokens
    for expected, actual in zip(
        reference_model.parameters(), local_model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
