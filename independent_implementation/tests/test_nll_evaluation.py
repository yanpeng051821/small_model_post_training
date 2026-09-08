from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import Qwen3Config, Qwen3ForCausalLM

from post_training_core.data import TokenizedSFTDataset, collate_sft_batch
from post_training_core.engine import evaluate_sft_nll
from post_training_core.nll_evaluation import evaluate_sft_nll_records


def _model():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
        )
    )


def _loader(batch_size=1):
    dataset = TokenizedSFTDataset(
        [
            {
                "sample_id": "a",
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
            },
            {
                "sample_id": "b",
                "input_ids": [3, 4, 5, 6],
                "labels": [-100, -100, 5, 6],
            },
        ]
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_sft_batch, pad_token_id=0),
    )


def test_per_record_nll_matches_existing_global_token_weighted_evaluation():
    torch.manual_seed(0)
    model = _model()

    summary, records = evaluate_sft_nll_records(
        model,
        _loader(),
        device=torch.device("cpu"),
        dtype="float32",
    )
    expected_nll, expected_tokens = evaluate_sft_nll(
        model,
        _loader(batch_size=2),
        device=torch.device("cpu"),
        dtype="float32",
    )

    assert [record["sample_id"] for record in records] == ["a", "b"]
    assert summary["valid_tokens"] == expected_tokens == 4
    assert summary["mean_nll"] == pytest.approx(expected_nll, rel=1e-6)


def test_per_record_nll_rejects_batches_larger_than_one():
    with pytest.raises(ValueError, match="batch_size=1"):
        evaluate_sft_nll_records(
            _model(),
            _loader(batch_size=2),
            device=torch.device("cpu"),
            dtype="float32",
        )
