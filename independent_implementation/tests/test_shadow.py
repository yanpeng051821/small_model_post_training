import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from post_training_core.data import collate_sft_batch
from post_training_core.shadow import compare_sft_batch_with_hf_reference


def test_shadow_matches_transformers_causal_lm_loss_and_gradients():
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(
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
    batch = collate_sft_batch(
        [
            {
                "sample_id": "a",
                "input_ids": [1, 2, 3, 4],
                "labels": [-100, -100, 3, 4],
            },
            {
                "sample_id": "b",
                "input_ids": [2, 3, 4],
                "labels": [-100, 3, 4],
            },
        ],
        pad_token_id=0,
    )

    result = compare_sft_batch_with_hf_reference(model, batch)

    assert result["passed"] is True
    assert result["valid_tokens"] == 4
    assert result["absolute_loss_difference"] < 1e-6
    assert result["maximum_probe_absolute_difference"] < 1e-6
