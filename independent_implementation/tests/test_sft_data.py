import copy

import pytest
import torch

from post_training_core import (
    build_assistant_only_labels,
    build_single_turn_sft_sample,
    collate_sft_batch,
)


def test_collates_variable_length_sft_samples():
    samples = [
        {
            "input_ids": [10, 11, 12, 13],
            "labels": [-100, -100, 12, 13],
        },
        {
            "input_ids": [20, 21, 22],
            "labels": [-100, 21, 22],
        },
    ]

    batch = collate_sft_batch(samples, pad_token_id=0)

    expected_input_ids = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 0],
        ],
        dtype=torch.long,
    )

    expected_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    expected_labels = torch.tensor(
        [
            [-100, -100, 12, 13],
            [-100, 21, 22, -100],
        ],
        dtype=torch.long,
    )

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "labels",
    }

    torch.testing.assert_close(batch["input_ids"], expected_input_ids)
    torch.testing.assert_close(batch["attention_mask"], expected_attention_mask)
    torch.testing.assert_close(batch["labels"], expected_labels)


def test_collator_preserves_stable_sample_ids_for_failure_evidence():
    samples = [
        {
            "sample_id": "sample-a",
            "input_ids": [1, 2],
            "labels": [-100, 2],
        },
        {
            "sample_id": "sample-b",
            "input_ids": [3, 4],
            "labels": [-100, 4],
        },
    ]

    batch = collate_sft_batch(samples, pad_token_id=0)

    assert batch["sample_ids"] == ["sample-a", "sample-b"]


def test_collator_rejects_partially_missing_sample_ids():
    samples = [
        {"sample_id": "sample-a", "input_ids": [1, 2], "labels": [-100, 2]},
        {"input_ids": [3, 4], "labels": [-100, 4]},
    ]

    with pytest.raises(ValueError, match="sample_id"):
        collate_sft_batch(samples, pad_token_id=0)


@pytest.mark.parametrize(
    "sample",
    [
        {
            "input_ids": [10, 11, 12],
            "labels": [-100, 11],
        },
        {
            "input_ids": [10, 11],
            "labels": [-100, 11, 12],
        },
    ],
)
def test_rejects_mismatched_input_and_label_lengths(sample):
    with pytest.raises(ValueError, match="input_ids and labels must have equal length"):
        collate_sft_batch([sample], pad_token_id=0)


def test_rejects_empty_sample_batch():
    with pytest.raises(
        ValueError,
        match="samples must not be empty",
    ):
        collate_sft_batch(
            [],
            pad_token_id=0,
        )


def test_uses_custom_padding_values_without_mutating_samples():
    samples = [
        {
            "input_ids": [5, 6, 7],
            "labels": [-7, 6, 7],
        },
        {
            "input_ids": [8, 9],
            "labels": [-7, 9],
        },
    ]

    original_samples = copy.deepcopy(samples)

    batch = collate_sft_batch(
        samples,
        pad_token_id=99,
        ignore_index=-7,
    )

    assert samples == original_samples

    torch.testing.assert_close(
        batch["input_ids"],
        torch.tensor(
            [
                [5, 6, 7],
                [8, 9, 99],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        batch["labels"],
        torch.tensor(
            [
                [-7, 6, 7],
                [-7, 9, -7],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        batch["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )


def test_builds_assistant_only_labels():
    input_ids = [
        101,
        102,  # system/header
        201,
        202,
        203,  # user
        301,  # assistant header
        401,
        402,  # assistant completion
        500,  # assistant end token
    ]

    assistant_mask = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]

    labels = build_assistant_only_labels(input_ids, assistant_mask)

    assert labels == [
        -100,
        -100,  # system/header
        -100,
        -100,
        -100,  # user
        -100,  # assistant header
        401,
        402,  # assistant completion
        500,  # assistant end token
    ]


def test_rejects_mismatched_input_and_assistant_mask_lengths():
    input_ids = [10, 11, 12]
    assistant_mask = [False, True]

    with pytest.raises(
        ValueError,
        match="input_ids and assistant_mask must have equal length",
    ):
        build_assistant_only_labels(
            input_ids,
            assistant_mask,
        )


def test_rejects_mask_without_assistant_token():
    input_ids = [101, 200, 310]
    assistant_mask = [False, False, False]

    with pytest.raises(
        ValueError,
        match="assistant_mask must have at least one True",
    ):
        build_assistant_only_labels(
            input_ids,
            assistant_mask,
        )


def test_rejects_empty_inputs_for_assistant_labels():
    with pytest.raises(ValueError, match="input_ids must not be empty"):
        build_assistant_only_labels([], [])


def test_uses_custom_ignore_index_without_mutating_inputs():
    input_ids = [10, 20, 30, 40]
    assistant_mask = [False, False, True, True]

    original_input_ids = input_ids.copy()
    original_assistant_mask = assistant_mask.copy()

    labels = build_assistant_only_labels(
        input_ids,
        assistant_mask,
        ignore_index=-7,
    )

    assert labels == [-7, -7, 30, 40]
    assert input_ids == original_input_ids
    assert assistant_mask == original_assistant_mask


class FakeChatTokenizer:
    def __init__(self, prompt_ids=None, full_ids=None):
        self.calls = []
        self.prompt_ids = [10, 11, 12] if prompt_ids is None else prompt_ids

        self.full_ids = [10, 11, 12, 20, 21, 99, 30] if full_ids is None else full_ids

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )

        if add_generation_prompt:
            return self.prompt_ids

        return self.full_ids

    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return 99

        return None


def test_renders_prompt_and_full_conversation_separately():
    tokenizer = FakeChatTokenizer()

    user_message = {
        "role": "user",
        "content": "Question",
    }
    assistant_message = {
        "role": "assistant",
        "content": "Answer",
    }

    build_single_turn_sft_sample(
        tokenizer,
        user_message,
        assistant_message,
    )

    assert tokenizer.calls == [
        {
            "messages": [user_message],
            "tokenize": True,
            "add_generation_prompt": True,
        },
        {
            "messages": [user_message, assistant_message],
            "tokenize": True,
            "add_generation_prompt": False,
        },
    ]


def test_rejects_full_conversation_that_does_not_share_prompt_prefix():
    tokenizer = FakeChatTokenizer(
        prompt_ids=[10, 11, 12],
        full_ids=[10, 99, 12, 20, 21, 99, 30],
    )

    user_message = {
        "role": "user",
        "content": "Question",
    }
    assistant_message = {
        "role": "assistant",
        "content": "Answer",
    }

    with pytest.raises(
        ValueError, match="prompt token ids must be a prefix of full token ids"
    ):
        build_single_turn_sft_sample(tokenizer, user_message, assistant_message)


def test_rejects_missing_assistant_end_token():
    tokenizer = FakeChatTokenizer(
        prompt_ids=[10, 99, 12],
        full_ids=[10, 99, 12, 20, 21, 30],
    )

    user_message = {
        "role": "user",
        "content": "Question",
    }
    assistant_message = {
        "role": "assistant",
        "content": "Answer",
    }

    with pytest.raises(
        ValueError, match="assistant end token must appear after assistant start"
    ):
        build_single_turn_sft_sample(tokenizer, user_message, assistant_message)


def test_builds_single_turn_sft_sample():
    tokenizer = FakeChatTokenizer(
        prompt_ids=[10, 11, 12],
        full_ids=[10, 11, 12, 20, 21, 99, 30],
    )

    user_message = {
        "role": "user",
        "content": "Question",
    }
    assistant_message = {
        "role": "assistant",
        "content": "Answer",
    }

    sample = build_single_turn_sft_sample(
        tokenizer,
        user_message,
        assistant_message,
        ignore_index=-7,
    )

    assert sample == {
        "input_ids": [10, 11, 12, 20, 21, 99, 30],
        "labels": [-7, -7, -7, 20, 21, 99, -7],
    }


def test_rejects_unknown_assistant_end_token():
    tokenizer = FakeChatTokenizer()

    user_message = {
        "role": "user",
        "content": "Question",
    }
    assistant_message = {
        "role": "assistant",
        "content": "Answer",
    }

    with pytest.raises(
        ValueError,
        match="unknown assistant end token",
    ):
        build_single_turn_sft_sample(
            tokenizer,
            user_message,
            assistant_message,
            assistant_end_token="<|wrong_end|>",
        )
