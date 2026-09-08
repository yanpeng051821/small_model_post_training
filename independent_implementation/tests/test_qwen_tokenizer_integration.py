import pytest
from transformers import AutoTokenizer

from post_training_core import build_single_turn_sft_sample

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "311c62e88814bff7206909ccd330bab0a784743b"


@pytest.mark.integration
def test_builds_labels_from_pinned_qwen_chat_template():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        local_files_only=True,
    )
    user_message = {"role": "user", "content": "What is 1 + 1?"}
    assistant_message = {"role": "assistant", "content": "1 + 1 = 2."}

    sample = build_single_turn_sft_sample(
        tokenizer,
        user_message,
        assistant_message,
    )
    prompt_ids = tokenizer.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [user_message, assistant_message],
        tokenize=True,
        add_generation_prompt=False,
    )
    assistant_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_start = len(prompt_ids)
    assistant_end = full_ids.index(
        assistant_end_token_id,
        assistant_start,
    )

    expected_labels = [-100] * len(full_ids)
    expected_labels[assistant_start : assistant_end + 1] = full_ids[
        assistant_start : assistant_end + 1
    ]

    assert "{% generation" not in tokenizer.chat_template
    assert sample == {
        "input_ids": full_ids,
        "labels": expected_labels,
    }
    assert sample["labels"][assistant_start] != -100
    assert sample["labels"][assistant_end] == assistant_end_token_id
    assert sample["labels"][assistant_end + 1] == -100
