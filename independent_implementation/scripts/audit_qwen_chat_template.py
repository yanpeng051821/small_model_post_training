"""Inspect Qwen3 chat-template rendering and assistant-token boundaries."""

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "311c62e88814bff7206909ccd330bab0a784743b"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    user_message = {"role": "user", "content": "What is 1 + 1?"}
    assistant_message = {"role": "assistant", "content": "1 + 1 = 2."}

    prompt_messages = [user_message]
    full_messages = [user_message, assistant_message]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    assistant_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print(f"model: {MODEL_ID}")
    print(f"revision: {REVISION}")
    print(f"has generation block: {'{% generation' in tokenizer.chat_template}")
    print(f"eos token: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
    print(f"assistant end token: '<|im_end|>' ({assistant_end_token_id})")
    print("\nprompt text:")
    print(repr(prompt_text))
    print("\nfull text:")
    print(repr(full_text))

    prefix_matches = full_ids[: len(prompt_ids)] == prompt_ids
    print(f"\nprompt token count: {len(prompt_ids)}")
    print(f"full token count: {len(full_ids)}")
    print(f"prompt is full prefix: {prefix_matches}")
    if not prefix_matches:
        raise RuntimeError("prompt token ids are not a prefix of full token ids")

    assistant_start = len(prompt_ids)
    assistant_end = next(
        index
        for index in range(assistant_start, len(full_ids))
        if full_ids[index] == assistant_end_token_id
    )

    print(f"assistant content starts at token index: {assistant_start}")
    print(f"assistant end token index: {assistant_end}")
    print("\nindex | region     | token_id | token")
    print("------|------------|----------|----------------")
    for index, token_id in enumerate(full_ids):
        if index < assistant_start:
            region = "prompt"
        elif index <= assistant_end:
            region = "assistant"
        else:
            region = "trailing"
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{index:>5} | {region:<10} | {token_id:>8} | {ascii(token)}")

    native_mask_result = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    native_mask = native_mask_result["assistant_masks"]
    print(f"\nnative assistant-mask true count: {sum(native_mask)}")


if __name__ == "__main__":
    main()
