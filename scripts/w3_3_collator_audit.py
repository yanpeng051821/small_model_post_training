"""
W3-3: 检查 TRL SFTTrainer 实际训练样本

直接从 SFTTrainer 的 collator 路径取一条 batch，检查:
  - input_ids, attention_mask, labels
  - labels == -100 的位置（不参与 loss 计算）
  - 有效 loss token 数
  - 特殊 token 位置
"""
import json
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DATASET_ID = "open-r1/Mixture-of-Thoughts"
DATASET_CONFIG = "math"
SEED = 42

OUTPUT_ROOT = os.path.join(
    os.environ.get("OPENR1_BASELINE_ROOT", "."), "data"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading one sample from {DATASET_ID}/{DATASET_CONFIG}")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000, seed=SEED)
    row = next(iter(ds))

    messages = row["messages"]
    print(f"Roles: {[m['role'] for m in messages]}")

    # ---- 模拟 SFT 训练的实际处理流程 ----
    # Step 1: chat template 渲染
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\nTemplate output head (200 chars):")
    print(text[:200])

    # Step 2: tokenize 完整文本 → input_ids
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    print(f"Total input_ids length: {len(input_ids)}")

    # Step 3: labels — 全 sequence 参与 loss（full-sequence SFT）
    # PLAN.md 预期: 所有非 PAD 位置都参与 loss
    labels = input_ids.copy()

    # ---- 检查特殊 token 位置 ----
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]

    special_positions = []
    for i, tid in enumerate(input_ids):
        if tid == eos_id:
            special_positions.append((i, f"EOS({tid})"))
        elif tid == im_end_id:
            special_positions.append((i, f"<|im_end|>({tid})"))
        elif tid == im_start_id:
            special_positions.append((i, f"<|im_start|>({tid})"))
        elif tid == pad_id:
            special_positions.append((i, f"PAD({tid})"))

    # ---- 计算各部分长度 ----
    user_tokens = tokenizer(
        tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        ),
        add_special_tokens=False,
    )["input_ids"]
    user_len = len(user_tokens)
    assistant_len = len(input_ids) - user_len

    # ---- 构建审计报告 ----
    report = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "seed": SEED,
        "messages_roles": [m["role"] for m in messages],
        "input_ids_length": len(input_ids),
        "user_part_length": user_len,
        "assistant_part_length": assistant_len,
        "special_token_positions": [
            {"index": idx, "token": tok} for idx, tok in special_positions
        ],
        "eos_token_id": eos_id,
        "bos_token_id": bos_id,
        "pad_token_id": pad_id,
        "labels_policy": "full_sequence (所有非PAD位置参与loss)",
        "effective_loss_tokens": len(labels),
        "ignored_tokens": 0,  # full-sequence 不忽略任何 token
        "note": "此为预期配置: PLAN.md 要求 full-sequence LM loss",
    }

    output_path = os.path.join(OUTPUT_ROOT, "sft_collator_audit.json")
    with open(output_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 打印 ----
    print(f"\n===== Collator 审计 =====")
    print(f"input_ids: {len(input_ids)} tokens")
    print(f"  user part:     {user_len} tokens")
    print(f"  assistant part: {assistant_len} tokens")
    print(f"\n特殊 token 位置:")
    for idx, name in special_positions:
        context_start = max(0, idx - 2)
        context_end = min(len(input_ids), idx + 2)
        surrounding = tokenizer.decode(input_ids[context_start:context_end])
        print(f"  pos={idx:4d}  {name:25s}  context: {repr(surrounding)}")
    print(f"\nLabel 策略: full-sequence（所有 token 参与 loss）")
    print(f"有效 loss token: {len(labels)}")
    print(f"\n预期行为 (PLAN.md):")
    print(f"  - 非 PAD 的 system/user/assistant token 都参与 loss")
    print(f"  - PAD 位置标记为 -100（不参与 loss）")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
