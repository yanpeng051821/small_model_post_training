"""
Week 2-3: SFT / GRPO 真实样本审计

目的：下载训练数据各一条，检查数据结构和 token 长度。
     - SFT: open-r1/Mixture-of-Thoughts (math config)
     - GRPO: open-r1/OpenR1-Math-220k
"""
import os
import sys
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
OUTPUT_FILE = os.path.join(
    os.environ.get("OPENR1_BASELINE_ROOT", "."),
    "setup", "real_sample_audit.txt"
)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ========================================
    # 1. SFT 样本检查
    # ========================================
    print("=" * 60)
    print("1. SFT 样本 (Mixture-of-Thoughts, math)")
    print("=" * 60)

    # streaming=True: 只下载一条，不加载整个数据集
    sft = load_dataset("open-r1/Mixture-of-Thoughts", "math", split="train", streaming=True)
    sft_row = next(iter(sft))

    print(f"SFT 字段名: {sorted(sft_row.keys())}")
    print(f"SFT 角色序列: {[m.get('role') for m in sft_row['messages']]}")

    # token 长度
    sft_text = tokenizer.apply_chat_template(sft_row["messages"], tokenize=False)
    sft_ids = tokenizer(sft_text, add_special_tokens=False)["input_ids"]
    print(f"SFT 总 token 数: {len(sft_ids)}")
    print(f"SFT 开头 300 字符: {repr(sft_text[:300])}")
    print(f"SFT 结尾 300 字符: {repr(sft_text[-300:])}")

    # 检查和 PLAN.md 里的预期是否一致
    assert "messages" in sft_row, "FAIL: SFT 数据缺少 messages 字段"
    assert len(sft_row["messages"]) >= 2, f"FAIL: SFT 消息数 {len(sft_row['messages'])} < 2"
    print("SFT 字段检查: PASSED")

    print()

    # ========================================
    # 2. GRPO 样本检查
    # ========================================
    print("=" * 60)
    print("2. GRPO 样本 (OpenR1-Math-220k)")
    print("=" * 60)

    grpo = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    grpo_row = next(iter(grpo))

    print(f"GRPO 字段名: {sorted(grpo_row.keys())}")

    problem = str(grpo_row.get("problem", ""))
    solution = str(grpo_row.get("solution", ""))
    print(f"Problem 开头 300 字符: {repr(problem[:300])}")
    print(f"Solution 开头 300 字符: {repr(solution[:300])}")

    # 检查和 PLAN.md 里的预期是否一致
    assert "problem" in grpo_row, "FAIL: GRPO 数据缺少 problem 字段"
    assert "solution" in grpo_row, "FAIL: GRPO 数据缺少 solution 字段"
    print("GRPO 字段检查: PASSED")

    print()

    # ========================================
    # 3. 总结：和 PLAN.md 里预期对比
    # ========================================
    print("=" * 60)
    print("3. 审计结论")
    print("=" * 60)
    plan_sft = "open-r1/Mixture-of-Thoughts, math config, train split"
    plan_grpo = "open-r1/OpenR1-Math-220k, default config, train split"
    print(f"PLAN.md SFT:  {plan_sft}")
    print(f"PLAN.md GRPO: {plan_grpo}")
    print()
    print(f"SFT 数据字段: messages (roles: {[m.get('role') for m in sft_row['messages']]})")
    print(f"SFT 示例 token 长度: {len(sft_ids)}")
    print(f"GRPO 数据字段: problem + solution")
    print(f"GRPO solution 仅作为 reward ground truth，不直接作为训练文本")
    print()
    print("审计结果: PASSED — 字段结构与 PLAN.md 一致，可以进入 W2-4")


if __name__ == "__main__":
    main()
