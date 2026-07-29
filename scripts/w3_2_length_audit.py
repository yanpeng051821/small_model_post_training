"""
W3-2: SFT 数据长度审计

用真实 tokenizer + chat template 测量 1000 条样本的 token 长度分布，
输出分位数和候选 max_seq_length 下的截断率。

输出: data/sft_length_audit.json
"""
import json
import os
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

DATASET_ID = "open-r1/Mixture-of-Thoughts"
DATASET_CONFIG = "math"
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42
N_SAMPLES = 1000

OUTPUT_ROOT = os.path.join(
    os.environ.get("OPENR1_BASELINE_ROOT", "."), "data"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading dataset: {DATASET_ID}/{DATASET_CONFIG} (streaming)")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000, seed=SEED)

    lengths = []
    assistant_lengths = []
    eos_presence = []
    think_presence = []

    print(f"Tokenizing {N_SAMPLES} samples ...")
    for i, row in enumerate(ds):
        if i >= N_SAMPLES:
            break

        messages = row["messages"]

        # 整条消息转 token
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        all_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(all_ids))

        # 只看 assistant 部分（模型需要学的内容）
        user_text = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
        assistant_len = len(all_ids) - len(user_ids)
        assistant_lengths.append(assistant_len)

        # EOS 标签是否在 assistant 结尾
        eos_presence.append(all_ids[-1] == tokenizer.eos_token_id)

        # 是否包含 </think>
        think_presence.append("</think>" in text)

        if i % 200 == 0 and i > 0:
            print(f"  ... {i}/{N_SAMPLES}")

    lengths = np.array(lengths)
    assistant_lengths = np.array(assistant_lengths)

    # ---- 分位数 ----
    percentiles = {
        "min": int(np.min(lengths)),
        "p10": int(np.percentile(lengths, 10)),
        "p25": int(np.percentile(lengths, 25)),
        "p50": int(np.percentile(lengths, 50)),
        "p75": int(np.percentile(lengths, 75)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
    }

    assistant_percentiles = {
        "p50": int(np.percentile(assistant_lengths, 50)),
        "p90": int(np.percentile(assistant_lengths, 90)),
        "p95": int(np.percentile(assistant_lengths, 95)),
        "max": int(np.max(assistant_lengths)),
    }

    # ---- 截断率评估 ----
    truncation = {}
    for max_len in [1024, 2048, 3072, 4096, 6144, 8192]:
        truncated = np.sum(lengths > max_len)
        trunc_rate = truncated / len(lengths) * 100
        # 在这些截断样本中，assistant 部分还保留多少
        if truncated > 0:
            keep_ratio = np.mean(
                [max_len / length for length in lengths if length > max_len]
            )
        else:
            keep_ratio = 1.0
        truncation[f"max_len_{max_len}"] = {
            "truncated_count": int(truncated),
            "truncated_pct": round(trunc_rate, 1),
            "assistant_retencion_ratio": round(keep_ratio, 3),
        }

    # ---- 统计 ----
    stats = {
        "num_samples": int(len(lengths)),
        "mean_length": round(float(np.mean(lengths)), 1),
        "std_length": round(float(np.std(lengths)), 1),
        "eos_at_end_pct": round(np.mean(eos_presence) * 100, 1),
        "has_think_tag_pct": round(np.mean(think_presence) * 100, 1),
    }

    # ---- 输出 ----
    result = {
        "tokenizer": MODEL_ID,
        "seed": SEED,
        "stats": stats,
        "percentiles": percentiles,
        "assistant_percentiles": assistant_percentiles,
        "truncation_analysis": truncation,
    }

    output_path = os.path.join(OUTPUT_ROOT, "sft_length_audit.json")
    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- 打印摘要 ----
    print(f"\n===== 长度审计 =====")
    print(f"Samples: {stats['num_samples']}")
    print(f"Mean: {stats['mean_length']:.0f} tokens | Std: {stats['std_length']:.0f}")
    print(f"EOS at end: {stats['eos_at_end_pct']:.1f}% | Has </think>: {stats['has_think_tag_pct']:.1f}%")
    print(f"\nTotal length percentiles:")
    print(f"  P50={percentiles['p50']}  P75={percentiles['p75']}  "
          f"P90={percentiles['p90']}  P95={percentiles['p95']}  "
          f"P99={percentiles['p99']}  max={percentiles['max']}")
    print(f"Assistant length percentiles:")
    print(f"  P50={assistant_percentiles['p50']}  P90={assistant_percentiles['p90']}  "
          f"P95={assistant_percentiles['p95']}  max={assistant_percentiles['max']}")
    print(f"\nTruncation analysis:")
    for key, val in truncation.items():
        print(f"  {key}: {val['truncated_count']}/{len(lengths)} "
              f"({val['truncated_pct']:.1f}%) truncated, "
              f"keep {val['assistant_retencion_ratio']:.2f}x")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
