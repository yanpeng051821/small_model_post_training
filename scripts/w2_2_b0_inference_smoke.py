"""
Week 2-2: B0 模型推理冒烟测试（不是 MATH-500 评测！）

验证: Qwen3-0.6B-Base 能在 T4 GPU 上正确加载、生成。
目的: 确认模型+GPU+tokenizer+flash_attn 链路打通，为后续评测做准备。
"""
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
OUTPUT_ROOT = os.environ.get(
    "OPENR1_BASELINE_ROOT",
    "/data/workspace/minimind-practice/small_model_post_training/open_r1_reproduction/baselines/local/open-r1-qwen3-0.6b",
)
OUTPUT_FILE = os.path.join(OUTPUT_ROOT, "setup", "b0_tokenizer_inference.txt")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def main():
    # ---- 1. Tokenizer 信息 ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    info = {
        "model_id": MODEL_ID,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "has_chat_template": tokenizer.chat_template is not None,
    }
    print("=== TOKENIZER INFO ===")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("=== CHAT TEMPLATE ===")
    print(tokenizer.chat_template)

    if tokenizer.chat_template is None:
        print("FATAL: chat template 为空，无法继续。", file=sys.stderr)
        sys.exit(1)

    # ---- 2. 构造 prompt ----
    messages = [{"role": "user", "content": "人生的意义是什么"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("=== RENDERED PROMPT ===")
    print(prompt)

    # ---- 3. 加载模型 ----
    print("Loading model to GPU (float16)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    vram_gb = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"Params: {n_params/1e6:.1f}M | GPU mem: {vram_gb:.2f} GB")

    # ---- 4. 推理 ----
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,           # 贪心解码，确定性输出
            temperature=1.0,           # do_sample=False 时忽略
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    prompt_len = inputs["input_ids"].shape[1]
    gen_len = new_tokens.shape[0]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)

    print("=== MODEL OUTPUT ===")
      # 改成，同时打印 token id
    print("TOKEN_IDS:", new_tokens.tolist())
    print("DECODED:", repr(tokenizer.decode(new_tokens, skip_special_tokens=False)))
    print(decoded)
    print(f"prompt_tokens: {prompt_len} | generated_tokens: {gen_len} | ratio: {gen_len/prompt_len:.1f}x")

    torch.cuda.empty_cache()
    print("Done. Output saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
