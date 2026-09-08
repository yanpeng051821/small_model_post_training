# Week 2：环境确认、真实样本与 B0 Baseline

状态：等待在训练机器执行
职责边界：硬件、驱动、CUDA、显存和安装适配由用户处理；本文件只规定实验步骤、验证证据和阶段门。

## 1. 本周目标

```text
确认固定源码和实际包版本
-> 检查 Qwen3 tokenizer/chat template/EOS/PAD
-> 加载一条真实 SFT/GRPO 数据
-> 跑 B0 inference smoke
-> 跑 LightEval 五题 smoke
-> 冻结并运行 B0 canonical eval
```

本周不启动 SFT 或 GRPO 训练。

## 2. 统一输出目录

在机器上替换为真实绝对路径：

```bash
export OPENR1_REPO="/absolute/path/to/open-r1"
export OPENR1_BASELINE_ROOT="/absolute/path/to/small_model_post_training/open_r1_reproduction/baselines/local/open-r1-qwen3-0.6b"

mkdir -p "$OPENR1_BASELINE_ROOT/setup"
mkdir -p "$OPENR1_BASELINE_ROOT/logs"
mkdir -p "$OPENR1_BASELINE_ROOT/evals/b0-smoke"
mkdir -p "$OPENR1_BASELINE_ROOT/evals/b0-canonical"
```

## 3. W2-1：记录源码与环境事实

```bash
{
  date -Is
  git -C "$OPENR1_REPO" rev-parse HEAD
  git -C "$OPENR1_REPO" status --short
  python --version
  uv --version || true
  nvidia-smi || true
  python -m pip freeze
} 2>&1 | tee "$OPENR1_BASELINE_ROOT/setup/environment_and_source.txt"
```

实验要求：

- Open-R1 HEAD 为 `1416fa0cf21595d2083b399a2a0bbddd7f6e9563`。
- 记录实际 `torch`、`transformers`、`trl`、`accelerate`、`datasets`、`vllm`、`lighteval`、`math-verify` 版本。
- 如果版本与 `PLAN.md` 不同，只记录偏差及原因；硬件/环境修复由用户处理。
- 不删除已有修改，不使用 `git reset --hard`。

## 4. W2-2：Tokenizer 与 B0 inference smoke

```bash
python - 2>&1 <<'PY' | tee "$OPENR1_BASELINE_ROOT/setup/b0_tokenizer_inference.txt"
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(json.dumps({
    "bos_token": tokenizer.bos_token,
    "bos_token_id": tokenizer.bos_token_id,
    "eos_token": tokenizer.eos_token,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token": tokenizer.pad_token,
    "pad_token_id": tokenizer.pad_token_id,
    "has_chat_template": tokenizer.chat_template is not None,
}, ensure_ascii=False, indent=2))
print("CHAT_TEMPLATE")
print(tokenizer.chat_template)

if tokenizer.chat_template is None:
    raise SystemExit("STOP: chat template 尚未确定")

messages = [{"role": "user", "content": "计算 2+3，并给出简短解释。"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("RENDERED_PROMPT")
print(prompt)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to("cuda").eval()
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
new_tokens = output[0, inputs["input_ids"].shape[1]:]
print("OUTPUT")
print(tokenizer.decode(new_tokens, skip_special_tokens=False))
print("prompt_tokens:", inputs["input_ids"].shape[1])
print("generated_tokens:", new_tokens.shape[0])
PY
```

检查：模型能加载和生成；完整 template 与特殊 token id 已保存。答案质量不是 smoke gate。若模板为空或 EOS 不明确，先停止并统一合同，不能给 B0/B1/B2 分别临时处理。

## 5. W2-3：真实 SFT/GRPO 样本

```bash
python - 2>&1 <<'PY' | tee "$OPENR1_BASELINE_ROOT/setup/real_sample_audit.txt"
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

sft = load_dataset(
    "open-r1/Mixture-of-Thoughts", "math", split="train", streaming=True
)
sft_row = next(iter(sft))
print("SFT_KEYS:", sorted(sft_row.keys()))
print("SFT_ROLES:", [m.get("role") for m in sft_row["messages"]])
sft_text = tokenizer.apply_chat_template(sft_row["messages"], tokenize=False)
sft_ids = tokenizer(sft_text, add_special_tokens=False)["input_ids"]
print("SFT_TOKEN_COUNT:", len(sft_ids))
print("SFT_HEAD:", repr(sft_text[:500]))
print("SFT_TAIL:", repr(sft_text[-500:]))

grpo = load_dataset(
    "open-r1/OpenR1-Math-220k", split="train", streaming=True
)
grpo_row = next(iter(grpo))
print("GRPO_KEYS:", sorted(grpo_row.keys()))
print("PROBLEM_HEAD:", repr(str(grpo_row.get("problem"))[:500]))
print("SOLUTION_HEAD:", repr(str(grpo_row.get("solution"))[:500]))
PY
```

检查：SFT 存在 `messages`；GRPO 存在 `problem/solution`；`solution` 只作为 reward ground truth。Streaming 样本只用于结构检查，不作为 canonical 子集 manifest。

## 6. W2-4：B0 LightEval 五题 smoke

以下参数只用于检查链路，不作为最终评测合同。硬件相关模型参数由用户按机器调整，但必须保存实际命令。

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export B0_MODEL="Qwen/Qwen3-0.6B-Base"
export B0_MODEL_ARGS="model_name=$B0_MODEL,dtype=float16,max_model_length=4096,gpu_memory_utilization=0.80,generation_parameters={max_new_tokens:512,temperature:0.0}"

lighteval vllm "$B0_MODEL_ARGS" "lighteval|math_500|0|0" \
  --use-chat-template \
  --max-samples 5 \
  --save-details \
  --output-dir "$OPENR1_BASELINE_ROOT/evals/b0-smoke" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/b0_lighteval_smoke.log"
```

检查：命令退出、生成结果文件和逐题 details、实际只处理 5 题。五题分数禁止写成 B0 baseline。

## 7. W2-5：冻结并运行 B0 canonical eval

五题 smoke 通过后，把以下字段写入 `PLAN.md` 和一份实际执行 YAML/命令文件：

- B0 模型和 tokenizer revision。
- chat template、system prompt、EOS/PAD。
- `max_model_length`、`max_new_tokens`、temperature、top-p。
- 每题响应数与 seed/seed list。
- LightEval task、版本和答案提取逻辑。

然后运行完整 MATH-500，输出到 `evals/b0-canonical/`。B1/B2 后续必须复用同一合同；若任何参数变化，B0 也必须按新合同重跑。

## 8. 本周需要保存的证据

```text
setup/environment_and_source.txt
setup/b0_tokenizer_inference.txt
setup/real_sample_audit.txt
logs/b0_lighteval_smoke.log
evals/b0-smoke/ 结果和 details
evals/b0-canonical/ 结果和 details
B0 canonical 实际命令或 YAML
```

## 9. 完成标准

- [ ] 固定源码和实际依赖有记录。
- [ ] tokenizer/template/EOS/PAD 已确认。
- [ ] B0 inference smoke 通过。
- [ ] SFT/GRPO 真实样本结构通过。
- [ ] LightEval 五题 smoke 通过。
- [ ] B0 canonical 合同已经冻结。
- [ ] 完整 B0 MATH-500 结果与逐题输出存在。

机器适配成功本身不等于 Week 2 完成；只有 B0 canonical 证据齐全才进入 Week 3。
