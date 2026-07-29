"""
理解 chat template：Jinja2 模板如何把 messages 列表拼成模型认识的字符串。

这不是实验脚本，是学习脚本——帮你理解 apply_chat_template 底层做了什么。
"""
from jinja2 import Template

# ============================================================
# 1. 极简版模板：只有最基本的拼接逻辑，没有 think/tools/空白控制
# ============================================================
simple_tpl = Template("""\
{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}\
""")


# ============================================================
# 2. 测试数据：覆盖 think 标签、tools、system prompt 三个特征
# ============================================================
messages_think = [
    {"role": "user", "content": "计算 2+3"}
]

messages_reasoning = [
    {"role": "user", "content": "计算 2+3"}
    # 注意：assistant 消息还没生成，后面我们用 enable_thinking 演示
]

messages_tools = [
    {
        "role": "system",
        "content": "你是一个数学助手，可以使用计算器工具。"
    },
    {"role": "user", "content": "计算 12345 * 67890"}
]

# 模拟 GRPO 训练后模型输出的消息——带 think 标签
messages_grpo_output = [
    {"role": "user", "content": "计算 2+3"},
    {
        "role": "assistant",
        "content": "<think>\n2+3 等于 5，因为 2 加上 3 就是 5。\n</think>\n\n答案是 5。"
    }
]


# ============================================================
# 3. 对比：逐个展示差异
# ============================================================
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

print("#" * 70)
print("# 差异 1：add_generation_prompt 控制该谁说话")
print("#" * 70)

result = simple_tpl.render(messages=messages_think, add_generation_prompt=False)
print("\n[极简模板] 不加 add_generation_prompt:\n", repr(result))

result = simple_tpl.render(messages=messages_think, add_generation_prompt=True)
print("\n[极简模板] 加 add_generation_prompt:\n", repr(result))

result = tokenizer.apply_chat_template(messages_think, tokenize=False, add_generation_prompt=True)
print("\n[真实 Qwen3] 加 add_generation_prompt:\n", repr(result))
print("   ↑ 末尾多了一个 \\n，这是 Qwen3 模板的细节差异")

print()
print("#" * 70)
print("# 差异 2：enable_thinking 参数——GRPO 推理用空 think 占位")
print("#" * 70)
print("说明: GRPO 训练时常 off thinking 以加速，但结构上需要 think 占位符。")

result = tokenizer.apply_chat_template(
    messages_reasoning, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print("\n[真实 Qwen3] enable_thinking=False (少参数，纯答案):\n", repr(result))

result = tokenizer.apply_chat_template(
    messages_reasoning, tokenize=False, add_generation_prompt=True
)
print("\n[真实 Qwen3] 默认（不传 enable_thinking）:\n", repr(result))
print("   ↑ 没有空 think 标签，留给模型自己决定要不要思考")
print("\n[极简模板] 不支持 enable_thinking 参数，无法控制这个行为")

print()
print("#" * 70)
print("# 差异 3：think 标签——真实模板会拆分思考过程和答案")
print("#" * 70)
print("说明: GRPO 训练后模型输出带 <think>...</think>，评测时也会遇到。")

# 极简模板直接把 think 标签当普通文本拼接
result = simple_tpl.render(messages=messages_grpo_output, add_generation_prompt=True)
print("\n[极简模板] 带 think 标签的输出:\n", repr(result))
print("   ^ think 被当成普通 content 塞进 assistant 消息，没有拆分")

# 真实模板识别 <think> 并正确处理
result = tokenizer.apply_chat_template(messages_grpo_output, tokenize=False)
print("\n[真实 Qwen3] 带 think 标签的输出:\n", repr(result))
print("   ↑ 真实模板识别到 </think>，会拆成 reasoning 和 answer 两部分")

print()
print("#" * 70)
print("# 差异 4：tools——真实模板支持工具调用，极简模板不支持")
print("#" * 70)

# 真实 Qwen3：传入 tools 参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行算术运算",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
        }
    }
]
result_with_tools = tokenizer.apply_chat_template(
    messages_tools,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)
result_no_tools = tokenizer.apply_chat_template(
    messages_tools,
    tokenize=False,
    add_generation_prompt=True
)
print("\n[真实 Qwen3] 带 tools 的 system prompt 开头是:\n", repr(result_with_tools[:200]))
print("\n[真实 Qwen3] 不带 tools 的 system prompt 开头是:\n", repr(result_no_tools[:200]))
print("\n[极简模板] 不支持 tools，无法渲染工具定义")
