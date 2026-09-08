import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from post_training_core import (
    collate_sft_batch,
    forward_sft_batch,
    masked_sft_loss,
    sft_optimizer_step,
)


def _global_grad_norm(model):
    total_grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]

    gradients = torch.cat(total_grads)

    return gradients.norm(p=2)


# 1 单步更新模型参数测试
def test_single_optimizer_step_updates_model_parameters():
    torch.manual_seed(0)

    # 1 模型配置
    vocal_size = 8
    hidden_size = 4

    model = nn.Sequential(
        nn.Embedding(vocal_size, hidden_size),
        nn.Linear(hidden_size, vocal_size),
    )

    learning_rate = 0.1

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    optimizer.zero_grad()

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    # 2 输入模拟
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    labels = input_ids.clone()

    # 3 前向传播计算logits
    logits = model(input_ids)
    loss = masked_sft_loss(logits, labels)

    # 4 反向传播计算中间层参数梯度和更新参数
    loss.backward()

    expected_parameters = [
        before - learning_rate * parameter.grad
        for before, parameter in zip(parameters_before, model.parameters(), strict=True)
    ]

    optimizer.step()

    for expected, actual in zip(expected_parameters, model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)

    assert torch.isfinite(loss)

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    )


def test_tiny_batch_can_be_overfit():
    torch.manual_seed(0)

    # 模型配置
    vocal_size = 8
    hidden_size = 4
    model = nn.Sequential(
        nn.Embedding(vocal_size, hidden_size), nn.Linear(hidden_size, vocal_size)
    )

    # 优化器配置
    learning_rate = 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    # 输入
    inputs_id = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    labels = inputs_id.clone()

    # 拿到初始的loss值
    with torch.no_grad():
        pre_logits = model(inputs_id)
        pre_loss = masked_sft_loss(pre_logits, labels).item()

    # 简单训练
    for _ in range(100):
        optimizer.zero_grad()

        logits = model(inputs_id)
        loss = masked_sft_loss(logits, labels)
        loss.backward()

        optimizer.step()

    # 训练后，再看看loss的值
    with torch.no_grad():
        after_logits = model(inputs_id)
        after_loss = masked_sft_loss(after_logits, labels).item()

        predicted_ids = after_logits[:, :-1, :].argmax(dim=-1)
        target_ids = labels[:, 1:]

    # 比较
    assert after_loss < pre_loss * 0.1
    assert after_loss < 0.05
    assert torch.equal(predicted_ids, target_ids)

    generated_ids = torch.tensor([[1]], dtype=torch.long)  # 1 , 1
    with torch.no_grad():
        for _ in range(4):
            logits = model(generated_ids)  # 1, 1, 8

            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    expected_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    assert torch.equal(generated_ids, expected_ids)


def test_gradient_accumulation_matches_full_batch_update():
    torch.manual_seed(0)

    vocal_size = 8
    hidden_size = 4
    learning_rate = 0.1

    # reference_model 和 accumulated_model 数值相同、对象不同、存储不同
    reference_model = RecordingCausalLM(vocab_size=vocal_size, hidden_size=hidden_size)

    accumulated_model = copy.deepcopy(reference_model)

    # 【检查1】训练之前比较两个模型的参数
    for reference_param, accumulated_param in zip(
        reference_model.parameters(), accumulated_model.parameters(), strict=True
    ):
        torch.testing.assert_close(reference_param, accumulated_param)
        assert reference_param is not accumulated_param

    micro_inputs = [
        torch.tensor(
            [[1, 2, 0, 0]],
            dtype=torch.long,
        ),
        torch.tensor(
            [[2, 3, 4, 5]],
            dtype=torch.long,
        ),
    ]  # 2 * [1, 4]

    micro_labels = [
        torch.tensor(
            [[1, 2, -100, -100]],
            dtype=torch.long,
        ),
        torch.tensor(
            [[2, 3, 4, 5]],
            dtype=torch.long,
        ),
    ]  # 2 * [1, 4]

    micro_attention_masks = [
        torch.tensor(
            [[1, 1, 0, 0]],
            dtype=torch.long,
        ),
        torch.tensor(
            [[1, 1, 1, 1]],
            dtype=torch.long,
        ),
    ]

    micro_batches = [
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}
        for input_ids, attention_mask, label in zip(
            micro_inputs, micro_attention_masks, micro_labels, strict=True
        )
    ]

    full_inputs = torch.cat([micro_inputs[0], micro_inputs[1]], dim=0)

    full_labels = torch.cat([micro_labels[0], micro_labels[1]], dim=0)

    full_attention_masks = torch.cat(
        [micro_attention_masks[0], micro_attention_masks[1]], dim=0
    )

    full_batch = {
        "input_ids": full_inputs,
        "attention_mask": full_attention_masks,
        "labels": full_labels,
    }

    # 【检查2】检查两个样本合并后的inputs 和 labels 的形状
    assert full_inputs.shape == (2, 4)
    assert full_labels.shape == (2, 4)
    assert full_attention_masks.shape == (2, 4)

    reference_optimizer = torch.optim.SGD(reference_model.parameters(), learning_rate)
    accumulated_optimizer = torch.optim.SGD(
        accumulated_model.parameters(), learning_rate
    )

    reference_optimizer.zero_grad()

    # 路径A，full batch
    logits_A, labels = forward_sft_batch(reference_model, full_batch)

    loss_A = masked_sft_loss(logits_A, labels)
    loss_A.backward()

    # 路径B
    # logits_B_0 = accumulated_model(micro_inputs[0])
    # loss_sum_B0, valid_token_count_B0 = masked_sft_loss_sum_and_count(logits_B_0, micro_labels[0])
    # loss_sum_B0.backward()

    # logits_B_1 = accumulated_model(micro_inputs[1])
    # loss_sum_B1, valid_token_count_B1 = masked_sft_loss_sum_and_count(logits_B_1,micro_labels[1])
    # loss_sum_B1.backward()

    # total_valid_token_count = valid_token_count_B0 + valid_token_count_B1
    # # 【检查项3】分批更新梯度的token的有效数量
    # assert valid_token_count_B0.item() == 1
    # assert valid_token_count_B1.item() == 3
    # assert total_valid_token_count.item() == 4

    # # 归一化累积梯度
    # for parameter in accumulated_model.parameters():
    #     assert parameter.grad is not None
    #     parameter.grad.div_(total_valid_token_count)

    # # 更新参数之前比较一下两条路径的梯度
    # for reference_param, accumulated_param in zip(reference_model.parameters(), accumulated_model.parameters()):
    #     assert reference_param.grad is not None
    #     assert accumulated_param.grad is not None
    #     torch.testing.assert_close(reference_param.grad, accumulated_param.grad)

    # # 统一更新参数
    # reference_optimizer.step()
    # accumulated_optimizer.step()

    accumulated_mean_loss, total_valid_token = sft_optimizer_step(
        accumulated_model, accumulated_optimizer, micro_batches
    )

    assert total_valid_token.item() == 4

    torch.testing.assert_close(loss_A.detach(), accumulated_mean_loss)

    reference_optimizer.step()

    for reference_param, accumulated_param in zip(
        reference_model.parameters(), accumulated_model.parameters(), strict=True
    ):
        torch.testing.assert_close(reference_param, accumulated_param)


def test_sft_optimizer_step_rejects_empty_micro_batches():
    model = nn.Sequential(nn.Embedding(8, 4), nn.Linear(4, 8))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="micro_batches must not be empty"):
        sft_optimizer_step(model, optimizer, [])


def test_sft_optimizer_step_skips_frozen_parameters():
    torch.manual_seed(0)

    # 构建模型
    model = RecordingCausalLM(vocab_size=8, hidden_size=4)

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # 冻结第一层参数
    model.embedding.weight.requires_grad_(False)

    # 提前保存每一层的参数
    frozen_before = model.embedding.weight.detach().clone()
    training_before = model.lm_head.weight.detach().clone()

    micro_batches = [
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1]],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [[1, 2, 3, 4]],
                dtype=torch.long,
            ),
        }
    ]

    sft_optimizer_step(model, optimizer, micro_batches)

    # 【开始检查】
    assert model.embedding.weight.grad is None
    assert model.lm_head.weight.grad is not None

    torch.testing.assert_close(model.embedding.weight, frozen_before)
    assert not torch.equal(model.lm_head.weight, training_before)


@pytest.mark.parametrize(
    "max_grad_norm",
    [0.0, -1.0],
)
def test_sft_optimizer_step_rejects_non_positive_grad_norm(max_grad_norm):
    model = nn.Sequential(nn.Embedding(8, 4), nn.Linear(4, 8))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="max_grad_norm must be positive"):
        sft_optimizer_step(model, optimizer, [], max_grad_norm=max_grad_norm)


def test_sft_optimizer_step_clips_global_gradient_norm():
    torch.manual_seed(0)

    uncliped_model = RecordingCausalLM(vocab_size=8, hidden_size=4)

    cliped_model = copy.deepcopy(uncliped_model)

    uncliped_optimizer = torch.optim.SGD(uncliped_model.parameters(), lr=0.05)
    cliped_optimizer = torch.optim.SGD(cliped_model.parameters(), lr=0.05)

    micro_batches = [
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1]],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [[1, 2, 3, 4]],
                dtype=torch.long,
            ),
        }
    ]

    sft_optimizer_step(uncliped_model, uncliped_optimizer, micro_batches)

    max_grad_norm = 0.1

    sft_optimizer_step(
        cliped_model, cliped_optimizer, micro_batches, max_grad_norm=max_grad_norm
    )

    # 计算每个模型的l2 norm
    uncliped_norm = _global_grad_norm(uncliped_model)
    cliped_norm = _global_grad_norm(cliped_model)

    assert uncliped_norm > cliped_norm
    torch.testing.assert_close(
        cliped_norm, torch.tensor(max_grad_norm), rtol=1e-5, atol=1e-6
    )


class RecordingCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.received_input_ids = None
        self.received_attention_mask = None

    def forward(self, *, input_ids, attention_mask):
        self.received_input_ids = input_ids
        self.received_attention_mask = attention_mask

        hidden_states = self.embedding(input_ids)
        logits = self.lm_head(hidden_states)

        return SimpleNamespace(logits=logits)


def test_forward_sft_batch_uses_hf_model_interface():
    model = RecordingCausalLM()

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        "labels": torch.tensor(
            [
                [-100, 2, 3, -100],
                [-100, 3, -100, -100],
            ],
            dtype=torch.long,
        ),
    }

    logits, labels = forward_sft_batch(model, batch)

    assert logits.shape == (2, 4, 8)
    assert labels is batch["labels"]

    assert model.received_attention_mask is batch["attention_mask"]
    assert model.received_input_ids is batch["input_ids"]


def test_sft_optimizer_step_accepts_structured_batch():
    torch.manual_seed(0)

    model = RecordingCausalLM()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 0],
                [2, 3, 0, 0],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [
                [-100, 2, 3, -100],
                [-100, 3, -100, -100],
            ],
            dtype=torch.long,
        ),
    }

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    mean_loss, valid_token_count = sft_optimizer_step(
        model,
        optimizer,
        [batch],
    )

    assert torch.isfinite(mean_loss)
    assert valid_token_count.item() == 3

    assert model.received_input_ids is batch["input_ids"]
    assert model.received_attention_mask is batch["attention_mask"]

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            model.parameters(),
            strict=True,
        )
    )


def test_collated_sft_batch_completes_optimizer_step():
    torch.manual_seed(0)

    samples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
        },
        {
            "input_ids": [2, 3, 4],
            "labels": [-100, 3, 4],
        },
    ]

    batch = collate_sft_batch(samples, pad_token_id=0)

    model = RecordingCausalLM(vocab_size=8, hidden_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    parameters_before = [param.detach().clone() for param in model.parameters()]

    mean_loss, total_valid_token = sft_optimizer_step(
        model,
        optimizer,
        [batch],
    )

    assert torch.isfinite(mean_loss)
    assert total_valid_token.item() == 4

    assert model.received_input_ids is batch["input_ids"]
    assert model.received_attention_mask is batch["attention_mask"]

    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    )
