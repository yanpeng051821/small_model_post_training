import pytest
import torch
import torch.nn.functional as F

from post_training_core import masked_sft_loss, masked_sft_loss_sum_and_count


def test_applies_causal_shift():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0],  # 位置 0，明显偏向类别 0
                [0.0, 2.0],  # 位置 1，明显偏向类别 1
                [100.0, -100.0],  # 最后一个 logits，应该被完全忽略
            ]
        ],
        requires_grad=True,
    )

    labels = torch.tensor(
        [[1, 0, 1]],
        dtype=torch.long,
    )

    actual = masked_sft_loss(logits, labels)
    expected = torch.tensor(0.126928, dtype=logits.dtype)
    torch.testing.assert_close(actual, expected)


def test_ignores_masked_targets():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0],  # 有效，正确类别 0
                [100.0, -100.0],  # 对应 -100，必须被忽略
                [0.0, 2.0],  # 有效，正确类别 1
                [-100.0, 100.0],  # 最后位置，不参与
            ]
        ],
        requires_grad=True,
    )

    labels = torch.tensor(
        [[1, 0, -100, 1]],
        dtype=torch.long,
    )

    actual = masked_sft_loss(logits, labels)
    expected = torch.tensor(0.126928, dtype=logits.dtype)
    torch.testing.assert_close(actual, expected)

    changed_logits = logits.detach().clone()
    changed_logits[:, 1, :] = torch.tensor([-999.0, 999.0])

    print("changed_logits:", changed_logits.shape)
    changed_loss = masked_sft_loss(changed_logits, labels)
    torch.testing.assert_close(actual, changed_loss)


def test_rejects_all_masked_batch():
    logits = torch.randn(
        1,
        3,
        2,
        requires_grad=True,
    )

    labels = torch.full(
        (1, 3),
        -100,
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="all tokens are ignored"):
        masked_sft_loss(logits, labels)


def test_backward_only_updates_supervised_positions():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0],  # 有效
                [100.0, -100.0],  # mask
                [0.0, 2.0],  # 有效
                [-100.0, 100.0],  # 最后一个位置
            ]
        ],
        requires_grad=True,
    )

    labels = torch.tensor(
        [[1, 0, -100, 1]],
        dtype=torch.long,
    )

    loss = masked_sft_loss(logits, labels)
    loss.backward()

    grad = logits.grad

    # 第一组断言：梯度对象正常
    assert grad is not None
    assert grad.shape == logits.shape
    assert torch.isfinite(grad).all()

    # 第二组断言：有效位置有梯度
    assert torch.count_nonzero(grad[:, 0, :]) > 0
    assert torch.count_nonzero(grad[:, 2, :]) > 0

    # 第三组断言：无效位置梯度全零
    torch.testing.assert_close(
        grad[:, 1, :],
        torch.zeros_like(grad[:, 1, :]),
    )

    torch.testing.assert_close(
        grad[:, 3, :],
        torch.zeros_like(grad[:, 3, :]),
    )


def test_matches_pytorch_reference():
    # 准备两份数值相同、但计算图相互独立的 logits：
    torch.manual_seed(0)

    logits = torch.randn(2, 5, 7)
    labels = torch.tensor(
        [
            [1, 2, -100, 4, 5],
            [3, -100, 1, 6, 0],
        ]
    )

    actual_logits = logits.clone().requires_grad_(True)
    reference_logits = logits.clone().requires_grad_(True)
    shift_reference_logits = reference_logits[:, :-1, :]
    shift_reference_labels = labels[:, 1:]
    valid = shift_reference_labels != -100
    if not valid.any():
        raise ValueError("all tokens are ignored")
    reference_loss = F.cross_entropy(
        shift_reference_logits.reshape(-1, shift_reference_logits.size(-1)),
        shift_reference_labels.reshape(-1),
        reduction="mean",
        ignore_index=-100,
    )
    reference_loss.backward()

    actual_loss = masked_sft_loss(actual_logits, labels)
    actual_loss.backward()

    # 从梯度和loss 比较核心计算能力是否欧克
    torch.testing.assert_close(actual_loss, reference_loss)
    torch.testing.assert_close(actual_logits.grad, reference_logits.grad)


def test_rejects_mismatched_batch_or_sequence_dimensions():
    logits = torch.randn(2, 4, 5)
    labels = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="logits and labels must match on batch and sequence dimensions",
    ):
        masked_sft_loss(logits, labels)


def test_rejects_sequence_shorter_than_two_tokens():
    logits = torch.randn(4, 1, 2)
    labels = torch.zeros(4, 1, dtype=torch.long)

    with pytest.raises(ValueError, match="sequence length must be at least 2"):
        masked_sft_loss(logits, labels)


def test_rejects_logits_with_wrong_rank():
    logits = torch.randn(2, 4)
    labels = torch.zeros(2, 4, dtype=torch.long)

    with pytest.raises(ValueError, match=r"logits must have shape \[B, T, V\]"):
        masked_sft_loss(logits, labels)


def test_rejects_labels_with_wrong_rank():
    logits = torch.randn(2, 4, 5)
    labels = torch.zeros(2, 4, 1, dtype=torch.long)

    with pytest.raises(
        ValueError,
        match=r"labels must have shape \[B, T\]",
    ):
        masked_sft_loss(logits, labels)


def test_supports_custom_ignore_index():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0],  # 预测 labels[1] = 0
                [100.0, -100.0],  # 对应 labels[2] = -1，应被忽略
                [0.0, 2.0],  # 预测 labels[3] = 1
                [-100.0, 100.0],  # 最后位置不参与预测
            ]
        ],
        requires_grad=True,
    )

    labels = torch.tensor(
        [[1, 0, -1, 1]],
        dtype=torch.long,
    )

    actual = masked_sft_loss(
        logits,
        labels,
        ignore_index=-1,
    )

    expected = torch.tensor(
        0.126928,
        dtype=logits.dtype,
    )

    torch.testing.assert_close(actual, expected)


def test_averages_over_all_valid_tokens_not_samples():

    # 输入
    logits = torch.tensor(
        [
            [
                [0.0, 0.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [0.0, 0.0],
            ],
            [
                [2.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ],
        ]
    )

    labels = torch.tensor(
        [
            [1, 0, -100, -100],
            [1, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    actual = masked_sft_loss(logits, labels)

    valid_logits = torch.stack(
        [logits[0, 0, :], logits[1, 0, :], logits[1, 1, :], logits[1, 2, :]]
    )  # [4, 2]

    valid_targets = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    expected = F.cross_entropy(valid_logits, valid_targets, reduction="mean")

    torch.testing.assert_close(actual, expected)


def test_rejects_non_floating_logits():
    logits = torch.ones(1, 3, 4, dtype=torch.long)

    labels = torch.zeros(1, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="logits must be floating point"):
        masked_sft_loss(logits, labels)


def test_rejects_non_long_labels():
    logits = torch.randn(1, 3, 4)
    labels = torch.zeros(1, 3, dtype=torch.float16)

    with pytest.raises(ValueError, match="labels must have dtype torch.long"):
        masked_sft_loss(logits, labels)


def test_loss_sum_and_count_reconstruct_mean_loss():
    logits = torch.tensor(
        [
            [
                [0.0, 0.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [0.0, 0.0],
            ],
            [
                [2.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ],
        ]
    )

    labels = torch.tensor(
        [
            [1, 0, -100, -100],
            [1, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    mean_loss = masked_sft_loss(logits, labels)

    sum_loss, valid_token_count = masked_sft_loss_sum_and_count(logits, labels)
    mean_loss_reconstruct = sum_loss / valid_token_count

    assert valid_token_count.item() == 4
    assert valid_token_count.ndim == 0
    assert valid_token_count.dtype == torch.long

    torch.testing.assert_close(mean_loss, mean_loss_reconstruct)
