from post_training_core.data import StatefulRandomSampler, TokenizedSFTDataset
from post_training_core.trl_reference import (
    OrderedCompletionMaskDataset,
    frozen_epoch_indices,
)


def _dataset():
    return TokenizedSFTDataset(
        [
            {
                "sample_id": str(index),
                "input_ids": [index + 1, index + 2],
                "labels": [-100, index + 2],
            }
            for index in range(5)
        ]
    )


def test_reference_order_matches_independent_sampler_first_epoch():
    dataset = _dataset()
    expected = list(iter(StatefulRandomSampler(dataset, seed=7)))[:3]

    assert frozen_epoch_indices(dataset, seed=7, limit=3) == expected


def test_reference_adapter_recreates_trl_completion_mask():
    dataset = _dataset()
    adapted = OrderedCompletionMaskDataset(dataset, [3, 1])

    assert adapted[0] == {
        "input_ids": [4, 5],
        "completion_mask": [0, 1],
    }
