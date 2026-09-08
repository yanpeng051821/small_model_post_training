"""Compare one frozen SFT batch with the TRL/HF reference loss path."""

from __future__ import annotations

import argparse
import json
from functools import partial
from importlib import metadata
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from post_training_core.config import ExperimentConfig
from post_training_core.data import (
    IndexedTokenizedSFTDataset,
    StatefulRandomSampler,
    collate_sft_batch,
)
from post_training_core.engine import move_batch_to_device, resolve_device
from post_training_core.experiment import atomic_write_json, utc_now
from post_training_core.runner import load_model
from post_training_core.shadow import compare_sft_batch_with_hf_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/first_batch_shadow.json"),
    )
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("first-batch shadow requires the configured CUDA device")

    dataset = IndexedTokenizedSFTDataset(
        config.train_artifact,
        limit=config.train_sample_limit,
    )
    sampler = StatefulRandomSampler(dataset, seed=config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=partial(collate_sft_batch, pad_token_id=config.pad_token_id),
        num_workers=0,
    )
    batch = move_batch_to_device(next(iter(loader)), device)
    model = load_model(config).to(device)
    result = compare_sft_batch_with_hf_reference(model, batch)
    report = {
        **result,
        "created_at": utc_now(),
        "config": str(args.config.resolve()),
        "reference": (
            "TRL 0.18.0 SFTTrainer -> Transformers 4.52.3 model causal-LM loss"
        ),
        "software": {
            "torch": torch.__version__,
            "transformers": metadata.version("transformers"),
            "trl": metadata.version("trl"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
