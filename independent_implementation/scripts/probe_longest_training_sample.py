"""Run one optimizer step on the longest audited sample before paid training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from post_training_core.config import ExperimentConfig
from post_training_core.data import collate_sft_batch
from post_training_core.engine import (
    build_optimizer,
    build_scheduler,
    resolve_device,
    run_optimizer_step,
)
from post_training_core.experiment import (
    atomic_write_json,
    set_reproducible_seed,
    utc_now,
)
from post_training_core.memory_probe import load_longest_tokenized_record
from post_training_core.runner import load_model, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    record = load_longest_tokenized_record(config.train_artifact)
    result = {
        "config": str(args.config.resolve()),
        "created_at": utc_now(),
        "sample_id": record["sample_id"],
        "sequence_length": len(record["input_ids"]),
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)

    started = time.perf_counter()
    try:
        set_reproducible_seed(config.seed)
        device = resolve_device(config.device)
        if device.type != "cuda":
            raise RuntimeError("the paid-host memory probe requires CUDA")
        tokenizer = load_tokenizer(config)
        model = load_model(config).to(device)
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, total_steps=1)
        torch.cuda.reset_peak_memory_stats(device)
        metrics = run_optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            micro_batches=[
                collate_sft_batch([record], pad_token_id=config.pad_token_id)
            ],
            device=device,
            dtype=config.dtype,
            max_grad_norm=config.max_grad_norm,
        )
        result.update(
            {
                "duration_seconds": time.perf_counter() - started,
                "max_cuda_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "max_cuda_reserved_mb": torch.cuda.max_memory_reserved(device)
                / (1024**2),
                "metrics": metrics.__dict__,
                "status": "passed",
            }
        )
    except BaseException as error:
        result.update(
            {
                "duration_seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "status": "failed",
            }
        )
        atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
