"""Run S1 or a single-epoch pilot using pinned TRL; retain native resume checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from post_training_core.config import ExperimentConfig
from post_training_core.data import IndexedTokenizedSFTDataset, sha256_file
from post_training_core.experiment import (
    atomic_write_json,
    collect_environment,
    set_reproducible_seed,
    utc_now,
)
from post_training_core.provenance import runtime_file_hashes, runtime_tree_hash
from post_training_core.trl_reference import (
    OrderedCompletionMaskDataset,
    frozen_epoch_indices,
)
from post_training_core.trl_training import (
    FrozenOrderSFTTrainer,
    SaveAndStopCallback,
    build_trl_sft_args,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("stop-after-steps must be positive")
    config = ExperimentConfig.from_yaml(args.config)
    if config.resume_from_checkpoint is not None:
        raise ValueError("use --resume-from-checkpoint for native TRL checkpoints")
    root = Path(__file__).resolve().parents[1]
    source = IndexedTokenizedSFTDataset(
        config.train_artifact, limit=config.train_sample_limit
    )
    indices = frozen_epoch_indices(source, seed=config.seed, limit=len(source))
    dataset = OrderedCompletionMaskDataset(source, indices)
    training_args = build_trl_sft_args(config, args.output_dir / "trainer", len(source))
    if training_args.world_size != 1 or training_args.n_gpu > 1:
        raise ValueError(
            "S1 is single GPU; set CUDA_VISIBLE_DEVICES to exactly one GPU"
        )
    order = [
        {"position": i, "sample_id": source[index]["sample_id"]}
        for i, index in enumerate(indices)
    ]
    order_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in order)
    identity = {
        "config_hash": config.semantic_hash(),
        "runtime_tree_sha256": runtime_tree_hash(runtime_file_hashes(root)),
        "experiment_contract_sha256": sha256_file(root / "EXPERIMENT_CONTRACT.md"),
        "train_sha256": sha256_file(config.train_artifact),
        "validation_sha256": sha256_file(config.validation_artifact),
        "sample_order_sha256": hashlib.sha256(order_text.encode()).hexdigest(),
        "record_count": len(source),
        "max_steps": training_args.max_steps,
        "warmup_steps": training_args.warmup_steps,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    if args.resume_from_checkpoint:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("resume contract, data, or runtime identity changed")
        if (
            args.resume_from_checkpoint.resolve().parent
            != (args.output_dir / "trainer").resolve()
        ):
            raise ValueError("resume checkpoint must belong to this run")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("fresh TRL run requires an empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sample_order.jsonl").write_text(
        order_text, encoding="utf-8", newline="\n"
    )
    manifest = {
        "identity": identity,
        "created_at": utc_now(),
        "backend": "trl-0.18.0",
        "resume_from": str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None,
        "status": "dry_run" if args.dry_run else "running",
    }
    attempt_path = args.output_dir / f"attempt-{utc_now().replace(':', '-')}.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(attempt_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    try:
        set_reproducible_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, revision=config.model_revision
        )
        if tokenizer.pad_token_id != config.pad_token_id:
            raise ValueError("tokenizer pad ID differs from the contract")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            revision=config.model_revision,
            torch_dtype=torch.float32,
            attn_implementation=config.attn_implementation,
            use_cache=False,
        )
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            raise ValueError("tokenizer has no assistant end token")
        model.generation_config.eos_token_id = eos_id
        trainer = FrozenOrderSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                config.pad_token_id, completion_only_loss=True
            ),
            callbacks=[SaveAndStopCallback(args.stop_after_steps)],
        )
        result = trainer.train(
            resume_from_checkpoint=str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
        completed = trainer.state.global_step == training_args.max_steps
        if completed:
            model.config.use_cache = True
            trainer.save_model(args.output_dir / "final_model")
            tokenizer.save_pretrained(args.output_dir / "final_model")
        trainer.save_state()
        manifest.update(
            status="completed" if completed else "paused",
            global_step=trainer.state.global_step,
            metrics=result.metrics,
            environment=collect_environment(root),
        )
    except BaseException:
        manifest.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        manifest["finished_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(attempt_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
