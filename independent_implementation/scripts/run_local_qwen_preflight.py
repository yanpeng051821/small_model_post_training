"""Run one real Qwen BF16 optimizer step on the shortest audited sample."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import yaml

from post_training_core.config import ExperimentConfig
from post_training_core.data import sha256_file
from post_training_core.experiment import atomic_write_json
from post_training_core.memory_probe import load_shortest_tokenized_record
from post_training_core.provenance import runtime_file_hashes, runtime_tree_hash

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "311c62e88814bff7206909ccd330bab0a784743b"


def _write_single_record(path: Path, record: dict) -> None:
    path.write_text(
        json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_config(path: Path, config: ExperimentConfig) -> None:
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _run_cli(
    config_path: Path,
    *,
    stop_after_steps: int | None = None,
) -> float:
    command = [
        sys.executable,
        "-m",
        "post_training_core.cli",
        "--config",
        str(config_path),
    ]
    if stop_after_steps is not None:
        command.extend(["--stop-after-steps", str(stop_after_steps)])
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    sample = load_shortest_tokenized_record(args.audited_train)
    with tempfile.TemporaryDirectory(prefix="gate0b-preflight-data-") as temporary:
        temporary_path = Path(temporary)
        train_artifact = temporary_path / "train.jsonl"
        validation_artifact = temporary_path / "validation.jsonl"
        _write_single_record(train_artifact, sample)
        _write_single_record(validation_artifact, sample)
        config = ExperimentConfig(
            run_name="local-real-qwen-preflight",
            model_name_or_path=MODEL_ID,
            model_revision=MODEL_REVISION,
            train_artifact=train_artifact,
            validation_artifact=validation_artifact,
            output_dir=args.output_dir,
            pad_token_id=151643,
            seed=42,
            device="cuda",
            dtype="bfloat16",
            parameter_dtype="bfloat16",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=2,
            max_steps=2,
            learning_rate=4.0e-5,
            max_grad_norm=0.2,
            gradient_checkpointing=True,
            log_every_steps=1,
            eval_every_steps=1,
            save_every_steps=1,
        )
        first_config_path = temporary_path / "first.yaml"
        _write_config(first_config_path, config)
        first_duration = _run_cli(first_config_path, stop_after_steps=1)
        run_dir = config.output_dir.resolve() / config.run_name
        checkpoint = run_dir / "checkpoints" / "step-00000001"
        resume_config_path = temporary_path / "resume.yaml"
        _write_config(
            resume_config_path,
            replace(config, resume_from_checkpoint=checkpoint),
        )
        resume_duration = _run_cli(resume_config_path)

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint_root.rglob("*") if path.is_file()
    )
    final_checkpoint = checkpoint_root / "step-00000002"
    required_inference_assets = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    summary = {
        "audited_train_sha256": sha256_file(args.audited_train),
        "checkpoint_bytes_before_cleanup": checkpoint_bytes,
        "checkpoint_inference_assets_complete": all(
            (final_checkpoint / name).is_file() for name in required_inference_assets
        ),
        "checkpoints_retained": args.keep_checkpoints,
        "config_hash": config.semantic_hash(),
        "final_status": manifest["status"],
        "first_process_seconds": first_duration,
        "max_cuda_memory_mb_by_step": [
            record["max_cuda_memory_mb"]
            for record in metrics
            if record["event"] == "train"
        ],
        "optimizer_steps": manifest["optimizer_step"],
        "project_tree_sha256": runtime_tree_hash(
            runtime_file_hashes(Path(__file__).resolve().parents[1])
        ),
        "resume_process_seconds": resume_duration,
        "sample_id": sample["sample_id"],
        "sequence_length": len(sample["input_ids"]),
    }
    atomic_write_json(run_dir / "preflight_summary.json", summary)
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_root)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
