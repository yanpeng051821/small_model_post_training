import subprocess
import sys
from pathlib import Path

import yaml

from post_training_core.config import ExperimentConfig


def test_resume_config_script_preserves_semantic_contract(tmp_path):
    base = tmp_path / "base.yaml"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "resume.yaml"
    payload = {
        "run_name": "resume-contract",
        "model_name_or_path": "model",
        "model_revision": "revision",
        "train_artifact": str(tmp_path / "train.jsonl"),
        "validation_artifact": str(tmp_path / "validation.jsonl"),
        "output_dir": str(tmp_path / "runs"),
        "dtype": "float32",
    }
    base.write_text(yaml.safe_dump(payload), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_resume_config.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-config",
            str(base),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    original = ExperimentConfig.from_yaml(base)
    resumed = ExperimentConfig.from_yaml(output)
    assert resumed.resume_from_checkpoint == checkpoint.resolve()
    assert resumed.semantic_hash() == original.semantic_hash()
