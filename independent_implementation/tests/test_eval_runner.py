import json
from pathlib import Path

from post_training_core.eval_runner import (
    build_lighteval_command,
    run_lighteval,
)

CONFIG = Path(__file__).parents[1] / "configs" / "gate0b" / "evaluation.yaml"


def test_math500_command_freezes_openr1_generation_contract(tmp_path):
    command, evidence = build_lighteval_command(
        config_path=CONFIG,
        suite_name="math500",
        model="Qwen/Qwen3-0.6B-Base",
        model_revision="fixed-revision",
        output_dir=tmp_path,
    )

    model_args = command[2]
    assert command[:2] == ["lighteval", "vllm"]
    assert command[3] == "lighteval|math_500|0|0"
    assert "dtype=bfloat16" in model_args
    assert "max_model_length=32768" in model_args
    assert "max_new_tokens:32768" in model_args
    assert "temperature:0.6" in model_args
    assert "top_p:0.95" in model_args
    assert "seed:1234" in model_args
    assert "--use-chat-template" in command
    assert "--save-details" in command
    assert evidence["lighteval_commit"].startswith("d3da6b9")


def test_regression_panel_has_59_non_chat_likelihood_tasks(tmp_path):
    command, evidence = build_lighteval_command(
        config_path=CONFIG,
        suite_name="regression",
        model="checkpoint",
        model_revision=None,
        output_dir=tmp_path,
    )

    assert len(evidence["tasks"]) == 59
    assert "leaderboard|arc:challenge|25|0" in evidence["tasks"]
    assert "leaderboard|hellaswag|10|0" in evidence["tasks"]
    assert sum("mmlu:" in task for task in evidence["tasks"]) == 57
    assert "--use-chat-template" not in command


def test_dry_run_writes_exact_invocation_without_starting_process(tmp_path):
    command, evidence = build_lighteval_command(
        config_path=CONFIG,
        suite_name="gsm8k",
        model="checkpoint",
        model_revision=None,
        output_dir=tmp_path,
        max_samples=2,
    )

    returncode = run_lighteval(
        command=command,
        evidence=evidence,
        output_dir=tmp_path,
        dry_run=True,
    )

    manifest = json.loads((tmp_path / "invocation_manifest.json").read_text())
    assert returncode == 0
    assert manifest["status"] == "dry_run"
    assert manifest["command"][-2:] == ["--max-samples", "2"]
