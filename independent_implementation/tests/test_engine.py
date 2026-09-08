import copy
import json
from dataclasses import replace
from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import Qwen3Config, Qwen3ForCausalLM

import post_training_core.engine as engine_module
from post_training_core.config import ExperimentConfig
from post_training_core.data import (
    StatefulRandomSampler,
    TokenizedSFTDataset,
    collate_sft_batch,
)
from post_training_core.engine import (
    StepMetrics,
    TrainingGuard,
    TrainingSoftStop,
    _restore_training_guard,
    build_optimizer,
    build_scheduler,
    run_optimizer_step,
    train,
)
from post_training_core.experiment import set_reproducible_seed
from post_training_core.sft import masked_sft_loss


def _tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    return Qwen3ForCausalLM(config)


def _records():
    return [
        {
            "sample_id": "a",
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
        },
        {
            "sample_id": "b",
            "input_ids": [2, 3, 4],
            "labels": [-100, 3, 4],
        },
        {
            "sample_id": "c",
            "input_ids": [3, 4, 5, 6],
            "labels": [-100, -100, 5, 6],
        },
        {
            "sample_id": "d",
            "input_ids": [4, 5, 6],
            "labels": [-100, 5, 6],
        },
    ]


def _config(tmp_path, run_name="run", resume=None):
    return ExperimentConfig(
        run_name=run_name,
        model_name_or_path="local-tiny-qwen",
        model_revision="test-revision",
        train_artifact=tmp_path / "train.jsonl",
        validation_artifact=tmp_path / "validation.jsonl",
        output_dir=tmp_path,
        dtype="float32",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        max_steps=4,
        learning_rate=1.0e-2,
        max_grad_norm=1.0,
        log_every_steps=1,
        eval_every_steps=2,
        save_every_steps=2,
        resume_from_checkpoint=resume,
    )


def _loaders(seed=42, num_workers=0):
    dataset = TokenizedSFTDataset(_records())
    sampler = StatefulRandomSampler(dataset, seed=seed)
    collator = partial(collate_sft_batch, pad_token_id=0)
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    return sampler, train_loader, validation_loader


def test_loss_matches_qwen_builtin_causal_lm_loss():
    set_reproducible_seed(0)
    model = _tiny_qwen()
    collator = partial(collate_sft_batch, pad_token_id=0)
    batch = collator(_records()[:2])

    outputs = model(**batch)
    local_loss = masked_sft_loss(outputs.logits, batch["labels"])

    torch.testing.assert_close(local_loss, outputs.loss)


def test_optimizer_step_uses_real_qwen_interface_and_updates_parameters(tmp_path):
    config = replace(_config(tmp_path), warmup_ratio=0.0)
    model = _tiny_qwen()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, total_steps=4)
    collator = partial(collate_sft_batch, pad_token_id=0)
    before = model.model.embed_tokens.weight.detach().clone()

    metrics = run_optimizer_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        micro_batches=[collator(_records()[:1]), collator(_records()[1:2])],
        device=torch.device("cpu"),
        dtype="float32",
        max_grad_norm=1.0,
    )

    assert metrics.valid_tokens == 4
    assert metrics.mean_loss > 0
    assert metrics.grad_norm > 0
    assert not torch.equal(before, model.model.embed_tokens.weight)
    state_dtypes = {
        value.dtype
        for state in optimizer.state.values()
        for name, value in state.items()
        if name in {"exp_avg", "exp_avg_sq"}
    }
    assert state_dtypes == {torch.float32}


def test_scheduler_rounds_warmup_up_like_transformers(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), warmup_ratio=0.03)
    model = _tiny_qwen()
    optimizer = build_optimizer(config, model)
    captured = {}

    def fake_get_scheduler(name, optimizer, **kwargs):
        captured.update(kwargs)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    monkeypatch.setattr(engine_module, "get_scheduler", fake_get_scheduler)

    build_scheduler(config, optimizer, total_steps=487)

    assert captured["num_warmup_steps"] == 15


@pytest.mark.parametrize("num_workers", [0, 2])
def test_interrupted_training_resumes_to_same_result(tmp_path, num_workers):
    set_reproducible_seed(7)
    initial = _tiny_qwen()
    uninterrupted = copy.deepcopy(initial)
    interrupted = copy.deepcopy(initial)

    full_config = _config(tmp_path / "full", run_name="same-contract")
    full_run_dir = full_config.output_dir / full_config.run_name
    full_run_dir.mkdir(parents=True)
    sampler, train_loader, validation_loader = _loaders(num_workers=num_workers)
    full_state = train(
        config=full_config,
        run_dir=full_run_dir,
        model=uninterrupted,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        sampler=sampler,
    )

    resume_config = _config(tmp_path / "resume", run_name="same-contract")
    resume_run_dir = resume_config.output_dir / resume_config.run_name
    resume_run_dir.mkdir(parents=True)
    sampler, train_loader, validation_loader = _loaders(num_workers=num_workers)
    first_state = train(
        config=resume_config,
        run_dir=resume_run_dir,
        model=interrupted,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        sampler=sampler,
        stop_after_steps=2,
    )
    checkpoint = resume_run_dir / "checkpoints" / "step-00000002"
    resumed_model = Qwen3ForCausalLM.from_pretrained(checkpoint)
    resumed_config = _config(
        tmp_path / "resume",
        run_name="same-contract",
        resume=checkpoint,
    )
    sampler, train_loader, validation_loader = _loaders(num_workers=num_workers)
    resumed_state = train(
        config=resumed_config,
        run_dir=resume_run_dir,
        model=resumed_model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        sampler=sampler,
    )

    assert first_state.optimizer_step == 2
    assert full_state.optimizer_step == resumed_state.optimizer_step == 4
    assert full_state.valid_tokens_seen == resumed_state.valid_tokens_seen
    for expected, actual in zip(
        uninterrupted.parameters(), resumed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)

    records = [
        json.loads(line)
        for line in (resume_run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [
        record["optimizer_step"] for record in records if record["event"] == "train"
    ] == [
        1,
        2,
        3,
        4,
    ]


def test_rejects_checkpoint_from_different_contract(tmp_path):
    config = _config(tmp_path)
    model = _tiny_qwen()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sampler, train_loader, validation_loader = _loaders()
    train(
        config=config,
        run_dir=run_dir,
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        sampler=sampler,
        stop_after_steps=2,
    )
    checkpoint = run_dir / "checkpoints" / "step-00000002"
    changed = _config(tmp_path, resume=checkpoint)
    object.__setattr__(changed, "learning_rate", 2.0e-2)
    resumed = Qwen3ForCausalLM.from_pretrained(checkpoint)
    sampler, train_loader, validation_loader = _loaders()

    with pytest.raises(ValueError, match="config hash"):
        train(
            config=changed,
            run_dir=run_dir,
            model=resumed,
            train_dataloader=train_loader,
            validation_dataloader=validation_loader,
            sampler=sampler,
        )


def test_training_records_final_validation_when_step_is_not_eval_interval(tmp_path):
    config = replace(
        _config(tmp_path),
        max_steps=3,
        eval_every_steps=2,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sampler, train_loader, validation_loader = _loaders()

    state = train(
        config=config,
        run_dir=run_dir,
        model=_tiny_qwen(),
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        sampler=sampler,
    )

    records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert state.optimizer_step == 3
    assert [
        record["optimizer_step"]
        for record in records
        if record["event"] == "validation"
    ] == [0, 2, 3]


def _step_metrics(*, loss=1.0, grad_norm=1.0):
    return StepMetrics(
        mean_loss=loss,
        valid_tokens=10,
        grad_norm=grad_norm,
        learning_rate=1.0e-4,
        max_cuda_memory_mb=0.0,
        step_seconds=1.0,
        valid_tokens_per_second=10.0,
    )


def test_training_guard_requires_history_and_three_consecutive_spikes():
    loss_guard = TrainingGuard()
    gradient_guard = TrainingGuard()
    for _ in range(20):
        assert loss_guard.observe(_step_metrics()) is None
        assert gradient_guard.observe(_step_metrics()) is None

    assert loss_guard.observe(_step_metrics(loss=6.0)) is None
    assert loss_guard.observe(_step_metrics(loss=6.0)) is None
    assert "loss exceeded" in loss_guard.observe(_step_metrics(loss=6.0))

    assert gradient_guard.observe(_step_metrics(grad_norm=11.0)) is None
    assert gradient_guard.observe(_step_metrics(grad_norm=11.0)) is None
    assert "gradient norm exceeded" in gradient_guard.observe(
        _step_metrics(grad_norm=11.0)
    )


def test_training_guard_restores_streak_from_durable_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [
        {
            "event": "train",
            "mean_loss": 1.0,
            "valid_tokens": 10,
            "grad_norm": 1.0,
            "learning_rate": 1.0e-4,
            "max_cuda_memory_mb": 0.0,
        }
        for _ in range(20)
    ]
    records.extend(
        {
            "event": "train",
            "mean_loss": 6.0,
            "valid_tokens": 10,
            "grad_norm": 1.0,
            "learning_rate": 1.0e-4,
            "max_cuda_memory_mb": 0.0,
        }
        for _ in range(2)
    )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    guard = _restore_training_guard(tmp_path)

    assert "loss exceeded" in guard.observe(_step_metrics(loss=6.0))


def test_soft_stop_saves_checkpoint_and_durable_event(tmp_path, monkeypatch):
    config = replace(
        _config(tmp_path),
        eval_on_start=False,
        eval_every_steps=100,
        max_steps=23,
        num_train_epochs=12,
        save_every_steps=100,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sampler, train_loader, validation_loader = _loaders()
    calls = 0

    def scripted_step(**_kwargs):
        nonlocal calls
        calls += 1
        return _step_metrics(loss=1.0 if calls <= 20 else 6.0)

    monkeypatch.setattr(engine_module, "run_optimizer_step", scripted_step)

    with pytest.raises(TrainingSoftStop, match="loss exceeded") as caught:
        train(
            config=config,
            run_dir=run_dir,
            model=_tiny_qwen(),
            train_dataloader=train_loader,
            validation_dataloader=validation_loader,
            sampler=sampler,
        )

    assert caught.value.state.optimizer_step == 23
    assert caught.value.checkpoint.is_dir()
    records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert records[-1]["event"] == "soft_stop"
