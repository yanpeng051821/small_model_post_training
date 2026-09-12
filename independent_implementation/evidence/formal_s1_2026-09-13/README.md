# Gate 0B Formal S1 Finalization

> Finalized: 2026-09-13 04:47 CST
> Status: training completed; partial paired evaluation completed
> Decision: preserve S1 and continue evaluation; do not claim overall success yet

## 1. Training outcome

The formal S1 run completed successfully on one NVIDIA A100-SXM4-80GB:

| Field | Value |
| --- | ---: |
| Model | `Qwen/Qwen3-0.6B-Base` |
| Training records | 61,224 |
| Epochs | 1 |
| Optimizer steps | 479 / 479 |
| Exit code | 0 |
| Runtime | 31,867.14 seconds, about 8 h 51 min |
| Aggregate train loss | 0.56893 |
| Tokens processed | 355,235,351 |
| Peak allocated CUDA memory | 38.97 GiB |
| Peak reserved CUDA memory | 76.63 GiB |
| Final checkpoint | `checkpoint-479` |

The final model and `checkpoint-479/model.safetensors` have the same SHA-256:

```text
d82a6c23c39574fecddefbe55f034db21551d331f54c7c663d9bc5eaf6a027b4
```

The final checkpoint also contains the optimizer, scheduler, RNG, trainer state,
training arguments, tokenizer, and model configuration required for inspection and
resume. The complete 16 GiB run remains on the persistent server data disk; model
weights and optimizer state are deliberately excluded from Git.

## 2. Training health

The run manifest reports `status: completed`, `global_step: 479`, and a final exit
code of zero. No OOM, traceback, NaN, or Inf failure was found. The log contains a
FlashAttention dtype warning emitted while the model was initially in FP32, but the
runner used BF16 autocast and completed all steps. This warning is retained as
evidence and should be cleaned up in the model-loading path later; it did not stop
this run.

The final logged step had:

```text
loss=0.5290
mean_token_accuracy=0.8417
grad_norm=0.5845
learning_rate=4.0004e-06
```

These values support numerical completion, not downstream capability by themselves.

## 3. Validation NLL

The full held-out validation artifact was evaluated with the same artifact hash and
all 1,968 records used by B0:

| Metric | B0 | S1 | Change |
| --- | ---: | ---: | ---: |
| Token-weighted mean NLL | 0.78048 | 0.54661 | -0.23387 (-30.0%) |
| Per-record mean NLL used by paired comparison | 0.76468 | 0.51176 | -0.25291 |

The paired per-record delta 95% bootstrap interval is
`[-0.25515, -0.25067]`. All 1,968 records improved under the per-record comparison.
This is strong evidence that S1 learned the target SFT distribution. It is not by
itself evidence that general mathematical reasoning improved.

## 4. Full GSM8K paired evaluation

The frozen full GSM8K contract completed on all 1,319 examples:

| Metric | B0 | S1 | Change |
| --- | ---: | ---: | ---: |
| `qem` | 0.47612 | 0.51478 | +0.03867 (+3.87 points) |

At sample level, 199 examples improved, 148 regressed, and 972 were unchanged. The
aggregate result is encouraging and is based on the full paired task, but it does
not remove the need for MATH-500 and the general-capability regression panel.

## 5. MATH-500 smoke warning

A paired 20-question smoke run used the same `gate0b-eval-smoke-v1` contract for B0
and S1 (`max_new_tokens=512`, four sampled responses per question):

| Metric | B0 | S1 | Change |
| --- | ---: | ---: | ---: |
| `math_pass@1:1_samples` | 0.25 | 0.05 | -0.20 |
| `math_pass@1:4_samples` | 0.30 | 0.075 | -0.225 |

This is a material regression signal, but only a smoke result with high sampling
uncertainty. It must not be reported as the final MATH-500 score. It changes the
next action: the frozen 500-question MATH-500 evaluation and changed-sample error
analysis are now mandatory before accepting or rejecting S1.

## 6. Generation health

The final model reloaded in a fresh process and produced non-empty outputs for all
four smoke prompts with no runtime error. All four generations reached the 256-token
limit without emitting the configured stop token, so `eos_rate=0.0`.

The command passed only because the frozen technical gate had
`minimum_eos_rate=0.0`. Therefore this result proves loadability and generation,
not satisfactory response termination. A longer and format-aware generation check
is still required.

## 7. Artifact locations

Server, including model weights and optimizer state:

```text
/workspace/s1-preflight-20260912/independent_implementation/runs/gate0b-16k-s1
```

Local complete lightweight bundle, extracted evidence, logs, and tokenizer files:

```text
D:\pythonlearning\training_artifacts\formal_s1_2026-09-13
```

Curated Git-safe evidence is in this directory. The source bundle SHA-256 is:

```text
fc5ae98df2e062b3ea99029b93b4b13ac9e4c6e498f7956d119ef7874d07a02d
```

## 8. Remaining evaluation work

1. Run the frozen full 500-question MATH-500 contract for S1.
2. Run the frozen 59-task regression panel for S1.
3. Run `compare_lighteval_results.py` against the corresponding full B0 artifacts.
4. Inspect the MATH smoke regressions and GSM8K improved/regressed subsets.
5. Repeat generation health with a justified token budget and explicit response
   completion checks.
6. Make the final accept/reject/iterate decision only after these results exist.

## 9. Evidence index

- `run_manifest.json`: immutable training identity, environment, runtime, and metrics.
- `cuda_memory_summary.json`: final and peak CUDA allocator measurements.
- `sha256sums.txt`: hashes for model, checkpoint, optimizer, scheduler, RNG, and logs.
- `artifact_inventory.tsv`: server-side run artifact inventory.
- `validation_nll_summary.json`: full S1 held-out NLL result.
- `b0_vs_s1_validation_nll.json`: paired NLL comparison.
- `b0_vs_s1_gsm8k.json`: full paired GSM8K comparison.
- `b0_vs_s1_math500_smoke.json`: paired 20-question warning signal.
- `generation_health_summary.json`: fresh-process generation gate result.
- `training.log` and `training.exit`: formal launch output and exit status.
