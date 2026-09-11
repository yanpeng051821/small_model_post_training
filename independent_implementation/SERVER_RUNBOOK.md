# Gate 0B Server Runbook

> Status: executed on 2026-09-11 on an A100-80GB host. See §11 for the launch recipe that actually worked, the required environment variables, and measured timings. Stop immediately when any earlier command returns nonzero. Commands assume a Linux CUDA host and the `independent_implementation` repository root.

After the full audit and human review pass, generate the handoff identity on the local machine from the completed full-audit manifest:

```powershell
uv run python scripts/build_server_bundle_manifest.py `
  --data-manifest E:\small_model_post_training_gate0b\artifacts\gate0b-full-openr1-a\data_manifest.json `
  --review-summary evidence\data_review_decisions.summary.json `
  --output evidence/server_bundle_manifest.json
```

Generate this file only after code and configuration are final. Any later runtime-file edit invalidates its `project.tree_sha256` and requires a new manifest.

## 1. Frozen Inputs

Copy the completed audit directory to `artifacts/gate0b/`. It is valid only when these five files exist and `data_manifest.json` says `frozen_candidate: true`:

```text
data_manifest.json
train.jsonl
validation.jsonl
rejected.jsonl
review_samples.jsonl
```

## 2. Environment

```bash
set -euo pipefail
uv sync --frozen --all-groups
uv pip install -r requirements-server.txt
uv pip install setuptools
uv pip install 'flash-attn==2.7.4.post1' --no-build-isolation
mkdir -p evidence/server
uv pip freeze > evidence/server/pip-freeze.txt
nvidia-smi -q > evidence/server/nvidia-smi.txt
```

FlashAttention is built against the installed Torch/CUDA environment. A failed build is a hard stop; do not silently change the frozen training backend in the same experiment.

## 3. Runtime Gate

```bash
uv run python scripts/server_preflight.py \
  --train-artifact artifacts/gate0b/train.jsonl \
  --validation-artifact artifacts/gate0b/validation.jsonl \
  --data-manifest artifacts/gate0b/data_manifest.json \
  --bundle-manifest evidence/server_bundle_manifest.json \
  --minimum-free-gib 30 \
  --output evidence/server/server_preflight.json
```

The report must say `passed: true`. It checks CUDA, BF16, package versions, the fixed LightEval commit, vLLM, Math Verify, FlashAttention, artifact hashes, lock hash, and free disk.

## 4. Longest-Sample Training Memory Gate

The current local 8 GB preflight used a 105-token sample. Before any baseline evaluation or multi-step training, exercise the actual maximum accepted length, including backward, clipping, AdamW state creation, and the frozen BF16/FlashAttention/gradient-checkpointing path:

```bash
uv run python scripts/probe_longest_training_sample.py \
  --config configs/gate0b/sft_train.yaml \
  --output evidence/server/longest_sample_memory.json
```

The report must say `status: passed` and `sequence_length: 22295`. This command uses the formal FP32 master-parameter/AdamW-state plus BF16-autocast contract, and creates optimizer moments during the step. An OOM or any other nonzero exit is a hard stop. Gate 0B is single-GPU; select a larger-memory GPU rather than changing max length, data, label mask, or introducing an unverified distributed path.

## 5. Evaluation Dry Run

```bash
for SUITE in math500 gsm8k regression; do
  uv run python scripts/run_frozen_eval.py \
    --suite "$SUITE" \
    --model Qwen/Qwen3-0.6B-Base \
    --model-revision 311c62e88814bff7206909ccd330bab0a784743b \
    --output-dir "runs/gate0b/b0/evals/$SUITE" \
    --dry-run
done
```

Review each `invocation_manifest.json`. Remove only `--dry-run` after model revision, tasks, chat-template mode, and generation parameters match `configs/gate0b/evaluation.yaml`.

## 6. B0 Baseline

```bash
uv run python scripts/evaluate_validation_nll.py \
  --config configs/gate0b/sft_train.yaml \
  --model Qwen/Qwen3-0.6B-Base \
  --model-revision 311c62e88814bff7206909ccd330bab0a784743b \
  --output-dir runs/gate0b/b0/validation-nll

uv run python scripts/verify_checkpoint_generation.py \
  --checkpoint Qwen/Qwen3-0.6B-Base \
  --checkpoint-revision 311c62e88814bff7206909ccd330bab0a784743b \
  --tokenizer Qwen/Qwen3-0.6B-Base \
  --tokenizer-revision 311c62e88814bff7206909ccd330bab0a784743b \
  --prompts configs/gate0b/smoke_prompts.jsonl \
  --output-dir runs/gate0b/b0/generation-health \
  --minimum-eos-rate 0.0

for SUITE in math500 gsm8k regression; do
  uv run python scripts/run_frozen_eval.py \
    --suite "$SUITE" \
    --model Qwen/Qwen3-0.6B-Base \
    --model-revision 311c62e88814bff7206909ccd330bab0a784743b \
    --output-dir "runs/gate0b/b0/evals/$SUITE"
done
```

Do not start training until all three runs retain detailed per-sample outputs.

## 7. First-Batch Shadow

```bash
uv run python scripts/compare_first_batch.py \
  --config configs/gate0b/sft_smoke.yaml \
  --output evidence/server/first_batch_shadow.json
```

This performs no optimizer step. Valid-token count, loss, global gradient norm, and deterministic gradient probes must match the Transformers causal-LM loss path used by TRL `0.18.0`.

## 8. 20-Step Smoke And Resume

The smoke config limits training to 128 records and validation to 32. It repeats this engineering fixture for 20 epochs, one optimizer step per epoch; it is not an effect estimate.

```bash
uv run sft-train \
  --config configs/gate0b/sft_smoke.yaml \
  --stop-after-steps 10

uv run python scripts/make_resume_config.py \
  --base-config configs/gate0b/sft_smoke.yaml \
  --checkpoint runs/gate0b/qwen3-0.6b-sft-smoke/checkpoints/step-00000010 \
  --output runs/gate0b/qwen3-0.6b-sft-smoke/resume.yaml

uv run sft-train \
  --config runs/gate0b/qwen3-0.6b-sft-smoke/resume.yaml

uv run python scripts/verify_checkpoint_generation.py \
  --checkpoint runs/gate0b/qwen3-0.6b-sft-smoke/checkpoints/step-00000020 \
  --prompts configs/gate0b/smoke_prompts.jsonl \
  --output-dir runs/gate0b/qwen3-0.6b-sft-smoke/generation-health \
  --minimum-eos-rate 1.0
```

Any NaN/Inf, hash mismatch, missing EOS, failed reload, or nonzero exit is a hard stop.

## 9. 100-Step Pilot

```bash
uv run sft-train \
  --config configs/gate0b/sft_pilot.yaml \
  --stop-after-steps 50

uv run python scripts/make_resume_config.py \
  --base-config configs/gate0b/sft_pilot.yaml \
  --checkpoint runs/gate0b/qwen3-0.6b-sft-pilot/checkpoints/step-00000050 \
  --output runs/gate0b/qwen3-0.6b-sft-pilot/resume.yaml

uv run sft-train \
  --config runs/gate0b/qwen3-0.6b-sft-pilot/resume.yaml

uv run python scripts/verify_checkpoint_generation.py \
  --checkpoint runs/gate0b/qwen3-0.6b-sft-pilot/checkpoints/step-00000100 \
  --prompts configs/gate0b/smoke_prompts.jsonl \
  --output-dir runs/gate0b/qwen3-0.6b-sft-pilot/generation-health \
  --minimum-eos-rate 1.0
```

Review sampler position, optimizer step, scheduler learning rate, valid tokens seen, validation NLL, and generation health before approving S1.

Run the pinned TRL reference as a separate 100-step job. It uses the same first 12,800 sample IDs and completion masks as the independent pilot, but writes to an unrelated output directory:

```bash
uv run python scripts/train_sft_trl_reference.py \
  --config configs/gate0b/sft_pilot.yaml \
  --output-dir runs/gate0b/qwen3-0.6b-sft-trl-reference \
  --max-steps 100
```

Compare semantics and trend, not byte-identical checkpoints: the independent runner and `Trainer` have different logging/checkpoint containers, while the frozen data order, effective batch, loss mask, optimizer, scheduler and precision contract must agree.

## 10. Formal S1

Start from the frozen base model, never from smoke or pilot:

```bash
uv run sft-train --config configs/gate0b/sft_train.yaml
```

Set the final checkpoint path and repeat the exact B0 contract:

```bash
S1_CHECKPOINT=runs/gate0b/qwen3-0.6b-sft-s1/checkpoints/step-XXXXXXXX
uv run python scripts/evaluate_validation_nll.py \
  --config configs/gate0b/sft_train.yaml \
  --model "$S1_CHECKPOINT" \
  --output-dir runs/gate0b/s1/validation-nll

uv run python scripts/verify_checkpoint_generation.py \
  --checkpoint "$S1_CHECKPOINT" \
  --prompts configs/gate0b/smoke_prompts.jsonl \
  --output-dir runs/gate0b/s1/generation-health \
  --minimum-eos-rate 0.0

for SUITE in math500 gsm8k regression; do
  uv run python scripts/run_frozen_eval.py \
    --suite "$SUITE" \
    --model "$S1_CHECKPOINT" \
    --output-dir "runs/gate0b/s1/evals/$SUITE"
done

uv run python scripts/compare_validation_nll.py \
  --baseline-records runs/gate0b/b0/validation-nll/records.jsonl \
  --trained-records runs/gate0b/s1/validation-nll/records.jsonl \
  --output runs/gate0b/b0-vs-s1-validation-nll.json

uv run python scripts/compare_lighteval_results.py \
  --baseline-root runs/gate0b/b0/evals \
  --trained-root runs/gate0b/s1/evals \
  --output-dir runs/gate0b/b0-vs-s1-lighteval
```

The LightEval comparison refuses different contracts, task sets, prompts, few-shot counts, input tokens, golds, or choices. It writes aggregate per-metric deltas plus every changed prediction and regression to `changed_samples.jsonl`. Do not change prompts, sampling, few-shot counts, scorer, or task versions between B0 and S1. Final claims require this paired per-sample analysis and the decision rules in `EXPERIMENT_CONTRACT.md`.

## 11. Measured Execution Notes (2026-09-11)

This section records the deviations observed on the first real execution. The commands above remain the contract; the notes below say what had to change in practice.

### 11.1 Use a local snapshot path instead of repo id + revision

`--model Qwen/Qwen3-0.6B-Base --model-revision <sha>` triggers a `repo_info` network call. The link to `huggingface.co` from this host is unreliable (the same day `scripts/audit_openr1_math.py` failed with `requests.exceptions.ReadTimeout`), and the call can stall for minutes. Pass the cached snapshot path instead:

```bash
MODEL=/workspace/hf-cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/311c62e88814bff7206909ccd330bab0a784743b
```

### 11.2 Required environment variables

```bash
cd /workspace/small_model_post_training/independent_implementation
setsid env \
  PATH="/workspace/small_model_post_training/independent_implementation/.venv/bin:$PATH" \
  HF_HOME=/workspace/hf-cache \
  HF_HUB_ETAG_TIMEOUT=120 \
  HF_HUB_DOWNLOAD_TIMEOUT=120 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
  .venv/bin/python scripts/run_frozen_eval.py \
    --suite <math500|gsm8k|regression> \
    --model "$MODEL" \
    --output-dir "runs/gate0b-16k/b0/evals/<suite>" \
  > /workspace/b0-<suite>.log 2>&1 < /dev/null &
```

| Variable | Why it is required |
| --- | --- |
| `PATH=.venv/bin:$PATH` | `lighteval` is installed only inside the venv; the default login `PATH` does not contain it |
| `HF_HOME=/workspace/hf-cache` | persistent model and dataset cache; avoids re-downloading on every rented session |
| `HF_HUB_ETAG_TIMEOUT` / `HF_HUB_DOWNLOAD_TIMEOUT=120` | the default 10 s read timeout is too short for this host's link to `huggingface.co` |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | after flash-attn is installed, importing it initialises CUDA in the parent; vLLM's default forked engine then dies with `RuntimeError: Cannot re-initialize CUDA in forked subprocess` |
| `LIBRARY_PATH=/usr/local/cuda/lib64/stubs` | the host has `libcuda.so.1` but no `libcuda.so` development symlink, so Triton's runtime compile fails with `/usr/bin/ld: cannot find -lcuda` |

Do **not** set `HF_HUB_OFFLINE=1`. `datasets` still needs the network to resolve the legacy dataset short names LightEval uses (for example `ai2_arc`) into full repository ids; offline mode fails with `ConnectionError: Couldn't reach 'ai2_arc' on the Hub (OfflineModeIsEnabled)`. The dataset payloads are already cached, so online access is used only for name resolution. A failed attempt is archived under `runs/gate0b-16k/b0/evals/failures/`.

### 11.3 Dataset prefetch

`cais/mmlu`, `allenai/ai2_arc`, `Rowan/hellaswag` and `openai/gsm8k` were not cached and had to be fetched once:

```bash
setsid env HF_HOME=/workspace/hf-cache HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=120 \
  .venv/bin/python -c "from huggingface_hub import snapshot_download; [print('OK', r, snapshot_download(r, repo_type='dataset')) for r in ['cais/mmlu','allenai/ai2_arc','Rowan/hellaswag','openai/gsm8k']]" \
  > /workspace/b0-dataset-prefetch.log 2>&1 < /dev/null &
```

`repo_type='dataset'` is required; omitting it makes `huggingface_hub` query the model API and fail with `RepositoryNotFoundError`. The resolved revisions matched the ones pinned in `artifacts/gate0b-16k-derived-a/data_manifest.json`.

### 11.4 Measured timings (A100-80GB, Qwen3-0.6B-Base)

| Suite | Wall time | Notes |
| --- | --- | --- |
| `regression` | ~50 min | ~15 min dataset preparation, ~20 min CPU-side construction of ~100k log-likelihood requests, remainder scoring. GPU stays idle during request construction. |
| `gsm8k` | ~3.5 min | 1319 prompts, 31 s of generation, ~5,020 aggregate output tokens/s |
| `math500` | hours | see 11.5 |

### 11.5 Long tail on generative suites

`math500` uses `max_new_tokens=32768`. Samples that do not emit a stop token run to that cap, so a single evaluation has a long tail: the first ~half of the 2,000 generations complete quickly (a few hundred tokens each), while the remainder can take hours. Budget for this, and note that the same cost applies again to the S1 evaluation. A model that stops reliably would make the same suite much cheaper.
