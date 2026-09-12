# Formal S1 State Audit

## Intake Summary

- User intent: normalize all pre-S1 state, retain the evidence required for a
  formal launch, and finish CPU-side work before renting the next GPU.
- Current state: `baseline_ready`, `analysis_ready`, and
  `main_result_ready` only for engineering pilots. Formal S1 itself does not
  exist yet.
- Recommended next anchor: telemetry-enabled GPU preflight after the source and
  scheduler contract are frozen.

## Asset Matrix

| Area | Current asset | Trust level | Why | Missing proof | Action |
| --- | --- | --- | --- | --- | --- |
| Baseline | Full B0 NLL, MATH-500, GSM8K, regression evidence | trusted | Fixed model revision, task contract, command, outputs, and metrics exist | None for reuse | Preserve and pair against S1 |
| Training data | `gate0b-16k-derived-a` | trusted | Fresh live hashes and counts match the server manifest | None for data identity | Bind S1 to these hashes |
| Core loss/update | CPU tests and first-batch shadow | trusted | PyTorch/TRL loss and gradient probes agree | None for H1 | Reuse; do not reimplement before S1 |
| 20-step smoke | A100 execution records | usable with verification | Real BF16 training, checkpointing, and generation path completed | Final source commit did not yet emit allocator telemetry | Repeat only a tiny telemetry smoke |
| 100-step pilot | resumed TRL checkpoint, NLL, limited evals | usable with verification | Resume and finite training complete; NLL dropped on identical validation data | MATH pilot failed without terminal failure manifest; not a formal candidate | Retain as engineering evidence only |
| Evaluation interface | B0 full runs plus pilot GSM8K/regression | usable with verification | B0 command completed, pilot non-math suites completed | MATH pilot invocation needs a reproducible preflight | Run disposable B0/pilot-like evaluation preflight |
| Source tree | `origin/main` plus this pre-launch branch | needs verification | Current remote code passed CPU tests after telemetry integration | Need GPU runtime evidence and final commit pin | Freeze after review and preflight |
| Documentation | existing execution/contract documents | stale or conflicting | Several sections still say B0/smoke/pilot are not executed or use old sample counts | Need a current prelaunch record | This audit and plan are the current route reference |

## Reuse Rules

- B0 is the only training-pre baseline. Historical MATH-500 values around 10%
  or 26%-28% are superseded by the recorded B0 run.
- The 100-step pilot must not be resumed, evaluated as S1, or used as a
  checkpoint selection candidate.
- The 16K derived artifact is fixed by file hashes. Its parent data split is
  reference-only for this S1 because it contains records rejected by the
  length policy.
- Any evaluation-contract change requires a versioned contract and a new B0;
  changing only S1-side decoding invalidates the comparison.

## Conflicts That Must Not Be Ignored

1. Some legacy documents still use pre-filter record counts and derive 486 or
   487 updates. The actual retained training count is 61,224, which gives 479
   updates.
2. The B0 MATH-500 run completed, but the pilot MATH invocation has no terminal
   failure state. A launch wrapper must always write a terminal status.
3. Existing pilot GPU CSV data cannot distinguish allocated tensors from the
   PyTorch reserved caching pool. The next GPU preflight must emit allocator
   telemetry from the current source tree.

## Route Decision

Do not rent a long formal-S1 window yet. First commit the current CPU-side
preparation, sync it to the persistent data volume, and use a short GPU window
to close telemetry, update-count, and evaluation-preflight gates. If those
three gates pass without changing the frozen data or objective, reserve the
formal one-epoch S1 window.
