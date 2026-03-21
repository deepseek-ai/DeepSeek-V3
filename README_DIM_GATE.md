# Dimension Gate Notes

This document explains the first round of changes made to this DeepSeek-V3 repo, why those changes were made, and how the hidden-dimension gate currently works.

The goal of this round was not to make full DeepSeek-V3 run on a Mac mini. That is not realistic. The goal was to turn the repo into a usable architecture donor for local experiments by:

- shrinking the model to a toy configuration
- inserting the exact hidden-dimension gate logic from the Kanae-Kocho prototype
- making the toy path run on Apple Silicon through PyTorch MPS

## First-Round Changes

The first round touched these files:

- `inference/model.py`
- `inference/generate.py`
- `inference/requirements.txt`
- `inference/dim_gate.py`
- `inference/toy_run.py`
- `inference/configs/config_toy_gate.json`

What changed in each file:

### `inference/model.py`

- Added optional import handling for Triton kernels so the repo can be imported on systems that do not support Triton.
- Extended `ModelArgs` with:
  - Apple-Silicon-friendly `fp32` mode
  - explicit `attn_impl` selection
  - a `moe_gate_bias` override
  - hidden-dimension gate hyperparameters
- Changed `MLA` to read `args.attn_impl` instead of only relying on the old global.
- Added clearer runtime errors if a quantized path is requested without Triton.
- Added `HiddenDimensionGate` to `Block`.
- Inserted the gate at the attention path only:
  - `attn_input = attn_norm(x)`
  - `attn_input = dim_gate(attn_input, start_pos)`
  - `x = x + attn(attn_input, ...)`
- Added `fp32` dtype handling for small-device experiments.
- Made the file-level smoke path stop assuming CUDA is always present.

### `inference/dim_gate.py`

- New file.
- Contains the port of the Kanae-Kocho hidden-dimension selector and clamp logic.
- Implements:
  - `HiddenGateStats`
  - `gate_bounds`
  - `PlasticContextSpiker`
  - `HiddenDimensionGate`

### `inference/generate.py`

- Added device selection that supports:
  - `cuda`
  - `mps`
  - `cpu`
- Removed hard-coded CUDA tensor allocations in the generation loop.
- Added default dtype selection so non-CUDA runs fall back to `fp32`.

### `inference/toy_run.py`

- New file.
- Provides a small smoke-test runner for the toy model without needing official weights.
- Runs:
  - a prompt forward pass
  - one incremental decode step
  - per-layer gate-stat reporting

### `inference/configs/config_toy_gate.json`

- New toy config intended for local architecture experiments.
- Keeps the model dense by setting `n_dense_layers == n_layers`.
- Enables the hidden-dimension gate.
- Uses `fp32` so the path is safe on Apple Silicon.

### `inference/requirements.txt`

- Triton is now marked Linux-only.
- This keeps `pip install -r inference/requirements.txt` workable on macOS.

## Exact Insertion Point

The gate is inserted at the same conceptual point used in the Rust prototype:

- after RMSNorm
- before attention projections

In this repo that means:

1. `x` enters a `Block`
2. `attn_norm(x)` is computed
3. the hidden-dimension gate masks that normalized hidden state
4. the masked tensor is sent into MLA
5. the attention output is added back through the residual path

This is important because the gate is deciding which hidden channels are allowed to influence:

- `wq`
- `wkv_a`
- downstream attention behavior

It is not replacing the MoE expert gate. The existing `Gate` class in this repo still gates experts. The new dimension gate controls hidden channels.

## What "Exact Gate Logic" Means Here

The current implementation preserves the selector dynamics that mattered in the original prototype.

### Per-dimension state

For each batch element and hidden dimension, the selector keeps:

- `u`: membrane state
- `context`: exponentially decayed running context
- `trace`: eligibility trace
- `threshold`: adaptive firing threshold
- `weights`: learned per-dimension gain

The inference port keeps these states because they are part of the mechanism, not just training scaffolding.

### Step update

For each token position, the selector does:

1. Update running context:
   - `context = context_leak * context + (1 - context_leak) * x`
2. Form a context-augmented hidden signal:
   - `combined = x + context_scale * context`
3. Apply learned per-dimension weights:
   - `weighted = combined * weights`
4. Update membrane:
   - `u = leak * u + weighted`
5. Compute a graded support signal:
   - `support = sigmoid((u - threshold) / scale) * abs(combined)`
6. Update the eligibility trace:
   - `trace = trace_decay * trace + support`
7. Mark active dimensions where:
   - `u > threshold`
8. Add an active bonus to the trace for fired dimensions.

The support signal and active bonus are preserved from the original gate logic.

### Adaptive threshold

After selecting active dimensions for a token, the threshold is adjusted toward a target firing rate:

- `rate = active_dims / dim`
- `threshold += threshold_lr * (rate - target_rate)`
- threshold is clamped at zero from below

This is what keeps the selector from trivially firing everywhere or nowhere forever.

## Clamp Logic

The selector output is not used raw.

It is passed through the same style of clamp logic as the Rust implementation:

1. Compute the requested active-dimension budget from:
   - `gate_frac`
   - `gate_softness`
2. Convert that into:
   - `min_active`
   - `max_active`
3. If too many dimensions fire:
   - truncate to `max_active`
4. If too few dimensions fire:
   - backfill dimensions using `abs(value) * abs(weight)` ranking until `min_active` is reached

This means the active set is always bounded, even when the raw selector is unstable.

## What the Gate Actually Does in This Port

The gate currently applies a dense mask:

- selected dimensions keep their normalized activation
- non-selected dimensions are set to zero

So the current mechanism changes information flow, but it does not yet reduce kernel FLOPs in a hardware-efficient way.

That distinction matters:

- behaviorally, the gate is real now
- computationally, it is still dense execution with masked activations

This is the right first step. It lets us test whether the mechanism is useful before attempting sparse kernels or structural pruning.

## Decode-Time State Handling

The gate state is decode-aware.

- If `start_pos == 0`, the selector state is reset.
- If decoding continues from a later position, the selector state is reused.

That means the gate can carry temporal context across incremental generation in the toy path.

## What Is Intentionally Not Implemented Yet

This round does not yet include:

- reward-based training updates for the gate
- batch-signal updates
- sparse QKV kernels
- sparse KV storage driven by gate indices
- FFN-path gating
- compatibility with official giant DeepSeek checkpoints under the toy config

The current port is inference-first, because this repo itself is inference-first.

## Apple Silicon Compatibility

The toy path is compatible with Apple Silicon through PyTorch MPS.

Validated path:

```bash
cd /Volumes/Blackhole/Github/DEEPSEEK-V3
./.venv/bin/python inference/toy_run.py \
  --config inference/configs/config_toy_gate.json \
  --device mps
```

This was validated with:

- PyTorch `2.4.1`
- `mps available == True`
- toy forward pass
- one incremental decode step

The official Triton and FP8 paths remain Linux/CUDA-oriented.

## How to Read the Current Port

If you want to inspect the implementation in order, read:

1. `inference/model.py`
2. `inference/dim_gate.py`
3. `inference/configs/config_toy_gate.json`
4. `inference/toy_run.py`

Focus first on:

- `Block.forward` for the insertion point
- `HiddenDimensionGate.forward` for masking behavior
- `PlasticContextSpiker.step` for selector dynamics

## Practical Meaning of This Round

After this first round, this repo can now do something it could not do before:

- instantiate a toy DeepSeek-style architecture locally
- run it on Apple Silicon
- apply the Kanae hidden-dimension gate at the correct transformer insertion point
- inspect gate statistics layer by layer

That gives us a local experimental base for the next step, which should be training or distillation rather than more inference-only plumbing.
