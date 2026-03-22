# Dimension Gate Notes

This document explains what changed in this DeepSeek-V3 fork, how the hidden-dimension gate works now, and what the current Apple Silicon execution path actually does.

The goal of this fork is not to run full DeepSeek-V3 on a Mac mini. The goal is to turn the repo into a usable architecture donor for local experiments by:

- shrinking the model to a toy configuration
- inserting the Kanae-Kocho hidden-dimension gate at the correct transformer location
- making the toy path run on Apple Silicon through PyTorch MPS
- porting the selector-learning logic
- adding native MPS fast paths where Python overhead became the bottleneck

## Current State

The fork now has three attention-input execution modes:

1. Dense baseline
   - no dimension gate
   - standard DeepSeek MLA
2. Gated dense
   - hidden-dimension gate masks the normalized hidden state
   - MLA still uses dense projections
3. Gated sparse projection
   - hidden-dimension gate emits structured metadata
   - MLA uses that metadata to sparsify the `wq`/`wq_a` and `wkv_a` projections
   - the rest of attention stays dense

That means the repo is past the original “dense mask only” stage. It now has real projection-side sparsity with a native MPS backend, but it does not yet have sparse MLA latent-state math or sparse KV cache compaction.

## Files Added Or Changed

The current fork centers on these files:

- `inference/model.py`
- `inference/dim_gate.py`
- `inference/gate_native.py`
- `inference/sparse_mla.py`
- `inference/mla_native.py`
- `inference/native/gate_scan_mps.mm`
- `inference/native/sparse_projection_mps.mm`
- `inference/train_gate_compare.py`
- `inference/sweep_gate_lr.py`
- `inference/sweep_seed_compare.py`
- `inference/bench_sparse_projection.py`
- `inference/toy_run.py`
- `inference/configs/config_toy_gate.json`
- `inference/requirements.txt`

What each one is responsible for:

### `inference/model.py`

- Adds gate-related flags to `ModelArgs`
- Inserts the gate at the attention input:
  - `attn_input = attn_norm(x)`
  - `attn_input = dim_gate(attn_input, start_pos)`
  - `x = x + attn(attn_input, ..., gate_metadata=...)`
- Threads structured gate metadata from the block into MLA
- Uses sparse projection only at the projection boundary:
  - `wq` and `wkv_a` when `q_lora_rank == 0`
  - `wq_a` and `wkv_a` when `q_lora_rank > 0`

### `inference/dim_gate.py`

- Holds the Python port of the Kanae-Kocho selector
- Exposes:
  - `selection_mask`
  - `active_indices`
  - `active_counts`
- Caches MPS-ready `int32` metadata for the native sparse path
- Contains both the selector dynamics and the gate-side learning hooks

### `inference/gate_native.py`

- Loads the MPS extension for the native gate scan
- Keeps the recurrent selector scan off the Python hot path on Apple Silicon

### `inference/sparse_mla.py`

- Isolates projection-side sparsity logic from MLA math
- Dispatches between:
  - masked dense fallback
  - padded compact sparse projection
  - native MPS sparse projection
  - fused native MPS sparse projection pair

### `inference/mla_native.py`

- Loads the native MPS sparse-projection extension
- Exposes single and fused projection calls
- Uses native forward plus exact dense backward formulas in PyTorch

### `inference/native/gate_scan_mps.mm`

- Metal-backed recurrent gate scan
- Replaces the worst Python token loop in the selector hot path

### `inference/native/sparse_projection_mps.mm`

- Metal-backed sparse projection kernels
- Includes:
  - single projection kernels
  - fused pair projection kernels
  - shared-memory and pointwise variants

### Training / benchmark helpers

- `inference/train_gate_compare.py`
  - toy compare harness
  - batch selector accumulator
  - base vs gated evaluation
- `inference/sweep_gate_lr.py`
  - gate LR sweeps
- `inference/sweep_seed_compare.py`
  - multi-seed regression sweeps
- `inference/bench_sparse_projection.py`
  - projection-only microbenchmarks
- `inference/toy_run.py`
  - smoke runner for prompt, decode, and gate stats

## Exact Insertion Point

The gate is still inserted at the same conceptual point as Kanae-Kocho:

- after RMSNorm
- before attention projections

In this repo that means:

1. `Block.forward()` computes `attn_norm(x)`
2. `HiddenDimensionGate` selects active hidden channels
3. MLA receives:
   - the masked activation tensor
   - optional gate metadata
4. MLA decides whether to use dense or sparse projection

This is important because the selector is operating on the hidden state that feeds:

- `wq`
- `wq_a`
- `wkv_a`

It is not replacing the MoE expert gate.

## How The Gate Works

The selector keeps the same core state as the Kanae-Kocho prototype:

- `u`: membrane
- `context`: running context
- `trace`: eligibility trace
- `threshold`: adaptive threshold
- `weights`: learned per-dimension gain

For each token, the selector:

1. Updates the context
   - `context = context_leak * context + (1 - context_leak) * x`
2. Builds a context-augmented signal
   - `combined = x + context_scale * context`
3. Applies learned weights
   - `weighted = combined * weights`
4. Updates membrane
   - `u = leak * u + weighted`
5. Forms a graded support signal
6. Updates the eligibility trace
7. Fires dimensions where `u > threshold`
8. Adjusts `threshold` toward the target firing rate

This is still the core mechanism. What changed is how the selected dimensions are consumed downstream.

## Clamp Logic

Raw selector firing is still clamped into an active-dimension budget:

- `gate_frac` sets the target active fraction
- `gate_softness` defines the min/max band
- underfiring dims are backfilled by `abs(value) * abs(weight)` ranking
- overfiring dims are truncated

That keeps the selector from drifting into “everything on” or “nothing on” behavior.

## Selector Learning Hooks

The DeepSeek port now includes the gate-side learning API from Kanae-Kocho:

- `apply_reward`
- `apply_token_reward`
- `apply_batch_signal`
- `apply_batch_signal_scaled`
- `apply_batch_signal_soft`
- `plasticity_scores`
- reward EMA / advantage tracking
- weight recentering

This means the gate is not just a static runtime mask. It can adapt over training in the toy harness.

## Gate Metadata

The gate no longer acts only by mutating a tensor and hoping MLA survives. It now exposes structured metadata:

- `selection_mask`
- `active_indices`
- `active_counts`
- cached MPS-native `int32` copies for the sparse backend

That gives the repo a proper control plane:

- gate decides what is active
- MLA decides how to execute using that decision

This boundary is what made the sparse projection work possible without rewriting attention math.

## Sparse Projection Path

The current sparse path is stage-2A style projection sparsity, not full sparse MLA.

### Dense fallback

When sparse projection is disabled or unsupported:

- gate output is applied as a dense mask
- MLA uses standard dense projection modules

### Compact sparse reference

When sparse projection is enabled but native MPS cannot be used:

- the code falls back to a padded compact gather/matmul path
- this exists mainly for parity and debugging

### Native MPS sparse projection

On supported MPS shapes:

- active dimension indices are passed into a Metal kernel
- the kernel accumulates only over active hidden dimensions
- single and fused pair kernels are available

### Fused pair projection

The current fast path is the fused pair path:

- `wq + wkv_a` when `q_lora_rank == 0`
- `wq_a + wkv_a` when `q_lora_rank > 0`

This matters because the same gated hidden state and active-index metadata feed both projections. Fusing them avoids paying the metadata and launch overhead twice.

## Native MPS Fast Paths

There are now two native MPS pieces:

### 1. Native gate scan

- lives in `inference/native/gate_scan_mps.mm`
- accelerates the recurrent selector scan
- removes the old Python token loop bottleneck from the gate

### 2. Native sparse projection

- lives in `inference/native/sparse_projection_mps.mm`
- accelerates sparse projection for supported MPS shapes
- supports both single and fused pair kernels

The sparse projection autograd path is intentionally hybrid:

- forward: native MPS kernel
- backward: exact dense formulas in PyTorch

That keeps the implementation correct and easy to validate while still moving the expensive forward path off Python.

## Apple Silicon Compatibility

The toy path is compatible with Apple Silicon through PyTorch MPS.

The repo now handles:

- missing Triton on macOS
- `fp32` toy runs
- MPS-safe causal mask construction
- native MPS gate scan
- native MPS sparse projection

Smoke-checked path:

```bash
cd /Volumes/Blackhole/Github/DEEPSEEK-V3
./.venv/bin/python inference/toy_run.py \
  --config inference/configs/config_toy_gate.json \
  --device mps \
  --simulate-learning
```

## What The Current Local Experiments Show

At the moment, the strongest conclusions are:

- the gate is real and adaptive
- the fused MPS sparse backend is the correct sparse backend for this repo
- fused sparse projection is slightly better than dense-gated on average in recent local sweeps
- the gated variants still do not clearly dominate the ungated base model across all quality metrics

So the engineering side is in a healthier place than before, but the model-quality question is still open.

## What Is Still Not Implemented

This fork still does not include:

- sparse MLA latent-state math
- sparse KV cache compaction driven by gate indices
- sparse score/value accumulation
- FFN-path gating
- official giant-checkpoint support under the toy config

So the current sparse story is:

- selector sparsity: yes
- projection sparsity: yes
- full attention/KV sparsity: not yet

## How To Read The Code

If you want to inspect the implementation in order, read:

1. `inference/model.py`
2. `inference/dim_gate.py`
3. `inference/sparse_mla.py`
4. `inference/mla_native.py`
5. `inference/native/gate_scan_mps.mm`
6. `inference/native/sparse_projection_mps.mm`
7. `inference/train_gate_compare.py`

Focus first on:

- `Block.forward()` for the insertion point
- `HiddenDimensionGate.forward()` for gate metadata emission
- `MLA.forward()` for sparse projection consumption
- `sparse_project_hidden_pair()` for the current fused sparse fast path

## Practical Meaning Of The Current Fork

This repo can now do something the original inference tree could not do:

- run a toy DeepSeek-style architecture locally on Apple Silicon
- learn and evaluate a Kanae-style hidden-dimension gate
- compare dense-gated vs sparse-gated execution
- run the sparse path through native MPS kernels instead of Python bookkeeping

That gives you a usable local platform for the next real step:

- either improving gated-vs-base model quality
- or pushing stage 2B toward latent sparsity and KV-cache compaction
