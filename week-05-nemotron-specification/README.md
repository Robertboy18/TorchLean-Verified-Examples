# Nemotron architecture, numerical arithmetic, and recurrent state in Lean

This week turns the state-cache discussion in the [NVIDIA Nemotron 3 Ultra technical
report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf#page=44)
into exact replay laws and a numerical error contract. Ultra is a hybrid Mamba-2,
attention, and LatentMoE model, and Section 4.5 of the report proposes keeping the prepared activations and
replaying them from an SSM checkpoint that is quantized less often. (Its Table 16 uses emulated
experiments on Nemotron 3 Super, while the released Ultra cache uses FP16 with stochastic
rounding.) We also look at the gradient-accumulation failure in §2.7 and the speculative-state
rollback in §5.1.

There's an [article](site/index.html) that explains the proofs, with an interactive arithmetic
example you can play with, and if you want to build it (checked Lean statements, types, and proof
panels included), follow the [site preview instructions](../README.md#build-the-example-site).

## What the theorems establish

In the architecture modules I transcribed the released BF16 block schedule, which has 48 Mamba blocks,
48 MoE blocks, and 12 attention blocks. Each entry is one residual block (not a mixer plus an
implicit feed-forward block). You'll find the configuration source pinned in the module comment.

Cache accounting works from tensor shapes, not from a memory estimate taken from one run.
In the main stack the FP32 SSM payload is 384 MiB, and the FP8 KV payload adds 6144 bytes per token. So
how long does a sequence have to get before the KV cache catches up? `released_fp32_fp8_crossover`
proves that these payloads are equal at 65,536 tokens, with KV no larger precisely up to that
boundary, and storing the SSM state in FP16 moves it to 32,768 tokens. These are payload counts for
one sequence (no convolution history, scale metadata, padding, MTP, or additional retained
checkpoints). They aren't total serving memory.

`Cache.Replay.observe_push` proves that replaying one more prepared input gives the same updated
state and output as stepping the reconstructed state, and `run_from_replay` proves that the full
output sequence is equal across a replay boundary. And if state storage is exact,
`quantizedChunks_id` proves that the final state is unchanged for every partition into chunks.

`Cache.quantizedChunks_within_of_writes` gives a coordinatewise tensor error bound for a specified storage
schedule, assuming a transition sensitivity contract that bounds one step by `L` times the incoming
error and quantization that contributes at most `δ` per checkpoint. For chunk lengths `n₁, …, nₘ`, the budget
starts at `E₀` and follows

```text
Eⱼ₊₁ = L ^ nⱼ₊₁ * Eⱼ + δ.
```

`checkpointBudget_le_count` proves `Eₘ ≤ E₀ + m δ` when `0 ≤ L ≤ 1`, `δ ≥ 0`, and `E₀ ≥ 0`, so
the number of storage writes shows up right there as a factor. When `L > 1`, the full weighted
budget still applies, but the linear bound need not hold. (Both bounds compare runs on the same
prepared inputs.)

When you compare two states under one prepared input, the rank-one write cancels, and
`SSM.update_sub_apply` proves it. `SSM.update_within` then gets sensitivity from the supplied
retention bound, and `SSM.checkpoint_error` instantiates the generic checkpoint theorem using
`BoundedInput`, whose type carries that retention evidence, so the stability hypothesis becomes
something you can check for each input instead of an assertion about all arbitrary real
coefficients.

For a fixed read vector, `SSM.read_within` bounds the output error by the state error times
the sum of the vector's absolute coefficients, and `step_output_within` combines that bound with the
retention factor. So a small state error only becomes a small output error if the read vector is
under control. Neither theorem assumes that learned coefficients are bounded.

`Architecture.Mamba2.prepareHead` connects grouped coefficients to that existing core using the
source's negative-exponential rate, positive softplus timestep, and timestep-scaled write vector,
and from that parameterization `prepareHead_bounded` derives unit state sensitivity, which
`updateHeads_within` then lifts to all heads of the state tensor. Its input still starts after
projection and convolution, though, and the comparison keeps the prepared token data fixed.

`WritesWithin`, the storage premise, only checks the states that the quantized run actually writes.
Why does that matter? On a floating grid, no finite absolute error cap can hold for every unbounded
real input. `quantizedChunks_within` is still there as a convenience corollary for a globally
bounded quantizer, such as fixed-grid nearest-integer rounding.

`Cache.roundTensor_within` reuses FloatLib's format-indexed half-ULP theorem, subnormal
absolute-error treatment included. Two kernel-decidable rational examples then use FloatLib's
actual nearest-even integer rounder. In one, repeated ties erase updates. In the other, four
storage writes end up with smaller actual error than two, because the errors cancel! That refutes
a universal monotonic-error interpretation of checkpointing (the idea that more writes always mean
more error), though it doesn't refute the report's measurements.

`Cache.Stochastic.roundQuot_sum` proves that FloatLib's executable stochastic quotient rounder
is unbiased under uniform entropy residues. `roundTensor_mean` lifts that result to tensors
whose coordinates each see uniform residues, and `roundTensor_within` bounds every outcome by one
grid spacing. Each coordinate gets its own signed grid cell from the adapter, with its quantum and
denominator supplied explicitly. Before this result can describe the released FP16 encoder,
saturation, overflow, flush-to-zero, and generating uniform residues all need their own refinement
arguments. If coordinates share entropy (`roundTensor_mean_shared`), the local means are preserved
but the errors aren't independent. For independent residues there's
`roundTensor_mean_independent`.

`Cache.Stochastic.update_mean` proves that the prepared affine SSM update commutes with a finite
ensemble mean, and `SSM.scan_mean` composes mean preservation across the full sequence from the
actual ensemble update and mean-zero storage perturbations at every step. This needs the writes to be
conditionally unbiased at the realized states, and local encoder unbiasedness alone doesn't supply
that premise.

`Training.accumulate_within` reuses FloatLib's reduction-tree error budget for gradient tensors
of any shape, scalar carrier, and supplied addition order, and it bounds rounded accumulation
against the exact sum of interpreted leaves (errors made in producing the leaves are kept
separate). Two finite-format examples prove that BF16 rounds `256 + 1` to `256`, while binary32
returns `257`. That's a lost contribution in miniature. It isn't a measurement of NVIDIA's actual
gradients, and it doesn't explain the report's unresolved second divergence.

For rollback, `Cache.Speculation.restore_eq_prefix` proves that selecting a retained snapshot gives
the state of the accepted draft prefix, and `run_after_restore` preserves the continuing state and
all committed outputs. `SSM.Sequence.snapshots` exposes the same states as a
`[tokens + 1, valueDim, stateDim]` tensor (the initial state is there too, for zero accepted
tokens), and `snapshots_unstack` identifies every bounded row with its corresponding prefix state.
These are rollback laws. They don't prove anything about the draft acceptance rule's sampling
distribution.

## Source layout

| Module | Role |
| --- | --- |
| [`Nemotron/Cache/Replay.lean`](Nemotron/Cache/Replay.lean) | Prepared-input replay, exact boundary laws, chunk quantization schedule |
| [`Nemotron/Cache/Error.lean`](Nemotron/Cache/Error.lean) | Arbitrary-shape tensor coordinate error and weighted checkpoint budget |
| [`Nemotron/SSM.lean`](Nemotron/SSM.lean) | Parameterized single-head tensor recurrence, sensitivity proof, public packed sequence interface |
| [`Nemotron/Cache/Rounding.lean`](Nemotron/Cache/Rounding.lean) | FloatLib rounding adapter and concrete arithmetic counterexamples |
| [`Nemotron/Cache/Stochastic.lean`](Nemotron/Cache/Stochastic.lean) | Executable stochastic rounding choice, local tensor mean and error, affine mean composition |
| [`Nemotron/Cache/Speculation.lean`](Nemotron/Cache/Speculation.lean) | Stored-snapshot rollback, committed continuation, public tensor snapshots |
| [`Nemotron/Training/Accumulation.lean`](Nemotron/Training/Accumulation.lean) | Arbitrary-order gradient accumulation bounds and finite-format lost-contribution examples |
| [`Nemotron/Architecture/Schedule.lean`](Nemotron/Architecture/Schedule.lean) | Released residual-block ordering and checked counts |
| [`Nemotron/Architecture/Dimensions.lean`](Nemotron/Architecture/Dimensions.lean) | Projection partitions, expert-matrix parameter counts, and tensor-shape cache accounting |
| [`Nemotron/Architecture/Mamba2.lean`](Nemotron/Architecture/Mamba2.lean) | Grouped coefficient preparation, recurrent update and skip readout, derived retention bound |

There's no second scan implementation here, because the generic causal scan is reused from
`KimiK3.Sequence.Scan` in week 04. The public `SSM.Sequence.run` accepts prepared tensors and returns a state tensor
and a `[tokens, valueDim]` output tensor. Inside the shared scan and the checkpoint statements,
lists represent the chronological inputs, but that's an internal detail.

Build from the repository root with `lake build Nemotron`. If you only want to check one source
module at a time, a cached `lake env lean` invocation can do that too.

## Mathematical scope

The prepared core uses `S' = a S + x Bᵀ` and `y = S' C`, with `S` a
`[valueDim, stateDim]` TorchLean tensor. Section 4.5 of the Ultra report doesn't give the complete
selective SSM equations, so for discretization, contiguous group sharing, and the output skip the
grouped adapter follows the pinned model source. Its `timeStepMin` parameter covers both source
paths (the decode update has no time-step floor, while the chunked path clamps at
`time_step_min`). Projection, convolution, gating, grouped normalization, output projection,
attention, MoE execution, and learned weights all stay outside that adapter, whose dimensions are
parameters rather than small numbers standing in for the published model. And mathematical
softplus isn't identified with PyTorch's finite-precision threshold implementation.

Replay equality holds for any deterministic transition as long as the same arithmetic and inputs
are replayed, but a stochastic transition needs either its random choices kept as part of the replay
data or an additional distributional theorem. The numerical theorem assumes a supplied local
storage-error allowance at the actual writes, and the stochastic module proves a local grid mean
and conditional affine composition (no concentration result).
FloatLib's nearest-even real grid has no upper exponent limit, so connecting it to finite
executable values takes the library's finiteness refinement premises.

The error theorem compares recurrent state under fixed prepared inputs, and the readout theorem
bounds the prepared core's output using the same read vector. Neither establishes a logit,
token-selection, rollout, or benchmark-accuracy guarantee. If you compare a rounded recurrence with
exact-real arithmetic, you also have to account for the arithmetic errors inside each replayed
transition. The article names these obligations and keeps them apart from the mistakes it
actually demonstrates.
