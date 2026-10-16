/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.Architecture.Schedule
public import NN.Spec.Core.Shape
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega

/-!
# Shapes, projection partitions, and cache payloads

Table 1 (report p. 4) and the released BF16 config give the dimensions below. For the Mamba
input projection partition, see `NemotronHMamba2Mixer.__init__`, lines 384–421 of:
https://github.com/huggingface/transformers/blob/c8b25ab3af135bcfd9496eff3ef507303da17f15/src/transformers/models/nemotron_h/modeling_nemotron_h.py

Payload counts cover one main-model sequence, not all the memory a serving system uses. They
leave out convolution history, scales, allocator padding, batch size, MTP caches, and rollback or
prefix snapshots. §4.5 (pp. 44–46) compares FP32 SSM storage to FP8 KV storage. An FP32 cache
(the BF16 config's default) and the deployed quantized model's FP16 stochastic cache are
different operating points, not conflicting claims about one configuration.
-/

@[expose] public section

namespace Nemotron.Architecture

open Spec

/-- A recurrent layer stores one matrix per head, independent of sequence length. -/
def ssmShape (heads headDim stateDim : Nat) : Shape := [heads, headDim, stateDim]

/-- One attention layer stores keys and values for every preceding token. -/
def kvShape (tokens kvHeads headDim : Nat) : Shape := [2, kvHeads, tokens, headDim]

/-- SSM payload bytes for the recurrent blocks of an arbitrary hybrid schedule. -/
def ssmBytes (schedule : List BlockKind) (heads headDim stateDim bytesPerScalar : Nat) : Nat :=
  schedule.count .mamba * (ssmShape heads headDim stateDim).size * bytesPerScalar

/-- KV payload bytes for the attention blocks of an arbitrary hybrid schedule. -/
def kvBytes (schedule : List BlockKind) (tokens kvHeads headDim bytesPerScalar : Nat) : Nat :=
  schedule.count .attention * (kvShape tokens kvHeads headDim).size * bytesPerScalar

/-- Each head contributes one headDim-by-stateDim matrix. -/
theorem ssmShape_size (heads headDim stateDim : Nat) :
    (ssmShape heads headDim stateDim).size = heads * headDim * stateDim := by
  simp [ssmShape, Shape.size, Nat.mul_assoc]

/-- Keys and values give the leading factor of two. -/
theorem kvShape_size (tokens kvHeads headDim : Nat) :
    (kvShape tokens kvHeads headDim).size = 2 * kvHeads * tokens * headDim := by
  simp [kvShape, Shape.size, Nat.mul_assoc]

/-- KV bytes add over a split of the token count. -/
theorem kvBytes_add_tokens (schedule : List BlockKind)
    (tokens more kvHeads headDim bytesPerScalar : Nat) :
    kvBytes schedule (tokens + more) kvHeads headDim bytesPerScalar =
      kvBytes schedule tokens kvHeads headDim bytesPerScalar +
        kvBytes schedule more kvHeads headDim bytesPerScalar := by
  simp only [kvBytes, kvShape_size]
  ring

/-- SSM bytes add over concatenated stacks, whatever their layer order. -/
theorem ssmBytes_append (left right : List BlockKind)
    (heads headDim stateDim bytesPerScalar : Nat) :
    ssmBytes (left ++ right) heads headDim stateDim bytesPerScalar =
      ssmBytes left heads headDim stateDim bytesPerScalar +
        ssmBytes right heads headDim stateDim bytesPerScalar := by
  simp [ssmBytes, List.count_append, Nat.add_mul]

/-- Channels convolved together: expanded values, grouped B, and grouped C. -/
def mambaConvChannels (heads headDim groups stateDim : Nat) : Nat :=
  heads * headDim + 2 * groups * stateDim

/-- Input projection width: a gate, the convolved channels, and one time step per head. -/
def mambaInputWidth (heads headDim groups stateDim : Nat) : Nat :=
  heads * headDim + mambaConvChannels heads headDim groups stateDim + heads

/-- Splitting as the source does, `[gate, x, B, C, dt]`, leaves no remainder. -/
theorem mambaInputWidth_partition (heads headDim groups stateDim : Nat) :
    mambaInputWidth heads headDim groups stateDim =
      heads * headDim + heads * headDim + groups * stateDim + groups * stateDim + heads := by
  simp only [mambaInputWidth, mambaConvChannels]
  ring

/-- Routed-expert parameters: two matrices per non-gated expert, both on the latent dimension.
Router, shared expert, and latent projections are not counted. -/
def routedExpertParameters (experts latentDim innerDim : Nat) : Nat :=
  (Shape.size [experts, innerDim, latentDim]) + (Shape.size [experts, latentDim, innerDim])

/-- Each routed expert contributes an up projection and a down projection of equal size. -/
theorem routedExpertParameters_eq (experts latentDim innerDim : Nat) :
    routedExpertParameters experts latentDim innerDim = 2 * experts * latentDim * innerDim := by
  simp only [routedExpertParameters, Shape.size]
  ring

/-- A latent dimension of 2048 instead of 8192 makes the routed-expert matrices four times
smaller. Because of the added latent projections and the shared expert, neither the MoE layer
nor the model shrinks fourfold. -/
theorem released_routed_parameter_ratio :
    4 * routedExpertParameters 512 2048 5120 = routedExpertParameters 512 8192 5120 := by
  decide +kernel

/-- Released dimensions agree with the value expansion, B/C grouping, and attention projections. -/
theorem released_projection_dimensions :
    256 * 64 = 2 * 8192 ∧
    mambaConvChannels 256 64 8 128 = 18432 ∧
    mambaInputWidth 256 64 8 128 = 35072 ∧
    64 * 128 = 8192 ∧
    2 * 128 = 256 := by
  decide +kernel

/-- Exactly 384 MiB of FP32 recurrent payload and 6144 bytes of FP8 KV payload per token. -/
theorem released_cache_payloads (tokens : Nat) :
    ssmBytes releasedSchedule 256 64 128 4 = 402653184 ∧
    kvBytes releasedSchedule tokens 2 128 1 = 6144 * tokens := by
  have hc := releasedSchedule_counts
  simp [ssmBytes, kvBytes, ssmShape_size, kvShape_size, hc.1, hc.2.1]
  ring

/-- FP8 KV payload stays at or below the FP32 recurrent payload up to 64 Ki tokens, where the
two are equal. -/
theorem released_fp32_fp8_crossover (tokens : Nat) :
    kvBytes releasedSchedule tokens 2 128 1 ≤ ssmBytes releasedSchedule 256 64 128 4 ↔
      tokens ≤ 65536 := by
  rw [(released_cache_payloads tokens).1, (released_cache_payloads tokens).2]
  omega

/-- Moving the recurrent payload to FP16 halves its crossover to 32 Ki tokens. -/
theorem released_fp16_fp8_crossover (tokens : Nat) :
    kvBytes releasedSchedule tokens 2 128 1 ≤ ssmBytes releasedSchedule 256 64 128 2 ↔
      tokens ≤ 32768 := by
  rw [(released_cache_payloads tokens).2]
  have hc := releasedSchedule_counts.1
  norm_num [ssmBytes, ssmShape_size, hc]
  omega

end Nemotron.Architecture
