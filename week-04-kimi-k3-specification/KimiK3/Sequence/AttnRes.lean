/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import KimiK3.Sequence.Scan

/-!
# Block Attention Residuals

Retrieve a weighted mixture of earlier depth representations. RMS normalization enters only the
scores, and the weighted values are the original representations. Each depth state keeps
completed block summaries and the current partial block, with the embedding as an initial
source. Both growing-list and fixed-shape retrieval functions are defined below.

Reference: Kimi K3 technical report, Sections 2.1--2.2, pp. 4--6, Eqs. 1--10.
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

namespace AttnRes

variable {α : Type} [Storage α] [Context α]
variable {modelDim : Nat}

/-- Positive softmax kernel `exp(qᵀ RMSNorm(k))` from Eq. 9. -/
def kernel (query key : Tensor α [modelDim]) : α :=
  MathFunctions.exp (Tensor.dotSpec query (RMSNorm.unit key))

/-- Sum of the exponential scores that normalize the depth-attention weights. -/
def normalizer (query : Tensor α [modelDim])
    (sources : List (Tensor α [modelDim])) : α :=
  sources.foldl (fun total source => total + kernel query source) 0

/-- Depth-attention normalizers are positive whenever at least one source is present, so
real-valued AttnRes retrieval never divides by zero. -/
theorem normalizer_pos_of_ne_nil (query : Tensor ℝ [modelDim])
    (sources : List (Tensor ℝ [modelDim])) (hSources : sources ≠ []) :
    0 < normalizer query sources := by
  cases sources with
  | nil => exact (hSources rfl).elim
  | cons first rest =>
      simp only [normalizer, List.foldl_cons]
      clear hSources
      have hFirst : 0 < (0 : ℝ) + kernel query first := by
        change 0 < 0 + Real.exp _
        positivity
      generalize (0 : ℝ) + kernel query first = accumulator at hFirst ⊢
      induction rest generalizing accumulator with
      | nil => simpa using hFirst
      | cons source rest ih =>
          simp only [List.foldl_cons]
          apply ih
          exact add_pos_of_pos_of_nonneg hFirst (Real.exp_pos _).le

/--
Attention over depth representations. Callers always include the embedding, so `sources` is
nonempty.
-/
def attend (query : Tensor α [modelDim])
    (sources : List (Tensor α [modelDim])) : Tensor α [modelDim] :=
  let denominator := normalizer query sources
  sources.foldl (fun output source =>
    let weight := kernel query source / denominator
    output + Tensor.map (fun value => weight * value) source)
    (Tensor.zeros [modelDim])

/-- Fixed-shape form of depth attention used at graph and backend boundaries.

Rows of `values` are the visible depth representations. They are RMS-normalized, scored against
the query, exponentiated, and normalized by their total mass, and the resulting weights multiply
the unnormalized sources in a vector--matrix product, as in Eq. 9--10.
-/
def attendPacked {sources : Nat} (hModel : 0 < modelDim)
    (query : Tensor α [modelDim])
    (values : Tensor α [sources, modelDim]) :
    Tensor α [modelDim] :=
  let unitScale := Tensor.ones [modelDim]
  let normalized := Tensor.stackLeading fun row =>
    RMSNorm.scalePositive hModel (Spec.get values row) unitScale
  let scores := vecMatMulSpec query (Tensor.swapAdjacentAxes normalized 0)
  let kernels := Tensor.expSpec scores
  let denominator := Tensor.sumSpec kernels
  let weights := Tensor.map (fun weight => weight / denominator) kernels
  vecMatMulSpec weights values

/-- State retained by Block AttnRes while traversing layers. -/
structure BlockState (α : Type) [Storage α] (modelDim : Nat) where
  /-- Token embedding, always visible to depth attention. -/
  embedding : Tensor α [modelDim]
  /-- Sums of the completed AttnRes blocks. -/
  completedBlocks : List (Tensor α [modelDim])
  /-- Sum accumulated in the current, incomplete block. -/
  partialBlock : Tensor α [modelDim]
  /-- Number of decoder layers represented by `partialBlock`. -/
  partialSize : Nat

namespace BlockState

/-- A valid partial AttnRes block has not yet reached its block boundary. -/
def WF (state : BlockState α modelDim) (blockSize : Nat) : Prop :=
  state.partialSize < blockSize

/-- Sources in Eq. 10: embedding, completed block sums, and an optional current partial sum. -/
def sources (state : BlockState α modelDim) : List (Tensor α [modelDim]) :=
  state.embedding :: state.completedBlocks ++
    if state.partialSize = 0 then [] else [state.partialBlock]

omit [Context α] in
/-- Because the embedding is always included, every AttnRes source list is nonempty. -/
theorem sources_ne_nil (state : BlockState α modelDim) : state.sources ≠ [] := by
  simp [sources]

omit [Context α] in
/-- Source count is the embedding, completed blocks, and an optional partial block. -/
theorem sources_length (state : BlockState α modelDim) :
    state.sources.length =
      1 + state.completedBlocks.length + if state.partialSize = 0 then 0 else 1 := by
  by_cases hPartial : state.partialSize = 0 <;> simp [sources, hPartial]
    <;> omega

/-- Outputs of the AttnRes blocks so far. A partial block that has at least one layer counts as
the current last block. Token embeddings are not block outputs. -/
def blockOutputs (state : BlockState α modelDim) : List (Tensor α [modelDim]) :=
  state.completedBlocks ++ if state.partialSize = 0 then [] else [state.partialBlock]

omit [Context α] in
/-- There is one visible block output per completed block, plus one for the current partial block
when at least one layer has contributed to it. -/
theorem blockOutputs_length (state : BlockState α modelDim) :
    state.blockOutputs.length =
      state.completedBlocks.length + if state.partialSize = 0 then 0 else 1 := by
  by_cases h : state.partialSize = 0 <;> simp [blockOutputs, h]

/-- Every depth state has a positive real-valued attention normalizer, because the embedding is
always a source. -/
theorem normalizer_pos (state : BlockState ℝ modelDim)
    (query : Tensor ℝ [modelDim]) :
    0 < AttnRes.normalizer query state.sources := by
  apply normalizer_pos_of_ne_nil
  simp [sources]

/-- Retrieve the input to a layer using its learned depth pseudo-query. -/
def retrieve (state : BlockState α modelDim) (query : Tensor α [modelDim]) :
    Tensor α [modelDim] :=
  attend query state.sources

/--
Depth representations visible after the sequence mixer but before the channel mixer. Adding the
current sequence output to the partial block does not advance the decoder-layer counter.
-/
def sourcesAfterSequence (state : BlockState α modelDim)
    (sequenceOutput : Tensor α [modelDim]) :
    List (Tensor α [modelDim]) :=
  state.embedding :: state.completedBlocks ++ [state.partialBlock + sequenceOutput]

/-- Retrieve the channel-mixer input after adding the current layer's sequence output. -/
def retrieveAfterSequence (state : BlockState α modelDim)
    (sequenceOutput query : Tensor α [modelDim]) :
    Tensor α [modelDim] :=
  attend query (state.sourcesAfterSequence sequenceOutput)

/--
Commit one complete decoder layer. Sequence-mixer and channel outputs belong to the same AttnRes
block, so `partialSize` advances once after both submodules have run.
-/
def finishLayer (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α [modelDim]) : BlockState α modelDim :=
  let partialSum := state.partialBlock + sequenceOutput + channelOutput
  if state.partialSize + 1 = blockSize then
    { state with
      completedBlocks := state.completedBlocks ++ [partialSum]
      partialBlock := Tensor.zeros [modelDim]
      partialSize := 0 }
  else
    { state with partialBlock := partialSum, partialSize := state.partialSize + 1 }

/-- At a block boundary, the accumulated sum is committed and the partial block is reset. -/
theorem finishLayer_of_boundary (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α [modelDim])
    (hBoundary : state.partialSize + 1 = blockSize) :
    state.finishLayer blockSize sequenceOutput channelOutput =
      { state with
        completedBlocks := state.completedBlocks ++
          [state.partialBlock + sequenceOutput + channelOutput]
        partialBlock := Tensor.zeros [modelDim]
        partialSize := 0 } := by
  simp [finishLayer, hBoundary]

/-- Away from a boundary, the layer output is accumulated without committing a block. -/
theorem finishLayer_of_not_boundary (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α [modelDim])
    (hBoundary : state.partialSize + 1 ≠ blockSize) :
    state.finishLayer blockSize sequenceOutput channelOutput =
      { state with
        partialBlock := state.partialBlock + sequenceOutput + channelOutput
        partialSize := state.partialSize + 1 } := by
  simp [finishLayer, hBoundary]

/-- Finishing a layer advances the block position by one, wrapping to zero at the boundary. -/
private theorem finishLayer_partialSize (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α [modelDim]) :
    (state.finishLayer blockSize sequenceOutput channelOutput).partialSize =
      if state.partialSize + 1 = blockSize then 0 else state.partialSize + 1 := by
  by_cases hFull : state.partialSize + 1 = blockSize
  · simp [finishLayer, hFull]
  · simp [finishLayer, hFull]

/-- Committing a decoder layer preserves the block-position invariant, so a state never grows past
its AttnRes block size. -/
theorem finishLayer_wf (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α [modelDim])
    (hState : state.WF blockSize) :
    (state.finishLayer blockSize sequenceOutput channelOutput).WF blockSize := by
  rw [WF] at hState
  rw [WF, finishLayer_partialSize]
  split
  · omega
  · omega

end BlockState

end AttnRes

end KimiK3
