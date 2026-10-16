/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Model
public import Mathlib.Data.List.OfFn

/-!
# Sequence evaluation and streaming

Full-sequence evaluation runs each layer over all tokens before the next layer, while a streaming
decoder runs all layers on one token before reading the next. Both share parameters, routing,
and the layer transition, and we prove that both orders give the same vocabulary logits. Streaming
state holds each layer's KDA recurrence or MLA cache. Input and output sequences are TorchLean
tensors.

This equivalence concerns the public real-valued model. GraphSpec's lowered tensor-cache graph
still needs its own packing/refinement theorem, and neither result verifies an external GPU kernel.
-/

@[expose] public section

namespace KimiK3.LanguageModel

open Spec TorchLean

variable {cfg : TextConfig} {decayRank : Nat}

namespace Layers

/-- Causal states for a list of layers, one per layer (KDA and MLA states differ in shape). -/
def State (model : LanguageModel ℝ cfg decayRank) : List (Fin cfg.numLayers) → Type
  | [] => Unit
  | index :: rest => (model.layer index).sequence.CausalState × State model rest

/-- Empty causal state for a segment of decoder layers. -/
noncomputable def initialState (model : LanguageModel ℝ cfg decayRank) :
    (indices : List (Fin cfg.numLayers)) → State model indices
  | [] => ()
  | index :: rest => ((model.layer index).sequence.initialState, initialState model rest)

/-- Advance a list of layers by one token in depth order, updating each layer's causal state. -/
noncomputable def step (model : LanguageModel ℝ cfg decayRank) :
    (indices : List (Fin cfg.numLayers)) → State model indices →
      AttnRes.BlockState ℝ cfg.hiddenDim →
      State model indices × AttnRes.BlockState ℝ cfg.hiddenDim
  | [], _, depth => ((), depth)
  | index :: rest, state, depth =>
      let first := (model.layer index).forwardToken model.activeExperts_le state.1 depth
      let second := step model rest state.2 first.2
      ((first.1, second.1), second.2)

/-- Layer-wise sequence evaluation equals token-wise evaluation of the same layers. -/
theorem scan_eq_fold (model : LanguageModel ℝ cfg decayRank)
    (indices : List (Fin cfg.numLayers))
    (inputs : List (AttnRes.BlockState ℝ cfg.hiddenDim)) :
    (CausalScan.run (step model indices) (initialState model indices) inputs).2 =
      indices.foldl (fun states index =>
        (model.layer index).forward model.activeExperts_le states) inputs := by
  induction indices generalizing inputs with
  | nil =>
      induction inputs with
      | nil => rfl
      | cons input rest ih => simpa [step, initialState, CausalScan.run] using ih
  | cons index rest ih =>
      -- Exchange the token and layer traversals by composing their causal scans.
      change (CausalScan.run (fun state input =>
        let first := (model.layer index).forwardToken model.activeExperts_le state.1 input
        let second := step model rest state.2 first.2
        ((first.1, second.1), second.2))
        ((model.layer index).sequence.initialState, initialState model rest) inputs).2 = _
      rw [CausalScan.run_compose]
      simp only
      rw [ih, ← BackboneLayer.forward_eq_scan]
      rfl

end Layers

/-- Causal state of the complete decoder, with one state for every configured layer. -/
abbrev CausalState (model : LanguageModel ℝ cfg decayRank) :=
  Layers.State model (List.finRange cfg.numLayers)

/-- Start a decoder with empty KDA histories and MLA caches. -/
noncomputable def initialState (model : LanguageModel ℝ cfg decayRank) : model.CausalState :=
  Layers.initialState model (List.finRange cfg.numLayers)

/-- Process one embedding through every decoder layer and return the next state with the token's
vocabulary logits. Repeating the call on the returned state gives autoregressive decoding. -/
noncomputable def step (model : LanguageModel ℝ cfg decayRank) (state : model.CausalState)
    (embedding : Tensor ℝ [cfg.hiddenDim]) : model.CausalState × Tensor ℝ [cfg.vocabSize] :=
  let result := Layers.step model (List.finRange cfg.numLayers) state
    (BackboneLayer.initialDepthState embedding)
  let hidden := RMSNorm.scale (result.2.retrieve model.finalQuery) model.finalNormScale
  (result.1, vecMatMulSpec hidden model.vocabularyHead)

/-- Run a fixed-length embedding tensor and return the state needed by the next chunk. -/
noncomputable def run {length : Nat} (model : LanguageModel ℝ cfg decayRank)
    (state : model.CausalState) (embeddings : Tensor ℝ [length, cfg.hiddenDim]) :
    model.CausalState × Tensor ℝ [length, cfg.vocabSize] :=
  let result := CausalScan.run model.step state (List.ofFn fun index => Spec.get embeddings index)
  have hLength : result.2.length = length := by
    rw [CausalScan.outputs_length, List.length_ofFn]
  (result.1, Tensor.stackLeading fun index => result.2.get ⟨index.val, by rw [hLength]; exact index.isLt⟩)

/-- Streaming from the initial state reproduces the full-sequence logits. It holds for every
sequence length and every parameter value, with sparse layers taking the model's own
deterministic routes. -/
theorem run_eq_logits {length : Nat} (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ [length, cfg.hiddenDim]) :
    (model.run model.initialState embeddings).2 = model.logits embeddings := by
  let inputs := List.ofFn fun index => Spec.get embeddings index
  let final := (List.finRange cfg.numLayers).foldl
    (fun states index => (model.layer index).forward model.activeExperts_le states)
    (inputs.map BackboneLayer.initialDepthState)
  let finish := fun depth : AttnRes.BlockState ℝ cfg.hiddenDim =>
    vecMatMulSpec (RMSNorm.scale (depth.retrieve model.finalQuery) model.finalNormScale)
      model.vocabularyHead
  have hOutputs : (CausalScan.run model.step model.initialState inputs).2 = final.map finish := by
    -- Separate embedding preparation and logit projection from the shared decoder scan.
    change (CausalScan.run (fun state input =>
      let result := Layers.step model (List.finRange cfg.numLayers) state
        (BackboneLayer.initialDepthState input)
      (result.1, finish result.2))
      (Layers.initialState model (List.finRange cfg.numLayers)) inputs).2 = _
    rw [CausalScan.run_map_output]
    simp only
    exact congrArg (List.map finish)
      (Layers.scan_eq_fold model (List.finRange cfg.numLayers) _)
  simp only [run, logits, forwardHidden, Tensor.stackLeading, Spec.get_dim]
  congr 1
  funext index
  change (CausalScan.run model.step model.initialState inputs).2.get _ =
    finish (final.get _)
  simp only [hOutputs, List.get_eq_getElem, List.getElem_map]

private theorem concat_rows {α : Type} [Storage α] {leftLength rightLength : Nat}
    {shape : Shape} (left : Tensor α (shape.prependDim leftLength))
    (right : Tensor α (shape.prependDim rightLength)) :
    Tensor.concat left right =
      Tensor.stackLeading (Fin.append (fun index => Spec.get left index)
        (fun index => Spec.get right index)) := by
  apply Tensor.Internal.Rep.ext
  rintro ⟨index, coordinate⟩
  by_cases h : index.val < leftLength
  · simp [Tensor.concat, Tensor.concatAxisSpec, Tensor.Internal.Rep.concatenateAxis,
      Tensor.Internal.Rep.castShape, Tensor.stackLeading, Tensor.dim, Spec.get, Tensor.unstack,
      Fin.append, Fin.addCases, h, Tensor.Internal.Coord.appendEquiv]
    rfl
  · simp [Tensor.concat, Tensor.concatAxisSpec, Tensor.Internal.Rep.concatenateAxis,
      Tensor.Internal.Rep.castShape, Tensor.stackLeading, Tensor.dim, Spec.get, Tensor.unstack,
      Fin.append, Fin.addCases, h, Tensor.Internal.Coord.appendEquiv]
    rfl

/-- Chunking an embedding tensor preserves every logit and the complete final decoder state.
The second chunk must start from the state returned by the first. -/
theorem run_append {leftLength rightLength : Nat} (model : LanguageModel ℝ cfg decayRank)
    (state : model.CausalState) (left : Tensor ℝ [leftLength, cfg.hiddenDim])
    (right : Tensor ℝ [rightLength, cfg.hiddenDim]) :
    model.run state (Tensor.concat left right) =
      let first := model.run state left
      let second := model.run first.1 right
      (second.1, Tensor.concat first.2 second.2) := by
  let first := CausalScan.run model.step state (List.ofFn fun index => Spec.get left index)
  let second := CausalScan.run model.step first.1 (List.ofFn fun index => Spec.get right index)
  have hFirst : first.2.length = leftLength := by
    rw [CausalScan.outputs_length, List.length_ofFn]
  have hInputs : (List.ofFn fun index => Spec.get (Tensor.concat left right) index) =
      (List.ofFn fun index => Spec.get left index) ++
        (List.ofFn fun index => Spec.get right index) := by
    rw [concat_rows]
    simp only [Tensor.stackLeading, Spec.get_dim, List.ofFn_fin_append]
  have hScan : CausalScan.run model.step state
      (List.ofFn fun index => Spec.get (Tensor.concat left right) index) =
      (second.1, first.2 ++ second.2) := by
    rw [hInputs, CausalScan.run_append]
  dsimp only [run]
  apply Prod.ext
  · change (CausalScan.run model.step state
      (List.ofFn fun index => Spec.get (Tensor.concat left right) index)).1 =
        second.1
    have hState := congrArg Prod.fst hScan
    exact hState
  · dsimp only
    conv_rhs => rw [concat_rows]
    apply congrArg Tensor.stackLeading
    funext index
    simp only [List.get_eq_getElem, Tensor.stackLeading, Spec.get_dim]
    rw [List.getElem_of_eq (congrArg Prod.snd hScan)]
    change (first.2 ++ second.2).get _ = _
    by_cases h : index.val < leftLength
    · simp [Fin.append, Fin.addCases, h, List.get_eq_getElem, hFirst]
      rfl
    · simp [Fin.append, Fin.addCases, h, List.get_eq_getElem, List.getElem_append, hFirst]
      rfl

end KimiK3.LanguageModel
