/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Decoder.Backbone

/-!
# Kimi K3 language-model graph

This file assembles the individually verified decoder graphs into a typed graph for one complete
autoregressive token step. It reuses the parameters and component equations of
`KimiK3.LanguageModel` and `KimiK3.BackboneLayer`, and represents causal state as fixed-shape
tensor packs suitable for GraphSpec.

The theorems here connect the graph to the packed semantics defined in this file. There is no
representation bridge yet to the list-based caches of `SequenceMixer.CausalState`, nor an
equivalence with the full-sequence `LanguageModel.logits`. Those are separate refinement
obligations, not consequences of shape checking.

Underneath, `DecoderLayer` gives every concrete backbone layer one dependent graph ABI, with
parameter and state shapes computed from the `SequenceMixer` and `ChannelMixer` constructors.
Pattern matching selects one of the four refinement theorems in the `GraphSpec.Backbone` modules.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean NN.GraphSpec.DAG

namespace TokenStep

open Spec TorchLean
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

variable {architecture : Config} {decayRank : Nat}

/-- Parameters used by one complete autoregressive token step.

The embedding table comes first, followed by every decoder-layer parameter and the three learned
objects used after the decoder: the final AttnRes query, RMSNorm scale, and vocabulary projection.
-/
abbrev Params (model : LanguageModel ℝ architecture.text decayRank) : List Shape :=
  [architecture.text.vocabSize, architecture.text.hiddenDim] ::
    (Decoder.ParamsFor model (List.finRange architecture.text.numLayers) ++
      [[architecture.text.hiddenDim],
        [architecture.text.hiddenDim],
        [architecture.text.hiddenDim, architecture.text.vocabSize]])

/-- Runtime inputs used by one complete autoregressive token step. -/
abbrev Inputs (pastTokens : Nat)
    (model : LanguageModel ℝ architecture.text decayRank) : List Shape :=
  Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers) ++
    (Decoder.ControlShapesFor model (List.finRange architecture.text.numLayers) ++
      [[], [], []])

/-- Updated causal states and the current token's vocabulary logits. -/
abbrev Outputs (pastTokens : Nat)
    (model : LanguageModel ℝ architecture.text decayRank) : List Shape :=
  Decoder.NextStateShapesFor pastTokens model (List.finRange architecture.text.numLayers) ++
    [[architecture.text.vocabSize]]

/-- Learned real-valued tensors in the `Params` order. -/
def parameters (model : LanguageModel ℝ architecture.text decayRank) :
    TorchLean.TensorPack ℝ (Params model) := by
  let decoder := Decoder.parametersFor model (List.finRange architecture.text.numLayers)
  let final : TorchLean.TensorPack ℝ
      [[architecture.text.hiddenDim],
        [architecture.text.hiddenDim],
        [architecture.text.hiddenDim, architecture.text.vocabSize]] :=
    .cons model.finalQuery <| .cons model.finalNormScale <| .cons model.vocabularyHead .nil
  exact .cons model.tokenEmbedding <|
    TorchLean.TensorPack.append
      (ss₁ := Decoder.ParamsFor model (List.finRange architecture.text.numLayers))
      (ss₂ :=
        [[architecture.text.hiddenDim],
          [architecture.text.hiddenDim],
          [architecture.text.hiddenDim, architecture.text.vocabSize]])
      decoder final

/-- Runtime values in the `Inputs` order. -/
noncomputable def inputs (pastTokens : Nat) (model : LanguageModel ℝ architecture.text decayRank)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : TorchLean.TensorPack ℝ (Inputs pastTokens model) :=
  TorchLean.TensorPack.append states <|
    TorchLean.TensorPack.append
      (Decoder.controlsFor model mlaScoreScale (List.finRange architecture.text.numLayers)) <|
        .cons (Tensor.scalar Normalize.l2Epsilon) <| .cons (Tensor.scalar architecture.text.situGateCap) <|
          .cons (Tensor.scalar architecture.text.situUpCap) .nil

/-- Pack a token embedding into the initial AttnRes state. -/
def initialDepth (embedding : Tensor ℝ [architecture.text.hiddenDim]) :
    TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize 0 architecture.text.hiddenDim) :=
  .cons (Tensor.reshapeSpec embedding (by
      simp [Shape.size, DepthSchedule.completedCount])) <|
    .cons (Tensor.zeros [architecture.text.hiddenDim]) .nil

/-- Graph terms for the initial AttnRes state of one token. -/
def initialDepthTerms {Γ : List Shape}
    (embedding : Term Γ [architecture.text.hiddenDim]) :
    Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize 0 architecture.text.hiddenDim) :=
  .cons
    (Term.op
      (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by
        simp [Shape.size, DepthSchedule.completedCount]))
      (.cons embedding .nil)) <|
    .cons (Term.op
      (NN.GraphSpec.DAG.PrimOp.zero [architecture.text.hiddenDim]) .nil) .nil

/-- Evaluating the initial-depth terms gives the mathematical initial AttnRes state. -/
theorem eval_initialDepthTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (embedding : Term Γ [architecture.text.hiddenDim]) :
    Term.evalArgs env (initialDepthTerms (architecture := architecture) embedding) =
      initialDepth (Term.eval env embedding) := by
  simp only [initialDepthTerms, initialDepth, Term.eval, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd, NN.GraphSpec.DAG.PrimOp.zero_specFwd]

/-- Number of depth representations visible to the final AttnRes query. -/
def finalSourceCount : Nat :=
  DepthSchedule.completedCount architecture.text.attnResBlockSize architecture.text.numLayers +
    if architecture.text.numLayers % architecture.text.attnResBlockSize = 0 then 0 else 1

/-- The final AttnRes query always sees at least the token embedding. -/
theorem finalSourceCount_pos : 0 < finalSourceCount (architecture := architecture) := by
  simp [finalSourceCount, DepthSchedule.completedCount]

/-- Convert the final packed depth state into the matrix consumed by AttnRes.

Completed depth blocks are already rows of the first tensor. If the final block is incomplete, its
partial sum is appended as one additional row.
-/
def finalSources
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
        architecture.text.hiddenDim)) :
    Tensor ℝ [finalSourceCount (architecture := architecture), architecture.text.hiddenDim] :=
  let completed := Env.tget depth Var.head
  let partialState := Env.tget depth (.tail .head)
  if hBoundary :
      architecture.text.numLayers % architecture.text.attnResBlockSize = 0 then
    Tensor.castShape completed (by simp [finalSourceCount, hBoundary])
  else
    let row : Tensor ℝ [1, architecture.text.hiddenDim] :=
      Tensor.reshapeSpec partialState (by simp [Shape.size])
    Tensor.castShape (Tensor.concat completed row)
      (by simp [finalSourceCount, hBoundary])

/-- Graph term for the depth rows visible to the final AttnRes query. -/
def finalSourcesTerm {Γ : List Shape}
    (depth : Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
        architecture.text.hiddenDim)) :
    Term Γ [finalSourceCount (architecture := architecture), architecture.text.hiddenDim] :=
  let completed := Args.get depth Var.head
  let partialState := Args.get depth (.tail .head)
  if hBoundary :
      architecture.text.numLayers % architecture.text.attnResBlockSize = 0 then
    Term.cast completed (by simp [finalSourceCount, hBoundary])
  else
    let row : Term Γ [1, architecture.text.hiddenDim] :=
      Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
        (.cons partialState .nil)
    let appended : Term Γ
        [DepthSchedule.completedCount architecture.text.attnResBlockSize architecture.text.numLayers + 1, architecture.text.hiddenDim] :=
      GraphSpec.concatLeadingTerm
        (DepthSchedule.completedCount architecture.text.attnResBlockSize
          architecture.text.numLayers) 1 [architecture.text.hiddenDim] completed row
    Term.cast appended (by simp [finalSourceCount, hBoundary])

/-- Evaluating `finalSourcesTerm` gives `finalSources`. -/
theorem eval_finalSourcesTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (depth : Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
        architecture.text.hiddenDim)) :
    Term.eval env (finalSourcesTerm (architecture := architecture) depth) =
      finalSources (Term.evalArgs env depth) := by
  by_cases hBoundary :
      architecture.text.numLayers % architecture.text.attnResBlockSize = 0
  · have hCompleted := Term.eval_get env depth (Var.head)
    simp only [finalSourcesTerm, finalSources, hBoundary, dite_true, Term.eval_cast]
    rw [hCompleted, Tensor.eqRec_eq_cast_shape]
  · have hCompleted := Term.eval_get env depth (Var.head)
    have hPartial := Term.eval_get env depth (.tail .head)
    simp only [finalSourcesTerm, finalSources, hBoundary, dite_false, Term.eval_cast]
    rw [GraphSpec.eval_concatLeadingTerm]
    simp only [Term.eval_op, Term.evalArgs, NN.GraphSpec.DAG.PrimOp.reshape_specFwd]
    rw [hCompleted, hPartial]
    rw [Tensor.eqRec_eq_cast_shape]
    rfl

private theorem processedAfter_full :
    Decoder.processedAfter 0 (List.finRange architecture.text.numLayers) =
      architecture.text.numLayers := by
  rw [Decoder.processedAfter_eq_add_length]
  simp

/-- A full decoder run returns every updated layer cache, then its final depth state. -/
theorem outputsFor_full (pastTokens : Nat)
    (model : LanguageModel ℝ architecture.text decayRank) :
    Decoder.OutputsFor pastTokens 0 model (List.finRange architecture.text.numLayers) =
      Decoder.NextStateShapesFor pastTokens model (List.finRange architecture.text.numLayers) ++
        DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
          architecture.text.hiddenDim := by
  simp only [Decoder.OutputsFor]
  rw [processedAfter_full]

/-- Apply K3's final AttnRes retrieval, normalization, and vocabulary projection to a decoder
run. -/
noncomputable def finishDecoder (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (pastTokens : Nat)
    (decoderResult : TorchLean.TensorPack ℝ
      (Decoder.OutputsFor pastTokens 0 model (List.finRange architecture.text.numLayers))) :
    TorchLean.TensorPack ℝ (Outputs pastTokens model) :=
  let normalizedResult : TorchLean.TensorPack ℝ
      (Decoder.NextStateShapesFor pastTokens model
          (List.finRange architecture.text.numLayers) ++
        DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
          architecture.text.hiddenDim) :=
    outputsFor_full pastTokens model ▸ decoderResult
  let parts := TorchLean.TensorPack.split normalizedResult
  let sources := finalSources parts.2
  let retrieved := AttnRes.attendPacked hcfg.hiddenDim_pos
    model.finalQuery sources
  let hidden := RMSNorm.scale retrieved model.finalNormScale
  let logits := vecMatMulSpec hidden model.vocabularyHead
  TorchLean.TensorPack.append parts.1 (.cons logits .nil)

/-- Semantics of one autoregressive token step under a supplied route trace. -/
noncomputable def runSpec (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : TorchLean.TensorPack ℝ (Outputs pastTokens model) :=
  let embedding := Spec.get model.tokenEmbedding token
  let decoderResult := Decoder.runSpec hcfg model route pastTokens 0
    (List.finRange architecture.text.numLayers) (initialDepth embedding) states mlaScoreScale
  finishDecoder hcfg model pastTokens decoderResult

/-- Complete token-step semantics with sparse routes computed at the layers where they are used. -/
noncomputable def runSpecAuto (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : TorchLean.TensorPack ℝ (Outputs pastTokens model) :=
  let embedding := Spec.get model.tokenEmbedding token
  let decoderResult := Decoder.runSpecAuto hcfg model pastTokens 0
    (List.finRange architecture.text.numLayers) (initialDepth embedding) states mlaScoreScale
  finishDecoder hcfg model pastTokens decoderResult

/-- Route-trace condition for the complete token graph at a concrete token and causal state. -/
noncomputable def RoutesAgree (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : Prop :=
  Decoder.RoutesAgree hcfg model route pastTokens 0
    (List.finRange architecture.text.numLayers)
    (initialDepth (Spec.get model.tokenEmbedding token)) states mlaScoreScale

/-- A route trace that passes `RoutesAgree` gives the automatically routed token step. -/
theorem runSpec_eq_runSpecAuto_of_routesAgree (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ)
    (hRoutes : RoutesAgree hcfg model route pastTokens token states mlaScoreScale) :
    runSpec hcfg model route pastTokens token states mlaScoreScale =
      runSpecAuto hcfg model pastTokens token states mlaScoreScale := by
  apply congrArg (finishDecoder hcfg model pastTokens)
  exact Decoder.runSpec_eq_runSpecAuto_of_routesAgree hcfg model route pastTokens 0
    (List.finRange architecture.text.numLayers)
    (initialDepth (Spec.get model.tokenEmbedding token)) states mlaScoreScale hRoutes

/-- Default tensors of the right shapes for the token-step graph.

They are for graph construction and compilation. A checkpoint supplies the learned values through
the same `Params` ABI.
-/
def initialParams (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) : TorchLean.TensorPack Float (Params languageModel) :=
  .cons (Tensor.zeros [architecture.text.vocabSize, architecture.text.hiddenDim]) <|
    TorchLean.TensorPack.append
      (ss₁ := Decoder.ParamsFor languageModel (List.finRange architecture.text.numLayers))
      (ss₂ :=
        [[architecture.text.hiddenDim],
          [architecture.text.hiddenDim],
          [architecture.text.hiddenDim, architecture.text.vocabSize]])
      (Decoder.fullModel hcfg languageModel route pastTokens).initParams <|
        .cons (Tensor.zeros [architecture.text.hiddenDim]) <|
          .cons (Tensor.zeros [architecture.text.hiddenDim]) <|
            .cons (Tensor.zeros [architecture.text.hiddenDim, architecture.text.vocabSize]) .nil

/-- Extract the token-embedding table from the token-step parameter terms. -/
def embeddingTableTerm {Γ : List Shape}
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Term Γ [architecture.text.vocabSize, architecture.text.hiddenDim] :=
  Args.get params Var.head

/-- Extract the per-layer decoder parameters from the token-step parameter terms. -/
def decoderParameterTerms {Γ : List Shape}
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Args Γ (Decoder.ParamsFor languageModel (List.finRange architecture.text.numLayers)) :=
  match params with
  | .cons _ rest => (splitArgs rest).1

/-- Extract the final-source, normalization, and vocabulary-head parameters. -/
def finalParameterTerms {Γ : List Shape}
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) : Args Γ
      [[architecture.text.hiddenDim],
        [architecture.text.hiddenDim],
        [architecture.text.hiddenDim, architecture.text.vocabSize]] :=
  match params with
  | .cons _ rest => (splitArgs rest).2

/-- Assemble the decoder inputs from the embedding, cached states, and shared controls. -/
def decoderInputTerms {Γ : List Shape} (pastTokens : Nat)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (embedding : Term Γ [architecture.text.hiddenDim])
    (allInputs : Args Γ (Inputs pastTokens languageModel)) :
    Args Γ (Decoder.InputsFor pastTokens 0 languageModel
      (List.finRange architecture.text.numLayers)) :=
  let stateAndRest := splitArgs
    (left := Decoder.StateShapesFor pastTokens languageModel
      (List.finRange architecture.text.numLayers))
    (right :=
      Decoder.ControlShapesFor languageModel (List.finRange architecture.text.numLayers) ++
        [[], [], []])
    allInputs
  let controlAndShared := splitArgs
    (left := Decoder.ControlShapesFor languageModel
      (List.finRange architecture.text.numLayers))
    (right := [[], [], []]) stateAndRest.2
  Args.append (initialDepthTerms embedding) <|
    Args.append stateAndRest.1 <| Args.append controlAndShared.1 controlAndShared.2

theorem eval_decoderParameterTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Term.evalArgs env (decoderParameterTerms languageModel params) =
      (TorchLean.TensorPack.split
        (match Term.evalArgs env params with
        | .cons _ rest => rest)).1 := by
  cases params with
  | cons table rest =>
      exact eval_splitArgs_left env rest

theorem eval_finalParameterTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Term.evalArgs env (finalParameterTerms languageModel params) =
      (TorchLean.TensorPack.split
        (match Term.evalArgs env params with
        | .cons _ rest => rest)).2 := by
  cases params with
  | cons table rest =>
      exact eval_splitArgs_right env rest

theorem eval_decoderInputTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (pastTokens : Nat) (languageModel : LanguageModel ℝ architecture.text decayRank)
    (embedding : Term Γ [architecture.text.hiddenDim])
    (allInputs : Args Γ (Inputs pastTokens languageModel)) :
    Term.evalArgs env (decoderInputTerms pastTokens languageModel embedding allInputs) =
      let stateAndRest := TorchLean.TensorPack.split
        (Term.evalArgs env allInputs)
      let controlAndShared := TorchLean.TensorPack.split stateAndRest.2
      TorchLean.TensorPack.append (initialDepth (Term.eval env embedding)) <|
        TorchLean.TensorPack.append stateAndRest.1 <|
          TorchLean.TensorPack.append controlAndShared.1 controlAndShared.2 := by
  simp only [decoderInputTerms, Term.evalArgs_append, eval_initialDepthTerms,
    eval_splitArgs_left, eval_splitArgs_right]

/-- Complete typed DAG for one autoregressive token step, from embedding lookup to logits. -/
def graph (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize) :
    NN.GraphSpec.DAG.MultiModel (Params languageModel) (Inputs pastTokens languageModel)
    (Outputs pastTokens languageModel) :=
  { initParams := initialParams hcfg languageModel route pastTokens
    body := by
      let all := Args.vars (Params languageModel ++ Inputs pastTokens languageModel)
      let parameterAndInput := splitArgs
        (left := Params languageModel) (right := Inputs pastTokens languageModel) all
      let decoderParameters := decoderParameterTerms languageModel parameterAndInput.1
      let finalParameters := finalParameterTerms languageModel parameterAndInput.1
      let embedding := StableLatentMoE.selectLeadingTerm architecture.text.vocabSize
        [architecture.text.hiddenDim] token
        (embeddingTableTerm languageModel parameterAndInput.1)
      let decoderInputs := decoderInputTerms pastTokens languageModel embedding parameterAndInput.2
      let decoder := Decoder.fullModel hcfg languageModel route pastTokens
      let decoderBlock := outputsFor_full pastTokens languageModel ▸
        decoder.inline decoderParameters decoderInputs
      exact decoderBlock.andThen <|
        let decoderOutputs :=
          Decoder.NextStateShapesFor pastTokens languageModel
              (List.finRange architecture.text.numLayers) ++
            DepthSchedule.Shapes architecture.text.attnResBlockSize
              architecture.text.numLayers architecture.text.hiddenDim
        let extended := Params languageModel ++ Inputs pastTokens languageModel
        let decoderResults : Args (extended ++ decoderOutputs) decoderOutputs :=
          Args.rename (Var.inRight extended) (Args.vars decoderOutputs)
        let resultParts := splitArgs decoderResults
        let liftOriginal := fun {shape : Shape} (term : Term extended shape) =>
          Term.weakenAppend decoderOutputs term
        let finalQuery := liftOriginal (Args.get finalParameters Var.head)
        let finalNormScale := liftOriginal (Args.get finalParameters (.tail .head))
        let vocabularyHead := liftOriginal
          (Args.get finalParameters (.tail (.tail .head)))
        let sources := finalSourcesTerm resultParts.2
        let retrieved := AttnRes.term finalSourceCount architecture.text.hiddenDim
          hcfg.hiddenDim_pos finalQuery sources
        let hidden := Term.op
          (NN.GraphSpec.DAG.PrimOp.rmsNorm [] architecture.text.hiddenDim
            hcfg.hiddenDim_pos)
          (.cons retrieved (.cons finalNormScale .nil))
        let logits := Term.op
          (PrimOp.broadcastVecMat [] [] []
            architecture.text.hiddenDim architecture.text.vocabSize
            (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
          (.cons hidden (.cons vocabularyHead .nil))
        Block.ret (Args.append resultParts.1 (.cons logits .nil)) }

/-- `graph` computes `runSpec` for the route trace it was built with. -/
theorem graph_specFwd_eq_runSpec (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens languageModel
        (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) :
    (graph hcfg languageModel route pastTokens token).specFwd
        (parameters languageModel) (inputs pastTokens languageModel states mlaScoreScale) =
      runSpec hcfg languageModel route pastTokens token states mlaScoreScale := by
  simp only [NN.GraphSpec.DAG.MultiModel.specFwd, graph, Params, Inputs, Outputs,
    parameters, inputs, Decoder.InputsFor]
  rw [Block.eval_andThen]
  rw [Block.eval_cast]
  rw [NN.GraphSpec.DAG.MultiModel.eval_inline]
  rw [eval_decoderParameterTerms, eval_decoderInputTerms]
  simp only [eval_splitArgs_left, eval_splitArgs_right,
    Term.evalArgs_vars, TorchLean.TensorPack.split_append,
    Term.eval_get, Env.tget, embeddingTableTerm,
    StableLatentMoE.eval_selectLeadingTerm]
  have hDecoder :
      (Decoder.fullModel hcfg languageModel route pastTokens).specFwd
          (Decoder.parametersFor languageModel (List.finRange architecture.text.numLayers))
          (TorchLean.TensorPack.append
            (initialDepth (Spec.get languageModel.tokenEmbedding token)) <|
              TorchLean.TensorPack.append states <|
                TorchLean.TensorPack.append
                  (Decoder.controlsFor languageModel mlaScoreScale
                    (List.finRange architecture.text.numLayers)) <|
                    .cons (Tensor.scalar Normalize.l2Epsilon) <|
                      .cons (Tensor.scalar architecture.text.situGateCap) <|
                        .cons (Tensor.scalar architecture.text.situUpCap) .nil) =
        Decoder.runSpec hcfg languageModel route pastTokens 0
          (List.finRange architecture.text.numLayers)
          (initialDepth (Spec.get languageModel.tokenEmbedding token)) states mlaScoreScale := by
    exact Decoder.model_specFwd_eq_runSpec hcfg languageModel route pastTokens 0
      (List.finRange architecture.text.numLayers)
      (initialDepth (Spec.get languageModel.tokenEmbedding token)) states mlaScoreScale
  rw [hDecoder]
  simp only [Block.eval, Term.evalArgs_append, Term.evalArgs,
    Term.evalArgs_rename_inRight, Term.evalArgs_vars, Term.eval_weakenAppend,
    eval_finalParameterTerms, eval_splitArgs_left,
    eval_splitArgs_right, TorchLean.TensorPack.split_append,
    Term.eval, Term.eval_get, Env.tget, eval_finalSourcesTerm, AttnRes.eval_term,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd,
    PrimOp.broadcastVecMat]
  rw [runSpec]
  rw [GraphSpec.rmsNormSpec_scalar_eq_scale]
  simp [finishDecoder, NN.GraphSpec.DAG.PrimOp.Internal.vecMatCommonBatchSpec, Tensor.zipEach]

/-- For a route trace that passes `RoutesAgree`, the token DAG computes K3's automatically routed
semantics `runSpecAuto`.

Top-k is not a node in the graph, so `hRoutes` supplies the link: every sparse-layer route equals
the route computed from that layer's actual hidden representation.
-/
theorem graph_specFwd_eq_runSpecAuto (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens languageModel
        (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ)
    (hRoutes : RoutesAgree hcfg languageModel route pastTokens token states mlaScoreScale) :
    (graph hcfg languageModel route pastTokens token).specFwd
        (parameters languageModel) (inputs pastTokens languageModel states mlaScoreScale) =
      runSpecAuto hcfg languageModel pastTokens token states mlaScoreScale := by
  rw [graph_specFwd_eq_runSpec]
  exact runSpec_eq_runSpecAuto_of_routesAgree hcfg languageModel route pastTokens token states
    mlaScoreScale hRoutes

end TokenStep

end GraphSpec
end KimiK3
