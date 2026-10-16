/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Decoder.Depth
public import KimiK3.GraphSpec.Decoder.Layer

/-!
# Composing the decoder backbone

Concatenate the heterogeneous layer parameters and causal states, thread the AttnRes depth
state through the configured schedule, and prove the resulting graph agrees with its reference
calculation. Sparse routes are static graph arguments, and `RoutesAgree` states separately when
they match the router.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean NN.GraphSpec.DAG

private theorem eval_splitArgs {α : Type} [Storage α] [Context α]
    {Γ left right : List Shape} (env : TorchLean.TensorPack α Γ)
    (args : Args Γ (left ++ right)) :
    (Term.evalArgs env (splitArgs args).1, Term.evalArgs env (splitArgs args).2) =
      TorchLean.TensorPack.split (Term.evalArgs env args) := by
  exact Prod.ext (eval_splitArgs_left env args) (eval_splitArgs_right env args)

namespace Decoder

open Spec TorchLean
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

variable {cfg : TextConfig} {decayRank : Nat}

/-!
### Typed ABI of a decoder schedule

The decoder has no uniform per-layer cache. A KDA layer owns a fixed convolution window and
recurrent matrix, while an MLA layer owns two caches whose leading dimension grows with the context.
The functions below concatenate the per-layer ABIs and keep every shape in the type, where an
untyped array would lose the distinction.
-/

/-- Parameter shapes of the selected layers, in execution order. -/
def ParamsFor (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | List.nil => []
  | index :: rest =>
      DecoderLayer.Params (model.layer index).sequence (model.layer index).channel ++
        ParamsFor model rest

/-- Causal-state shapes before the selected layers process the current token. -/
def StateShapesFor (pastTokens : Nat) (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | List.nil => []
  | index :: rest =>
      DecoderLayer.StateShapes pastTokens (model.layer index).sequence ++
        StateShapesFor pastTokens model rest

/-- Causal-state shapes after the selected layers process the current token. -/
def NextStateShapesFor (pastTokens : Nat) (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | List.nil => []
  | index :: rest =>
      DecoderLayer.NextStateShapes pastTokens (model.layer index).sequence ++
        NextStateShapesFor pastTokens model rest

/-- One scalar control per sequence mixer (log floor for KDA, score scale for MLA). -/
def ControlShapesFor (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | List.nil => []
  | _index :: rest => [] :: ControlShapesFor model rest

/-- Number of processed layers after executing a concrete schedule segment. -/
def processedAfter {Layer : Type} : Nat → List Layer → Nat
  | processedLayers, List.nil => processedLayers
  | processedLayers, _layer :: rest => processedAfter (processedLayers + 1) rest

/-- Executing a schedule advances the depth counter by its length. -/
theorem processedAfter_eq_add_length {Layer : Type} (processedLayers : Nat)
    (layers : List Layer) :
    processedAfter processedLayers layers = processedLayers + layers.length := by
  induction layers generalizing processedLayers with
  | nil => simp [processedAfter]
  | cons layer rest ih =>
      simp only [processedAfter, ih, List.length_cons]
      omega

/-- Inputs to a decoder segment: depth state, causal states, numerical controls, and shared
scalars. -/
abbrev InputsFor (pastTokens processedLayers : Nat) (model : LanguageModel ℝ cfg decayRank)
    (indices : List (Fin cfg.numLayers)) : List Shape :=
  DepthSchedule.Shapes cfg.attnResBlockSize processedLayers cfg.hiddenDim ++
    (StateShapesFor pastTokens model indices ++
      (ControlShapesFor model indices ++ [[], [], []]))

/-- Outputs of a decoder segment: updated causal states followed by the final packed depth state. -/
abbrev OutputsFor (pastTokens processedLayers : Nat) (model : LanguageModel ℝ cfg decayRank)
    (indices : List (Fin cfg.numLayers)) : List Shape :=
  NextStateShapesFor pastTokens model indices ++
    DepthSchedule.Shapes cfg.attnResBlockSize (processedAfter processedLayers indices) cfg.hiddenDim

/-- Extract the selected layers' parameters from the existing language-model value. -/
def parametersFor (model : LanguageModel ℝ cfg decayRank) :
    (indices : List (Fin cfg.numLayers)) → TorchLean.TensorPack ℝ (ParamsFor model indices)
  | List.nil => .nil
  | index :: rest =>
      TorchLean.TensorPack.append
        (DecoderLayer.parameters (model.layer index)) (parametersFor model rest)

/-- Pack one causal state per layer into the decoder's heterogeneous state ABI. -/
def packStates (pastTokens : Nat) (model : LanguageModel ℝ cfg decayRank)
    (state : ∀ index, TorchLean.TensorPack ℝ
      (DecoderLayer.StateShapes pastTokens (model.layer index).sequence)) :
    (indices : List (Fin cfg.numLayers)) →
      TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices)
  | [] => .nil
  | index :: rest =>
      TorchLean.TensorPack.append (state index)
        (packStates pastTokens model state rest)

/-- Empty fixed-shape cache corresponding to a concrete sequence mixer. -/
def emptyState (sequence : SequenceMixer ℝ cfg decayRank) :
    TorchLean.TensorPack ℝ (DecoderLayer.StateShapes 0 sequence) :=
  match sequence with
  | .kda _ _ =>
      .cons (Tensor.zeros [cfg.shortConvWidth, cfg.hiddenDim]) <|
        .cons (Tensor.zeros [cfg.numHeads, cfg.kdaHeadDim, cfg.kdaValueDim]) .nil
  | .mla _ =>
      .cons (Tensor.zeros [0, cfg.kvLatentDim]) <|
        .cons (Tensor.zeros [0, cfg.qkReservedHeadDim]) .nil

/-- Empty causal states for a fresh decoder run. -/
def initialStatesFor (model : LanguageModel ℝ cfg decayRank) :
    (indices : List (Fin cfg.numLayers)) →
      TorchLean.TensorPack ℝ (StateShapesFor 0 model indices) :=
  packStates 0 model (fun index => emptyState (model.layer index).sequence)

/-- Pack the numerical control of every selected sequence mixer. -/
def controlsFor (model : LanguageModel ℝ cfg decayRank) (mlaScoreScale : ℝ) :
    (indices : List (Fin cfg.numLayers)) → TorchLean.TensorPack ℝ (ControlShapesFor model indices)
  | List.nil => .nil
  | index :: rest =>
      .cons (Tensor.scalar (DecoderLayer.controlValue (model.layer index).sequence mlaScoreScale))
        (controlsFor model mlaScoreScale rest)

/-- Package the values consumed by a decoder segment in its public input order. -/
noncomputable def inputsFor (pastTokens processedLayers : Nat)
    (model : LanguageModel ℝ cfg decayRank)
    (indices : List (Fin cfg.numLayers))
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes cfg.attnResBlockSize processedLayers cfg.hiddenDim))
    (states : TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices))
    (mlaScoreScale : ℝ) :
    TorchLean.TensorPack ℝ (InputsFor pastTokens processedLayers model indices) :=
  TorchLean.TensorPack.append depth <|
    TorchLean.TensorPack.append states <|
      TorchLean.TensorPack.append (controlsFor model mlaScoreScale indices) <|
        .cons (Tensor.scalar Normalize.l2Epsilon) <| .cons (Tensor.scalar cfg.situGateCap) <|
          .cons (Tensor.scalar cfg.situUpCap) .nil

/-- An empty decoder schedule returns its depth state unchanged. -/
theorem outputsFor_nil (pastTokens processedLayers : Nat)
    (model : LanguageModel ℝ cfg decayRank) :
    DepthSchedule.Shapes cfg.attnResBlockSize processedLayers cfg.hiddenDim =
      OutputsFor pastTokens processedLayers model [] := by
  simp [OutputsFor, NextStateShapesFor, processedAfter]

/-- One decoder step prepends its updated causal state to the remaining schedule outputs. -/
theorem outputsFor_cons (pastTokens processedLayers : Nat)
    (model : LanguageModel ℝ cfg decayRank) (index : Fin cfg.numLayers)
    (rest : List (Fin cfg.numLayers)) :
    DecoderLayer.NextStateShapes pastTokens (model.layer index).sequence ++
        OutputsFor pastTokens (processedLayers + 1) model rest =
      OutputsFor pastTokens processedLayers model (index :: rest) := by
  simp [OutputsFor, NextStateShapesFor, processedAfter, List.append_assoc]

/-- Pure mathematical execution of a decoder schedule segment.

This is the schedule-level counterpart of `runBlock`, with no graph syntax and no backend
operations: it runs each layer's `specStep`, advances the AttnRes state, and recurses.
-/
noncomputable def runSpec {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF) (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      TorchLean.TensorPack ℝ
        (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
          architecture.text.hiddenDim) →
      TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices) → ℝ →
      TorchLean.TensorPack ℝ (OutputsFor pastTokens processedLayers model indices)
  | List.nil, depth, _states, _mlaScoreScale =>
      outputsFor_nil pastTokens processedLayers model ▸ depth
  | index :: rest, depth, states, mlaScoreScale => by
      let layer := model.layer index
      let stateParts := TorchLean.TensorPack.split
        (ss₁ := DecoderLayer.StateShapes pastTokens layer.sequence)
        (ss₂ := StateShapesFor pastTokens model rest) states
      let completed := Env.tget depth Var.head
      let partialState := Env.tget depth (.tail .head)
      let result := DecoderLayer.specStep pastTokens layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos completed partialState stateParts.1
        mlaScoreScale
      let nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed
        (DecoderLayer.nextPartialValue pastTokens layer.sequence result)
      let restResult := runSpec hcfg model route pastTokens (processedLayers + 1) rest nextDepth
        stateParts.2 mlaScoreScale
      exact outputsFor_cons pastTokens processedLayers model index rest ▸
        TorchLean.TensorPack.append
          (DecoderLayer.nextStateValues pastTokens layer.sequence result) restResult
termination_by indices => indices.length

/-- Pure decoder semantics with every sparse route recomputed from the intermediate channel input.

This is the architecture-level meaning of K3 routing. The route-staged graph below specializes
this function, and `RoutesAgree` states when that specialization is valid for the current token and
causal state.
-/
noncomputable def runSpecAuto {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF) (model : LanguageModel ℝ architecture.text decayRank)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      TorchLean.TensorPack ℝ
        (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
          architecture.text.hiddenDim) →
      TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices) → ℝ →
      TorchLean.TensorPack ℝ (OutputsFor pastTokens processedLayers model indices)
  | List.nil, depth, _states, _mlaScoreScale =>
      outputsFor_nil pastTokens processedLayers model ▸ depth
  | index :: rest, depth, states, mlaScoreScale => by
      let layer := model.layer index
      let stateParts := TorchLean.TensorPack.split
        (ss₁ := DecoderLayer.StateShapes pastTokens layer.sequence)
        (ss₂ := StateShapesFor pastTokens model rest) states
      let completed := Env.tget depth Var.head
      let partialState := Env.tget depth (.tail .head)
      let result := DecoderLayer.specStepAuto pastTokens layer
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos hcfg.activeExperts_le completed partialState
        stateParts.1 mlaScoreScale
      let nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed
        (DecoderLayer.nextPartialValue pastTokens layer.sequence result)
      let restResult := runSpecAuto hcfg model pastTokens (processedLayers + 1) rest nextDepth
        stateParts.2 mlaScoreScale
      exact outputsFor_cons pastTokens processedLayers model index rest ▸
        TorchLean.TensorPack.append
          (DecoderLayer.nextStateValues pastTokens layer.sequence result) restResult
termination_by indices => indices.length

/-- Every staged route agrees with the deterministic router along the packed decoder run.

The recursive call follows `runSpecAuto`, so the route for layer `i + 1` is compared with scores
computed from the representation produced through layer `i`. Dense layers contribute `True`, and
sparse layers require equality with `StableLatentMoE.route`. This is a hypothesis of the final
theorem, not a route check run inside the floating-point DAG.
-/
noncomputable def RoutesAgree {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF) (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      TorchLean.TensorPack ℝ
        (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
          architecture.text.hiddenDim) →
      TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices) → ℝ → Prop
  | List.nil, _depth, _states, _mlaScoreScale => True
  | index :: rest, depth, states, mlaScoreScale =>
      let layer := model.layer index
      let stateParts := TorchLean.TensorPack.split
        (ss₁ := DecoderLayer.StateShapes pastTokens layer.sequence)
        (ss₂ := StateShapesFor pastTokens model rest) states
      let completed := Env.tget depth Var.head
      let partialState := Env.tget depth (.tail .head)
      let routeAgrees := DecoderLayer.RouteAgrees pastTokens layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.activeExperts_le completed partialState stateParts.1 mlaScoreScale
      let result := DecoderLayer.specStepAuto pastTokens layer
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos hcfg.activeExperts_le completed partialState
        stateParts.1 mlaScoreScale
      let nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed
        (DecoderLayer.nextPartialValue pastTokens layer.sequence result)
      routeAgrees ∧ RoutesAgree hcfg model route pastTokens (processedLayers + 1) rest nextDepth
        stateParts.2 mlaScoreScale
termination_by indices => indices.length

/-- A route-staged decoder run equals K3's automatically routed semantics when its trace passes
`RoutesAgree`. -/
theorem runSpec_eq_runSpecAuto_of_routesAgree
    {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) (indices : List (Fin architecture.text.numLayers))
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim))
    (states : TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices)) (mlaScoreScale : ℝ)
    (hRoutes : RoutesAgree hcfg model route pastTokens processedLayers indices depth states
      mlaScoreScale) :
    runSpec hcfg model route pastTokens processedLayers indices depth states mlaScoreScale =
      runSpecAuto hcfg model pastTokens processedLayers indices depth states mlaScoreScale := by
  induction indices generalizing processedLayers with
  | nil =>
      rw [runSpec.eq_def, runSpecAuto.eq_def]
  | cons index rest ih =>
      simp only [RoutesAgree] at hRoutes
      rw [runSpec, runSpecAuto]
      rw [DecoderLayer.specStep_eq_specStepAuto_of_routeAgrees
        (hRoute := hRoutes.1)]
      rw [ih _ _ _ hRoutes.2]

/-!
### Recursive decoder graph
-/

/-- Inline a schedule segment into one typed DAG block.

Each recursive step runs the selected decoder-layer graph, advances the packed AttnRes state, and
continues with the remaining layers. The route is static graph data, and `RoutesAgree` relates it
to the router, so no sorting primitive sits inside the numerical graph.
-/
def runBlock {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    {Γ : List Shape} (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      Args Γ (ParamsFor model indices) →
      Args Γ (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim) →
      Args Γ (StateShapesFor pastTokens model indices) →
      Args Γ (ControlShapesFor model indices) →
      Term Γ [] → Term Γ [] → Term Γ [] →
      Block Γ (OutputsFor pastTokens processedLayers model indices)
  | List.nil, params, depth, states, controls, _epsilon, _gateCap, _upCap => by
      cases params
      cases states
      cases controls
      exact outputsFor_nil pastTokens processedLayers model ▸ Block.ret depth
  | index :: rest, params, depth, states, controls, epsilon, gateCap, upCap => by
      let layer := model.layer index
      let paramParts := splitArgs
        (left := DecoderLayer.Params layer.sequence layer.channel)
        (right := ParamsFor model rest) params
      let stateParts := splitArgs
        (left := DecoderLayer.StateShapes pastTokens layer.sequence)
        (right := StateShapesFor pastTokens model rest) states
      let completed := Args.get depth Var.head
      let partialState := Args.get depth (.tail .head)
      let controlParts := splitArgs
        (left := [[]]) (right := ControlShapesFor model rest) controls
      let sequenceControl := Args.get controlParts.1 Var.head
      let layerGraph := DecoderLayer.model pastTokens
        (DepthSchedule.completedCount architecture.text.attnResBlockSize processedLayers)
        layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
        hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
        hcfg.routedLatentDim_pos
      let layerInputs := DecoderLayer.inputTerms pastTokens layer.sequence completed partialState
        stateParts.1 sequenceControl epsilon gateCap upCap
      let layerBlock := layerGraph.inline paramParts.1 layerInputs
      let layerOutputs := DecoderLayer.Outputs pastTokens layer.sequence
      exact layerBlock.andThen <|
        let results : Args (Γ ++ layerOutputs) layerOutputs :=
          Args.rename (Var.inRight Γ) (Args.vars layerOutputs)
        let nextStates := DecoderLayer.nextStateTerms pastTokens layer.sequence results
        let nextPartial := DecoderLayer.nextPartialTerm pastTokens layer.sequence results
        let completed' := Term.weakenAppend layerOutputs completed
        let depthBlock := DepthSchedule.advanceBlock architecture.text.attnResBlockSize
          processedLayers architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed'
          nextPartial
        depthBlock.andThen <|
          let nextDepthShapes := DepthSchedule.Shapes architecture.text.attnResBlockSize
            (processedLayers + 1) architecture.text.hiddenDim
          let extended := Γ ++ layerOutputs
          let depthResults : Args (extended ++ nextDepthShapes) nextDepthShapes :=
            Args.rename (Var.inRight extended) (Args.vars nextDepthShapes)
          let liftOriginal := fun {sh : Shape} (term : Term Γ sh) =>
            Term.weakenAppend nextDepthShapes (Term.weakenAppend layerOutputs term)
          let liftOriginalArgs := fun {shapes : List Shape} (args : Args Γ shapes) =>
            Args.rename (Var.inLeft nextDepthShapes)
              (Args.rename (Var.inLeft layerOutputs) args)
          let recursive := runBlock hcfg model route pastTokens (processedLayers + 1) rest
            (liftOriginalArgs paramParts.2) depthResults (liftOriginalArgs stateParts.2)
            (liftOriginalArgs controlParts.2) (liftOriginal epsilon) (liftOriginal gateCap)
            (liftOriginal upCap)
          let recursiveOutputs := OutputsFor pastTokens (processedLayers + 1) model rest
          recursive.andThen <|
            let recursiveResults : Args ((extended ++ nextDepthShapes) ++ recursiveOutputs)
                recursiveOutputs :=
              Args.rename (Var.inRight (extended ++ nextDepthShapes))
                (Args.vars recursiveOutputs)
            let nextStates' := Args.rename (Var.inLeft nextDepthShapes) nextStates
            let nextStates'' := Args.rename (Var.inLeft recursiveOutputs) nextStates'
            outputsFor_cons pastTokens processedLayers model index rest ▸
              Block.ret (Args.append nextStates'' recursiveResults)
termination_by indices => indices.length

/-- Evaluating `runBlock` gives the pure decoder schedule `runSpec`.

The hypotheses fix the learned parameters, the per-layer controls, epsilon, and the two SiTU caps in
the surrounding graph environment. Causal and AttnRes states are arbitrary, so the theorem covers a
fresh prompt as well as continuation from any well-typed cache. -/
theorem eval_runBlock {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ) (pastTokens processedLayers : Nat)
    (indices : List (Fin architecture.text.numLayers))
    (params : Args Γ (ParamsFor languageModel indices))
    (depth : Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim))
    (states : Args Γ (StateShapesFor pastTokens languageModel indices))
    (controls : Args Γ (ControlShapesFor languageModel indices))
    (epsilon gateCap upCap : Term Γ []) (mlaScoreScale : ℝ)
    (hParams : Term.evalArgs env params = parametersFor languageModel indices)
    (hControls : Term.evalArgs env controls =
      controlsFor languageModel mlaScoreScale indices)
    (hEpsilon : Term.eval env epsilon = Tensor.scalar Normalize.l2Epsilon)
    (hGateCap : Term.eval env gateCap =
      Tensor.scalar (architecture.text.situGateCap : ℝ))
    (hUpCap : Term.eval env upCap = Tensor.scalar (architecture.text.situUpCap : ℝ)) :
    Block.eval env
        (runBlock hcfg languageModel route pastTokens processedLayers indices params depth states
          controls epsilon gateCap upCap) =
      runSpec hcfg languageModel route pastTokens processedLayers indices
        (Term.evalArgs env depth) (Term.evalArgs env states) mlaScoreScale := by
  induction indices generalizing Γ processedLayers with
  | nil =>
      rw [runBlock.eq_def, runSpec.eq_def]
      cases params
      cases states
      cases controls
      simp only [Block.eval]
      rfl
  | cons index rest ih =>
      rw [runBlock.eq_def]
      let layer := languageModel.layer index
      change Args Γ
        (DecoderLayer.Params layer.sequence layer.channel ++ ParamsFor languageModel rest) at params
      change Args Γ
        (DecoderLayer.StateShapes pastTokens layer.sequence ++
          StateShapesFor pastTokens languageModel rest) at states
      change Args Γ ([[]] ++ ControlShapesFor languageModel rest) at controls
      let paramParts := splitArgs
        (left := DecoderLayer.Params layer.sequence layer.channel)
        (right := ParamsFor languageModel rest) params
      let stateParts := splitArgs
        (left := DecoderLayer.StateShapes pastTokens layer.sequence)
        (right := StateShapesFor pastTokens languageModel rest) states
      let controlParts := splitArgs
        (left := [[]]) (right := ControlShapesFor languageModel rest) controls
      let completed := Args.get depth Var.head
      let partialState := Args.get depth (.tail .head)
      let control := Args.get controlParts.1 Var.head
      have hParamParts := eval_splitArgs env params
      have hStateParts := eval_splitArgs env states
      have hControlParts := eval_splitArgs env controls
      dsimp [paramParts, stateParts, controlParts] at hParamParts hStateParts hControlParts
      have hParamParts' := hParamParts.trans <| congrArg
        (TorchLean.TensorPack.split
          (ss₁ := DecoderLayer.Params layer.sequence layer.channel)
          (ss₂ := ParamsFor languageModel rest)) hParams
      simp only [parametersFor,
        TorchLean.TensorPack.split_append] at hParamParts'
      have hControlParts' := hControlParts.trans <| congrArg
        (TorchLean.TensorPack.split
          (ss₁ := [[]]) (ss₂ := ControlShapesFor languageModel rest)) hControls
      simp only [controlsFor] at hControlParts'
      have hLayerParams : Term.evalArgs env paramParts.1 =
          DecoderLayer.parameters layer := congrArg Prod.fst hParamParts'
      have hRestParams : Term.evalArgs env paramParts.2 =
          parametersFor languageModel rest := congrArg Prod.snd hParamParts'
      have hControlValues : Term.evalArgs env controlParts.1 =
          .cons (Tensor.scalar (DecoderLayer.controlValue layer.sequence mlaScoreScale)) .nil :=
        congrArg Prod.fst hControlParts'
      have hRestControls : Term.evalArgs env controlParts.2 =
          controlsFor languageModel mlaScoreScale rest := congrArg Prod.snd hControlParts'
      have hControl : Term.eval env control =
          Tensor.scalar (DecoderLayer.controlValue layer.sequence mlaScoreScale) := by
        calc
          Term.eval env control = Env.tget (Term.evalArgs env controlParts.1) Var.head :=
            Term.eval_get env controlParts.1 Var.head
          _ = Tensor.scalar (DecoderLayer.controlValue layer.sequence mlaScoreScale) := by
            rw [hControlValues]
            rfl
      have hLayerResult :
          Block.eval env
              ((DecoderLayer.model pastTokens
                (DepthSchedule.completedCount architecture.text.attnResBlockSize processedLayers)
                layer (route index)
                (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
                (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize
                  processedLayers)
                hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
                hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
                hcfg.routedLatentDim_pos).inline paramParts.1
                  (DecoderLayer.inputTerms pastTokens layer.sequence completed partialState
                    stateParts.1 control epsilon gateCap upCap)) =
            DecoderLayer.specStep pastTokens layer (route index)
              (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
              (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
              hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
              hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos (Term.eval env completed)
              (Term.eval env partialState) (Term.evalArgs env stateParts.1)
              mlaScoreScale := by
        rw [NN.GraphSpec.DAG.MultiModel.eval_inline, hLayerParams]
        rw [DecoderLayer.evalArgs_inputTerms env pastTokens layer.sequence completed partialState
          stateParts.1 control epsilon gateCap upCap mlaScoreScale]
        · exact DecoderLayer.model_specFwd_eq_specStep pastTokens layer (route index)
            (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
            (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
            hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
            hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
            hcfg.routedLatentDim_pos hcfg.activeExperts_pos (Term.eval env completed)
            (Term.eval env partialState) (Term.evalArgs env stateParts.1) mlaScoreScale
        · exact hControl
        · exact hEpsilon
        · exact hGateCap
        · exact hUpCap
      rw [Block.eval_andThen, hLayerResult]
      rw [Block.eval_andThen]
      rw [DepthSchedule.eval_advanceBlock]
      simp only [Term.eval_weakenAppend, DecoderLayer.eval_nextPartialTerm,
        Term.evalArgs_rename_inRight, Term.evalArgs_vars]
      set layerResult := DecoderLayer.specStep pastTokens layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos (Term.eval env completed)
        (Term.eval env partialState) (Term.evalArgs env stateParts.1) mlaScoreScale
      set nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos (Term.eval env completed)
        (DecoderLayer.nextPartialValue pastTokens layer.sequence layerResult)
      set envDepth := TorchLean.TensorPack.append
        (TorchLean.TensorPack.append env layerResult) nextDepth
      let layerOutputs := DecoderLayer.Outputs pastTokens layer.sequence
      let nextDepthShapes := DepthSchedule.Shapes architecture.text.attnResBlockSize
        (processedLayers + 1) architecture.text.hiddenDim
      let extended := Γ ++ layerOutputs
      let depthResults : Args (extended ++ nextDepthShapes) nextDepthShapes :=
        Args.rename (Var.inRight extended) (Args.vars nextDepthShapes)
      let liftOriginal := fun {shape : Shape} (term : Term Γ shape) =>
        Term.weakenAppend nextDepthShapes (Term.weakenAppend layerOutputs term)
      let liftOriginalArgs := fun {shapes : List Shape} (args : Args Γ shapes) =>
        Args.rename (Var.inLeft nextDepthShapes) (Args.rename (Var.inLeft layerOutputs) args)
      have hDepthResults : Term.evalArgs envDepth depthResults = nextDepth := by
        dsimp only [envDepth, depthResults]
        rw [Term.evalArgs_rename_inRight, Term.evalArgs_vars]
      have hLiftedParams :
          Term.evalArgs envDepth (liftOriginalArgs paramParts.2) =
            parametersFor languageModel rest := by
        dsimp only [envDepth, liftOriginalArgs]
        rw [Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inLeft]
        exact hRestParams
      have hLiftedStates :
          Term.evalArgs envDepth (liftOriginalArgs stateParts.2) =
            Term.evalArgs env stateParts.2 := by
        dsimp only [envDepth, liftOriginalArgs]
        rw [Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inLeft]
      have hLiftedControls :
          Term.evalArgs envDepth (liftOriginalArgs controlParts.2) =
            controlsFor languageModel mlaScoreScale rest := by
        dsimp only [envDepth, liftOriginalArgs]
        rw [Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inLeft]
        exact hRestControls
      have hLiftedEpsilon :
          Term.eval envDepth (liftOriginal epsilon) = Tensor.scalar Normalize.l2Epsilon := by
        dsimp only [envDepth, liftOriginal]
        rw [Term.eval_weakenAppend, Term.eval_weakenAppend]
        exact hEpsilon
      have hLiftedGateCap :
          Term.eval envDepth (liftOriginal gateCap) =
            Tensor.scalar (architecture.text.situGateCap : ℝ) := by
        dsimp only [envDepth, liftOriginal]
        rw [Term.eval_weakenAppend, Term.eval_weakenAppend]
        exact hGateCap
      have hLiftedUpCap :
          Term.eval envDepth (liftOriginal upCap) =
            Tensor.scalar (architecture.text.situUpCap : ℝ) := by
        dsimp only [envDepth, liftOriginal]
        rw [Term.eval_weakenAppend, Term.eval_weakenAppend]
        exact hUpCap
      have hRecursive :
          Block.eval envDepth
              (runBlock hcfg languageModel route pastTokens (processedLayers + 1) rest
                (liftOriginalArgs paramParts.2) depthResults
                (liftOriginalArgs stateParts.2) (liftOriginalArgs controlParts.2)
                (liftOriginal epsilon) (liftOriginal gateCap) (liftOriginal upCap)) =
            runSpec hcfg languageModel route pastTokens (processedLayers + 1) rest nextDepth
              (Term.evalArgs env stateParts.2) mlaScoreScale := by
        rw [ih envDepth (processedLayers + 1) (liftOriginalArgs paramParts.2)
          depthResults (liftOriginalArgs stateParts.2) (liftOriginalArgs controlParts.2)
          (liftOriginal epsilon) (liftOriginal gateCap) (liftOriginal upCap)
          hLiftedParams hLiftedControls hLiftedEpsilon hLiftedGateCap hLiftedUpCap,
          hDepthResults, hLiftedStates]
      rw [Block.eval_andThen]
      rw [hRecursive]
      simp only [Block.eval_cast, Block.eval, Term.evalArgs_append, envDepth,
        Term.evalArgs_rename_inLeft,
        DecoderLayer.evalArgs_nextStateTerms, Term.evalArgs_rename_inRight,
        Term.evalArgs_vars, layer]
      have hCompletedValue :
          Env.tget (Term.evalArgs env depth) Var.head = Term.eval env completed :=
        (Term.eval_get env depth Var.head).symm
      have hPartialValue :
          Env.tget (Term.evalArgs env depth) (.tail .head) =
            Term.eval env partialState :=
        (Term.eval_get env depth (.tail .head)).symm
      rw [runSpec]
      rw [hCompletedValue, hPartialValue]
      simp only [StateShapesFor, layer] at states hStateParts ⊢
      rw [← hStateParts]

/-- Default GraphSpec parameters for a decoder segment.

These are the component models' initial values, meant for graph construction and shape checks.
A released checkpoint replaces them through the same `ParamsFor` ABI.
-/
def initialParamsFor {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      TorchLean.TensorPack Float (ParamsFor model indices)
  | [] => .nil
  | index :: rest =>
      let layer := model.layer index
      let layerGraph := DecoderLayer.model pastTokens
        (DepthSchedule.completedCount architecture.text.attnResBlockSize processedLayers)
        layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
        hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
        hcfg.routedLatentDim_pos
      TorchLean.TensorPack.append layerGraph.initParams
        (initialParamsFor hcfg model route pastTokens (processedLayers + 1) rest)

/-- A complete typed GraphSpec model for a decoder schedule segment. -/
def model {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat)
    (indices : List (Fin architecture.text.numLayers)) :
    NN.GraphSpec.DAG.MultiModel (ParamsFor languageModel indices)
      (InputsFor pastTokens processedLayers languageModel indices)
      (OutputsFor pastTokens processedLayers languageModel indices) :=
  { initParams := initialParamsFor hcfg languageModel route pastTokens processedLayers indices
    body := by
      let all := Args.vars
        (ParamsFor languageModel indices ++
          InputsFor pastTokens processedLayers languageModel indices)
      let parameterAndInput := splitArgs
        (left := ParamsFor languageModel indices)
        (right := InputsFor pastTokens processedLayers languageModel indices) all
      let depthAndRest := splitArgs
        (left := DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
          architecture.text.hiddenDim)
        (right := StateShapesFor pastTokens languageModel indices ++
          (ControlShapesFor languageModel indices ++ [[], [], []]))
        parameterAndInput.2
      let stateAndRest := splitArgs
        (left := StateShapesFor pastTokens languageModel indices)
        (right := ControlShapesFor languageModel indices ++ [[], [], []])
        depthAndRest.2
      let controlAndShared := splitArgs
        (left := ControlShapesFor languageModel indices)
        (right := [[], [], []]) stateAndRest.2
      let epsilon := Args.get controlAndShared.2 Var.head
      let gateCap := Args.get controlAndShared.2 (.tail .head)
      let upCap := Args.get controlAndShared.2 (.tail (.tail .head))
      exact runBlock hcfg languageModel route pastTokens processedLayers indices
        parameterAndInput.1 depthAndRest.1 stateAndRest.1 controlAndShared.1
        epsilon gateCap upCap }

/-- `model` computes `runSpec` on arbitrary typed state.

The statement covers a heterogeneous list of KDA and MLA layers. Its output lists every updated
causal state, then the packed AttnRes depth state.
-/
theorem model_specFwd_eq_runSpec {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat)
    (indices : List (Fin architecture.text.numLayers))
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim))
    (states : TorchLean.TensorPack ℝ (StateShapesFor pastTokens languageModel indices))
    (mlaScoreScale : ℝ) :
    (model hcfg languageModel route pastTokens processedLayers indices).specFwd
        (parametersFor languageModel indices)
        (inputsFor pastTokens processedLayers languageModel indices depth states mlaScoreScale) =
      runSpec hcfg languageModel route pastTokens processedLayers indices depth states
        mlaScoreScale := by
  simp only [NN.GraphSpec.DAG.MultiModel.specFwd, model, InputsFor, inputsFor]
  rw [eval_runBlock (mlaScoreScale := mlaScoreScale)]
  all_goals
    simp only [eval_splitArgs_left, eval_splitArgs_right,
      Term.evalArgs_vars, TorchLean.TensorPack.split_append,
      Term.eval_get, Env.tget]

/-- Decoder model for the full configured schedule, in canonical zero-based layer order. -/
def fullModel {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) :
    NN.GraphSpec.DAG.MultiModel
      (ParamsFor languageModel (List.finRange architecture.text.numLayers))
      (InputsFor pastTokens 0 languageModel (List.finRange architecture.text.numLayers))
      (OutputsFor pastTokens 0 languageModel (List.finRange architecture.text.numLayers)) :=
  model hcfg languageModel route pastTokens 0 (List.finRange architecture.text.numLayers)

end Decoder

end GraphSpec
end KimiK3
