/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import NN.API
public import NN.API.Models.CausalTransformer
public import NN.Spec.Core.Shape

/-!
# The language model

The Week 3 model uses TorchLean's public causal-Transformer constructor. Its shape is familiar:

1. look up each GPT-2 token in an embedding table;
2. add a learned position embedding;
3. run a stack of pre-normalized, causally masked Transformer blocks;
4. apply a final LayerNorm and project through the token table's transpose.

The large preset uses GPT-2-small's context length, width, head count, and depth. OpenAI's GPT-2
also shares its token table with the output projection. The TorchLean constructor below does the
same, giving this preset about 124.4 million stored parameters. It retains TorchLean's current
bias-free query, key, and value projections, so this is a GPT-2-small-sized training experiment
rather than a claim of bit-for-bit checkpoint compatibility with OpenAI's model.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT

/-- Model dimensions that remain independent of minibatch size. -/
structure ModelConfig where
  context : Nat
  vocab : Nat
  width : Nat
  heads : Nat
  layers : Nat
  dropout : Float := 0.0
  deriving Repr

namespace ModelConfig

/-- Small end-to-end run that still uses the full GPT-2 vocabulary. -/
def quick : ModelConfig :=
  { context := 16
    vocab := 50257
    width := 64
    heads := 4
    layers := 2 }

/-- GPT-2-small dimensions: context 1,024, width 768, 12 heads, and 12 blocks. -/
def gpt2Small : ModelConfig :=
  { context := 1024
    vocab := 50257
    width := 768
    heads := 12
    layers := 12 }

/-- Parse the two documented presets. Every field can still be overridden by the runner. -/
def ofName (name : String) : Except String ModelConfig :=
  match name.trimAscii.toString.toLower with
  | "quick" => .ok quick
  | "gpt2-small" => .ok gpt2Small
  | other => .error s!"unknown preset '{other}'; expected quick or gpt2-small"

/-- Reject dimensions that cannot define a nondegenerate multi-head Transformer. -/
def validate (cfg : ModelConfig) : Except String Unit := do
  if cfg.context = 0 then
    throw "context length must be positive"
  if cfg.vocab = 0 then
    throw "vocabulary size must be positive"
  if cfg.width = 0 then
    throw "model width must be positive"
  if cfg.heads = 0 then
    throw "attention head count must be positive"
  if cfg.layers = 0 then
    throw "Transformer layer count must be positive"
  if cfg.width % cfg.heads != 0 then
    throw s!"attention heads ({cfg.heads}) must divide model width ({cfg.width})"
  if cfg.dropout.isNaN || cfg.dropout.isInf then
    throw "dropout must be finite"
  if cfg.dropout < 0.0 || cfg.dropout >= 1.0 then
    throw "dropout must lie in [0, 1)"

/--
Convert the readable Week 3 dimensions to TorchLean's public model configuration.

GPT-2 initializes ordinary projection and embedding weights with standard deviation `0.02`. Its
attention-output and feed-forward-output projections write directly to residual streams, so their
standard deviation is reduced to `0.02 / sqrt (2 * layers)`. The separate initializer keeps this
depth-dependent convention explicit in the model configuration.
-/
def toTorchLean (cfg : ModelConfig) : nn.models.CausalTransformer.Config :=
  let residualStd := 0.02 / Float.sqrt (Float.ofNat (2 * cfg.layers))
  { sequenceLength := cfg.context
    vocabularySize := cfg.vocab
    headCount := cfg.heads
    headWidth := cfg.width / cfg.heads
    feedForwardWidth := 4 * cfg.width
    layerCount := cfg.layers
    activation := .gelu
    dropout? := if cfg.dropout == 0.0 then none else some cfg.dropout
    normalizeFirst := true
    attentionOutputBias := true
    parameterInitialization? := some (.normal 0.0 0.02)
    residualProjectionInitialization? := some (.normal 0.0 residualStd) }

end ModelConfig

/-- Preserve the normalization used by the original Week 3 checkpoints on each backend. -/
def normalizationEpsilon (usesCuda : Bool) : Rat :=
  if usesCuda then 1e-6 else 1e-5

/--
Build the complete tied-weight language model through TorchLean's public API.

The same model owns token lookup, the hidden Transformer, and vocabulary projection. Its builder
initializes the embedding table and body from one seed stream; training and generation therefore
share one parameter layout. The operation adapter preserves the original checkpoints' LayerNorm
epsilon until the constructor exposes that setting; all other operations use the library defaults.
-/
def buildModel (cfg : nn.models.CausalTransformer.Config) (batch : Nat)
    (runtime : Runtime.Config) :
    nn.Builder (nn.IndexedModel (cfg.tokenShape [batch]) (cfg.vocabularyShape [batch])
      (Fin cfg.vocabularySize)) := do
  let model ← nn.models.CausalTransformer.tied cfg [batch]
  pure <| nn.IndexedModel.Internal.create model.stateShapes model.initialState
    (fun mode {α} _ _ {m} _ operations =>
      letI : _root_.Runtime.Autograd.Torch.Ops m α :=
        { operations with
          layerNorm := fun hRows hWidth input scale bias _ =>
            operations.layerNorm hRows hWidth input scale bias
              (Context.ofRat (normalizationEpsilon runtime.usesCuda)) }
      nn.IndexedModel.Internal.program model mode (α := α) (m := m))
    (kind := model.kind)
    (initializationPlan := nn.IndexedModel.Internal.initializationPlan model)
    (trainableMask := model.requiresGrad)
    (validateModel := model.validate)
    (validateInput := nn.IndexedModel.Internal.validateInput model)

/--
Next-token cross entropy with one weight per prediction row.

Tokens and targets remain discrete `Fin cfg.vocabularySize` tensors. Row weights use the model
scalar type; the data loader normalizes the active weights to sum to one and gives padding and
prompt-only positions weight zero. Both ordinary training and instruction tuning use the same
indexed model, initialization plan, and parameter order.
-/
def weightedObjective
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocabularySize]
    (batch : Nat)
    (model : nn.IndexedModel (cfg.tokenShape [batch]) (cfg.vocabularyShape [batch])
      (Fin cfg.vocabularySize))
    (mode : nn.Mode := .train) :
    Module.ObjectiveDefinition (Fin cfg.vocabularySize)
      model.stateShapes [cfg.tokenShape [batch]] [cfg.tokenShape [batch], cfg.tokenShape [batch]] :=
  { initState := nn.State.Internal.toTensorPack model.initialState
    runtimeInit := nn.IndexedModel.Internal.initializationPlan model
    requiresGrad := model.requiresGrad
    validate := model.validate
    validateDataInputs := fun
      | .cons tokens (.cons _targets .nil) => nn.IndexedModel.Internal.validateInput model tokens
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := Runtime.ValueRef (m := m) (α := α))
          (ss := model.stateShapes ++ [cfg.tokenShape [batch]])
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.DataRef
              (m := m) (α := α) (Fin cfg.vocabularySize) s)
            [cfg.tokenShape [batch], cfg.tokenShape [batch]]
            (m (Runtime.ValueRef (m := m) (α := α) [])))
          (fun arguments => fun tokens => fun targets => (do
            let (state, rowWeights) := _root_.Runtime.Autograd.Torch.RefList.splitLast
              (Ref := Runtime.ValueRef (m := m) (α := α))
              (ss := model.stateShapes) (τ := cfg.tokenShape [batch]) arguments
            let forward := _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
              (Ref := Runtime.ValueRef (m := m) (α := α))
              (ss := model.stateShapes)
              (β := _root_.Runtime.Autograd.Torch.CurriedRef
                (fun s => _root_.Runtime.Autograd.Torch.DataRef
                  (m := m) (α := α) (Fin cfg.vocabularySize) s)
                [cfg.tokenShape [batch]]
                (m (Runtime.ValueRef (m := m) (α := α) (cfg.vocabularyShape [batch]))))
              (nn.IndexedModel.Internal.program model mode (α := α) (m := m)) state
            let logits ← forward tokens
            let logitsIndexed : Runtime.ValueRef (m := m) (α := α)
                ((cfg.tokenShape [batch]).concat [cfg.vocabularySize]) := by
              simpa [nn.models.CausalTransformer.Config.vocabularyShape,
                nn.models.CausalTransformer.Config.tokenShape] using logits
            Loss.crossEntropyWeighted (m := m) (α := α)
              (leading := cfg.tokenShape [batch]) (trailing := [])
              (classes := cfg.vocabularySize) (cfg.tokenShape [batch]).rank rfl
              logitsIndexed targets rowWeights : m (Runtime.ValueRef (m := m) (α := α) []))) }

end TorchLeanGPT
