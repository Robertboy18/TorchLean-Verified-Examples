/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLean.NN.Models.CausalTransformer.LanguageModel

/-!
# The language model

The Week 3 model uses TorchLean's public causal-Transformer constructor. Its shape is familiar:

1. look up each GPT-2 token in an embedding table
2. add a learned position embedding
3. run a stack of pre-normalized, causally masked Transformer blocks
4. apply a final LayerNorm and project through the token table's transpose

The `gpt2-small` preset uses GPT-2-small's context length, width, head count, and depth. Like
OpenAI's GPT-2, the constructor below shares the token table with the output projection, which
gives this preset about 124.4 million stored parameters. The query, key, and value projections keep
TorchLean's current bias-free form, so this is a GPT-2-small-sized training experiment without
bit-for-bit checkpoint compatibility with OpenAI's model.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT

/-- Model dimensions and dropout rate, independent of the minibatch size. -/
structure ModelConfig where
  context : Nat
  vocab : Nat
  width : Nat
  heads : Nat
  layers : Nat
  dropout : Float := 0.0
  deriving Repr

namespace ModelConfig

/-- Small preset for quick end-to-end runs, with the full GPT-2 vocabulary. -/
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

/--
Look up a preset by name, ignoring case and surrounding whitespace. The runner can still override
every field.
-/
def ofName (name : String) : Except String ModelConfig :=
  match name.trimAscii.toString.toLower with
  | "quick" => .ok quick
  | "gpt2-small" => .ok gpt2Small
  | other => .error s!"unknown preset '{other}'; expected quick or gpt2-small"

/--
Reject dimensions that cannot define a nondegenerate multi-head Transformer, and dropout outside
`[0, 1)`.
-/
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
Convert the Week 3 dimensions to TorchLean's public model configuration.

GPT-2 initializes projection and embedding weights with standard deviation `0.02`, except the
attention-output and feed-forward-output projections. Those write into the residual stream and get
`0.02 / sqrt (2 * layers)`, which goes in `residualInit?`.
-/
def toTorchLean (cfg : ModelConfig) : nn.CausalTransformer.Config :=
  let residualStd := 0.02 / Float.sqrt (Float.ofNat (2 * cfg.layers))
  { sequenceLength := cfg.context
    vocabularySize := cfg.vocab
    headCount := cfg.heads
    headWidth := cfg.width / cfg.heads
    feedForwardWidth := 4 * cfg.width
    layerCount := cfg.layers
    activation := .geluTanh
    dropout? := if cfg.dropout == 0.0 then none else some cfg.dropout
    normalizeFirst := true
    attentionOutputBias := true
    parameterInit? := some (.normal 0.0 0.02)
    residualInit? := some (.normal 0.0 residualStd) }

end ModelConfig

/-- LayerNorm epsilon used by the Week 3 checkpoints on each backend. -/
def normalizationEpsilon (usesCuda : Bool) : Rat :=
  if usesCuda then 1e-6 else 1e-5

/--
Build the tied-weight language model from TorchLean's `nn.CausalTransformer.tied`.

One model owns token lookup, the Transformer body, and the vocabulary projection, and its builder
initializes the embedding table and body from one seed stream. Training and every generator call
this function, so they share one parameter layout. The operation adapter changes only the LayerNorm
epsilon (to `normalizationEpsilon`) until the constructor exposes that setting.
-/
def buildModel (cfg : nn.CausalTransformer.Config) (batch : Nat)
    (runtime : Session.Config) :
    nn.Builder (nn.IndexedModel (cfg.tokenShape [batch]) (cfg.vocabularyShape [batch])
      (Fin cfg.vocabularySize)) := do
  let model ← nn.CausalTransformer.tied cfg [batch]
  pure <| nn.IndexedModel.create model.stateShapes model.initialState
    (fun mode {α} _ _ {m} _ operations =>
      letI : Ops m α :=
        { operations with
          apply := fun op _supported args =>
            match op with
            | .layerNorm hRows hWidth hLead hShape =>
              let epsilon := operations.dataConst
                (Tensor.scalar (Rat.cast (normalizationEpsilon runtime.usesCUDA) : α))
              operations.apply (.layerNorm hRows hWidth hLead hShape)
                (args.1, args.2.1, args.2.2.1, .data epsilon, ())
            | op => operations.apply op args }
      nn.IndexedModel.program model mode (α := α) (m := m))
    (kind := model.kind)
    (initializationPlan := nn.IndexedModel.initializationPlan model)
    (trainableMask := model.requiresGrad)
    (validateModel := model.validate)
    (validateInput := nn.IndexedModel.validateInput model)

/--
Next-token cross entropy with one weight per prediction row.

Tokens and targets stay discrete `Fin cfg.vocabularySize` tensors, while row weights use the model
scalar type. The data loader gives padding and prompt-only positions weight zero and normalizes the
active weights to sum to one (`Run.activeRowWeight`). Ordinary training and instruction tuning use
the same indexed model, initialization plan, and parameter order.
-/
def weightedObjective
    (cfg : nn.CausalTransformer.Config) [NeZero cfg.vocabularySize]
    (batch : Nat)
    (model : nn.IndexedModel (cfg.tokenShape [batch]) (cfg.vocabularyShape [batch])
      (Fin cfg.vocabularySize))
    (mode : nn.Mode := .train) :
    ObjectiveDef (Fin cfg.vocabularySize)
      model.stateShapes [cfg.tokenShape [batch]] [cfg.tokenShape [batch], cfg.tokenShape [batch]] :=
  { initState := model.initialState
    runtimeInit := nn.IndexedModel.initializationPlan model
    requiresGrad := model.requiresGrad
    validate := model.validate
    validateDataInputs := fun
      | .cons tokens (.cons _targets .nil) => nn.IndexedModel.validateInput model tokens
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        TorchLean.CurriedRef.curry
          (Ref := Ops.Ref (m := m) (α := α))
          (ss := model.stateShapes ++ [cfg.tokenShape [batch]])
          (β := TorchLean.CurriedRef
            (fun s => Ops.DataRef
              (m := m) (α := α) (Fin cfg.vocabularySize) s)
            [cfg.tokenShape [batch], cfg.tokenShape [batch]]
            (m (Ops.Ref (m := m) (α := α) [])))
          (fun arguments => fun tokens => fun targets => (do
            let (state, .cons rowWeights .nil) := Pack.split
              (ss₁ := model.stateShapes) (ss₂ := [cfg.tokenShape [batch]]) arguments
            let forward := TorchLean.CurriedRef.uncurry
              (Ref := Ops.Ref (m := m) (α := α))
              (ss := model.stateShapes)
              (β := TorchLean.CurriedRef
                (fun s => Ops.DataRef
                  (m := m) (α := α) (Fin cfg.vocabularySize) s)
                [cfg.tokenShape [batch]]
                (m (Ops.Ref (m := m) (α := α) (cfg.vocabularyShape [batch]))))
              (nn.IndexedModel.program model mode (α := α) (m := m)) state
            let logits ← forward tokens
            let axis : Fin (cfg.vocabularyShape [batch]).rank :=
              ⟨(cfg.tokenShape [batch]).rank, by
                simp [Shape.rank, nn.CausalTransformer.Config.tokenShape,
                  nn.CausalTransformer.Config.vocabularyShape]⟩
            let logProbs ← Ops.apply (m := m) (α := α) (.logSoftmax axis) (logits, ())
            let losses ← Loss.nllUnreduced (m := m) (α := α)
              (leading := cfg.tokenShape [batch]) (trailing := [])
              (classes := cfg.vocabularySize) axis.val rfl logProbs targets
            let weighted ← Ops.apply (m := m) (α := α) (.mul (s := cfg.tokenShape [batch]))
              (losses, rowWeights, ())
            Loss.reduce (m := m) (α := α) (s := cfg.tokenShape [batch]) weighted .sum :
              m (Ops.Ref (m := m) (α := α) []))) }

end TorchLeanGPT
