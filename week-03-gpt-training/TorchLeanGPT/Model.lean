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
  { seqLen := cfg.context
    vocab := cfg.vocab
    numHeads := cfg.heads
    headDim := cfg.width / cfg.heads
    ffnHidden := 4 * cfg.width
    layers := cfg.layers
    activation := .gelu
    dropout? := if cfg.dropout == 0.0 then none else some cfg.dropout
    normFirst := true
    attentionOutputBias := true
    parameterInit? := some (.normal 0.0 0.02)
    residualProjectionInit? := some (.normal 0.0 residualStd) }

end ModelConfig

/-- Flat token-id input used by TorchLean's dynamic minibatch path. -/
abbrev tokenShape (cfg : nn.models.CausalTransformer.Config) (batch : Nat) : List Nat :=
  nn.models.CausalTransformer.tokenShape cfg [batch]

/-- Vocabulary logits for every batch row and context position. -/
abbrev logitShape (cfg : nn.models.CausalTransformer.Config) (batch : Nat) : List Nat :=
  nn.models.CausalTransformer.vocabularyShape cfg [batch]

/--
Build the model through TorchLean's public API.

The returned body starts after token lookup and ends before the tied vocabulary projection.
TorchLean's tied-token module owns the shared embedding table and uses this body for both training
and evaluation.
-/
def buildModel
    (cfg : nn.models.CausalTransformer.Config)
    (batch : Nat)
    (hContext : cfg.seqLen ≠ 0)
    (hWidth : cfg.dModel ≠ 0) :
    nn.Builder (nn.Sequential
      (nn.models.CausalTransformer.embeddingShape cfg [batch])
      (nn.models.CausalTransformer.embeddingShape cfg [batch])) :=
  nn.models.CausalTransformer.hidden cfg [batch] hContext hWidth

/--
Tied-token causal-language-model loss with one floating-point weight per prediction row.

Token ids and targets remain discrete `Fin cfg.vocab` tensors. Only the row weights use the model
scalar type, which lets instruction tuning mask prompt tokens without passing token ids through
floating point. Active weights are normalized by the data loader, so this definition returns their
weighted mean cross entropy.
-/
def weightedTiedTokenScalarModuleDefWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocab]
    (batch : Nat)
    (body : nn.Sequential
      (nn.models.CausalTransformer.embeddingShape cfg [batch])
      (nn.models.CausalTransformer.embeddingShape cfg [batch])) :
    Module.ObjectiveDef (Fin cfg.vocab)
      (nn.models.CausalTransformer.Tied.stateShapes cfg body)
      [tokenShape cfg batch] [tokenShape cfg batch, tokenShape cfg batch] :=
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  { initState := .cons
      (_root_.Runtime.Autograd.Torch.Init.tensor embeddingInit (seed := 0)) (nn.initState body)
    runtimeInit :=
      match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? body with
      | some bodyPlan => some (.cons
          (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
            embeddingInit 0) bodyPlan)
      | none => none
    requiresGrad := #[true] ++ nn.requiresGrad body
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
          (ss := nn.models.CausalTransformer.Tied.stateShapes cfg body ++
            [Shape.ofList (tokenShape cfg batch)])
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.DataRef
              (m := m) (α := α) (Fin cfg.vocab) s)
            [tokenShape cfg batch, tokenShape cfg batch]
            (m (_root_.TorchLean.Runtime.ValueRef
              (m := m) (α := α) .scalar)))
          (fun args => fun tokens => fun targets => (do
            let (params, rowWeights) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
                (ss := nn.models.CausalTransformer.Tied.stateShapes cfg body)
                (τ := tokenShape cfg batch) args
            let logits ← nn.models.CausalTransformer.Tied.forward
              (m := m) (α := α) mode cfg body params tokens
            let logitsIndexed : _root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
                ((Shape.ofList (tokenShape cfg batch)).concat
                  (Shape.ofList [cfg.vocab])) := by
              simpa [logitShape, tokenShape, nn.models.CausalTransformer.vocabularyShape,
                nn.models.CausalTransformer.tokenShape, Shape.ofList_append] using logits
            let targetsIndexed : _root_.Runtime.Autograd.Torch.DataRef
                (m := m) (α := α) (Fin cfg.vocab)
                ((Shape.ofList (tokenShape cfg batch)).concat .scalar) := by
              simpa using targets
            let weightsIndexed : _root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
                ((Shape.ofList (tokenShape cfg batch)).concat .scalar) := by
              simpa using rowWeights
            _root_.TorchLean.Loss.crossEntropyWeighted
              (m := m) (α := α)
              (leading := Shape.ofList (tokenShape cfg batch)) (trailing := .scalar)
              (classes := cfg.vocab) (Shape.ofList (tokenShape cfg batch)).rank rfl
              logitsIndexed targetsIndexed weightsIndexed :
              m (_root_.TorchLean.Runtime.ValueRef
                (m := m) (α := α) .scalar))) }

/-- Training-mode weighted objective used by the instruction-tuning run. -/
def weightedTiedTokenScalarModuleDef
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocab]
    (batch : Nat)
    (body : nn.Sequential
      (nn.models.CausalTransformer.embeddingShape cfg [batch])
      (nn.models.CausalTransformer.embeddingShape cfg [batch])) :
    Module.ObjectiveDef (Fin cfg.vocab)
      (nn.models.CausalTransformer.Tied.stateShapes cfg body)
      [tokenShape cfg batch] [tokenShape cfg batch, tokenShape cfg batch] :=
  weightedTiedTokenScalarModuleDefWithMode .train cfg batch body

end TorchLeanGPT
