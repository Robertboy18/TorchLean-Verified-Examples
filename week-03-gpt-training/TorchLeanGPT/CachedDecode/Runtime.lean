/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Model
public import TorchLeanGPT.CachedDecode.Native

/-!
# Checkpoint-compatible cached decoder

This module reads the live CUDA parameter buffers owned by a TorchLean GPT module. It checks every
parameter position and shape before constructing a decoder, transposes the three matrices whose
stored layout is inconvenient for one-row multiplication, and then evaluates one token at a time.

The decoder is deliberately tied to the tied-token model assembled around
`TorchLeanGPT.buildModel`: pre-normalized blocks, bias-free Q/K/V projections, a biased attention
output projection, tanh-approximate GELU, and a shared token embedding/output matrix. A checkpoint
with any other shape or parameter count is rejected before execution.

Shape and layout validation protect the foreign-function boundary, but they do not prove that the
CUDA kernels refine `CachedDecode.Semantics`. That mathematical model and its cache-equivalence
theorems are backend independent. `CachedDecode.Check` is the executable comparison currently used
to audit this native realization.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace CachedDecode
namespace Runtime

open _root_.Runtime.Autograd.Cuda

/-- A borrowed parameter buffer together with the shape checked by TorchLean. -/
structure ParameterBuffer where
  shape : Shape
  buffer : Buffer

/--
Return a parameter's CUDA mirror, uploading its host value once when necessary.

The mirror remains owned by the TorchLean module. The cached decoder borrows it and never releases
it.
-/
def parameterBuffer {shape : Shape}
    (parameter : _root_.Runtime.Autograd.Torch.Param Float shape) :
    IO ParameterBuffer := do
  let any ←
    match ← parameter.cudaValue.get with
    | some value =>
        if value.s = shape then
          pure value
        else
          throw <| IO.userError <|
            "cached decoder: parameter CUDA shape mismatch (expected " ++
              Spec.Shape.pretty shape ++ ", got " ++ Spec.Shape.pretty value.s ++ ")"
    | none =>
        let host ← parameter.value.get
        let uploaded ←
          _root_.Runtime.Autograd.Torch.Internal.CudaBridge.toAnyBuffer
            (α := Float) (s := shape) host
        parameter.cudaValue.set (some uploaded)
        parameter.hostCurrent.set true
        pure uploaded
  let actualSize ← Buffer.sizeIO any.buf
  let expectedSize := UInt32.ofNat (Shape.size shape)
  if expectedSize.toNat != Shape.size shape then
    throw <| IO.userError
      "cached decoder: parameter is too large for the native UInt32 buffer ABI"
  if actualSize != expectedSize then
    throw <| IO.userError <|
      "cached decoder: parameter buffer length mismatch for " ++ Spec.Shape.pretty shape
  pure { shape, buffer := any.buf }

/-- Flatten TorchLean's heterogeneous parameter list without discarding shape metadata. -/
def collectParameterBuffers :
    {shapes : List Shape} →
      _root_.Runtime.Autograd.Torch.ParamList Float shapes →
      IO (Array ParameterBuffer)
  | [], .nil => pure #[]
  | _ :: _, .cons parameter parameters => do
      let head ← parameterBuffer parameter
      let tail ← collectParameterBuffers parameters
      pure <| #[head] ++ tail

/-- Checked conversion used for every dimension passed through the native ABI. -/
def natToUInt32 (label : String) (value : Nat) : IO UInt32 := do
  let result := UInt32.ofNat value
  if result.toNat = value then
    pure result
  else
    throw <| IO.userError s!"cached decoder: {label} does not fit in UInt32"

/-- Diagnostic labels and shapes for the tied-token model's fixed parameter order. -/
def expectedLayout
    (cfg : nn.models.CausalTransformer.Config) : Array (String × Shape) := Id.run do
  let width := cfg.dModel
  let mut result := #[
    ("token_embedding", ([cfg.vocab, width] : Shape)),
    ("position_embedding", ([cfg.seqLen, width] : Shape))
  ]
  for layer in [0:cfg.layers] do
    let stem := s!"blocks.{layer}"
    result := result ++ #[
      (stem ++ ".norm1.gamma", ([width] : Shape)),
      (stem ++ ".norm1.beta", ([width] : Shape)),
      (stem ++ ".attention.wq", ([width, width] : Shape)),
      (stem ++ ".attention.wk", ([width, width] : Shape)),
      (stem ++ ".attention.wv", ([width, width] : Shape)),
      (stem ++ ".attention.wo", ([width, width] : Shape)),
      (stem ++ ".attention.bo", ([width] : Shape)),
      (stem ++ ".norm2.gamma", ([width] : Shape)),
      (stem ++ ".norm2.beta", ([width] : Shape)),
      (stem ++ ".ffn.w1", ([cfg.ffnHidden, width] : Shape)),
      (stem ++ ".ffn.b1", ([cfg.ffnHidden] : Shape)),
      (stem ++ ".ffn.w2", ([width, cfg.ffnHidden] : Shape)),
      (stem ++ ".ffn.b2", ([width] : Shape))
    ]
  result := result ++ #[
    ("final_norm.gamma", ([width] : Shape)),
    ("final_norm.beta", ([width] : Shape))
  ]
  return result

/-- Validate every parameter position and shape before interpreting the flat pack. -/
def validateLayout
    (cfg : nn.models.CausalTransformer.Config)
    (parameters : Array ParameterBuffer) : IO Unit := do
  let expected := expectedLayout cfg
  if parameters.size != expected.size then
    throw <| IO.userError <|
      s!"cached decoder: parameter count mismatch (got {parameters.size}, expected {expected.size})"
  for index in [0:expected.size] do
    match expected[index]?, parameters[index]? with
    | some (name, shape), some actual =>
        if actual.shape != shape then
          throw <| IO.userError <|
            s!"cached decoder: {name} has shape {Spec.Shape.pretty actual.shape}; " ++
              s!"expected {Spec.Shape.pretty shape}"
    | _, _ =>
        throw <| IO.userError
          s!"cached decoder: internal layout index {index} disappeared during validation"

/-- Retrieve a parameter after `validateLayout`, retaining a defensive bounds check. -/
def parameterAt
    (parameters : Array ParameterBuffer) (index : Nat) : IO ParameterBuffer :=
  match parameters[index]? with
  | some parameter => pure parameter
  | none =>
      throw <| IO.userError s!"cached decoder: missing parameter at index {index}"

/-- Runtime buffers for one pre-normalized Transformer block. -/
structure Layer where
  norm1Gamma : Buffer
  norm1Beta : Buffer
  queryWeight : Buffer
  keyWeight : Buffer
  valueWeight : Buffer
  outputWeight : Buffer
  outputBias : Buffer
  norm2Gamma : Buffer
  norm2Beta : Buffer
  ffnInputWeight : Buffer
  ffnInputBias : Buffer
  ffnOutputWeight : Buffer
  ffnOutputBias : Buffer

/--
State needed for one incremental decoding stream.

Parameter buffers are borrowed from the TorchLean module. `nativeCache` is owned by the decoder.
The `closed` flag makes repeated cleanup harmless and rejects later use.
-/
structure Decoder where
  config : nn.models.CausalTransformer.Config
  tokenEmbedding : Buffer
  positionEmbedding : Buffer
  tokenOutputWeight : Buffer
  layers : Array Layer
  finalNormGamma : Buffer
  finalNormBeta : Buffer
  nativeCache : Native.Cache
  nextPosition : IO.Ref Nat
  closed : IO.Ref Bool
  widthU32 : UInt32
  hiddenU32 : UInt32
  vocabU32 : UInt32
  contextU32 : UInt32

/-- Obtain a borrowed buffer from a validated flat parameter pack. -/
def bufferAt (parameters : Array ParameterBuffer) (index : Nat) : IO Buffer := do
  pure (← parameterAt parameters index).buffer

/-- Reject model variants whose parameterization differs from the decoder equations. -/
def validateModelConfig (cfg : nn.models.CausalTransformer.Config) : IO Unit := do
  if cfg.seqLen = 0 || cfg.vocab = 0 || cfg.numHeads = 0 ||
      cfg.headDim = 0 || cfg.layers = 0 || cfg.ffnHidden = 0 then
    throw <| IO.userError "cached decoder: all model dimensions must be positive"
  if cfg.dModel != cfg.numHeads * cfg.headDim then
    throw <| IO.userError "cached decoder: inconsistent attention width"
  if !cfg.normFirst then
    throw <| IO.userError "cached decoder: this executable expects pre-normalized blocks"
  if !cfg.attentionOutputBias then
    throw <| IO.userError "cached decoder: this executable expects attention output biases"
  match cfg.activation with
  | .gelu => pure ()
  | _ =>
      throw <| IO.userError "cached decoder: this executable expects GELU feed-forward blocks"

/--
Construct a decoder from a live TorchLean parameter list.

The model module must remain alive until `close` because most buffers are borrowed from it.
-/
def Decoder.initialize
    (cfg : nn.models.CausalTransformer.Config)
    {parameterShapes : List Shape}
    (parameterList : _root_.Runtime.Autograd.Torch.ParamList Float parameterShapes) :
    IO Decoder := do
  validateModelConfig cfg
  let parameters ← collectParameterBuffers parameterList
  validateLayout cfg parameters

  let widthU32 ← natToUInt32 "model width" cfg.dModel
  let hiddenU32 ← natToUInt32 "feed-forward width" cfg.ffnHidden
  let vocabU32 ← natToUInt32 "vocabulary size" cfg.vocab
  let contextU32 ← natToUInt32 "context length" cfg.seqLen
  let headsU32 ← natToUInt32 "attention head count" cfg.numHeads
  let headDimU32 ← natToUInt32 "attention head width" cfg.headDim
  let layersU32 ← natToUInt32 "layer count" cfg.layers

  let tokenEmbedding ← bufferAt parameters 0
  let positionEmbedding ← bufferAt parameters 1
  let tokenOutputWeight := tokenEmbedding

  let mut layers := #[]
  for layerIndex in [0:cfg.layers] do
    let base := 2 + 13 * layerIndex
    let storedInputWeight ← bufferAt parameters (base + 9)
    let storedOutputWeight ← bufferAt parameters (base + 11)
    let layer : Layer :=
      { norm1Gamma := ← bufferAt parameters base
        norm1Beta := ← bufferAt parameters (base + 1)
        queryWeight := ← bufferAt parameters (base + 2)
        keyWeight := ← bufferAt parameters (base + 3)
        valueWeight := ← bufferAt parameters (base + 4)
        outputWeight := ← bufferAt parameters (base + 5)
        outputBias := ← bufferAt parameters (base + 6)
        norm2Gamma := ← bufferAt parameters (base + 7)
        norm2Beta := ← bufferAt parameters (base + 8)
        ffnInputWeight := storedInputWeight
        ffnInputBias := ← bufferAt parameters (base + 10)
        ffnOutputWeight := storedOutputWeight
        ffnOutputBias := ← bufferAt parameters (base + 12) }
    layers := layers.push layer

  let finalBase := 2 + 13 * cfg.layers
  let finalNormGamma ← bufferAt parameters finalBase
  let finalNormBeta ← bufferAt parameters (finalBase + 1)
  let nativeCache ← Native.create layersU32 headsU32 contextU32 headDimU32
  let nextPosition ← IO.mkRef 0
  let closed ← IO.mkRef false
  pure
    { config := cfg
      tokenEmbedding
      positionEmbedding
      tokenOutputWeight
      layers
      finalNormGamma
      finalNormBeta
      nativeCache
      nextPosition
      closed
      widthU32
      hiddenU32
      vocabU32
      contextU32 }

/-- Ensure an operation is not attempted after `close`. -/
def Decoder.requireOpen (decoder : Decoder) : IO Unit := do
  if ← decoder.closed.get then
    throw <| IO.userError "cached decoder: decoder is closed"

/-- Release a list of owned temporary buffers after an effectful native call has consumed them. -/
def releaseBuffers (buffers : List Buffer) : IO Unit := do
  for buffer in buffers do
    let _ ← Buffer.releaseIO buffer

/-- Multiply one row by a stored affine weight whose layout is `(outputWidth, inputWidth)`. -/
def Decoder.matmul
    (_decoder : Decoder) (input weight : Buffer)
    (inputWidth outputWidth : UInt32) : Buffer :=
  Buffer.bmmRightTranspose input weight 1 1 inputWidth outputWidth

/-- Look up a single row from a row-major table. -/
def gatherRow
    (table : Buffer) (rows columns : UInt32) (row : Nat) : Buffer :=
  Buffer.gatherRows table rows columns #[row] 1

/-- Reset position and native storage before decoding an unrelated prompt. -/
def Decoder.reset (decoder : Decoder) : IO Unit := do
  decoder.requireOpen
  Native.reset decoder.nativeCache
  decoder.nextPosition.set 0

/--
Consume one token and optionally return vocabulary logits for the resulting prefix.

During prompt prefill, callers set `produceLogits := false` except on the final prompt token. This
avoids the expensive tied vocabulary projection when no sample will be drawn from that row.
-/
def Decoder.push
    (decoder : Decoder) (token : Nat) (produceLogits : Bool := true) :
    IO (Option (Array Float)) := do
  decoder.requireOpen
  let position ← decoder.nextPosition.get
  if position >= decoder.config.seqLen then
    throw <| IO.userError <|
      s!"cached decoder: context exhausted at {position} tokens"
  if token >= decoder.config.vocab then
    throw <| IO.userError <|
      s!"cached decoder: token id {token} is outside vocabulary {decoder.config.vocab}"

  let tokenRow :=
    gatherRow decoder.tokenEmbedding decoder.vocabU32 decoder.widthU32 token
  let positionRow :=
    gatherRow decoder.positionEmbedding decoder.contextU32 decoder.widthU32 position
  let hidden0 := Buffer.add tokenRow positionRow
  let mut hidden := Buffer.releaseManyThen #[tokenRow, positionRow] hidden0

  for layerIndex in [0:decoder.layers.size] do
    let layer ←
      match decoder.layers[layerIndex]? with
      | some layer => pure layer
      | none =>
          throw <| IO.userError
            s!"cached decoder: missing layer {layerIndex} during execution"
    let norm1 ← Native.layerNorm hidden layer.norm1Gamma layer.norm1Beta
      decoder.widthU32 1e-6
    let query := decoder.matmul norm1 layer.queryWeight decoder.widthU32 decoder.widthU32
    let key := decoder.matmul norm1 layer.keyWeight decoder.widthU32 decoder.widthU32
    let value := decoder.matmul norm1 layer.valueWeight decoder.widthU32 decoder.widthU32
    let attended ← Native.attention decoder.nativeCache query key value
      (← natToUInt32 "layer index" layerIndex)
      (← natToUInt32 "cache position" position)
    releaseBuffers [norm1, query, key, value]

    let projected0 :=
      decoder.matmul attended layer.outputWeight decoder.widthU32 decoder.widthU32
    let projected := Buffer.releaseThen attended projected0
    let biased0 := Buffer.add projected layer.outputBias
    let biased := Buffer.releaseThen projected biased0
    let afterAttention0 := Buffer.add hidden biased
    let afterAttention := Buffer.releaseManyThen #[hidden, biased] afterAttention0

    let norm2 ← Native.layerNorm afterAttention layer.norm2Gamma layer.norm2Beta
      decoder.widthU32 1e-6
    let expanded0 :=
      decoder.matmul norm2 layer.ffnInputWeight decoder.widthU32 decoder.hiddenU32
    let expanded1 := Buffer.add expanded0 layer.ffnInputBias
    let expanded := Buffer.releaseThen expanded0 expanded1
    let activated ← Native.gelu expanded decoder.hiddenU32
    releaseBuffers [norm2, expanded]
    let contracted0 :=
      decoder.matmul activated layer.ffnOutputWeight decoder.hiddenU32 decoder.widthU32
    let contracted := Buffer.releaseThen activated contracted0
    let shifted0 := Buffer.add contracted layer.ffnOutputBias
    let shifted := Buffer.releaseThen contracted shifted0
    let nextHidden0 := Buffer.add afterAttention shifted
    hidden := Buffer.releaseManyThen #[afterAttention, shifted] nextHidden0

  decoder.nextPosition.set (position + 1)
  if produceLogits then
    let normalized ← Native.layerNorm hidden decoder.finalNormGamma decoder.finalNormBeta
      decoder.widthU32 1e-6
    let _ ← Buffer.releaseIO hidden
    let logits :=
      decoder.matmul normalized decoder.tokenOutputWeight decoder.widthU32 decoder.vocabU32
    let host ← Buffer.toFloatArrayIO logits
    releaseBuffers [normalized, logits]
    pure (some host.data)
  else
    let _ ← Buffer.releaseIO hidden
    pure none

/-- Current number of cached tokens. -/
def Decoder.length (decoder : Decoder) : IO Nat :=
  decoder.nextPosition.get

/--
Reset the decoder, consume a nonempty prefix, and return logits from its final row.

Earlier rows skip the tied vocabulary projection because their logits cannot affect the next
sample.
-/
def Decoder.prefill (decoder : Decoder) (tokens : List Nat) : IO (Array Float) := do
  if tokens.isEmpty then
    throw <| IO.userError "cached decoder: cannot prefill an empty token sequence"
  decoder.reset
  let mut finalLogits : Option (Array Float) := none
  for (token, index) in tokens.zipIdx do
    finalLogits ← decoder.push token (index + 1 = tokens.length)
  match finalLogits with
  | some logits => pure logits
  | none =>
      throw <| IO.userError "cached decoder: final prefill row produced no logits"

/-- Release native cache storage. Borrowed module parameters are left untouched. -/
def Decoder.close (decoder : Decoder) : IO Unit := do
  if ← decoder.closed.get then
    pure ()
  else
    Native.close decoder.nativeCache
    decoder.closed.set true

end Runtime
end CachedDecode
end TorchLeanGPT
