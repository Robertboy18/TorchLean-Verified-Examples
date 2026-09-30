/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Buffer
public import NN.Runtime.Autograd.Engine.LibTorch.Kernels

/-!
# Native storage for incremental decoding

The mathematical cache lives in `CachedDecode.Semantics`. This module contains only the foreign
interface used by the executable decoder:

* a persistent key/value table;
* one-token causal attention over the populated prefix;
* one-row LayerNorm and GELU through TorchLean's LibTorch operations.

The LibTorch adapter lives under `week-03-gpt-training/csrc/cached_decode`. It owns persistent
ATen tensors and uses LibTorch for attention; it does not implement CUDA kernels. This is an
executable boundary, not part of the Lean proof. Builds without LibTorch report cached decoding
as unavailable. `CachedDecode.Check` compares this path with the ordinary TorchLean model on the
same checkpoint.
-/

@[expose] public section

namespace TorchLeanGPT
namespace CachedDecode
namespace Native

open Runtime.Autograd.LibTorch

/-- Opaque, reference-counted handle to the native key/value storage. -/
opaque CacheImpl : NonemptyType

/-- Runtime type of a native key/value cache. -/
def Cache : Type := CacheImpl.val

instance : Nonempty Cache := CacheImpl.property

/-- Allocate storage for every layer, head, context position, and head coordinate. -/
@[extern "torchlean_gpt_kv_cache_create"]
opaque create (layers heads capacity headDim : UInt32) : IO Cache

/-- Clear a cache before decoding a new prefix. -/
@[extern "torchlean_gpt_kv_cache_reset"]
opaque reset (cache : @& Cache) : IO Unit

/-- Release native cache storage before the Lean handle itself is collected. -/
@[extern "torchlean_gpt_kv_cache_close"]
opaque close (cache : @& Cache) : IO Unit

/--
Append one key/value row and evaluate causal attention for one query.

All three input buffers have length `heads * headDim`; the result has the same layout. The native
implementation checks the layer, position, buffer lengths, and device before computing attention.
-/
@[extern "torchlean_gpt_kv_cache_attention"]
opaque attention
    (cache : @& Cache) (query key value : @& Buffer)
    (layer position : UInt32) : IO Buffer

/-- LayerNorm over one width-sized row, including learned scale and shift. -/
def layerNorm
    (input gamma beta : Buffer) (width : UInt32) (epsilon : Float) : IO Buffer := do
  if width == 0 || (← Buffer.sizeIO input) != width || (← Buffer.sizeIO gamma) != width ||
      (← Buffer.sizeIO beta) != width then
    throw <| IO.userError "cached decoder: LayerNorm width mismatch"
  let (output, normalized, inverseStd) ← IO.lazyPure fun _ =>
    Buffer.layerNormFwd input gamma beta 1 width (1 / Float.ofNat width.toNat) epsilon
  let _ ← Buffer.releaseIO normalized
  let _ ← Buffer.releaseIO inverseStd
  pure output

/-- Tanh-approximate GELU over a contiguous vector. -/
def gelu (input : Buffer) (count : UInt32) : IO Buffer := do
  if (← Buffer.sizeIO input) != count then
    throw <| IO.userError "cached decoder: GELU size mismatch"
  IO.lazyPure fun _ => Buffer.gelu input

end Native
end CachedDecode
end TorchLeanGPT
