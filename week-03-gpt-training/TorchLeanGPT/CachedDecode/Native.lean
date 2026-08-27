/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Kernels

/-!
# Native storage for incremental decoding

The mathematical cache lives in `CachedDecode.Semantics`. This module contains only the foreign
interface used by the executable decoder:

* a persistent key/value table;
* one-token causal attention over the populated prefix;
* one-row LayerNorm and GELU kernels.

The CUDA implementation and its host-memory parity stub live under
`week-03-gpt-training/csrc/cached_decode`. They are an executable boundary, not part of the Lean
proof. `CachedDecode.Check` compares this path with the ordinary TorchLean model on the same
checkpoint.
-/

@[expose] public section

namespace TorchLeanGPT
namespace CachedDecode
namespace Native

open Runtime.Autograd.Cuda

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
implementation checks the layer, position, and buffer lengths before launching a kernel.
-/
@[extern "torchlean_gpt_kv_cache_attention"]
opaque attention
    (cache : @& Cache) (query key value : @& Buffer)
    (layer position : UInt32) : IO Buffer

/-- LayerNorm over one width-sized row, including learned scale and shift. -/
@[extern "torchlean_gpt_layer_norm"]
opaque layerNorm
    (input gamma beta : @& Buffer) (width : UInt32) (epsilon : Float) : IO Buffer

/-- Tanh-approximate GELU over a contiguous vector. -/
@[extern "torchlean_gpt_gelu"]
opaque gelu (input : @& Buffer) (count : UInt32) : IO Buffer

end Native
end CachedDecode
end TorchLeanGPT
