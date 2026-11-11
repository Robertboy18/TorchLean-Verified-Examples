/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import KimiK3.Sequence.Scan
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Softmax

/-!
# Gated Multi-head Latent Attention

Each cache entry stores a compressed key/value vector and an unrotated shared key. Heads
reconstruct their keys and values from those entries. The current token is appended before
attention. `step` uses a growing list cache for the causal proofs, and `stepFixed` uses packed
tensors for the fixed-context graph. Each is proved against its own consumer, and no theorem
relates the two caches.

Reference: Kimi K3 technical report, Sections 2.1--2.2, pp. 4--6, Eqs. 1--10.
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open TorchLean
open FloatLib.Numerics (MathFunctions)
open Tensor

/-- One cached MLA token: a normalized KV latent and a shared, unrotated key component. -/
structure GatedMLA.Entry (α : Type) [Storage α] (kvLatentDim sharedKeyDim : Nat) where
  /-- Shared normalized KV latent from which every head reconstructs keys and values. -/
  latent : Tensor α [kvLatentDim]
  /-- Shared unrotated key coordinates retained for attention scoring. -/
  sharedKey : Tensor α [sharedKeyDim]

/-- Gated MLA parameters.  K3 uses NoPE attention, so there are no positional-encoding weights. -/
structure GatedMLA (α : Type) [Storage α]
    (modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat) where
  /-- Low-rank query projection. -/
  queryDown : Tensor α [modelDim, queryLatentDim]
  /-- RMSNorm scale for the compressed query. -/
  queryNormScale : Tensor α [queryLatentDim]
  /-- Shared compressed key/value projection. -/
  kvDown : Tensor α [modelDim, kvLatentDim]
  /-- RMSNorm scale for the compressed key/value latent. -/
  kvNormScale : Tensor α [kvLatentDim]
  /-- Projection producing the unrotated shared key. -/
  sharedKeyDown : Tensor α [modelDim, sharedKeyDim]
  /-- Content-query up-projection, with one matrix per head. -/
  queryContentUp : Tensor α [heads, queryLatentDim, contentKeyDim]
  /-- Shared-query up-projection, with one matrix per head. -/
  querySharedUp : Tensor α [heads, queryLatentDim, sharedKeyDim]
  /-- Content-key up-projection, with one matrix per head. -/
  keyUp : Tensor α [heads, kvLatentDim, contentKeyDim]
  /-- Value up-projection, with one matrix per head. -/
  valueUp : Tensor α [heads, kvLatentDim, valueDim]
  /-- Full-rank per-head output gate. -/
  gateWeight : Tensor α [modelDim, heads, valueDim]
  /-- Per-head value projection back to model width. -/
  outputWeight : Tensor α [heads, valueDim, modelDim]

namespace GatedMLA

variable {α : Type} [Storage α] [Context α]
variable {modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat}

/-- A list of cache entries, each holding the two shared representations every head is
reconstructed from. -/
abbrev Cache (α : Type) [Storage α] (kvLatentDim sharedKeyDim : Nat) :=
  List (Entry α kvLatentDim sharedKeyDim)

/-- Normalized global attention for one head over its compressed cache. -/
def attend
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (head : Fin heads)
    (queryLatent : Tensor α [queryLatentDim])
    (cache : Cache α kvLatentDim sharedKeyDim) : Tensor α [valueDim] :=
  let queryContent := Tensor.vecmat queryLatent (Tensor.unstack layer.queryContentUp head)
  let queryShared := Tensor.vecmat queryLatent (Tensor.unstack layer.querySharedUp head)
  let score := fun entry =>
    MathFunctions.exp ((Tensor.dot queryContent
      (Tensor.vecmat entry.latent (Tensor.unstack layer.keyUp head)) +
      Tensor.dot queryShared entry.sharedKey) /
      MathFunctions.sqrt (contentKeyDim + sharedKeyDim : α))
  let denominator := cache.foldl (fun total entry => total + score entry) 0
  cache.foldl (fun output entry =>
    let weight := score entry / denominator
    output + Tensor.map (fun value => weight * value)
      (Tensor.vecmat entry.latent (Tensor.unstack layer.valueUp head)))
    (Tensor.zeros [valueDim])

/-- One complete Gated MLA step at a fixed context length.

Here the caches are rank-two tensors and all heads run in one batched calculation, which is the form
the typed executable graph is proved against.  Returns both caches with the current token
appended, together with the projected model output.
-/
def stepFixed (pastTokens : Nat)
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (pastLatentCache : Tensor α [pastTokens, kvLatentDim])
    (pastSharedKeyCache : Tensor α [pastTokens, sharedKeyDim])
    (x : Tensor α [modelDim]) (scoreScale : α) :
    Tensor α [pastTokens + 1, kvLatentDim] ×
      Tensor α [pastTokens + 1, sharedKeyDim] ×
      Tensor α [modelDim] :=
  let queryProjected := Tensor.vecmat x layer.queryDown
  let queryLatent := RMSNorm.scale queryProjected layer.queryNormScale
  let kvProjected := Tensor.vecmat x layer.kvDown
  let currentKV := RMSNorm.scale kvProjected layer.kvNormScale
  let currentShared := Tensor.vecmat x layer.sharedKeyDown
  let currentKVRow : Tensor α [1, kvLatentDim] :=
    Tensor.reshape currentKV _ (by simp [Shape.size])
  let currentSharedRow : Tensor α [1, sharedKeyDim] :=
    Tensor.reshape currentShared _ (by simp [Shape.size])
  let latentCache := Tensor.concat pastLatentCache currentKVRow
  let sharedKeyCache := Tensor.concat pastSharedKeyCache currentSharedRow
  let queryRow : Tensor α [1, queryLatentDim] := Tensor.dim fun _ => queryLatent
  let queryBatch := Tensor.repeatLeading heads queryRow
  let latentBatch := Tensor.repeatLeading heads latentCache
  let sharedBatch := Tensor.repeatLeading heads sharedKeyCache
  let sameBatch := Shape.CanBroadcastTo.refl [heads]
  let queryContent := Tensor.batchMatmul sameBatch sameBatch queryBatch layer.queryContentUp
  let queryShared := Tensor.batchMatmul sameBatch sameBatch queryBatch layer.querySharedUp
  let keys := Tensor.batchMatmul sameBatch sameBatch latentBatch layer.keyUp
  let values := Tensor.batchMatmul sameBatch sameBatch latentBatch layer.valueUp
  let keysTransposed := keys.permute (Shape.adjacentSwap [heads, pastTokens + 1, contentKeyDim] 1)
    [heads, contentKeyDim, pastTokens + 1]
    (Shape.permute_adjacentSwap [heads, pastTokens + 1, contentKeyDim] 1).symm
  let sharedTransposed := sharedBatch.permute (Shape.adjacentSwap [heads, pastTokens + 1, sharedKeyDim] 1)
    [heads, sharedKeyDim, pastTokens + 1]
    (Shape.permute_adjacentSwap [heads, pastTokens + 1, sharedKeyDim] 1).symm
  let contentScores := Tensor.batchMatmul sameBatch sameBatch queryContent keysTransposed
  let sharedScores := Tensor.batchMatmul sameBatch sameBatch queryShared sharedTransposed
  let scoreShape : Shape := [heads, 1, pastTokens + 1]
  let scores := Tensor.zipWith (· * ·) (Tensor.full scoreShape scoreScale)
    (Tensor.zipWith (· + ·) contentScores sharedScores)
  let headOutput3 := Tensor.batchMatmul sameBatch sameBatch
    (Tensor.softmax scores ⟨2, by simp⟩) values
  let headOutput : Tensor α [heads, valueDim] :=
    Tensor.reshape headOutput3 _ (by simp [Shape.size])
  let gateMatrix : Tensor α [modelDim, heads * valueDim] :=
    Tensor.reshape layer.gateWeight _ (by simp [Shape.size])
  let gateFlat := Tensor.vecmat x gateMatrix
  let gate : Tensor α [heads, valueDim] :=
    Tensor.sigmoid (Tensor.reshape gateFlat _ (by simp [Shape.size]))
  let gatedHeads := Tensor.zipWith (· * ·) gate headOutput
  let gatedFlat : Tensor α [heads * valueDim] :=
    Tensor.reshape gatedHeads _ (by simp [Shape.size])
  let outputMatrix : Tensor α [heads * valueDim, modelDim] :=
    Tensor.reshape layer.outputWeight _ (by simp [Shape.size, Nat.mul_assoc])
  (latentCache, sharedKeyCache, Tensor.vecmat gatedFlat outputMatrix)

/-- One causal MLA token step, which appends the current entry before evaluating attention. -/
def step
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim) (x : Tensor α [modelDim]) :
    Cache α kvLatentDim sharedKeyDim × Tensor α [modelDim] :=
  let queryLatent := RMSNorm.scale (Tensor.vecmat x layer.queryDown) layer.queryNormScale
  let currentKV := RMSNorm.scale (Tensor.vecmat x layer.kvDown) layer.kvNormScale
  let current : Entry α kvLatentDim sharedKeyDim :=
    { latent := currentKV, sharedKey := Tensor.vecmat x layer.sharedKeyDown }
  let cache' := cache ++ [current]
  let headOutput := fun h => layer.attend h queryLatent cache'
  let gate := fun h channel => Context.sigmoid <|
    (List.finRange modelDim).foldl (fun total input =>
      total + Tensor.getScalar x input * Tensor.get2 (Tensor.unstack layer.gateWeight input) h channel) 0
  let output := Tensor.dim (fun feature => Tensor.scalar <|
    (List.finRange heads).foldl (fun acrossHeads h =>
      (List.finRange valueDim).foldl (fun total channel =>
        total + gate h channel * Tensor.getScalar (headOutput h) channel *
          Tensor.get2 (Tensor.unstack layer.outputWeight h) channel feature) acrossHeads) 0)
  (cache', output)

/-- One MLA step appends one entry to the streaming cache. -/
theorem step_cache_length
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim) (x : Tensor α [modelDim]) :
    (layer.step cache x).1.length = cache.length + 1 := by
  simp [step]

/-- An MLA token step keeps the old cache as a prefix. -/
theorem step_cache_prefix
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim) (x : Tensor α [modelDim]) :
    cache <+: (layer.step cache x).1 := by
  simp [step]

/-- Running MLA over a token list grows the compressed cache by the list's length. -/
theorem run_cache_length
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim)
    (tokens : List (Tensor α [modelDim])) :
    (CausalScan.run layer.step cache tokens).1.length = cache.length + tokens.length := by
  induction tokens generalizing cache with
  | nil => simp
  | cons token rest ih =>
      simp only [CausalScan.run]
      rw [ih]
      rw [step_cache_length]
      simp [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

/-- Chunking a Gated MLA causal pass preserves its cache and outputs. -/
theorem run_append
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim)
    (left right : List (Tensor α [modelDim])) :
    CausalScan.run layer.step cache (left ++ right) =
      let first := CausalScan.run layer.step cache left
      let second := CausalScan.run layer.step first.1 right
      (second.1, first.2 ++ second.2) :=
  CausalScan.run_append layer.step cache left right

end GatedMLA

end KimiK3
