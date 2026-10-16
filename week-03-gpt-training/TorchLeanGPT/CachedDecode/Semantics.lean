/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import Mathlib.Data.List.Basic

/-!
# Incremental causal decoding

Autoregressive decoding repeatedly asks for the output at the newest token. Recomputing the whole
prefix is a valid reference implementation, but it repeats every earlier key and value projection.
A key-value cache stores those projections and appends one entry per token.

This file describes that algorithm without fixing a tensor representation, scalar type, attention
kernel, or execution device. The abstraction is small:

* `Kernel.query`, `Kernel.key`, and `Kernel.value` project one token state
* `Kernel.attend` consumes the current query and the complete key/value prefix
* `Cache` stores the projected prefix
* `step` appends one token and computes its causal output
* `recompute` is the slower reference path that rebuilds every prefix from tokens

`CachedDecode.Correctness` proves that both paths return the same output stream whenever the
initial cache represents the supplied prefix. The native decoder in `Runtime.lean` does not use
these definitions.
-/

@[expose] public section

namespace TorchLeanGPT
namespace CachedDecode

/--
Parts of one causal decoder layer that incremental attention needs.

`Token` may be a hidden-state vector, and `Query`, `Key`, `Value`, and `Output` typed tensors. With
the types left independent, one cache theorem covers single-head and multi-head attention in exact
or floating-point arithmetic.
-/
structure Kernel (Token Query Key Value Output : Type) where
  query : Token → Query
  key : Token → Key
  value : Token → Value
  attend : Query → List Key → List Value → Output

/-- Projected keys and values for an already processed token prefix. -/
structure Cache (Key Value : Type) where
  keys : List Key
  values : List Value
  deriving Repr

namespace Cache

/-- The empty cache used before the first token. -/
def empty : Cache Key Value :=
  { keys := [], values := [] }

/-- A cache with as many keys as values. -/
def WellFormed (cache : Cache Key Value) : Prop :=
  cache.keys.length = cache.values.length

/--
`cache` holds the key and value projections of `history`, in token order.

This is the cache invariant. It says more than matching lengths, since it rules out stale entries,
swapped layers, and keys or values computed from a different prefix.
-/
def Represents
    (kernel : Kernel Token Query Key Value Output)
    (history : List Token) (cache : Cache Key Value) : Prop :=
  cache.keys = history.map kernel.key ∧
    cache.values = history.map kernel.value

/-- Number of processed tokens represented by a well-formed cache. -/
def length (cache : Cache Key Value) : Nat :=
  cache.keys.length

end Cache

/-- Result of one incremental decoder step. -/
structure StepResult (Key Value Output : Type) where
  cache : Cache Key Value
  output : Output
  deriving Repr

/--
Append one token's key and value, then attend over the resulting causal prefix.

The current token is visible to itself. Future tokens cannot appear because they have not yet been
appended.
-/
def step
    (kernel : Kernel Token Query Key Value Output)
    (cache : Cache Key Value) (token : Token) :
    StepResult Key Value Output :=
  let keys := cache.keys ++ [kernel.key token]
  let values := cache.values ++ [kernel.value token]
  { cache := { keys, values }
    output := kernel.attend (kernel.query token) keys values }

/--
Reference output for `token`, computed by projecting the complete prefix again.

It is slow on purpose, because it serves as the comparison target for the incremental decoder.
-/
def recomputeOne
    (kernel : Kernel Token Query Key Value Output)
    (history : List Token) (token : Token) : Output :=
  let visible := history ++ [token]
  kernel.attend (kernel.query token)
    (visible.map kernel.key) (visible.map kernel.value)

/--
Reference autoregressive outputs for a suffix following the prefix `history`.

Each recursive call extends the visible prefix by one token.
-/
def recompute
    (kernel : Kernel Token Query Key Value Output) :
    List Token → List Token → List Output
  | _history, [] => []
  | history, token :: rest =>
      recomputeOne kernel history token ::
        recompute kernel (history ++ [token]) rest

/--
Run the incremental decoder over a suffix, returning the final cache and each causal output.
-/
def run
    (kernel : Kernel Token Query Key Value Output) :
    Cache Key Value → List Token → Cache Key Value × List Output
  | cache, [] => (cache, [])
  | cache, token :: rest =>
      let current := step kernel cache token
      let tail := run kernel current.cache rest
      (tail.1, current.output :: tail.2)

end CachedDecode
end TorchLeanGPT
