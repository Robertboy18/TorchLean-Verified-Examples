/-
TorchLean runtime versions of the custom upstream layers.

The checkpoint replay in `Replay/UpstreamFloatReplay` answers: "what did the
exported upstream model do on the finite task?"  The definitions below answer a different
question: "which of those custom layers can we run through TorchLean today?"

LeakyReLU and sparsemax fit the current TorchLean primitive surface directly.
Signed-L1-BandNorm is trickier because the upstream layer branches on a hard
support mask; the TorchLean layer below uses a bounded differentiable support
weight.  That makes it suitable for training experiments, but the exact upstream
checkpoint claim still belongs to the finite replay.

The implementation follows `scripts/small/train.py`, especially `LeakyReLU`,
`sparsemax`, `sparsemax_attention_forward`, and `SignedL1BandNorm`.
-/

import NN

open Spec Tensor
open NN.API

namespace VerifiableTransformers.TorchLean.Ops

namespace TL

abbrev Ref (m : Type → Type) (α : Type)
    [Context α] [DecidableEq Shape] [TorchLean.Ops (m := m) (α := α)]
    (s : Shape) : Type :=
  TorchLean.RefTy (m := m) (α := α) s

/-- Zero tensor as a TorchLean ref. -/
def zerosRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (s : Shape) : m (Ref m α s) :=
  TorchLean.const (m := m) (α := α) (s := s) (Tensor.fill (0 : α) s)

/-- Scalar constant as a TorchLean ref. -/
def scalarRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (x : α) : m (Ref m α Shape.scalar) :=
  TorchLean.const (m := m) (α := α) (s := Shape.scalar) (Tensor.scalar x)

/--
TorchLean-native LeakyReLU, implemented from existing differentiable primitives:

`max(x, 0) + slope * min(x, 0)`.

Direct runtime analogue of the upstream `LeakyReLU` replacement for GPT-2's
GELU.
-/
def leakyReluRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {s : Shape} (negativeSlope : α) (x : Ref m α s) : m (Ref m α s) := do
  let z ← zerosRef (m := m) (α := α) s
  let pos ← TorchLean.max (m := m) (α := α) (s := s) x z
  let neg ← TorchLean.min (m := m) (α := α) (s := s) x z
  let negScaled ← TorchLean.scale (m := m) (α := α) (s := s) neg negativeSlope
  TorchLean.add (m := m) (α := α) (s := s) pos negScaled

/--
Insert one scalar into a vector at a fixed coordinate.

The result is built with `scatterAddVec`, so gradients still flow to the scalar
that is inserted.
-/
def scatterScalarIntoVec {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {n : Nat} (base : Ref m α (.dim n .scalar)) (x : Ref m α Shape.scalar)
    (i : Nat) (h : i < n) : m (Ref m α (.dim n .scalar)) :=
  TorchLean.scatterAddVec (m := m) (α := α) (n := n) base x ⟨i, h⟩

/-- Pack a list of scalar refs into a vector. Extra list entries are ignored. -/
def packScalars {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (n : Nat) (xs : List (Ref m α Shape.scalar)) : m (Ref m α (.dim n .scalar)) := do
  let mut out ← zerosRef (m := m) (α := α) (.dim n .scalar)
  for p in List.zip (List.range xs.length) xs do
    let i := p.1
    if h : i < n then
      out ← scatterScalarIntoVec (m := m) (α := α) out p.2 i h
  pure out

/-- Gather the entries of a vector into a Lean list of scalar refs. -/
def unpackScalars {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {n : Nat} (x : Ref m α (.dim n .scalar)) : m (List (Ref m α Shape.scalar)) := do
  let mut out : List (Ref m α Shape.scalar) := []
  for i in [0:n] do
    let xi ← TorchLean.gatherScalarNat (m := m) (α := α) (n := n) x i
    out := out ++ [xi]
  pure out

/--
Insert one scalar ref into a descending list using only `max` and `min`.

A sorting-network style insertion, not a Lean-side branch on tensor values. The
branch-free construction is what makes it a genuine TorchLean program rather
than an external Python helper.
-/
partial def insertDescRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (x : Ref m α Shape.scalar) :
    List (Ref m α Shape.scalar) → m (List (Ref m α Shape.scalar))
  | [] => pure [x]
  | y :: ys => do
      let hi ← TorchLean.max (m := m) (α := α) (s := Shape.scalar) x y
      let lo ← TorchLean.min (m := m) (α := α) (s := Shape.scalar) x y
      let rest ← insertDescRef (m := m) (α := α) lo ys
      pure (hi :: rest)

/-- Sort scalar refs in descending order using a branch-free insertion network. -/
partial def sortDescRefs {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)] :
    List (Ref m α Shape.scalar) → m (List (Ref m α Shape.scalar))
  | [] => pure []
  | x :: xs => do
      let sorted ← sortDescRefs (m := m) (α := α) xs
      insertDescRef (m := m) (α := α) x sorted

/-- Prefix sums of scalar refs. -/
partial def prefixSums {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (acc : Ref m α Shape.scalar) :
    List (Ref m α Shape.scalar) → m (List (Ref m α Shape.scalar))
  | [] => pure []
  | x :: xs => do
      let acc' ← TorchLean.add (m := m) (α := α) (s := Shape.scalar) acc x
      let rest ← prefixSums (m := m) (α := α) acc' xs
      pure (acc' :: rest)

/-- Maximum of scalar refs, using `fallback` for the empty list. -/
partial def maxList {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (fallback : Ref m α Shape.scalar) :
    List (Ref m α Shape.scalar) → m (Ref m α Shape.scalar)
  | [] => pure fallback
  | x :: xs => do
      let rest ← maxList (m := m) (α := α) fallback xs
      TorchLean.max (m := m) (α := α) (s := Shape.scalar) x rest

/--
Sparsemax threshold for a vector.

For sorted values `z_(1) ≥ ... ≥ z_(n)`, sparsemax uses
`tau = max_k ((sum_{j≤k} z_(j) - 1) / k)`.  Then
`sparsemax(z)_i = max(z_i - tau, 0)`.

This avoids a hard support-count indicator in the runtime program; the support
appears implicitly through the final ReLU.
-/
def sparsemaxTau {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (xs : List (Ref m α Shape.scalar)) : m (Ref m α Shape.scalar) := do
  let zero ← scalarRef (m := m) (α := α) (0 : α)
  let one ← scalarRef (m := m) (α := α) (1 : α)
  let sorted ← sortDescRefs (m := m) (α := α) xs
  let sums ← prefixSums (m := m) (α := α) zero sorted
  let mut candidates : List (Ref m α Shape.scalar) := []
  for p in List.zip (List.range sums.length) sums do
    let k : α := (p.1 + 1 : Nat)
    let shifted ← TorchLean.sub (m := m) (α := α) (s := Shape.scalar) p.2 one
    let cand ← TorchLean.scale (m := m) (α := α) (s := Shape.scalar) shifted ((1 : α) / k)
    candidates := candidates ++ [cand]
  maxList (m := m) (α := α) zero candidates

/-- Sparsemax over a vector using the branch-free threshold construction above. -/
def sparsemaxVecRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {n : Nat} (x : Ref m α (.dim n .scalar)) : m (Ref m α (.dim n .scalar)) := do
  let xs ← unpackScalars (m := m) (α := α) x
  let tau ← sparsemaxTau (m := m) (α := α) xs
  let zero ← scalarRef (m := m) (α := α) (0 : α)
  let mut ys : List (Ref m α Shape.scalar) := []
  for xi in xs do
    let shifted ← TorchLean.sub (m := m) (α := α) (s := Shape.scalar) xi tau
    let yi ← TorchLean.max (m := m) (α := α) (s := Shape.scalar) shifted zero
    ys := ys ++ [yi]
  packScalars (m := m) (α := α) n ys

/-- Sum a list of scalar refs. -/
partial def sumList {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    (xs : List (Ref m α Shape.scalar)) : m (Ref m α Shape.scalar) := do
  let zero ← scalarRef (m := m) (α := α) (0 : α)
  let mut acc := zero
  for x in xs do
    acc ← TorchLean.add (m := m) (α := α) (s := Shape.scalar) acc x
  pure acc

/--
Projection of a nonnegative vector onto the L1 ball with radius `radius`.

The construction is branch-free: if the mass is already inside the ball, the
threshold computed below is zero and the result is exactly the input.
-/
def projectNonnegativeL1BallRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {n : Nat} (radius : α) (y : Ref m α (.dim n .scalar)) :
    m (Ref m α (.dim n .scalar)) := do
  let ys ← unpackScalars (m := m) (α := α) y
  let zero ← scalarRef (m := m) (α := α) (0 : α)
  let radiusRef ← scalarRef (m := m) (α := α) radius
  let sorted ← sortDescRefs (m := m) (α := α) ys
  let sums ← prefixSums (m := m) (α := α) zero sorted
  let mut candidates : List (Ref m α Shape.scalar) := []
  for p in List.zip (List.range sums.length) sums do
    let k : α := (p.1 + 1 : Nat)
    let shifted ← TorchLean.sub (m := m) (α := α) (s := Shape.scalar) p.2 radiusRef
    let cand ← TorchLean.scale (m := m) (α := α) (s := Shape.scalar) shifted ((1 : α) / k)
    candidates := candidates ++ [cand]
  let tau ← maxList (m := m) (α := α) zero candidates
  let mut out : List (Ref m α Shape.scalar) := []
  for yi in ys do
    let shifted ← TorchLean.sub (m := m) (α := α) (s := Shape.scalar) yi tau
    let zi ← TorchLean.max (m := m) (α := α) (s := Shape.scalar) shifted zero
    out := out ++ [zi]
  packScalars (m := m) (α := α) n out

/-- Even/odd fallback active mask used by upstream Signed-L1-BandNorm. -/
def parityFallbackTensor (α : Type) [Context α] (n : Nat) (positive : Bool) :
    Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i =>
    let even := i.val % 2 = 0
    Tensor.scalar <|
      if positive then
        if even then (1 : α) else (0 : α)
      else
        if even then (0 : α) else (1 : α))

/-- Bounded support weight used by the TorchLean BandNorm runtime approximation. -/
def positiveSupportRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {s : Shape} (x : Ref m α s) : m (Ref m α s) := do
  let scaled ← TorchLean.scale (m := m) (α := α) (s := s) x (((1000000 : Nat) : α))
  TorchLean.clamp (m := m) (α := α) (s := s) scaled (0 : α) (1 : α)

/--
Additive lift used by the TorchLean Signed-L1-BandNorm runtime layer.

The exact upstream layer branches on whether a projected coordinate is positive.
The current generic TorchLean op surface has no comparator/select primitive, so
this runtime path uses `positiveSupportRef` as a bounded support weight.  Exact
finite traces from Neel's checkpoint are replayed separately in the certificate modules.
-/
def additiveLiftRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {n : Nat} (target : α) (fallback : Tensor α (.dim n .scalar))
    (y : Ref m α (.dim n .scalar)) : m (Ref m α (.dim n .scalar)) := do
  let ys ← unpackScalars (m := m) (α := α) y
  let mass ← sumList (m := m) (α := α) ys
  let targetRef ← scalarRef (m := m) (α := α) target
  let deficitRaw ← TorchLean.sub (m := m) (α := α) (s := Shape.scalar) targetRef mass
  let zero ← scalarRef (m := m) (α := α) (0 : α)
  let one ← scalarRef (m := m) (α := α) (1 : α)
  let deficit ← TorchLean.max (m := m) (α := α) (s := Shape.scalar) deficitRaw zero
  let active ← positiveSupportRef (m := m) (α := α) (s := .dim n .scalar) y
  let activeCount ← sumList (m := m) (α := α) =<< unpackScalars (m := m) (α := α) active
  let fallbackDeficit ← TorchLean.sub (m := m) (α := α) (s := Shape.scalar) one activeCount
  let useFallback ← positiveSupportRef (m := m) (α := α) (s := Shape.scalar) fallbackDeficit
  let useFallbackVec ← TorchLean.broadcastTo (m := m) (α := α)
    (s₁ := Shape.scalar) (s₂ := .dim n .scalar)
    (Shape.CanBroadcastTo.scalar_to_any (.dim n .scalar)) useFallback
  let fallbackRef ← TorchLean.const (m := m) (α := α) (s := .dim n .scalar) fallback
  let fallbackDiff ← TorchLean.sub (m := m) (α := α) (s := .dim n .scalar) fallbackRef active
  let fallbackPatch ← TorchLean.mul (m := m) (α := α) (s := .dim n .scalar) useFallbackVec fallbackDiff
  let activeAdjusted ← TorchLean.add (m := m) (α := α) (s := .dim n .scalar) active fallbackPatch
  let activeAdjustedCount ← sumList (m := m) (α := α) =<<
    unpackScalars (m := m) (α := α) activeAdjusted
  let activeCountClamped ← TorchLean.max (m := m) (α := α) (s := Shape.scalar) activeAdjustedCount one
  let delta ← TorchLean.mul (m := m) (α := α) (s := Shape.scalar) deficit
    =<< TorchLean.inv (m := m) (α := α) (s := Shape.scalar) activeCountClamped
  let deltaVec ← TorchLean.broadcastTo (m := m) (α := α)
    (s₁ := Shape.scalar) (s₂ := .dim n .scalar)
    (Shape.CanBroadcastTo.scalar_to_any (.dim n .scalar)) delta
  let patch ← TorchLean.mul (m := m) (α := α) (s := .dim n .scalar) deltaVec activeAdjusted
  TorchLean.add (m := m) (α := α) (s := .dim n .scalar) y patch

/-- TorchLean runtime approximation to Signed-L1-BandNorm on one hidden vector. -/
def signedL1BandNormVecRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {n : Nat} (gamma beta : Ref m α (.dim n .scalar))
    (x : Ref m α (.dim n .scalar)) : m (Ref m α (.dim n .scalar)) := do
  let xs ← unpackScalars (m := m) (α := α) x
  let sumX ← sumList (m := m) (α := α) xs
  let meanX ← TorchLean.scale (m := m) (α := α) (s := Shape.scalar) sumX
    ((1 : α) / ((n : Nat) : α))
  let meanVec ← TorchLean.broadcastTo (m := m) (α := α)
    (s₁ := Shape.scalar) (s₂ := .dim n .scalar)
    (Shape.CanBroadcastTo.scalar_to_any (.dim n .scalar)) meanX
  let centered ← TorchLean.sub (m := m) (α := α) (s := .dim n .scalar) x meanVec
  let zeroVec ← zerosRef (m := m) (α := α) (.dim n .scalar)
  let negCentered ← TorchLean.scale (m := m) (α := α) (s := .dim n .scalar) centered (-1 : α)
  let pos ← TorchLean.max (m := m) (α := α) (s := .dim n .scalar) centered zeroVec
  let neg ← TorchLean.max (m := m) (α := α) (s := .dim n .scalar) negCentered zeroVec
  let halfLow : α := ((n : Nat) : α) * ((11 : Nat) : α) / ((40 : Nat) : α)
  let halfHigh : α := ((n : Nat) : α) * ((21 : Nat) : α) / ((40 : Nat) : α)
  let posProj ← projectNonnegativeL1BallRef (m := m) (α := α) (n := n) halfHigh pos
  let negProj ← projectNonnegativeL1BallRef (m := m) (α := α) (n := n) halfHigh neg
  let posExpanded ← additiveLiftRef (m := m) (α := α) (n := n) halfLow
    (parityFallbackTensor α n true) posProj
  let negExpanded ← additiveLiftRef (m := m) (α := α) (n := n) halfLow
    (parityFallbackTensor α n false) negProj
  let y ← TorchLean.sub (m := m) (α := α) (s := .dim n .scalar) posExpanded negExpanded
  let ys ← unpackScalars (m := m) (α := α) y
  let sumY ← sumList (m := m) (α := α) ys
  let meanY ← TorchLean.scale (m := m) (α := α) (s := Shape.scalar) sumY
    ((1 : α) / ((n : Nat) : α))
  let meanYVec ← TorchLean.broadcastTo (m := m) (α := α)
    (s₁ := Shape.scalar) (s₂ := .dim n .scalar)
    (Shape.CanBroadcastTo.scalar_to_any (.dim n .scalar)) meanY
  let recentered ← TorchLean.sub (m := m) (α := α) (s := .dim n .scalar) y meanYVec
  let scaled ← TorchLean.mul (m := m) (α := α) (s := .dim n .scalar) recentered gamma
  TorchLean.add (m := m) (α := α) (s := .dim n .scalar) scaled beta

/-- Apply the BandNorm vector runtime layer independently to matrix rows. -/
def signedL1BandNormRowsRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {rows n : Nat} (gamma beta : Ref m α (.dim n .scalar))
    (x : Ref m α (.dim rows (.dim n .scalar))) :
    m (Ref m α (.dim rows (.dim n .scalar))) := do
  let mut out ← zerosRef (m := m) (α := α) (.dim rows (.dim n .scalar))
  for i in [0:rows] do
    if h : i < rows then
      let row ← TorchLean.gatherRow (m := m) (α := α) (rows := rows) (cols := n) x ⟨i, h⟩
      let normed ← signedL1BandNormVecRef (m := m) (α := α) (n := n) gamma beta row
      out ← TorchLean.scatterAddRow (m := m) (α := α) (rows := rows) (cols := n) out normed ⟨i, h⟩
  pure out

/--
Apply sparsemax independently to each row of a matrix.

Attention scores use this row-wise form after flattening all non-key axes. The
operation is still built entirely from TorchLean primitives, so it can run in
the same eager/CUDA training path as the rest of the model.
-/
def sparsemaxRowsRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {rows keyLen : Nat}
    (x : Ref m α (.dim rows (.dim keyLen .scalar))) :
    m (Ref m α (.dim rows (.dim keyLen .scalar))) := do
  let mut out ← zerosRef (m := m) (α := α) (.dim rows (.dim keyLen .scalar))
  for i in [0:rows] do
    if h : i < rows then
      let row ← TorchLean.gatherRow (m := m) (α := α)
        (rows := rows) (cols := keyLen) x ⟨i, h⟩
      let sm ← sparsemaxVecRef (m := m) (α := α) (n := keyLen) row
      out ← TorchLean.scatterAddRow (m := m) (α := α)
        (rows := rows) (cols := keyLen) out sm ⟨i, h⟩
  pure out

/-- Sparsemax over the last axis of a 4D attention-score tensor. -/
def sparsemaxAttentionScoresRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {batch heads queryLen keyLen : Nat}
    (x : Ref m α (.dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar))))) :
    m (Ref m α (.dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar))))) := do
  let rows := batch * heads * queryLen
  let xRows ← TorchLean.reshape (m := m) (α := α)
    (s₁ := .dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar))))
    (s₂ := .dim rows (.dim keyLen .scalar))
    x (by
      simp [rows, Spec.Shape.size, Nat.mul_left_comm, Nat.mul_comm])
  let yRows ← sparsemaxRowsRef (m := m) (α := α)
    (rows := rows) (keyLen := keyLen) xRows
  TorchLean.reshape (m := m) (α := α)
    (s₁ := .dim rows (.dim keyLen .scalar))
    (s₂ := .dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar))))
    yRows (by
      simp [rows, Spec.Shape.size, Nat.mul_left_comm, Nat.mul_comm])

end TL

/-! ## TorchLean layer wrappers -/

/--
Trainable TorchLean layer wrapper for LeakyReLU.

The upstream default slope is `0.01`, represented here as `1 / 100` in the
current scalar backend.  Keeping it as a rational literal avoids needing a
runtime `Float → α` cast in `LayerDef.forward`.
-/
def leakyReluLayer {s : Shape} (negativeSlopeDenom : Nat := 100) :
    TorchLean.NN.LayerDef s s :=
  { paramShapes := []
    initParams := .nil
    paramRequiresGrad := []
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          let slope : α := (1 : α) / ((negativeSlopeDenom : Nat) : α)
          TL.leakyReluRef (m := m) (α := α) (s := s)
            slope x
  }

/-- Sequential wrapper for `leakyReluLayer`. -/
def leakyRelu {s : Shape} (negativeSlopeDenom : Nat := 100) :
    nn.Sequential s s :=
  nn.of (leakyReluLayer (s := s) negativeSlopeDenom)

/--
Sparsemax layer on a single vector.

The TorchLean runtime can train and execute this layer because it is expressed
with existing differentiable primitives (`gather`, `max`, `min`, `scale`,
`scatterAdd`).  Batched attention uses the wrapper below to map the vector layer
over the last axis of the attention-score tensor.
-/
def sparsemaxVectorLayer (n : Nat) :
    TorchLean.NN.LayerDef (.dim n .scalar) (.dim n .scalar) :=
  { paramShapes := []
    initParams := .nil
    paramRequiresGrad := []
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TL.sparsemaxVecRef (m := m) (α := α) (n := n) x
  }

/-- Sequential wrapper for sparsemax on one vector. -/
def sparsemaxVector (n : Nat) :
    nn.Sequential (.dim n .scalar) (.dim n .scalar) :=
  nn.of (sparsemaxVectorLayer n)

/-- Sparsemax applied independently to each row of a `rows × keyLen` matrix. -/
def sparsemaxRowsLayer (rows keyLen : Nat) :
    TorchLean.NN.LayerDef (.dim rows (.dim keyLen .scalar)) (.dim rows (.dim keyLen .scalar)) :=
  { paramShapes := []
    initParams := .nil
    paramRequiresGrad := []
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TL.sparsemaxRowsRef (m := m) (α := α) (rows := rows) (keyLen := keyLen) x
  }

/-- Sequential wrapper for row-wise sparsemax. -/
def sparsemaxRows (rows keyLen : Nat) :
    nn.Sequential (.dim rows (.dim keyLen .scalar)) (.dim rows (.dim keyLen .scalar)) :=
  nn.of (sparsemaxRowsLayer rows keyLen)

/--
Sparsemax over the last axis of a 4D attention-score tensor.

Shape convention:

`scores : batch × heads × queryLen × keyLen`

The layer flattens `(batch, heads, queryLen)` into one row axis, applies the
row-wise sparsemax extension, then reshapes back.  That is the structural slot
occupied by the upstream repo's sparsemax attention weights.
-/
def sparsemaxAttentionScoresLayer
    (batch heads queryLen keyLen : Nat) :
    TorchLean.NN.LayerDef
      (.dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar))))
      (.dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar)))) :=
  { paramShapes := []
    initParams := .nil
    paramRequiresGrad := []
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TL.sparsemaxAttentionScoresRef (m := m) (α := α)
          (batch := batch) (heads := heads) (queryLen := queryLen) (keyLen := keyLen) x
  }

/-- Sequential wrapper for sparsemax over attention-score tensors. -/
def sparsemaxAttentionScores
    (batch heads queryLen keyLen : Nat) :
    nn.Sequential
      (.dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar))))
      (.dim batch (.dim heads (.dim queryLen (.dim keyLen .scalar)))) :=
  nn.of (sparsemaxAttentionScoresLayer batch heads queryLen keyLen)

/-- TorchLean runtime approximation to Signed-L1-BandNorm on one hidden vector. -/
def signedL1BandNormVectorLayer (n : Nat) :
    TorchLean.NN.LayerDef (.dim n .scalar) (.dim n .scalar) :=
  let gammaShape : Shape := .dim n .scalar
  let betaShape : Shape := .dim n .scalar
  let gamma0 : Tensor Float gammaShape := Spec.fill (α := Float) 1.0 gammaShape
  let beta0 : Tensor Float betaShape := Spec.fill (α := Float) 0.0 betaShape
  { paramShapes := [gammaShape, betaShape]
    initParams := TorchLean.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TL.signedL1BandNormVecRef (m := m) (α := α) (n := n) gamma beta x
  }

/-- Sequential wrapper for the BandNorm runtime vector layer. -/
def signedL1BandNormVector (n : Nat) :
    nn.Sequential (.dim n .scalar) (.dim n .scalar) :=
  nn.of (signedL1BandNormVectorLayer n)

/-- TorchLean runtime approximation to Signed-L1-BandNorm independently on each row of a matrix. -/
def signedL1BandNormRowsLayer (rows n : Nat) :
    TorchLean.NN.LayerDef (.dim rows (.dim n .scalar)) (.dim rows (.dim n .scalar)) :=
  let gammaShape : Shape := .dim n .scalar
  let betaShape : Shape := .dim n .scalar
  let gamma0 : Tensor Float gammaShape := Spec.fill (α := Float) 1.0 gammaShape
  let beta0 : Tensor Float betaShape := Spec.fill (α := Float) 0.0 betaShape
  { paramShapes := [gammaShape, betaShape]
    initParams := TorchLean.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TL.signedL1BandNormRowsRef (m := m) (α := α) (rows := rows) (n := n) gamma beta x
  }

/-- Sequential wrapper for row-wise BandNorm runtime approximation. -/
def signedL1BandNormRows (rows n : Nat) :
    nn.Sequential (.dim rows (.dim n .scalar)) (.dim rows (.dim n .scalar)) :=
  nn.of (signedL1BandNormRowsLayer rows n)

/-! ## Exact-upstream custom op status -/

/--
Native TorchLean layer coverage for the upstream custom operators.

The booleans make the status explicit in Lean: LeakyReLU and sparsemax are
available as runtime layers; exact hard-support Signed-L1-BandNorm is still a
future runtime extension.
-/
structure ExactOpRuntimeStatus where
  leakyReluLayer : Bool
  sparsemaxVectorLayer : Bool
  sparsemaxBatchedAttentionLayer : Bool
  signedL1BandNormExactLayer : Bool
deriving Repr, DecidableEq

/-- Current extension status for native TorchLean runtime layers. -/
def currentRuntimeStatus : ExactOpRuntimeStatus :=
  { leakyReluLayer := true
    sparsemaxVectorLayer := true
    sparsemaxBatchedAttentionLayer := true
    signedL1BandNormExactLayer := false }

/-- LeakyReLU is available as a direct TorchLean runtime layer. -/
theorem leakyRelu_runtime_available :
    currentRuntimeStatus.leakyReluLayer = true := rfl

/-- Vector sparsemax is available as a direct TorchLean runtime layer. -/
theorem sparsemaxVector_runtime_available :
    currentRuntimeStatus.sparsemaxVectorLayer = true := rfl

/-- Batched attention-score sparsemax is available as a direct TorchLean runtime layer. -/
theorem sparsemaxBatchedAttention_runtime_available :
    currentRuntimeStatus.sparsemaxBatchedAttentionLayer = true := rfl

/-- Exact upstream BandNorm remains in replay/certificate land, not the generic runtime layer. -/
theorem exactBandNorm_runtime_pending :
    currentRuntimeStatus.signedL1BandNormExactLayer = false := rfl

end VerifiableTransformers.TorchLean.Ops
