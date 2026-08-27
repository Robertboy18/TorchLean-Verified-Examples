/-
Rational contracts for the small `verifiable-transformers` GPT.

TorchLean already has the general tensor/runtime story.  Here we record the
finite rational semantics that Neel's SMT and circuit certificates refer to.

The two unusual pieces are what make this model family solver-friendly:

* Signed-L1-BandNorm, including the branch data used by the certificate trace;
* sparsemax attention, represented through the simplex/L1 projection threshold.

The source map is explicit: `scripts/small/config.py` for dimensions,
`scripts/small/train.py` for the custom layers, and
`scripts/smt/{encoders,trace}.py` for the rational/Z3 branch encodings.  The
TorchLean runtime version lives next door in `VerifiableTransformers/TorchLean`;
these definitions are the finite mathematical contract the exported
certificates point at.
-/

import Mathlib.Tactic

namespace VerifiableTransformers.Spec.UpstreamSmallGPT

open Finset

/-! ## Small rational containers

Neel's SMT encoding works over finite rational vectors. We use tiny
function-backed containers here because the certificate statements are about
finite symbolic coordinates, not TorchLean runtime tensors. Runtime/training
code should use TorchLean tensors instead.
-/

abbrev RVec (d : Nat) := Fin d -> Rat
abbrev RMatrix (m n : Nat) := Fin m -> Fin n -> Rat

/-- Finite rational coordinate sum used by the SMT-facing vector semantics. -/
def sumRVec {d : Nat} (x : RVec d) : Rat :=
  ∑ i, x i

/-- Rational ReLU for branch/certificate statements. -/
def reluRat (x : Rat) : Rat :=
  max x 0

/--
Scalar LeakyReLU for the rational circuit.

TorchLean has the general tensor-level `Activation.leakyReluSpec`; this helper
keeps the SMT-facing spec in simple `Fin d -> Rat` form.
-/
def leakyReluRat (negativeSlope x : Rat) : Rat :=
  if x >= 0 then x else negativeSlope * x

/-- Mean of a finite rational vector. -/
def meanRVec {d : Nat} (x : RVec d) : Rat :=
  sumRVec x / d

/-- Center a rational vector by subtracting its mean from each coordinate. -/
def centerRVec {d : Nat} (x : RVec d) : RVec d :=
  fun i => x i - meanRVec x

/-- Per-coordinate affine scale and shift. -/
def affineRVec {d : Nat} (gamma beta x : RVec d) : RVec d :=
  fun i => gamma i * x i + beta i

/--
Row-major affine map used by the SMT export.

TorchLean's real model/training path uses its own tensor linear layers; this is
only the rational circuit contract for exported weights.
-/
def affineMap {m n : Nat} (W : RMatrix m n) (b : RVec m) (x : RVec n) : RVec m :=
  fun i => (∑ j, W i j * x j) + b i

/-- Positive half of a centered rational vector. -/
def positivePart {d : Nat} (x : RVec d) : RVec d :=
  fun i => reluRat (x i)

/-- Negative half of a centered rational vector. -/
def negativePart {d : Nat} (x : RVec d) : RVec d :=
  fun i => reluRat (-(x i))

/-! ## Projection primitive shared by sparsemax and BandNorm

Both sparsemax and Signed-L1-BandNorm use the same sorted-threshold
characterization of projection onto a nonnegative L1 ball/simplex.
-/

/--
Projection with an externally supplied threshold.

Branch certificates supply `tau`; Lean then checks the local consequences needed
by the certificate.
-/
def nonnegativeL1ProjectWithTau {d : Nat} (tau : Rat) (y : RVec d) : RVec d :=
  fun i => reluRat (y i - tau)

/-- Insert one rational value into a descending list. -/
def insertDesc (x : Rat) : List Rat -> List Rat
  | [] => [x]
  | y :: ys => if x >= y then x :: y :: ys else y :: insertDesc x ys

/-- Descending sort used by the sparsemax/L1 projection threshold scan. -/
def sortDesc : List Rat -> List Rat
  | [] => []
  | x :: xs => insertDesc x (sortDesc xs)

/-- Enumerate a finite rational vector as a list for threshold computation. -/
def valuesOfRVec {d : Nat} (x : RVec d) : List Rat :=
  (List.finRange d).map x

/-- Threshold scan over sorted projection values. -/
def thresholdFromSortedAux (radius : Rat) : List Rat -> Nat -> Rat -> Rat -> Rat
  | [], _, _, best => best
  | y :: ys, k, pref, best =>
      let k' := k + 1
      let pref' := pref + y
      let tau := (pref' - radius) / k'
      let best' := if y > tau then tau else best
      thresholdFromSortedAux radius ys k' pref' best'

/-- Threshold used by the top-k characterization for sparsemax and L1 projection. -/
def projectionTauFromValues (radius : Rat) (values : List Rat) : Rat :=
  thresholdFromSortedAux radius (sortDesc values) 0 0 0

/-- Executable projection used when no branch certificate is supplied. -/
def nonnegativeL1ProjectExecutable {d : Nat} (radius : Rat) (y : RVec d) : RVec d :=
  if sumRVec y <= radius then
    y
  else
    let tau := projectionTauFromValues radius (valuesOfRVec y)
    nonnegativeL1ProjectWithTau tau y

/-- Local conditions checked for an externally supplied L1 projection threshold. -/
def ProjectionCertificate {d : Nat} (radius tau : Rat) (y : RVec d) : Prop :=
  0 <= tau ∧
  (sumRVec y <= radius ∧ tau = 0 ∨
    sumRVec (nonnegativeL1ProjectWithTau tau y) = radius)

/-- Projected nonnegative L1 coordinates are nonnegative for any threshold. -/
theorem nonnegativeL1ProjectWithTau_nonneg {d : Nat} (tau : Rat) (y : RVec d) :
    ∀ i, 0 <= nonnegativeL1ProjectWithTau tau y i := by
  intro i
  unfold nonnegativeL1ProjectWithTau reluRat
  exact le_max_right _ _

/-! ## Signed-L1-BandNorm

These definitions capture the upstream custom normalization layer as exact
rational branch/certificate semantics for the SMT-facing proof objects.
-/

/--
Low-mass correction used by Signed-L1-BandNorm.

The active mask is a rational 0/1 vector chosen by the upstream branch trace
or by the fallback policy.
-/
def additiveLiftWithMask {d : Nat} (target : Rat) (active y : RVec d) : RVec d :=
  let mass := sumRVec y
  let activeCount := sumRVec active
  let delta := (target - mass) / activeCount
  fun i => if mass < target then y i + active i * delta else y i

/-- Select the positive support, falling back when the support is empty. -/
def activeMaskFromPositiveOrFallback {d : Nat} (fallback y : RVec d) : RVec d :=
  if ∃ i, y i > 0 then
    fun i => if y i > 0 then 1 else 0
  else
    fallback

/-- Parameters for the upstream Signed-L1-BandNorm layer. -/
structure BandNormParams (d : Nat) where
  gamma : RVec d
  beta : RVec d
  halfLow : Rat
  halfHigh : Rat
  posFallback : RVec d
  negFallback : RVec d

/-- Branch data exported by the upstream SMT/Python trace for one BandNorm cell. -/
structure BandNormTrace (d : Nat) where
  posTau : Rat
  negTau : Rat
  posActive : RVec d
  negActive : RVec d

/--
Signed-L1-BandNorm with an explicit branch trace:

1. center;
2. split into positive and negative halves;
3. project each half using the exported threshold;
4. lift low-mass coordinates using the exported active mask;
5. recombine, recenter, and apply affine parameters.
-/
def signedL1BandNormWithTrace {d : Nat} (params : BandNormParams d)
    (trace : BandNormTrace d) (x : RVec d) : RVec d :=
  let c := centerRVec x
  let p := positivePart c
  let n := negativePart c
  let pProjected := nonnegativeL1ProjectWithTau trace.posTau p
  let nProjected := nonnegativeL1ProjectWithTau trace.negTau n
  let pNorm := additiveLiftWithMask params.halfLow trace.posActive pProjected
  let nNorm := additiveLiftWithMask params.halfLow trace.negActive nProjected
  let z : RVec d := fun i => pNorm i - nNorm i
  affineRVec params.gamma params.beta (centerRVec z)

/-- Executable Signed-L1-BandNorm matching the upstream forward pass. -/
def signedL1BandNormExecutable {d : Nat} (params : BandNormParams d) (x : RVec d) : RVec d :=
  let c := centerRVec x
  let p := positivePart c
  let n := negativePart c
  let pProjected := nonnegativeL1ProjectExecutable params.halfHigh p
  let nProjected := nonnegativeL1ProjectExecutable params.halfHigh n
  let pMass := sumRVec pProjected
  let nMass := sumRVec nProjected
  let pActive := activeMaskFromPositiveOrFallback params.posFallback pProjected
  let nActive := activeMaskFromPositiveOrFallback params.negFallback nProjected
  let pNorm :=
    if pMass < params.halfLow then
      additiveLiftWithMask params.halfLow pActive pProjected
    else
      pProjected
  let nNorm :=
    if nMass < params.halfLow then
      additiveLiftWithMask params.halfLow nActive nProjected
    else
      nProjected
  let z : RVec d := fun i => pNorm i - nNorm i
  affineRVec params.gamma params.beta (centerRVec z)

/-! ## Sparsemax attention

The upstream model replaces softmax attention with sparsemax. These definitions
state that attention weight computation as simplex projection, then apply the
weights to value vectors.
-/

def sparsemaxWithTau {d : Nat} (tau : Rat) (logits : RVec d) : RVec d :=
  fun i => reluRat (logits i - tau)

/-- Executable sparsemax using the same sorted-threshold rule as the Python/Z3 code. -/
def sparsemaxExecutable {d : Nat} (logits : RVec d) : RVec d :=
  let tau := projectionTauFromValues 1 (valuesOfRVec logits)
  sparsemaxWithTau tau logits

/-- Local sparsemax certificate: simplex mass and coordinate nonnegativity. -/
def SparsemaxCertificate {d : Nat} (tau : Rat) (logits : RVec d) : Prop :=
  sumRVec (sparsemaxWithTau tau logits) = 1 ∧
  ∀ i, 0 <= sparsemaxWithTau tau logits i

/-- Sparsemax coordinates are nonnegative for any threshold. -/
theorem sparsemax_nonneg {d : Nat} (tau : Rat) (logits : RVec d) :
    ∀ i, 0 <= sparsemaxWithTau tau logits i := by
  intro i
  unfold sparsemaxWithTau reluRat
  exact le_max_right _ _

/-- Extract the simplex mass equation from a sparsemax certificate. -/
theorem sparsemax_sum_eq_one_of_certificate {d : Nat} {tau : Rat} {logits : RVec d}
    (h : SparsemaxCertificate tau logits) :
    sumRVec (sparsemaxWithTau tau logits) = 1 :=
  h.1

/-- Coordinates below the sparsemax threshold are inactive. -/
theorem sparsemax_inactive_zero {d : Nat} (tau : Rat) (logits : RVec d) (i : Fin d)
    (h : logits i <= tau) :
    sparsemaxWithTau tau logits i = 0 := by
  unfold sparsemaxWithTau reluRat
  rw [max_eq_right]
  linarith

/-- Weighted sum of value vectors for one attention row. -/
def attentionValueReduction {seq d : Nat} (weights : RVec seq)
    (values : Fin seq -> RVec d) : RVec d :=
  fun j => ∑ t, weights t * values t j

/-- Sparse attention with an externally supplied sparsemax threshold. -/
def sparseAttentionWithTau {seq d : Nat} (tau : Rat) (scores : RVec seq)
    (values : Fin seq -> RVec d) : RVec d :=
  attentionValueReduction (sparsemaxWithTau tau scores) values

/-- Executable sparse attention using the sorted-threshold sparsemax rule. -/
def sparseAttentionExecutable {seq d : Nat} (scores : RVec seq)
    (values : Fin seq -> RVec d) : RVec d :=
  attentionValueReduction (sparsemaxExecutable scores) values

/-! ## Upstream small-model configuration -/

/-- Exact operator and dimension choices from `scripts/small/config.py`. -/
structure SmallGPTConfig where
  vocabSize : Nat
  maxSeqLen : Nat
  padTokenId : Nat
  bosTokenId : Nat
  dModel : Nat
  nLayers : Nat
  nHeads : Nat
  dMlp : Nat
  normVariant : String
  attnVariant : String
  activationVariant : String
  normL1LowPerDim : Rat
  normL1HighPerDim : Rat
  leakyReluNegativeSlope : Rat
  tieEmbeddings : Bool
  useBias : Bool
  dropout : Rat
  attnPdrop : Rat
  residPdrop : Rat
  embdPdrop : Rat
deriving Repr

/-- Exact transcription of the upstream default small GPT configuration. -/
def repoDefaultSmallConfig : SmallGPTConfig where
  vocabSize := 32
  maxSeqLen := 6
  padTokenId := 0
  bosTokenId := 1
  dModel := 16
  nLayers := 2
  nHeads := 1
  dMlp := 64
  normVariant := "signed_l1_band_norm"
  attnVariant := "sparsemax"
  activationVariant := "leaky_relu"
  normL1LowPerDim := 55 / 100
  normL1HighPerDim := 105 / 100
  leakyReluNegativeSlope := 1 / 100
  tieEmbeddings := false
  useBias := true
  dropout := 0
  attnPdrop := 0
  residPdrop := 0
  embdPdrop := 0

/-- Lower half-band target induced by the upstream per-dimension default. -/
def repoDefaultHalfLow : Rat :=
  (repoDefaultSmallConfig.normL1LowPerDim * repoDefaultSmallConfig.dModel) / 2

/-- Upper half-band target induced by Neel's per-dimension default. -/
def repoDefaultHalfHigh : Rat :=
  (repoDefaultSmallConfig.normL1HighPerDim * repoDefaultSmallConfig.dModel) / 2

/-- Sanity check that the embedded default config matches Neel's small model. -/
theorem repoDefaultSmallConfig_matches_upstream :
    repoDefaultSmallConfig.vocabSize = 32 ∧
    repoDefaultSmallConfig.maxSeqLen = 6 ∧
    repoDefaultSmallConfig.dModel = 16 ∧
    repoDefaultSmallConfig.nLayers = 2 ∧
    repoDefaultSmallConfig.nHeads = 1 ∧
    repoDefaultSmallConfig.dMlp = 64 ∧
    repoDefaultSmallConfig.normVariant = "signed_l1_band_norm" ∧
    repoDefaultSmallConfig.attnVariant = "sparsemax" ∧
    repoDefaultSmallConfig.activationVariant = "leaky_relu" := by
  simp [repoDefaultSmallConfig]

/-- Closed form of the default lower half-band target. -/
theorem repoDefaultHalfLow_eq : repoDefaultHalfLow = 22 / 5 := by
  norm_num [repoDefaultHalfLow, repoDefaultSmallConfig]

/-- Closed form of the default upper half-band target. -/
theorem repoDefaultHalfHigh_eq : repoDefaultHalfHigh = 42 / 5 := by
  norm_num [repoDefaultHalfHigh, repoDefaultSmallConfig]

end VerifiableTransformers.Spec.UpstreamSmallGPT
