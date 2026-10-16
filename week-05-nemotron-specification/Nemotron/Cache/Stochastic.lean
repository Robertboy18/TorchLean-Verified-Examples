/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.SSM
public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Policy.Runtime
public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
public import Mathlib.Order.Interval.Finset.Fin

import Mathlib.Data.Int.CardIntervalMod

/-!
# Stochastic writes on a fixed rounding grid

The Ultra report's §4.5 (pp. 44–46) uses stochastic FP16 cache writes. We reuse FloatLib's
executable stochastic quotient rounder and prove its local mean under uniform residues.
Each tensor coordinate gets its own signed grid cell, before overflow, saturation, or underflow
policies. Quanta and denominators come from the caller, so this is not an FP16
encoder. Uniform fixed-width words reduced `% den` need not give uniform residues, but they do
when `den` divides the number of words (`uniformResidues_range`).
-/

@[expose] public section

namespace Nemotron.Cache.Stochastic

open Spec TorchLean FloatLib.Floats.Formats.BinaryInterchange

private theorem roundQuot_eq (sign : Bool) (entropy num den : Nat) :
    Model.Policy.roundQuot .stochastic sign entropy num den =
      num / den + if entropy % den < num % den then 1 else 0 := by
  by_cases h : num % den = 0
  · simp [Model.Policy.roundQuot, h]
  · simp [Model.Policy.roundQuot, h]
    split <;> simp

/-- Summed over all `den` entropy residues, FloatLib's executable stochastic quotients add up
to `num`. -/
theorem roundQuot_sum (sign : Bool) (num den : Nat) (hden : 0 < den) :
    (∑ entropy : Fin den, Model.Policy.roundQuot .stochastic sign entropy.val num den) = num := by
  classical
  let remainder : Fin den := ⟨num % den, Nat.mod_lt num hden⟩
  have hfilter : Finset.univ.filter (fun e : Fin den => e.val < num % den) =
      Finset.Iio remainder := by
    ext e
    simp [Fin.lt_def, remainder]
  have hcount : (∑ e : Fin den, if e.val < num % den then 1 else 0) = num % den := by
    rw [← Finset.card_filter, hfilter]
    exact Fin.card_Iio remainder
  simp_rw [roundQuot_eq, Nat.mod_eq_of_lt (Fin.isLt _)]
  rw [Finset.sum_add_distrib, hcount]
  simpa [Nat.mul_comm] using Nat.div_add_mod num den

/-- One signed-grid write. Entropy is an argument, so a replay makes the same choice. -/
noncomputable def round (quantum : ℝ) (sign : Bool) (entropy num den : Nat) : ℝ :=
  (if sign then -1 else 1) * quantum *
    (Model.Policy.roundQuot .stochastic sign entropy num den : ℝ)

/-- Averaged over all `den` residues, a write equals the unrounded value of its grid cell. -/
theorem round_mean (quantum : ℝ) (sign : Bool) (num den : Nat) (hden : 0 < den) :
    (∑ entropy : Fin den, round quantum sign entropy.val num den) / (den : ℝ) =
      (if sign then -1 else 1) * quantum * ((num : ℝ) / den) := by
  have hsum : (∑ entropy : Fin den,
      (Model.Policy.roundQuot .stochastic sign entropy.val num den : ℝ)) = (num : ℝ) := by
    exact_mod_cast roundQuot_sum sign num den hden
  simp only [round, ← Finset.mul_sum, hsum]
  ring

/-- For every entropy word, a write is within one grid spacing `|quantum|` of the unrounded
value. Unbiasedness on average does not make a single write exact. -/
theorem round_error (quantum : ℝ) (sign : Bool) (entropy num den : Nat)
    (hden : 0 < den) :
    |round quantum sign entropy num den -
      (if sign then -1 else 1) * quantum * ((num : ℝ) / den)| ≤ |quantum| := by
  have hd : (0 : ℝ) < den := by exact_mod_cast hden
  have hdecomp : (den : ℝ) * (num / den : Nat) + (num % den : Nat) = num := by
    exact_mod_cast Nat.div_add_mod num den
  have hrem : (num % den : Nat) < (den : ℝ) := by
    exact_mod_cast Nat.mod_lt num hden
  have hrem0 : (0 : ℝ) ≤ (num % den : Nat) := Nat.cast_nonneg _
  have hlo : (num / den : Nat) ≤ (num : ℝ) / den := by
    apply (le_div_iff₀ hd).2
    nlinarith
  have hhi : (num : ℝ) / den ≤ (num / den : Nat) + 1 := by
    apply (div_le_iff₀ hd).2
    nlinarith
  have hunit : |(Model.Policy.roundQuot .stochastic sign entropy num den : ℝ) -
      (num : ℝ) / den| ≤ 1 := by
    rw [roundQuot_eq]
    split <;> simp only [Nat.cast_add, Nat.cast_one, add_zero]
    all_goals rw [abs_le]; constructor <;> linarith
  have hsign : |(if sign then (-1 : ℝ) else 1)| = 1 := by cases sign <;> norm_num
  calc
    _ = |(if sign then (-1 : ℝ) else 1) * quantum| *
        |(Model.Policy.roundQuot .stochastic sign entropy num den : ℝ) -
          (num : ℝ) / den| := by rw [round, ← mul_sub, abs_mul]
    _ ≤ |(if sign then (-1 : ℝ) else 1) * quantum| * 1 :=
      mul_le_mul_of_nonneg_left hunit (abs_nonneg _)
    _ = |quantum| := by rw [abs_mul, hsign, one_mul, mul_one]

/-- A finite family of entropy words with uniform residues modulo `den`: every residue below
`den` is hit by `samples / den` words. -/
def UniformResidues {samples : Nat} (den : Nat) (entropy : Fin samples → Nat) : Prop :=
  ∀ r < den, den * (Finset.univ.filter fun i => entropy i % den = r).card = samples

private theorem card_residue_lt {samples den : Nat} {entropy : Fin samples → Nat}
    (huniform : UniformResidues den entropy) :
    ∀ m ≤ den, den * (Finset.univ.filter fun i => entropy i % den < m).card = m * samples := by
  intro m
  induction m with
  | zero => intro _; simp
  | succ m ih =>
      intro hm
      have hsplit : (Finset.univ.filter fun i => entropy i % den < m + 1) =
          (Finset.univ.filter fun i => entropy i % den < m) ∪
            (Finset.univ.filter fun i => entropy i % den = m) := by
        ext i
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union]
        omega
      have hdisjoint : Disjoint (Finset.univ.filter fun i => entropy i % den < m)
          (Finset.univ.filter fun i => entropy i % den = m) := by
        rw [Finset.disjoint_filter]
        intro i _ h
        omega
      rw [hsplit, Finset.card_union_of_disjoint hdisjoint, mul_add, ih (by omega),
        huniform m (by omega)]
      ring

/-- `roundQuot_sum` for any finite ensemble with uniform residues. -/
theorem roundQuot_ensemble_sum (sign : Bool) (num den : Nat) (hden : 0 < den)
    {samples : Nat} (entropy : Fin samples → Nat) (huniform : UniformResidues den entropy) :
    den * (∑ i, Model.Policy.roundQuot .stochastic sign (entropy i) num den) =
      samples * num := by
  classical
  simp_rw [roundQuot_eq]
  rw [Finset.sum_add_distrib, ← Finset.card_filter, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, smul_eq_mul, mul_add, card_residue_lt huniform _ (Nat.mod_lt num hden).le]
  conv_rhs => rw [← Nat.div_add_mod num den]
  ring

/-- Over a nonempty ensemble with uniform residues, the mean write is the unrounded value. -/
theorem round_ensemble_mean (quantum : ℝ) (sign : Bool) (num den : Nat) (hden : 0 < den)
    {samples : Nat} (hsamples : 0 < samples) (entropy : Fin samples → Nat)
    (huniform : UniformResidues den entropy) :
    (∑ i, round quantum sign (entropy i) num den) / (samples : ℝ) =
      (if sign then -1 else 1) * quantum * ((num : ℝ) / den) := by
  have hsum : (den : ℝ) *
      (∑ i, (Model.Policy.roundQuot .stochastic sign (entropy i) num den : ℝ)) =
        samples * num := by
    exact_mod_cast roundQuot_ensemble_sum sign num den hden entropy huniform
  have hd : (den : ℝ) ≠ 0 := by positivity
  have hs : (samples : ℝ) ≠ 0 := by positivity
  have hrq : (∑ i, (Model.Policy.roundQuot .stochastic sign (entropy i) num den : ℝ)) =
      samples * num / den := by
    rw [eq_div_iff hd, mul_comm]
    exact hsum
  simp only [round, ← Finset.mul_sum, hrq]
  field_simp

/-- Apply `round` coordinatewise. Each coordinate has its own sign, quantum, denominator,
numerator, and entropy word, so mixed signs and cells are allowed. -/
noncomputable def roundTensor {shape : Shape} (quantum : Tensor ℝ shape)
    (sign : Tensor Bool shape) (den numerators entropy : Tensor Nat shape) : Tensor ℝ shape :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    round (quantum coordinate) (sign coordinate) (entropy coordinate) (numerators coordinate)
      (den coordinate)

/-- Signed value of each coordinate's grid cell before rounding. -/
noncomputable def unrounded {shape : Shape} (quantum : Tensor ℝ shape)
    (sign : Tensor Bool shape) (den numerators : Tensor Nat shape) : Tensor ℝ shape :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    (if sign coordinate then -1 else 1) * quantum coordinate *
      ((numerators coordinate : ℝ) / den coordinate)

/-- If every coordinate's grid spacing is at most `δ`, a stochastic tensor write is `Within δ`
of the unrounded tensor, however the entropy words are correlated. -/
theorem roundTensor_within {shape : Shape} (quantum : Tensor ℝ shape)
    (sign : Tensor Bool shape) (den numerators entropy : Tensor Nat shape) (δ : ℝ)
    (hden : ∀ coordinate, 0 < den coordinate)
    (hδ : ∀ coordinate, |quantum coordinate| ≤ δ) :
    Within δ (roundTensor quantum sign den numerators entropy)
      (unrounded quantum sign den numerators) := by
  intro coordinate
  simpa [roundTensor, unrounded] using
    (round_error (quantum coordinate) (sign coordinate) (entropy coordinate)
      (numerators coordinate) (den coordinate) (hden coordinate)).trans (hδ coordinate)

private theorem sum_apply {ι : Type} [Fintype ι] {shape : Shape}
    (values : ι → Tensor ℝ shape) (coordinate : Shape.Coord shape) :
    (∑ i, values i) coordinate = ∑ i, values i coordinate := by
  classical
  let evaluation : Tensor ℝ shape →+ ℝ :=
    { toFun := fun x => x coordinate
      map_zero' := by simp
      map_add' := by intros; simp }
  exact map_sum evaluation values Finset.univ

/-- Coordinatewise mean of a finite tensor ensemble. Mean laws require a nonempty sample axis. -/
noncomputable def mean {samples : Nat} {shape : Shape}
    (values : Tensor ℝ (shape.prependDim samples)) : Tensor ℝ shape :=
  (samples : ℝ)⁻¹ • ∑ i : Fin samples, Tensor.unstack values i

private theorem mean_apply {samples : Nat} {shape : Shape}
    (values : Tensor ℝ (shape.prependDim samples)) (coordinate : Shape.Coord shape) :
    mean values coordinate =
      (∑ i : Fin samples, Tensor.unstack values i coordinate) / samples := by
  simp [mean, sum_apply, div_eq_mul_inv, mul_comm]

private theorem mean_add {samples : Nat} {shape : Shape}
    (x y : Tensor ℝ (shape.prependDim samples)) :
    mean (Tensor.addSpec x y) = Tensor.addSpec (mean x) (mean y) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [mean_apply, Tensor.addSpec, Tensor.map2Spec, Tensor.unstack,
    Finset.sum_add_distrib, add_div]

/-- Each coordinate is unbiased when its own entropy residues are uniform. Nothing is assumed
about how the words of different coordinates are related. One word shared by every coordinate
and independent words both satisfy the premise (`roundTensor_mean_shared`,
`roundTensor_mean_independent`), though shared words give correlated coordinate errors. -/
theorem roundTensor_mean {samples : Nat} {shape : Shape} (quantum : Tensor ℝ shape)
    (sign : Tensor Bool shape) (den numerators : Tensor Nat shape)
    (entropy : Fin samples → Tensor Nat shape) (hsamples : 0 < samples)
    (hden : ∀ coordinate, 0 < den coordinate)
    (huniform : ∀ coordinate,
      UniformResidues (den coordinate) fun i => entropy i coordinate) :
    mean (Tensor.stackLeading fun i => roundTensor quantum sign den numerators (entropy i)) =
      unrounded quantum sign den numerators := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simpa [mean_apply, roundTensor, unrounded] using
    round_ensemble_mean (quantum coordinate) (sign coordinate) (numerators coordinate)
      (den coordinate) (hden coordinate) hsamples (fun i => entropy i coordinate)
      (huniform coordinate)

/-- Words `0, …, words - 1` have uniform residues modulo every divisor of `words`. -/
theorem uniformResidues_range {words den : Nat} (hden : 0 < den) (hdvd : den ∣ words) :
    UniformResidues den fun i : Fin words => i.val := by
  intro r hr
  have hcount := Nat.count_modEq_card (b := words) hden r
  rw [Nat.mod_eq_zero_of_dvd hdvd] at hcount
  simp only [Nat.not_lt_zero, ite_false, add_zero] at hcount
  have hmap : (Finset.univ.filter fun i : Fin words => i.val % den = r).map Fin.valEmbedding =
      (Finset.range words).filter fun k => k ≡ r [MOD den] := by
    ext k
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and,
      Fin.valEmbedding_apply, Finset.mem_range, Nat.ModEq, Nat.mod_eq_of_lt hr]
    constructor
    · rintro ⟨i, hi, rfl⟩
      exact ⟨i.isLt, hi⟩
    · rintro ⟨hk, hmod⟩
      exact ⟨⟨k, hk⟩, hmod, rfl⟩
  rw [← Finset.card_map Fin.valEmbedding, hmap, ← Nat.count_eq_card_filter_range, hcount]
  exact Nat.mul_div_cancel' hdvd

/-- Mean law for one word shared by every coordinate, drawn uniformly from `words` values that
every denominator divides (say a `2^k`-valued word and power-of-two denominators up to `2^k`).
Coordinate means are exact although coordinate errors are strongly correlated. -/
theorem roundTensor_mean_shared {shape : Shape} (quantum : Tensor ℝ shape)
    (sign : Tensor Bool shape) (den numerators : Tensor Nat shape) (words : Nat)
    (hwords : 0 < words) (hden : ∀ coordinate, 0 < den coordinate)
    (hdvd : ∀ coordinate, den coordinate ∣ words) :
    mean (Tensor.stackLeading fun e : Fin words =>
      roundTensor quantum sign den numerators (Tensor.full shape e.val)) =
      unrounded quantum sign den numerators := by
  apply roundTensor_mean quantum sign den numerators _ hwords hden
  intro coordinate
  simpa using uniformResidues_range (hden coordinate) (hdvd coordinate)

private theorem card_fiber_eq {ι : Type} [Fintype ι] [DecidableEq ι] {d : ι → Nat}
    (c : ι) (r r' : Fin (d c)) :
    (Finset.univ.filter fun ω : (j : ι) → Fin (d j) => ω c = r).card =
      (Finset.univ.filter fun ω : (j : ι) → Fin (d j) => ω c = r').card := by
  apply Finset.card_nbij' (fun ω => Function.update ω c r') (fun ω => Function.update ω c r)
  · intro ω _
    simp
  · intro ω _
    simp
  · intro ω hω
    have hω' : ω c = r := by simpa using hω
    funext j
    by_cases hj : j = c
    · subst hj
      simp [hω']
    · simp [hj]
  · intro ω hω
    have hω' : ω c = r' := by simpa using hω
    funext j
    by_cases hj : j = c
    · subst hj
      simp [hω']
    · simp [hj]

private theorem card_fiber_mul {ι : Type} [Fintype ι] [DecidableEq ι] {d : ι → Nat}
    (c : ι) (r : Fin (d c)) :
    d c * (Finset.univ.filter fun ω : (j : ι) → Fin (d j) => ω c = r).card =
      Fintype.card ((j : ι) → Fin (d j)) := by
  have hfibers := Finset.card_eq_sum_card_fiberwise
    (s := (Finset.univ : Finset ((j : ι) → Fin (d j))))
    (f := fun ω : (j : ι) → Fin (d j) => ω c) (t := Finset.univ) (by simp)
  rw [← Finset.card_univ, hfibers, Finset.sum_congr rfl fun r' _ => card_fiber_eq c r' r]
  simp

/-- Product ensemble: every assignment of one residue to each coordinate occurs once, so
different coordinates get independent uniform words. -/
noncomputable def independentEntropy {shape : Shape} (den : Tensor Nat shape) :
    Fin (Fintype.card ((coordinate : Shape.Coord shape) → Fin (den coordinate))) →
      Tensor Nat shape :=
  fun i => TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    ((Fintype.equivFin _).symm i coordinate).val

/-- Independent uniform residues, one per coordinate, also give exact coordinate means. -/
theorem roundTensor_mean_independent {shape : Shape} (quantum : Tensor ℝ shape)
    (sign : Tensor Bool shape) (den numerators : Tensor Nat shape)
    (hden : ∀ coordinate, 0 < den coordinate) :
    mean (Tensor.stackLeading fun i =>
      roundTensor quantum sign den numerators (independentEntropy den i)) =
      unrounded quantum sign den numerators := by
  classical
  have hsamples : 0 < Fintype.card ((coordinate : Shape.Coord shape) → Fin (den coordinate)) :=
    Fintype.card_pos_iff.mpr ⟨fun coordinate => ⟨0, hden coordinate⟩⟩
  apply roundTensor_mean quantum sign den numerators _ hsamples hden
  intro coordinate r hr
  refine Eq.trans ?_ (card_fiber_mul (d := fun coordinate => den coordinate) coordinate ⟨r, hr⟩)
  congr 1
  apply Finset.card_equiv (Fintype.equivFin _).symm
  intro i
  simp [independentEntropy, Nat.mod_eq_of_lt (Fin.isLt _), Fin.ext_iff]

/-- A fixed prepared SSM update is affine, so it commutes with the mean of a nonempty ensemble.
`scan_mean` uses this to compose conditionally unbiased state writes. Input-dependent token
selection, normalization, and nonlinear decoder layers fall outside the identity. -/
theorem update_mean {samples valueDim stateDim : Nat}
    (states : Tensor ℝ [samples, valueDim, stateDim])
    (input : SSM.Input ℝ valueDim stateDim) (hsamples : 0 < samples) :
    SSM.update (mean states) input =
      mean (Tensor.stackLeading fun i => SSM.update (Tensor.unstack states i) input) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  have hn : (samples : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hsamples
  simp only [SSM.update, Tensor.addSpec, Tensor.scaleSpec, Tensor.map2Spec_apply,
    Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply,
    mean_apply, Tensor.unstack_stackLeading]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

end Nemotron.Cache.Stochastic

namespace Nemotron.SSM

open Spec TorchLean KimiK3
open Cache.Stochastic

/-- If the ensemble mean follows one `update` per token, it follows the whole scan. `states`
can index all finite random histories, with repetitions for their weights. This needs the mean
over those histories at every step, which an encoder unbiased in isolation does not supply. -/
private theorem scan_mean_of_step {samples valueDim stateDim : Nat}
    (inputs : List (Input ℝ valueDim stateDim))
    (states : Nat → Tensor ℝ [samples, valueDim, stateDim])
    (hstep : ∀ i : Fin inputs.length, mean (states (i.val + 1)) =
      update (mean (states i.val)) inputs[i.val]) :
    mean (states inputs.length) = (CausalScan.run step (mean (states 0)) inputs).1 := by
  induction inputs generalizing states with
  | nil => rfl
  | cons input rest ih =>
      have hfirst := hstep ⟨0, by simp⟩
      have hrest := ih (fun n => states (n + 1)) (fun i => by
        simpa [Nat.add_assoc] using hstep i.succ)
      simpa [CausalScan.run, step, hfirst] using hrest

/-- If each step updates every sample and then adds storage errors of mean zero, the ensemble
mean follows the exact scan from the initial mean. Conditionally unbiased writes can supply
`herrors` on finite histories. Neither independence nor zero error on a single history is
assumed. -/
theorem scan_mean {samples valueDim stateDim : Nat}
    (inputs : List (Input ℝ valueDim stateDim))
    (states errors : Nat → Tensor ℝ [samples, valueDim, stateDim]) (hsamples : 0 < samples)
    (hstep : ∀ i : Fin inputs.length, states (i.val + 1) =
      Tensor.addSpec (Tensor.stackLeading fun sample =>
        update (Tensor.unstack (states i.val) sample) inputs[i.val]) (errors i.val))
    (herrors : ∀ i : Fin inputs.length,
      mean (errors i.val) = Tensor.full [valueDim, stateDim] 0) :
    mean (states inputs.length) = (CausalScan.run step (mean (states 0)) inputs).1 := by
  apply scan_mean_of_step inputs states
  intro i
  rw [hstep i, Cache.Stochastic.mean_add, herrors i]
  rw [← update_mean (states i.val) inputs[i.val] hsamples]
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [Tensor.addSpec, Tensor.map2Spec]

end Nemotron.SSM
