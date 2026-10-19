/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import Mathlib.Probability.CDF
public import Mathlib.Probability.Distributions.Gaussian.Real
public import NN.Spec.Layers.Activation

/-!
# ReLU moments of a Gaussian preactivation

Let `Z` have law `gaussianReal μ (σ ^ 2)` with `0 < σ`. Write `φ` for the standard normal density
`gaussianPDFReal 0 1` and `Φ` for mathlib's distribution function `cdf (gaussianReal 0 1)`, so
`Φ t` is the probability that a standard normal variable is at most `t`. Then

* `E[ReLU Z] = μ Φ(μ / σ) + σ φ(μ / σ)` (`relu_mean`),
* `E[(ReLU Z)²] = (μ² + σ²) Φ(μ / σ) + μ σ φ(μ / σ)` (`relu_second_moment`).

With `μ = 0` these are `σ / √(2π)` and `σ² / 2`. Both moments are integrals over `(0, ∞)`
against the density. Since `x - μ` times the density is `-σ²` times its derivative, the
fundamental theorem of calculus evaluates them.

Covariance propagation passes means and covariances through affine layers exactly
(`Moments.affine_mean`, `Moments.affine_covariance`) and applies these formulas at each ReLU.
This makes the per-neuron ReLU moment rule exact whenever the preactivation is Gaussian, so
its error in the estimator comes only from treating non-Gaussian preactivations as Gaussian,
which is the closure. The covariance of two different ReLU outputs needs the bivariate
Gaussian formula, which is not proved here.
-/

@[expose] public section

namespace HeuristicEstimation.Gaussian

open MeasureTheory ProbabilityTheory Set Filter Topology Real
open scoped NNReal

/-- The Gaussian density solves `p' = -((x - μ) / v) p`. -/
private theorem hasDerivAt_gaussianPDFReal (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    HasDerivAt (gaussianPDFReal μ v) (-((x - μ) / v) * gaussianPDFReal μ v x) x := by
  have hv' : (v : ℝ) ≠ 0 := by exact_mod_cast hv
  have hsq : HasDerivAt (fun x => (x - μ) ^ 2) (2 * (x - μ)) x := by
    simpa using ((hasDerivAt_id x).sub_const μ).fun_pow 2
  have hexponent : HasDerivAt (fun x => -(x - μ) ^ 2 / (2 * (v : ℝ))) (-((x - μ) / v)) x := by
    convert hsq.neg.div_const (2 * (v : ℝ)) using 1
    field_simp
  rw [gaussianPDFReal_def]
  convert hexponent.exp.const_mul (√(2 * π * v))⁻¹ using 1
  ring

/-- A function integrable for `gaussianReal μ v` is Lebesgue integrable after weighting by the
density. -/
private theorem integrable_gaussianPDFReal_mul {μ : ℝ} {v : ℝ≥0} (hv : v ≠ 0) {g : ℝ → ℝ}
    (hg : Integrable g (gaussianReal μ v)) :
    Integrable (fun x => gaussianPDFReal μ v x * g x) := by
  rw [gaussianReal_of_var_ne_zero _ hv, integrable_withDensity_iff_integrable_smul'
    (measurable_gaussianPDF _ _) (ae_of_all _ fun _ => gaussianPDF_lt_top)] at hg
  simpa using hg

private theorem integrable_gaussianPDFReal_mul_id (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    Integrable (fun x => gaussianPDFReal μ v x * x) :=
  integrable_gaussianPDFReal_mul hv ((memLp_id_gaussianReal 1).integrable le_rfl)

private theorem integrable_gaussianPDFReal_mul_sq (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    Integrable (fun x => gaussianPDFReal μ v x * x ^ 2) :=
  integrable_gaussianPDFReal_mul hv (memLp_id_gaussianReal 2).integrable_sq

private theorem integrable_gaussianPDFReal_mul_sub (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    Integrable (fun x => gaussianPDFReal μ v x * (x - μ)) := by
  refine ((integrable_gaussianPDFReal_mul_id μ hv).sub
    ((integrable_gaussianPDFReal μ v).mul_const μ)).congr (ae_of_all _ fun x => ?_)
  simp only [Pi.sub_apply]
  ring

/-- Above any threshold `a`, the first moment about the mean is `v` times the density at `a`. -/
private theorem integral_Ioi_gaussianPDFReal_mul_sub (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (a : ℝ) :
    ∫ x in Ioi a, gaussianPDFReal μ v x * (x - μ) = v * gaussianPDFReal μ v a := by
  have hv' : (v : ℝ) ≠ 0 := by exact_mod_cast hv
  have hderiv : ∀ x, HasDerivAt (fun x => -(v * gaussianPDFReal μ v x))
      (gaussianPDFReal μ v x * (x - μ)) x := by
    intro x
    convert ((hasDerivAt_gaussianPDFReal μ hv x).const_mul (v : ℝ)).neg using 1
    field_simp
  have hint := integrable_gaussianPDFReal_mul_sub μ hv
  have hlim : Tendsto (fun x => -(v * gaussianPDFReal μ v x)) atTop (𝓝 0) :=
    tendsto_zero_of_hasDerivAt_of_integrableOn_Ioi (a := a) (fun x _ => hderiv x)
      hint.integrableOn ((integrable_gaussianPDFReal μ v).const_mul _).neg.integrableOn
  rw [integral_Ioi_of_hasDerivAt_of_tendsto' (fun x _ => hderiv x) hint.integrableOn hlim]
  ring

/-- Above zero, the second moment is `v` times the mass plus `μ` times the first moment. -/
private theorem integral_Ioi_zero_gaussianPDFReal_mul_sq (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    ∫ x in Ioi 0, gaussianPDFReal μ v x * x ^ 2 =
      v * (∫ x in Ioi 0, gaussianPDFReal μ v x) +
        μ * ∫ x in Ioi 0, gaussianPDFReal μ v x * x := by
  have hv' : (v : ℝ) ≠ 0 := by exact_mod_cast hv
  set p := gaussianPDFReal μ v
  have hderiv : ∀ x, HasDerivAt (fun x => -(v * (p x * x)))
      (p x * x ^ 2 - (v * p x + μ * (p x * x))) x := by
    intro x
    convert (((hasDerivAt_gaussianPDFReal μ hv x).fun_mul (hasDerivAt_id' x)).const_mul
      (v : ℝ)).neg using 1
    field_simp
    ring
  have hmass : Integrable (fun x => (v : ℝ) * p x) :=
    (integrable_gaussianPDFReal μ v).const_mul _
  have hfirst : Integrable (fun x => μ * (p x * x)) :=
    (integrable_gaussianPDFReal_mul_id μ hv).const_mul _
  have hlower : Integrable (fun x => (v : ℝ) * p x + μ * (p x * x)) := hmass.add hfirst
  have hint : Integrable (fun x => p x * x ^ 2 - (v * p x + μ * (p x * x))) :=
    (integrable_gaussianPDFReal_mul_sq μ hv).sub hlower
  have hlim : Tendsto (fun x => -(v * (p x * x))) atTop (𝓝 0) :=
    tendsto_zero_of_hasDerivAt_of_integrableOn_Ioi (a := 0) (fun x _ => hderiv x)
      hint.integrableOn ((integrable_gaussianPDFReal_mul_id μ hv).const_mul _).neg.integrableOn
  have h := integral_Ioi_of_hasDerivAt_of_tendsto' (a := 0) (fun x _ => hderiv x)
    hint.integrableOn hlim
  rw [integral_sub (integrable_gaussianPDFReal_mul_sq μ hv).integrableOn hlower.integrableOn,
    integral_add hmass.integrableOn hfirst.integrableOn, integral_const_mul,
    integral_const_mul] at h
  simp only [mul_zero, neg_zero, sub_zero] at h
  linarith

/-- The mass of `gaussianReal μ (σ ^ 2)` above zero is `Φ(μ / σ)`. -/
private theorem integral_Ioi_zero_gaussianPDFReal (μ : ℝ) {σ : ℝ≥0} (hσ : 0 < σ) :
    ∫ x in Ioi 0, gaussianPDFReal μ (σ ^ 2) x = cdf (gaussianReal 0 1) (μ / σ) := by
  have hσ' : (0 : ℝ) < σ := hσ
  have hv : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ.ne'
  have hstandard :
      ((gaussianReal μ (σ ^ 2)).map (μ - ·)).map (· / (σ : ℝ)) = gaussianReal 0 1 := by
    rw [gaussianReal_map_const_sub, gaussianReal_map_div_const, sub_self, zero_div]
    congr 1
    ext
    simp [hσ'.ne']
  have hpreimage : (fun x => (μ - x) / (σ : ℝ)) ⁻¹' Iic (μ / σ) = Ici 0 := by
    ext x
    simp [div_le_div_iff_of_pos_right hσ']
  rw [cdf_eq_real, ← hstandard, Measure.map_map (by fun_prop) (by fun_prop),
    map_measureReal_apply (by fun_prop) measurableSet_Iic]
  rw [show ((· / (σ : ℝ)) ∘ (μ - ·)) = fun x => (μ - x) / (σ : ℝ) from rfl, hpreimage,
    measureReal_def, gaussianReal_apply_eq_integral _ hv, ENNReal.toReal_ofReal
      (setIntegral_nonneg measurableSet_Ici fun x _ => gaussianPDFReal_nonneg _ _ _),
    integral_Ici_eq_integral_Ioi]

/-- The density of `gaussianReal μ (σ ^ 2)` at zero, times `σ ^ 2`, is `σ φ(μ / σ)`. -/
private theorem var_mul_gaussianPDFReal_zero (μ : ℝ) {σ : ℝ≥0} (hσ : 0 < σ) :
    ((σ ^ 2 : ℝ≥0) : ℝ) * gaussianPDFReal μ (σ ^ 2) 0 = σ * gaussianPDFReal 0 1 (μ / σ) := by
  have hσ' : (0 : ℝ) < σ := hσ
  simp only [gaussianPDFReal, NNReal.coe_pow, NNReal.coe_one]
  rw [show 2 * π * (σ : ℝ) ^ 2 = 2 * π * 1 * (σ : ℝ) ^ 2 by ring,
    Real.sqrt_mul (by positivity), Real.sqrt_sq hσ'.le]
  have hexponent : -(0 - μ) ^ 2 / (2 * (σ : ℝ) ^ 2) = -(μ / σ - 0) ^ 2 / (2 * 1) := by
    field_simp
    ring
  rw [hexponent]
  field_simp

/-- A positive power of ReLU, weighted by `f`, is the same power of the identity restricted to
`(0, ∞)`. -/
private theorem mul_relu_pow_eq_indicator (f : ℝ → ℝ) {n : ℕ} (hn : n ≠ 0) :
    (fun x => f x * Activation.Math.reluSpec x ^ n) =
      (Ioi 0).indicator (fun x => f x * x ^ n) := by
  funext x
  by_cases hx : 0 < x
  · simp [hx, hx.le]
  · simp [hx, le_of_not_gt hx, hn]

/-- `relu_mean` as an integral against `gaussianReal μ (σ ^ 2)`. -/
private theorem integral_gaussianReal_relu (μ : ℝ) {σ : ℝ≥0} (hσ : 0 < σ) :
    ∫ x, Activation.Math.reluSpec x ∂gaussianReal μ (σ ^ 2) =
      μ * cdf (gaussianReal 0 1) (μ / σ) + σ * gaussianPDFReal 0 1 (μ / σ) := by
  have hv : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ.ne'
  have hrelu := mul_relu_pow_eq_indicator (gaussianPDFReal μ (σ ^ 2)) one_ne_zero
  simp only [pow_one] at hrelu
  have hsplit : ∫ x in Ioi 0, gaussianPDFReal μ (σ ^ 2) x * x =
      ∫ x in Ioi 0,
        (μ * gaussianPDFReal μ (σ ^ 2) x + gaussianPDFReal μ (σ ^ 2) x * (x - μ)) := by
    congr 1
    funext x
    ring
  rw [integral_gaussianReal_eq_integral_smul hv]
  simp only [smul_eq_mul]
  rw [hrelu, integral_indicator measurableSet_Ioi, hsplit,
    integral_add ((integrable_gaussianPDFReal μ _).const_mul _).integrableOn
      (integrable_gaussianPDFReal_mul_sub μ hv).integrableOn,
    integral_const_mul, integral_Ioi_gaussianPDFReal_mul_sub μ hv,
    integral_Ioi_zero_gaussianPDFReal μ hσ, var_mul_gaussianPDFReal_zero μ hσ]

/-- `relu_second_moment` as an integral against `gaussianReal μ (σ ^ 2)`. -/
private theorem integral_gaussianReal_relu_sq (μ : ℝ) {σ : ℝ≥0} (hσ : 0 < σ) :
    ∫ x, Activation.Math.reluSpec x ^ 2 ∂gaussianReal μ (σ ^ 2) =
      (μ ^ 2 + σ ^ 2) * cdf (gaussianReal 0 1) (μ / σ) +
        μ * σ * gaussianPDFReal 0 1 (μ / σ) := by
  have hv : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ.ne'
  have hmean := integral_gaussianReal_relu μ hσ
  have hrelu := mul_relu_pow_eq_indicator (gaussianPDFReal μ (σ ^ 2)) one_ne_zero
  simp only [pow_one] at hrelu
  rw [integral_gaussianReal_eq_integral_smul hv] at hmean ⊢
  simp only [smul_eq_mul] at hmean ⊢
  rw [hrelu, integral_indicator measurableSet_Ioi] at hmean
  rw [mul_relu_pow_eq_indicator _ two_ne_zero, integral_indicator measurableSet_Ioi,
    integral_Ioi_zero_gaussianPDFReal_mul_sq μ hv, hmean, integral_Ioi_zero_gaussianPDFReal μ hσ]
  push_cast
  ring

/-- Half of a centered Gaussian lies below zero: `Φ(0) = 1 / 2`. -/
private theorem cdf_gaussianReal_zero : cdf (gaussianReal 0 1) 0 = 1 / 2 := by
  have := nullSingletonClass_gaussianReal (μ := 0) (v := 1) one_ne_zero
  have hneg : (gaussianReal 0 1).map (fun x => -x) = gaussianReal 0 1 := by
    rw [gaussianReal_map_neg, neg_zero]
  have hsymm : (gaussianReal 0 1).real (Iic 0) = (gaussianReal 0 1).real (Ioi 0) := by
    calc (gaussianReal 0 1).real (Iic 0)
        = ((gaussianReal 0 1).map (fun x => -x)).real (Iic 0) := by rw [hneg]
      _ = (gaussianReal 0 1).real (Ici 0) := by
        rw [map_measureReal_apply measurable_neg measurableSet_Iic]
        congr 1
        ext x
        simp
      _ = (gaussianReal 0 1).real (Ioi 0) := measureReal_congr Ioi_ae_eq_Ici.symm
  have htotal := probReal_compl_eq_one_sub (μ := gaussianReal 0 1) (measurableSet_Iic (a := 0))
  rw [compl_Iic] at htotal
  rw [cdf_eq_real]
  linarith

/-- ReLU is continuous, so it is a.e. strongly measurable for every measure. -/
private theorem aestronglyMeasurable_relu (ν : Measure ℝ) :
    AEStronglyMeasurable (Activation.Math.reluSpec : ℝ → ℝ) ν := by
  have hrelu : (Activation.Math.reluSpec : ℝ → ℝ) = fun x => max x 0 :=
    funext Activation.Math.reluSpec_eq_max
  rw [hrelu]
  fun_prop

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {Z : Ω → ℝ}

/-- For a preactivation `Z ~ N(μ, σ²)` with `0 < σ`,
`E[ReLU Z] = μ Φ(μ / σ) + σ φ(μ / σ)`, where `Φ = cdf (gaussianReal 0 1)` and
`φ = gaussianPDFReal 0 1`. -/
theorem relu_mean {μ : ℝ} {σ : ℝ≥0} (hσ : 0 < σ) (hZ : HasLaw Z (gaussianReal μ (σ ^ 2)) P) :
    ∫ ω, Activation.Math.reluSpec (Z ω) ∂P =
      μ * cdf (gaussianReal 0 1) (μ / σ) + σ * gaussianPDFReal 0 1 (μ / σ) := by
  rw [← integral_gaussianReal_relu μ hσ]
  exact hZ.integral_comp (aestronglyMeasurable_relu _)

/-- For a preactivation `Z ~ N(μ, σ²)` with `0 < σ`,
`E[(ReLU Z)²] = (μ² + σ²) Φ(μ / σ) + μ σ φ(μ / σ)`. -/
theorem relu_second_moment {μ : ℝ} {σ : ℝ≥0} (hσ : 0 < σ)
    (hZ : HasLaw Z (gaussianReal μ (σ ^ 2)) P) :
    ∫ ω, Activation.Math.reluSpec (Z ω) ^ 2 ∂P =
      (μ ^ 2 + σ ^ 2) * cdf (gaussianReal 0 1) (μ / σ) +
        μ * σ * gaussianPDFReal 0 1 (μ / σ) := by
  rw [← integral_gaussianReal_relu_sq μ hσ]
  exact hZ.integral_comp ((aestronglyMeasurable_relu _).pow 2)

/-- A centered Gaussian preactivation `Z ~ N(0, σ²)` has `E[ReLU Z] = σ / √(2π)`. -/
theorem relu_mean_centered {σ : ℝ≥0} (hσ : 0 < σ) (hZ : HasLaw Z (gaussianReal 0 (σ ^ 2)) P) :
    ∫ ω, Activation.Math.reluSpec (Z ω) ∂P = σ / √(2 * π) := by
  rw [relu_mean hσ hZ]
  simp [gaussianPDFReal, div_eq_mul_inv]

/-- A centered Gaussian preactivation `Z ~ N(0, σ²)` has `E[(ReLU Z)²] = σ² / 2`. -/
theorem relu_second_moment_centered {σ : ℝ≥0} (hσ : 0 < σ)
    (hZ : HasLaw Z (gaussianReal 0 (σ ^ 2)) P) :
    ∫ ω, Activation.Math.reluSpec (Z ω) ^ 2 ∂P = σ ^ 2 / 2 := by
  rw [relu_second_moment hσ hZ, zero_div, cdf_gaussianReal_zero]
  ring

end HeuristicEstimation.Gaussian
