/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Basic
public import Mathlib.Probability.Moments.Variance

/-!
# Correcting an estimate with residual samples

A known surrogate mean removes bias; the residual variance determines the cost of the
remaining sampling. Pairwise zero covariance suffices for the variance calculation.
The tensor coordinates are sample values, not necessarily independent model activations.
-/

@[expose] public section

namespace HeuristicEstimation.Sampling

open TorchLean MeasureTheory ProbabilityTheory
open scoped BigOperators

/-- A nonempty tensor's public mean is its coordinate sum divided by its size. -/
theorem mean_eq_sum {n : Nat} (hn : 0 < n) (values : Tensor ℝ [n]) :
    Tensor.mean values = (∑ i : Fin n, values.getScalar i) / (n : ℝ) := by
  simp [Tensor.mean, Tensor.meanSpec, Spec.sum_spec_vec, Tensor.meanDenominator,
    Spec.Shape.size, Nat.ne_of_gt hn]

/-- The expectation of a finite sample average is the average of its expectations. -/
theorem mean_expectation {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Nat}
    (hn : 0 < n) (values : Ω → Tensor ℝ [n])
    (h : ∀ i, Integrable (fun ω => (values ω).getScalar i) μ) :
    (∫ ω, Tensor.mean (values ω) ∂μ) =
      (∑ i : Fin n, ∫ ω, (values ω).getScalar i ∂μ) / (n : ℝ) := by
  simp_rw [mean_eq_sum hn]
  rw [integral_div]
  congr 1
  exact integral_finsetSum Finset.univ (fun i _ => h i)

/-- Uncorrelated sample coordinates give the familiar inverse-sample-count variance.
Their variances may differ; no identical-distribution premise is needed. -/
theorem mean_variance {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : Nat} (hn : 0 < n) (values : Ω → Tensor ℝ [n])
    (h : ∀ i, MemLp (fun ω => (values ω).getScalar i) 2 μ)
    (hcov : ∀ i j : Fin n, i ≠ j →
      covariance (fun ω => (values ω).getScalar i)
        (fun ω => (values ω).getScalar j) μ = 0) :
    variance (fun ω => Tensor.mean (values ω)) μ =
      (∑ i : Fin n, variance (fun ω => (values ω).getScalar i) μ) / (n : ℝ) ^ 2 := by
  classical
  simp_rw [mean_eq_sum hn]
  simp_rw [div_eq_mul_inv]
  rw [variance_mul_const, variance_fun_sum h]
  simp only [inv_pow]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_eq_single i]
  · exact covariance_self (h i).aemeasurable
  · intro j _ hji
    exact hcov i j hji.symm
  · simp

/-- Equal coordinate means make the sample average unbiased. A positive sample count
is explicit because an empty tensor has no statistical average. -/
theorem mean_unbiased {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Nat}
    (hn : 0 < n) (values : Ω → Tensor ℝ [n]) (mean : ℝ)
    (h : ∀ i, Integrable (fun ω => (values ω).getScalar i) μ)
    (hm : ∀ i, (∫ ω, (values ω).getScalar i ∂μ) = mean) :
    (∫ ω, Tensor.mean (values ω) ∂μ) = mean := by
  rw [mean_expectation hn values h]
  simp only [hm, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact mul_div_cancel_left₀ mean (Nat.cast_ne_zero.mpr (Nat.ne_of_gt hn))

/-- Finite second moments are preserved by averaging a finite sample tensor. -/
theorem mean_memLp {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : Nat}
    (hn : 0 < n) (values : Ω → Tensor ℝ [n])
    (h : ∀ i, MemLp (fun ω => (values ω).getScalar i) 2 μ) :
    MemLp (fun ω => Tensor.mean (values ω)) 2 μ := by
  simp_rw [mean_eq_sum hn, div_eq_mul_inv]
  exact (memLp_finsetSum Finset.univ (fun i _ => h i)).mul_const _

/-- Squared error separates into variance and squared bias. This uses mathlib's variance,
with integrability supplied by a finite second moment. -/
theorem mse_eq_variance_add_bias {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {value : Ω → ℝ} (h : MemLp value 2 μ) (target : ℝ) :
    (∫ ω, (value ω - target) ^ 2 ∂μ) =
      variance value μ + ((∫ ω, value ω ∂μ) - target) ^ 2 := by
  have hexpand : (fun ω => (value ω - target) ^ 2) =
      (fun ω => value ω ^ 2 + (-2 * target) * value ω + target ^ 2) := by
    funext ω
    ring
  have hint := h.integrable (by norm_num)
  rw [hexpand,
    integral_add (f := fun ω => value ω ^ 2 + (-2 * target) * value ω)
      (g := fun _ => target ^ 2)
      (h.integrable_sq.add (hint.const_mul (-2 * target))) (integrable_const _),
    integral_add (f := fun ω => value ω ^ 2) (g := fun ω => (-2 * target) * value ω)
      h.integrable_sq (hint.const_mul (-2 * target)),
    integral_const_mul, integral_const, variance_eq_sub h]
  simp only [probReal_univ, one_smul, Pi.pow_apply]
  ring

/-- Residual sampling has exactly the residual variance divided by the sample count,
plus the squared error in the supplied surrogate mean. Pairwise uncorrelated samples
suffice; independence is one way to discharge that premise. -/
theorem residual_mse {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : Nat} (hn : 0 < n)
    (residuals : Ω → Tensor ℝ [n]) (residualMean variance : ℝ)
    (knownMean surrogateMean : ℝ)
    (h : ∀ i, MemLp (fun ω => (residuals ω).getScalar i) 2 μ)
    (hm : ∀ i, (∫ ω, (residuals ω).getScalar i ∂μ) = residualMean)
    (hv : ∀ i, ProbabilityTheory.variance (fun ω => (residuals ω).getScalar i) μ = variance)
    (hcov : ∀ i j : Fin n, i ≠ j →
      covariance (fun ω => (residuals ω).getScalar i)
        (fun ω => (residuals ω).getScalar j) μ = 0) :
    (∫ ω, (knownMean + Tensor.mean (residuals ω) -
      (surrogateMean + residualMean)) ^ 2 ∂μ) =
        variance / (n : ℝ) + (knownMean - surrogateMean) ^ 2 := by
  have havg := mean_memLp hn residuals h
  have havgMean := mean_unbiased hn residuals residualMean
    (fun i => (h i).integrable (by norm_num)) hm
  have havgVariance : ProbabilityTheory.variance (fun ω => Tensor.mean (residuals ω)) μ =
      variance / (n : ℝ) := by
    rw [mean_variance hn residuals h hcov]
    simp only [hv, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  have hexpand : (fun ω => (knownMean + Tensor.mean (residuals ω) -
      (surrogateMean + residualMean)) ^ 2) =
      (fun ω => (Tensor.mean (residuals ω) -
        (surrogateMean + residualMean - knownMean)) ^ 2) := by
    funext ω
    ring
  rw [hexpand, mse_eq_variance_add_bias havg, havgMean, havgVariance]
  ring

end HeuristicEstimation.Sampling
