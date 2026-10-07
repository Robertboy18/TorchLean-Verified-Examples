/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward
public import Mathlib.Order.Interval.Set.ProjIcc

/-!
# Histogram quantiles for Kimi K3 routing

Appendix D of the Kimi K3 report estimates each Quantile Balancing update from additive histogram
counts.  This file proves the two mathematical guarantees used there: pooling is independent of
how counts are sharded, and interpolation inside the selected bin differs from any true quantile
in that bin by at most one bin width.
-/

@[expose] public section

namespace KimiK3
namespace QuantileHistogram

/-- Per-bin counts for one routed expert. -/
abbrev Histogram (bins : ℕ) := Fin bins → ℕ

/-- Count entries through and including `bin`. -/
def cumulativeThrough {bins : ℕ} (histogram : Histogram bins) (bin : Fin bins) : ℕ :=
  ∑ index ∈ Finset.Iic bin, histogram index

/-- Count entries strictly before `bin`. -/
def cumulativeBefore {bins : ℕ} (histogram : Histogram bins) (bin : Fin bins) : ℕ :=
  ∑ index ∈ Finset.Iio bin, histogram index

/-- A selected quantile bin is the first interval whose cumulative count reaches the target rank. -/
def IsSelectedBin {bins : ℕ} (histogram : Histogram bins) (targetRank : ℕ)
    (bin : Fin bins) : Prop :=
  cumulativeBefore histogram bin < targetRank ∧
    targetRank ≤ cumulativeThrough histogram bin

/-- The cumulative count through a bin adds exactly that bin to the strict prefix. -/
theorem cumulativeThrough_eq_before_add {bins : ℕ} (histogram : Histogram bins)
    (bin : Fin bins) :
    cumulativeThrough histogram bin = cumulativeBefore histogram bin + histogram bin := by
  rw [cumulativeThrough, cumulativeBefore, Finset.Iic_eq_cons_Iio, Finset.sum_cons]
  exact Nat.add_comm _ _

/-- A bin satisfying the rank-crossing criterion has a nonzero interpolation denominator. -/
theorem count_pos_of_isSelectedBin {bins : ℕ} {histogram : Histogram bins}
    {targetRank : ℕ} {bin : Fin bins} (hSelected : IsSelectedBin histogram targetRank bin) :
    0 < histogram bin := by
  rw [IsSelectedBin, cumulativeThrough_eq_before_add] at hSelected
  omega

/-- Taking a cumulative count commutes with pointwise histogram addition. -/
theorem cumulativeThrough_add {bins : ℕ} (left right : Histogram bins) (bin : Fin bins) :
    cumulativeThrough (left + right) bin =
      cumulativeThrough left bin + cumulativeThrough right bin := by
  simp only [cumulativeThrough, Pi.add_apply, Finset.sum_add_distrib]

/-- Cumulative pooled counts are exactly the sum of the local cumulative counts. -/
theorem cumulativeThrough_sum {bins : ℕ} (histograms : List (Histogram bins))
    (bin : Fin bins) :
    cumulativeThrough histograms.sum bin =
      (histograms.map fun histogram => cumulativeThrough histogram bin).sum := by
  induction histograms with
  | nil => simp [cumulativeThrough]
  | cons histogram histograms ih =>
      rw [List.sum_cons, cumulativeThrough_add]
      simp only [List.map_cons, List.sum_cons]
      rw [ih]

/-- The required bias `cutoff - score` lies in the report's adaptive histogram range. -/
theorem requiredBias_mem_range {score cutoff biasMin biasMax : ℝ}
    (hScoreLower : 0 ≤ score) (hScoreUpper : score ≤ 1)
    (hCutoffLower : biasMin ≤ cutoff) (hCutoffUpper : cutoff ≤ 1 + biasMax) :
    biasMin - 1 ≤ cutoff - score ∧ cutoff - score ≤ biasMax + 1 := by
  constructor <;> linarith

/-- Lower endpoint of a uniform histogram bin. -/
def binLower (lower width : ℝ) (bin : ℕ) : ℝ :=
  lower + bin * width

/-- Upper endpoint of a uniform histogram bin. -/
def binUpper (lower width : ℝ) (bin : ℕ) : ℝ :=
  lower + (bin + 1) * width

/-- Interpolate inside one bin using the report's clipped within-bin fraction. -/
noncomputable def interpolate (lower width : ℝ) (bin : ℕ) (fraction : ℝ) : ℝ :=
  lower + (bin + (Set.projIcc 0 1 zero_le_one fraction : ℝ)) * width

/-- Histogram interpolation always remains inside the selected bin. -/
theorem interpolate_mem_bin {lower width : ℝ} (bin : ℕ) (fraction : ℝ)
    (hWidth : 0 ≤ width) :
    binLower lower width bin ≤ interpolate lower width bin fraction ∧
      interpolate lower width bin fraction ≤ binUpper lower width bin := by
  obtain ⟨hFractionLower, hFractionUpper⟩ :=
    (Set.projIcc 0 1 zero_le_one fraction).property
  constructor <;>
    simp only [binLower, binUpper, interpolate] <;>
    nlinarith

theorem binUpper_eq_binLower_add {lower width : ℝ} (bin : ℕ) :
    binUpper lower width bin = binLower lower width bin + width := by
  simp only [binLower, binUpper]
  ring

/-- Appendix D's interpolation formula for an integral target load.

For a nonintegral load `q`, the report selects a bin using `ceil q` but interpolates using `q`
itself. The natural-number argument here covers only the integral case.
-/
noncomputable def estimate (lower width : ℝ) (bin : ℕ)
    (targetRank cumulativeBefore countInBin : ℕ) : ℝ :=
  interpolate lower width bin
    (((targetRank : ℝ) - cumulativeBefore) / countInBin)

/-- Any true quantile in the selected bin is within one bin width of the estimate. -/
theorem abs_estimate_sub_le_width {lower width trueQuantile : ℝ} {bin : ℕ}
    (targetRank cumulative countInBin : ℕ) (hWidth : 0 ≤ width)
    (hTrueLower : binLower lower width bin ≤ trueQuantile)
    (hTrueUpper : trueQuantile ≤ binUpper lower width bin) :
    |estimate lower width bin targetRank cumulative countInBin - trueQuantile| ≤ width := by
  obtain ⟨hEstimateLower, hEstimateUpper⟩ :=
    interpolate_mem_bin bin
      (((targetRank : ℝ) - cumulative) / countInBin) hWidth
  have hEstimateLower' :
      binLower lower width bin ≤
        estimate lower width bin targetRank cumulative countInBin := by
    simpa only [estimate] using hEstimateLower
  have hEstimateUpper' :
      estimate lower width bin targetRank cumulative countInBin ≤
        binUpper lower width bin := by
    simpa only [estimate] using hEstimateUpper
  have hBinWidth := binUpper_eq_binLower_add (lower := lower) (width := width) bin
  rw [abs_le]
  constructor <;> linarith

end QuantileHistogram
end KimiK3
