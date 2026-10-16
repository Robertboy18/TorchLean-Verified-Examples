/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Common
public import Mathlib.Order.Interval.Set.ProjIcc
public import Mathlib.Algebra.Order.Floor.Semiring

/-!
# Histogram quantiles for Kimi K3 routing

Appendix D of the Kimi K3 report estimates each Quantile Balancing update from additive histogram
counts. This file proves the two guarantees used there: pooling does not depend on how counts are
sharded, and interpolation inside the selected bin is within one bin width of any true quantile in
that bin.
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

/-- `bin` is the selected quantile bin: the first bin whose cumulative count reaches
`targetRank`. -/
def IsSelectedBin {bins : ℕ} (histogram : Histogram bins) (targetRank : ℕ)
    (bin : Fin bins) : Prop :=
  cumulativeBefore histogram bin < targetRank ∧
    targetRank ≤ cumulativeThrough histogram bin

/-- Counting through a bin adds the bin's own count to the count before it. -/
theorem cumulativeThrough_eq_before_add {bins : ℕ} (histogram : Histogram bins)
    (bin : Fin bins) :
    cumulativeThrough histogram bin = cumulativeBefore histogram bin + histogram bin := by
  rw [cumulativeThrough, cumulativeBefore, Finset.Iic_eq_cons_Iio, Finset.sum_cons]
  exact Nat.add_comm _ _

/-- A selected bin is nonempty, so its interpolation denominator is nonzero. -/
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

/-- Pooling commutes with cumulative counts: the cumulative count of a sum of histograms is the
sum of their cumulative counts. -/
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

/-- For a score in `[0, 1]` and a cutoff in `[biasMin, 1 + biasMax]`, the required bias
`cutoff - score` lies in the report's adaptive histogram range `[biasMin - 1, biasMax + 1]`. -/
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

/-- For a nonnegative bin width, the interpolated value stays inside its bin. -/
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

/-- Appendix D's quantile estimate, which interpolates with the real target load rather than the
rounded-up rank used to select the bin. -/
noncomputable def estimate (lower width : ℝ) (bin : ℕ)
    (targetLoad : ℝ) (cumulativeBefore countInBin : ℕ) : ℝ :=
  interpolate lower width bin
    ((targetLoad - cumulativeBefore) / countInBin)

/-- If the bin is selected with rank `⌈targetLoad⌉₊`, the within-bin fraction computed from
`targetLoad` itself lies in `(0, 1]`. -/
theorem fraction_mem_of_isSelectedBin {bins : ℕ} {histogram : Histogram bins}
    {targetLoad : ℝ} {bin : Fin bins}
    (hSelected : IsSelectedBin histogram ⌈targetLoad⌉₊ bin) :
    0 < (targetLoad - cumulativeBefore histogram bin) / histogram bin ∧
      (targetLoad - cumulativeBefore histogram bin) / histogram bin ≤ 1 := by
  have hCount : (0 : ℝ) < histogram bin := by
    exact_mod_cast count_pos_of_isSelectedBin hSelected
  have hLower : (cumulativeBefore histogram bin : ℝ) < targetLoad :=
    Nat.lt_ceil.mp hSelected.1
  have hUpper : targetLoad ≤ (cumulativeThrough histogram bin : ℝ) :=
    Nat.ceil_le.mp hSelected.2
  rw [cumulativeThrough_eq_before_add, Nat.cast_add] at hUpper
  constructor
  · exact div_pos (sub_pos.mpr hLower) hCount
  · apply (div_le_iff₀ hCount).mpr
    linarith

/-- Any true quantile in the same bin is within one bin width of the estimate. -/
theorem abs_estimate_sub_le_width {lower width trueQuantile : ℝ} {bin : ℕ}
    (targetRank : ℝ) (cumulative countInBin : ℕ) (hWidth : 0 ≤ width)
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
