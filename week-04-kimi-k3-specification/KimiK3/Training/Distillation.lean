/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# MOPD rewards

The MOPD reward of a sampled token is the clipped teacher-to-student log probability ratio.
The probability subtype excludes zero, so the logarithm is always defined. Theorems below prove
the reward interval and show the reward vanishes when the two probabilities agree. Teacher selection
and the runtime stop-gradient are outside this file.

Reference: Kimi K3 technical report, Section 2.5 and Sections 3--4, pp. 10--14.
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor
open scoped BigOperators

namespace MOPD

/-- A strictly positive probability, safe inside the logarithm of the MOPD reward.

Teacher and student probabilities in Eq. 15 come from softmax distributions, so they are
strictly positive. This is Mathlib's interval subtype.
-/
abbrev StrictProbability := Set.Ioc (0 : ℝ) 1

/-- Symmetric clipping to `[-bound, bound]`. -/
noncomputable def clip (bound value : ℝ) : ℝ := max (-bound) (min bound value)

/-- Per-token teacher/student reward from Eq. 15, before the stop-gradient annotation. -/
noncomputable def reward (maxReward : ℝ)
    (teacherProbability studentProbability : StrictProbability) : ℝ :=
  clip maxReward (Real.log ((teacherProbability : ℝ) / studentProbability))

/-- Clipping does not change a value already inside the requested symmetric interval. -/
theorem clip_eq_self {bound value : ℝ} (hlower : -bound ≤ value) (hupper : value ≤ bound) :
    clip bound value = value := by
  rw [clip, min_eq_right hupper, max_eq_right hlower]

/-- Clipping enforces the report's reward interval for every log-ratio. -/
theorem reward_mem_interval {maxReward : ℝ} (h : 0 ≤ maxReward)
    (teacherProbability studentProbability : StrictProbability) :
    -maxReward ≤ reward maxReward teacherProbability studentProbability ∧
      reward maxReward teacherProbability studentProbability ≤ maxReward := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by linarith) (min_le_left _ _)

/-- A teacher and student assigning the same probability receive zero MOPD reward. -/
@[simp]
theorem reward_self {maxReward : ℝ} (hMaxReward : 0 ≤ maxReward)
    (probability : StrictProbability) :
    reward maxReward probability probability = 0 := by
  have hne : (probability : ℝ) ≠ 0 := ne_of_gt probability.property.1
  rw [reward, div_self hne, Real.log_one]
  exact clip_eq_self (neg_nonpos.mpr hMaxReward) hMaxReward

end MOPD

end KimiK3
