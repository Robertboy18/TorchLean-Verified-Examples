/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import Mathlib.Data.Fintype.Card
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Basic.Real.Basic

/-!
# Kimi K3 post-training control protocols

The Kimi K3 report describes three domain-specialized policies at each of three reasoning-effort
levels, a hard reward override for trajectories exceeding their token budget, and an analogous
verbosity rule for pairwise judging.  These rules are deterministic mathematics even though the
learned teachers, generated trajectories, and judge scores are empirical inputs.
-/

@[expose] public section

namespace KimiK3
namespace PostTraining

/-- The three policy domains used by multi-teacher on-policy distillation. -/
inductive Domain where
  | general
  | agentic
  | coding
  deriving DecidableEq

instance : Fintype Domain where
  elems := {.general, .agentic, .coding}
  complete domain := by cases domain <;> simp

/-- The three reasoning-effort levels named in the report. -/
inductive Effort where
  | low
  | high
  | max
  deriving DecidableEq

instance : Fintype Effort where
  elems := {.low, .high, .max}
  complete effort := by cases effort <;> simp

/-- A teacher is selected by one domain and one reasoning-effort level. -/
abbrev TeacherIndex := Domain × Effort

/-- The report's Cartesian product contains exactly nine specialized teachers. -/
theorem teacherIndex_card : Fintype.card TeacherIndex = 9 := by
  decide

/-- Apply the report's hard budget override to a task reward.

The threshold is real-valued because the curriculum multiplier need not be integral, while token
usage and the cold-start budget are natural counts.
-/
noncomputable def budgetControlledReward
    (initialBudget usedTokens : ℕ) (multiplier taskReward : ℝ) : ℝ :=
  if multiplier * initialBudget < usedTokens then -1 else taskReward

/-- The hard override preserves the usual normalized reward interval. -/
theorem budgetControlledReward_mem_interval
    (initialBudget usedTokens : ℕ) (multiplier taskReward : ℝ)
    (hTaskLower : -1 ≤ taskReward) (hTaskUpper : taskReward ≤ 1) :
    -1 ≤ budgetControlledReward initialBudget usedTokens multiplier taskReward ∧
      budgetControlledReward initialBudget usedTokens multiplier taskReward ≤ 1 := by
  by_cases hExceeded : multiplier * initialBudget < usedTokens
  · simp [budgetControlledReward, hExceeded]
  · simpa [budgetControlledReward, hExceeded] using And.intro hTaskLower hTaskUpper

/-- Reapplying the same budget rule cannot change an already controlled reward. -/
theorem budgetControlledReward_idempotent
    (initialBudget usedTokens : ℕ) (multiplier taskReward : ℝ) :
    budgetControlledReward initialBudget usedTokens multiplier
        (budgetControlledReward initialBudget usedTokens multiplier taskReward) =
      budgetControlledReward initialBudget usedTokens multiplier taskReward := by
  by_cases hExceeded : multiplier * initialBudget < usedTokens <;>
    simp [budgetControlledReward, hExceeded]

/-- Enforce the report's verbosity limit for the first candidate in a binary comparison. -/
noncomputable def enforceFirstCandidateVerbosity
    (initialLength candidateLength : ℕ) (multiplier : ℝ)
    (uncontrolled : Ordering) : Ordering :=
  if multiplier * initialLength < candidateLength then .lt else uncontrolled

/-- If the controlled comparison ranks the first candidate higher, it was within budget.

`Ordering.gt` means that the first argument ranks above the second; `Ordering.lt` is the forced-loss
result used when the first candidate is too verbose.
-/
theorem firstCandidateWins_only_if_within_budget
    {initialLength candidateLength : ℕ} {multiplier : ℝ} {uncontrolled : Ordering}
    (hWins : enforceFirstCandidateVerbosity initialLength candidateLength multiplier
      uncontrolled = .gt) :
    (candidateLength : ℝ) ≤ multiplier * initialLength := by
  by_contra hWithin
  have hExceeded : multiplier * initialLength < candidateLength := lt_of_not_ge hWithin
  simp [enforceFirstCandidateVerbosity, hExceeded] at hWins

end PostTraining
end KimiK3
