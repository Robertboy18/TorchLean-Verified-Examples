/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import ProgramEquilibrium.Guards

/-!
# The policy recovered by both samplers

The primary selection uses the supplied delayed-reciprocity control. The model-only
selections are Boolean rewrites of that control, as checked below. These theorems
concern modal semantics, not the neural models that proposed the formulas.
-/

@[expose] public section

namespace ProgramEquilibrium.Inkling

open ProgramEquilibrium Form

/-- Literal translation of the selected JSON constructor tree. -/
def control {Agent : Type} : Form (Target Agent) :=
  and (box (imp (not (box bot)) (atom .me))) top

/-- Literal constructor tree proposed by the base sampler. -/
def baseProposal {Agent : Type} : Form (Target Agent) :=
  and (box (imp (not (box bot)) (atom .me))) control

/-- Literal constructor tree proposed by the trained sampler. -/
def trainedProposal {Agent : Type} : Form (Target Agent) :=
  and (box (imp (not (box bot)) (atom .me)))
    (or (box (atom .me)) (not (box (atom .me))))

/-- The base sampler duplicated an existing conjunct rather than changing the rule. -/
theorem eval_base_eq_control {Agent : Type} (V : ℕ → Target Agent → Bool) (n : ℕ) :
    eval V n baseProposal = eval V n control := by
  simp [baseProposal, control, eval]

/-- The trained sampler's added guard is a Boolean tautology. -/
theorem eval_trained_eq_control {Agent : Type} (V : ℕ → Target Agent → Bool) (n : ℕ) :
    eval V n trainedProposal = eval V n control := by
  simp [trainedProposal, control, eval]

/-- The selected policy cannot eventually cooperate with an eventual defector. -/
theorem control_not_exploited (U : Universe) (f b : U.Agent)
    (hf : U.schema f = control) :
    ¬ (U.Cooperates f b ∧ U.Defects b f) := by
  exact Universe.delayed_guard_not_exploited (k := 1) (guard := top) hf

/-- Self-cooperation is not merely observed on the finite benchmark populations. -/
theorem control_self_cooperates (U : Universe) (f : U.Agent)
    (hf : U.schema f = control) : U.Cooperates f f := by
  refine ⟨0, fun n _ => ?_⟩
  induction n using Nat.strongRecOn with
  | _ n ih =>
    rw [U.play_eq]
    simp only [Universe.rule, hf, control, map, eval, Bool.and_true]
    apply List.all_eq_true.mpr
    intro m hm
    have hprior := ih m (List.mem_range.mp hm)
    simp [hprior]

end ProgramEquilibrium.Inkling

