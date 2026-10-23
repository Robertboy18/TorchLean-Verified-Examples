/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import ProgramEquilibrium.Modal

/-!
# Modal agents and the outcome table

A `Universe` is a population of agents in the style of Barasz et al. (2014). Each agent has a
`schema`: a modalized formula whose atoms are `Target.me` ("my opponent cooperates with me") and
`Target.agent c` ("my opponent cooperates with `c`"). The same schema is used against every
opponent, so an agent cannot look at who it is playing, only at what is provable about how the
opponent plays. Their definition also orders agents by rank, so that an agent only mentions
simpler ones; the computation here does not need that.

`play n a b` is whether `a` cooperates with `b` at world `n`. It is computed world by world, which
works because a modalized formula at world `n` only reads worlds below `n`. `play_eq` says the
table satisfies every agent's defining equation at every world, `play_unique` says it is the only
valuation that does, and `play_stabilizes` says every matchup settles from some world on, so that
`cooperates_or_defects` gives each matchup one eventual outcome.

For agents given by arithmetic formulas whose defining equations Peano arithmetic proves, ordered
by rank, and assuming arithmetic is sound, that eventual value is what an agent does in the
standard model. Getting there takes the closed modal representatives of Barasz et al. (Theorem
4.7, from the modal fixed point theorem, Theorems 4.2 and 4.3, after Lindström, and the
arithmetic soundness of GL, Theorem 4.1), the soundness of the chain for GL together with
`play_unique`, and the arithmetic truth of the resulting letterless formulas. Only the chain part
is proved here.
-/

@[expose] public section

namespace ProgramEquilibrium

open Form

/-- What an agent's formula can mention: whether its opponent cooperates with the agent itself,
or with the fixed agent `c`. -/
inductive Target (Agent : Type) where
  | me
  | agent (c : Agent)
  deriving DecidableEq, Repr

/-- A population of modal agents. -/
structure Universe where
  /-- The agents. -/
  Agent : Type
  /-- Each agent's formula, the same against every opponent. -/
  schema : Agent → Form (Target Agent)
  /-- Every atom in an agent's schema occurs under a box. -/
  modalized : ∀ a, (schema a).modalized = true

namespace Universe

variable (U : Universe)

/-- `a`'s formula against `b`, over the atoms "`x` cooperates with `y`". -/
def rule (a b : U.Agent) : Form (U.Agent × U.Agent) :=
  (U.schema a).map fun
    | .me => (b, a)
    | .agent c => (b, c)

theorem rule_modalized (a b : U.Agent) : (U.rule a b).modalized = true := by
  simp [rule, U.modalized a]

/-- `play n a b`: whether `a` cooperates with `b` at world `n`. -/
def play (n : ℕ) (a b : U.Agent) : Bool :=
  eval (fun m p => if _ : m < n then play m p.1 p.2 else false) n (U.rule a b)
termination_by n

/-- The table satisfies every agent's defining equation at every world. -/
theorem play_eq (n : ℕ) (a b : U.Agent) :
    U.play n a b = eval (fun m p => U.play m p.1 p.2) n (U.rule a b) := by
  rw [play]
  exact eval_congr_of_modalized (fun m hm p => by simp [hm]) (U.rule_modalized a b)

/-- Any valuation that satisfies the agents' equations at every world is the table. -/
theorem play_unique (V : ℕ → U.Agent → U.Agent → Bool)
    (hV : ∀ n a b, V n a b = eval (fun m p => V m p.1 p.2) n (U.rule a b)) :
    ∀ n a b, V n a b = U.play n a b := by
  intro n
  induction n using Nat.strongRecOn with
  | _ n ih =>
    intro a b
    rw [hV, U.play_eq]
    exact eval_congr_of_modalized (fun m hm p => ih m hm p.1 p.2) (U.rule_modalized a b)

/-- Every matchup settles: from some world on, its entry in the table no longer changes. -/
theorem play_stabilizes (a b : U.Agent) : ∃ K, ∀ n ≥ K, U.play n a b = U.play K a b := by
  obtain ⟨K, hK⟩ := stabilizes_of_modalized (fun m p => U.play m p.1 p.2) (U.rule_modalized a b)
  exact ⟨K, fun n hn => by rw [U.play_eq, U.play_eq K, hK n hn]⟩

/-- From some world on, `a` cooperates with `b`. -/
def Cooperates (a b : U.Agent) : Prop :=
  ∃ K, ∀ n ≥ K, U.play n a b = true

/-- From some world on, `a` defects against `b`. -/
def Defects (a b : U.Agent) : Prop :=
  ∃ K, ∀ n ≥ K, U.play n a b = false

/-- Every matchup has an eventual outcome. -/
theorem cooperates_or_defects (a b : U.Agent) : U.Cooperates a b ∨ U.Defects a b := by
  obtain ⟨K, hK⟩ := U.play_stabilizes a b
  cases h : U.play K a b
  · exact Or.inr ⟨K, fun n hn => (hK n hn).trans h⟩
  · exact Or.inl ⟨K, fun n hn => (hK n hn).trans h⟩

/-- and only one. -/
theorem not_cooperates_and_defects (a b : U.Agent) : ¬ (U.Cooperates a b ∧ U.Defects a b) := by
  rintro ⟨⟨K, hK⟩, ⟨L, hL⟩⟩
  have := (hK (max K L) (by omega)).symm.trans (hL (max K L) (by omega))
  simp at this

section FairBot

variable {U} {f : U.Agent}

/-- FairBot's formula, `□ [opponent cooperates with me]`, against `b`. -/
theorem fair_rule (hf : U.schema f = box (atom .me)) (b : U.Agent) :
    U.rule f b = box (atom (b, f)) := by
  simp [rule, hf, map]

/-- FairBot defects at every world above one where `b` defects against it. -/
theorem fair_defects_above (hf : U.schema f = box (atom .me)) {b : U.Agent} {m : ℕ}
    (hm : U.play m b f = false) : ∀ n > m, U.play n f b = false := by
  intro n hn
  rw [U.play_eq, fair_rule hf]
  apply Bool.eq_false_iff.2
  intro h
  have := (eval_box _ n _).1 h m hn
  simp [eval, hm] at this

/-- FairBot cannot be exploited: if `b` eventually defects against it, it eventually defects
against `b`. -/
theorem fair_unexploitable (hf : U.schema f = box (atom .me)) {b : U.Agent}
    (h : U.Defects b f) : U.Defects f b := by
  obtain ⟨K, hK⟩ := h
  exact ⟨K + 1, fun n hn => fair_defects_above hf (hK K (Nat.le_refl K)) n (by omega)⟩

/-- FairBot eventually cooperates with `b` exactly when `b` cooperates with FairBot at every
world. -/
theorem fair_cooperates_iff (hf : U.schema f = box (atom .me)) (b : U.Agent) :
    U.Cooperates f b ↔ ∀ m, U.play m b f = true := by
  constructor
  · rintro ⟨K, hK⟩ m
    have h := hK (max K (m + 1)) (by omega)
    rw [U.play_eq, fair_rule hf] at h
    exact (eval_box _ _ _).1 h m (by omega)
  · intro h
    refine ⟨0, fun n _ => ?_⟩
    rw [U.play_eq, fair_rule hf]
    exact (eval_box _ _ _).2 fun m _ => h m

end FairBot

end Universe

end ProgramEquilibrium
