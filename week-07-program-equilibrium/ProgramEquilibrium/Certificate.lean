/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import ProgramEquilibrium.Agents

/-!
# Certificates for finite modal populations

A certificate supplies a world and an outcome table. Agreement of every nested box
between that world and its successor proves that the entire table has settled.
The executable checker binds the supplied table to those permanent outcomes.
Acceptance additionally requires non-exploitation, self-cooperation and a partner.
Its conclusions cover exactly the supplied population, in chain semantics.
-/

@[expose] public section

namespace ProgramEquilibrium
namespace Form

variable {α : Type}

/-- Every nested box agrees between the two worlds. Atoms outside boxes are not compared. -/
def BoxesAgree (V : ℕ → α → Bool) (m n : ℕ) : Form α → Prop
  | atom _ | top | bot => True
  | not φ => BoxesAgree V m n φ
  | and φ ψ | or φ ψ | imp φ ψ => BoxesAgree V m n φ ∧ BoxesAgree V m n ψ
  | box φ => eval V m (box φ) = eval V n (box φ) ∧ BoxesAgree V m n φ

theorem eval_eq_of_boxes {V : ℕ → α → Bool} {m n : ℕ}
    (hV : ∀ a, V m a = V n a) (φ : Form α) (h : BoxesAgree V m n φ) :
    eval V m φ = eval V n φ := by
  induction φ with
  | atom a => exact hV a
  | top | bot => rfl
  | not φ ih => exact congrArg (! ·) (ih h)
  | and φ ψ ihφ ihψ => exact congrArg₂ (· && ·) (ihφ h.1) (ihψ h.2)
  | or φ ψ ihφ ihψ => exact congrArg₂ (· || ·) (ihφ h.1) (ihψ h.2)
  | imp φ ψ ihφ ihψ => exact congrArg₂ (fun x y => !x || y) (ihφ h.1) (ihψ h.2)
  | box φ => exact h.1

theorem eval_eq_of_modalized_boxes {V : ℕ → α → Bool} {m n : ℕ}
    {φ : Form α} (hφ : φ.modalized = true) (h : BoxesAgree V m n φ) :
    eval V m φ = eval V n φ := by
  induction φ with
  | atom a => simp [modalized] at hφ
  | top | bot => rfl
  | not φ ih => exact congrArg (! ·) (ih hφ h)
  | and φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    exact congrArg₂ (· && ·) (ihφ hφ.1 h.1) (ihψ hφ.2 h.2)
  | or φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    exact congrArg₂ (· || ·) (ihφ hφ.1 h.1) (ihψ hφ.2 h.2)
  | imp φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    exact congrArg₂ (fun x y => !x || y) (ihφ hφ.1 h.1) (ihψ hφ.2 h.2)
  | box φ => exact h.1

theorem eval_box_succ (V : ℕ → α → Bool) (n : ℕ) (φ : Form α) :
    eval V (n + 1) (box φ) = (eval V n (box φ) && eval V n φ) := by
  simp [eval, List.range_succ, List.all_append]

theorem boxes_agree_succ {V : ℕ → α → Bool} {m n : ℕ}
    (hV : ∀ a, V m a = V n a) (φ : Form α) (h : BoxesAgree V m n φ) :
    BoxesAgree V (m + 1) (n + 1) φ := by
  induction φ with
  | atom | top | bot => trivial
  | not φ ih => exact ih h
  | and φ ψ ihφ ihψ | or φ ψ ihφ ihψ | imp φ ψ ihφ ihψ =>
    exact ⟨ihφ h.1, ihψ h.2⟩
  | box φ ih =>
    refine ⟨?_, ih h.2⟩
    rw [eval_box_succ, eval_box_succ, h.1, eval_eq_of_boxes hV φ h.2]

end Form

namespace Universe

variable (U : Universe)

/-- A certificate records agreement of every nested box, not merely the visible play table. -/
def Stopped (n : ℕ) : Prop :=
  ∀ a b, Form.BoxesAgree (fun k p => U.play k p.1 p.2) n (n + 1) (U.rule a b)

theorem play_eq_succ_of_stopped {n : ℕ} (h : U.Stopped n) (a b : U.Agent) :
    U.play n a b = U.play (n + 1) a b := by
  rw [U.play_eq, U.play_eq]
  exact Form.eval_eq_of_modalized_boxes (U.rule_modalized a b) (h a b)

theorem stopped_succ {n : ℕ} (h : U.Stopped n) : U.Stopped (n + 1) := by
  intro a b
  exact Form.boxes_agree_succ (fun p => U.play_eq_succ_of_stopped h p.1 p.2)
    (U.rule a b) (h a b)

/-- Once the complete box state stops changing, every matchup is permanently settled. -/
theorem play_eq_of_stopped {n : ℕ} (h : U.Stopped n) {m : ℕ} (hm : n ≤ m)
    (a b : U.Agent) : U.play m a b = U.play n a b := by
  have step : ∀ k, U.Stopped (n + k) := by
    intro k
    induction k with
    | zero => exact h
    | succ k ih => exact U.stopped_succ ih
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hm
  clear hm
  induction k with
  | zero => rfl
  | succ k ih =>
    exact (U.play_eq_succ_of_stopped (step k) a b).symm.trans ih

end Universe
end ProgramEquilibrium

namespace ProgramEquilibrium.Certificate

open ProgramEquilibrium

namespace Form

/-- Executable comparison of all box occurrences, including those nested below other boxes. -/
def boxesAgree {α : Type} (V : ℕ → α → Bool) (m n : ℕ) :
    ProgramEquilibrium.Form α → Bool
  | .atom _ | .top | .bot => true
  | .not φ => boxesAgree V m n φ
  | .and φ ψ | .or φ ψ | .imp φ ψ => boxesAgree V m n φ && boxesAgree V m n ψ
  | .box φ => (ProgramEquilibrium.Form.eval V m (.box φ) ==
      ProgramEquilibrium.Form.eval V n (.box φ)) && boxesAgree V m n φ

theorem boxesAgree_sound {α : Type} {V : ℕ → α → Bool} {m n : ℕ}
    (φ : ProgramEquilibrium.Form α) (h : boxesAgree V m n φ = true) :
    ProgramEquilibrium.Form.BoxesAgree V m n φ := by
  induction φ with
  | atom | top | bot => trivial
  | not φ ih => exact ih h
  | and φ ψ ihφ ihψ | or φ ψ ihφ ihψ | imp φ ψ ihφ ihψ =>
    exact ⟨ihφ (Bool.and_eq_true_iff.mp h).1, ihψ (Bool.and_eq_true_iff.mp h).2⟩
  | box φ ih =>
    exact ⟨by simpa only [beq_iff_eq] using
      (Bool.and_eq_true_iff.mp h).1, ih (Bool.and_eq_true_iff.mp h).2⟩

end Form

variable (U : Universe) (agents : List U.Agent)

/-- The row agent acts against the column agent. -/
def table (world : ℕ) : List (List Bool) :=
  agents.map fun a => agents.map fun b => U.play world a b

/-- Check the complete box state rather than only equality of successive outcome tables. -/
def stoppedCheck (world : ℕ) : Bool :=
  agents.all fun a => agents.all fun b =>
    Form.boxesAgree (fun k p => U.play k p.1 p.2) world (world + 1) (U.rule a b)

theorem stoppedCheck_sound (complete : ∀ a, a ∈ agents) {world : ℕ}
    (h : stoppedCheck U agents world = true) : U.Stopped world := by
  intro a b
  exact Form.boxesAgree_sound _
    ((List.all_eq_true.mp ((List.all_eq_true.mp h) a (complete a))) b (complete b))

/-- A valid certificate binds the supplied table to the permanent table of this population. -/
def check (world : ℕ) (claimed : List (List Bool)) : Bool :=
  stoppedCheck U agents world && (claimed == table U agents world)

theorem check_stable (complete : ∀ a, a ∈ agents) {world : ℕ}
    {claimed : List (List Bool)} (h : check U agents world claimed = true)
    {later : ℕ} (hlater : world ≤ later) (a b : U.Agent) :
    U.play later a b = U.play world a b :=
  U.play_eq_of_stopped
    (stoppedCheck_sound U agents complete (Bool.and_eq_true_iff.mp h).1) hlater a b

theorem check_table {world : ℕ} {claimed : List (List Bool)}
    (h : check U agents world claimed = true) :
    claimed = table U agents world := by
  simpa only [beq_iff_eq] using (Bool.and_eq_true_iff.mp h).2

/-- The candidate never cooperates unilaterally in the certified table. -/
def safeCheck (world : ℕ) (candidate : U.Agent) : Bool :=
  agents.all fun b => !U.play world candidate b || U.play world b candidate

/-- Mutual cooperation with another agent; self-play alone does not satisfy this condition. -/
def externalCooperation [DecidableEq U.Agent] (world : ℕ) (candidate : U.Agent) : Bool :=
  agents.any fun b => decide (b ≠ candidate) &&
    U.play world candidate b && U.play world b candidate

/-- Certificate validity and desirability are different checks: an always-defecting candidate
can submit a valid certificate but does not pass this policy criterion. -/
def accepted [DecidableEq U.Agent] (world : ℕ) (claimed : List (List Bool))
    (candidate : U.Agent) : Bool :=
  check U agents world claimed && safeCheck U agents world candidate &&
    U.play world candidate candidate && externalCooperation U agents world candidate

/-- Accepted candidates cannot be exploited by any member of the supplied population in the
eventual chain outcomes. This does not quantify over agents absent from that population. -/
theorem accepted_safe [DecidableEq U.Agent] (complete : ∀ a, a ∈ agents)
    {world : ℕ} {claimed : List (List Bool)} {candidate : U.Agent}
    (h : accepted U agents world claimed candidate = true) (b : U.Agent)
    (hc : U.Cooperates candidate b) : U.Cooperates b candidate := by
  have parts := Bool.and_eq_true_iff.mp h
  have parts' := Bool.and_eq_true_iff.mp parts.1
  have parts'' := Bool.and_eq_true_iff.mp parts'.1
  have hs := stoppedCheck_sound U agents complete (Bool.and_eq_true_iff.mp parts''.1).1
  obtain ⟨k, hk⟩ := hc
  have forward : U.play world candidate b = true :=
    (U.play_eq_of_stopped hs (Nat.le_max_left world k) candidate b).symm.trans
      (hk _ (Nat.le_max_right world k))
  have backward := (List.all_eq_true.mp parts''.2) b (complete b)
  have hb : U.play world b candidate = true := by simpa [forward] using backward
  exact ⟨world, fun n hn => (U.play_eq_of_stopped hs hn b candidate).trans hb⟩

theorem accepted_self_cooperation [DecidableEq U.Agent] (complete : ∀ a, a ∈ agents)
    {world : ℕ} {claimed : List (List Bool)} {candidate : U.Agent}
    (h : accepted U agents world claimed candidate = true) : U.Cooperates candidate candidate := by
  have parts := Bool.and_eq_true_iff.mp h
  have parts' := Bool.and_eq_true_iff.mp parts.1
  have hs := stoppedCheck_sound U agents complete
    (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp parts'.1).1).1
  exact ⟨world, fun n hn => (U.play_eq_of_stopped hs hn candidate candidate).trans parts'.2⟩

theorem accepted_external_cooperation [DecidableEq U.Agent] (complete : ∀ a, a ∈ agents)
    {world : ℕ} {claimed : List (List Bool)} {candidate : U.Agent}
    (h : accepted U agents world claimed candidate = true) :
    ∃ b, b ≠ candidate ∧ U.Cooperates candidate b ∧ U.Cooperates b candidate := by
  have parts := Bool.and_eq_true_iff.mp h
  have hs := stoppedCheck_sound U agents complete
    (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp parts.1).1).1).1
  obtain ⟨b, _, hb⟩ := List.any_eq_true.mp parts.2
  have hpair := Bool.and_eq_true_iff.mp hb
  have hleft := Bool.and_eq_true_iff.mp hpair.1
  refine ⟨b, of_decide_eq_true hleft.1, ?_, ?_⟩
  · exact ⟨world, fun n hn => (U.play_eq_of_stopped hs hn candidate b).trans hleft.2⟩
  · exact ⟨world, fun n hn => (U.play_eq_of_stopped hs hn b candidate).trans hpair.2⟩

end ProgramEquilibrium.Certificate

