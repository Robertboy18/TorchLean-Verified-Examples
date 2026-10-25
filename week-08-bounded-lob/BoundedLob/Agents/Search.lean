/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Agents.Candidates
public import BoundedLob.Checker.Parse
public import BoundedLob.Checker.Worlds

/-!
# Agents that search for proofs

`run n a b` plays the agent with source `a` against the agent with source `b` at the budget
parameter `k = n`: it reads `a` with `opp := b` and `me := a`, and decides each box `□_t σ` by
`search`, which looks for a proof of `σ` with at most `t` characters in three stages:

1. if `σ` fails at one of the worlds `0, …, worldDepth - 1` of the chain, there is no proof of
   any length;
2. otherwise it checks the candidates from `candidates`;
3. otherwise it checks every certificate string of at most `t - |σ|` characters.

The first two stages only make it faster. `search_spec`: `search t σ` is exactly "`σ` has a proof
of at most `t` characters", and `run_spec`: `run n a b` is exactly the truth of `coop a b`. The
third stage lists every string, so it is a guarantee and not a practical search. For DUPOC
against itself, `dupoc_fast` proves that the second stage decides; the tournament in `Play`
checks it for each search it plays.
-/

@[expose] public section

namespace BoundedLob

open Form

/-- How many worlds `search` tries, `0` to `5`. A heuristic: the complete search keeps the
answer exact when the worlds find nothing. -/
def worldDepth : ℕ := 6

/-- `c` is a proof of `σ` with at most `t` characters. -/
def affordable (t : ℕ) (σ : Sen) (c : Cert) : Bool := decide (len c σ ≤ t) && check c σ

/-- The complete search over strings. -/
def exhaustive (t : ℕ) (σ : Sen) : Bool :=
  (strings (t - σ.size)).any fun s =>
    match Cert.parse s with
    | some c => affordable t σ c
    | none => false

/-- Some candidate is a proof of `σ` with at most `t` characters. -/
def byCandidate (t : ℕ) (σ : Sen) : Bool := (candidates σ).any (affordable t σ)

/-- Bounded proof search: is there a proof of `σ` with at most `t` characters? -/
def search (t : ℕ) (σ : Sen) : Bool :=
  if refuted worldDepth σ.form then false
  else if byCandidate t σ then true
  else exhaustive t σ

/-- `affordable` as a proposition. -/
theorem affordable_iff {t : ℕ} {σ : Sen} {c : Cert} :
    affordable t σ c = true ↔ check c σ = true ∧ len c σ ≤ t := by
  simp [affordable, and_comm]

/-- The complete search finds a proof exactly when there is one. -/
theorem exhaustive_spec {t : ℕ} {σ : Sen} : exhaustive t σ = true ↔ Provable t σ := by
  constructor
  · intro h
    obtain ⟨s, -, hs⟩ := List.any_eq_true.1 h
    split at hs
    · rename_i c _
      exact ⟨c, affordable_iff.1 hs⟩
    · simp at hs
  · rintro ⟨c, hc, hl⟩
    refine List.any_eq_true.2 ⟨c.text, mem_strings ?_ (fun _ h => Cert.text_subset c h), ?_⟩
    · have : c.text.length = c.size := rfl
      simp only [len] at hl
      omega
    · simp [Cert.parse_text, affordable_iff, hc, hl]

/-- **The search is exact**: `search t σ` holds exactly when `σ` has a proof of at most `t`
characters. -/
theorem search_spec {t : ℕ} {σ : Sen} : search t σ = true ↔ Provable t σ := by
  unfold search
  split
  · rename_i hr
    simp only [Bool.false_eq_true, false_iff]
    exact not_provable_of_refuted hr t
  · split
    · rename_i hh
      simp only [true_iff]
      obtain ⟨c, -, hc⟩ := List.any_eq_true.1 hh
      exact ⟨c, affordable_iff.1 hc⟩
    · exact exhaustive_spec

/-- Read a formula as an agent does, deciding each box by `search`. -/
def eval (n : ℕ) : Form → Option (Form × Form) → Bool
  | .bot, _ => false
  | .imp φ ψ, e => !eval n φ e || eval n ψ e
  | .iff φ ψ, e => eval n φ e == eval n ψ e
  | .box t φ, e => search (t.eval id n) (.inst n (φ.within e))
  | .coop a b, _ => !(a.isPlaceholder || b.isPlaceholder) && eval n a (some (b, a))
  | .opp, _ => false
  | .me, _ => false
  | .diag g p, e => !search (g.eval id n) (.inst n ((Form.diag g p).within e)) || eval n p e

/-- The agent with source `a` against the agent with source `b`, at `k = n`: `true` is
cooperation. -/
def run (n : ℕ) (a b : Form) : Bool := eval n (.coop a b) none

/-- `eval` computes the truth of a formula. -/
theorem eval_spec (n : ℕ) :
    ∀ (φ : Form) (e : Option (Form × Form)), eval n φ e = true ↔ holds n φ e
  | .bot, _ => by simp [eval, holds]
  | .imp φ ψ, e => by
    rw [eval, holds, ← eval_spec n φ e, ← eval_spec n ψ e]
    cases eval n φ e <;> simp
  | .iff φ ψ, e => by
    rw [eval, holds, ← eval_spec n φ e, ← eval_spec n ψ e]
    cases eval n φ e <;> cases eval n ψ e <;> simp
  | .box t φ, e => by rw [eval, holds, search_spec]
  | .coop a b, _ => by
    rw [eval, holds, ← eval_spec n a (some (b, a))]
    simp
  | .opp, _ => by simp [eval, holds]
  | .me, _ => by simp [eval, holds]
  | .diag g p, e => by
    rw [eval, holds, ← eval_spec n p e, ← search_spec]
    cases search (g.eval id n) (.inst n ((Form.diag g p).within e)) <;> simp

/-- **`run` plays exactly what the source says.** -/
theorem run_spec (n : ℕ) (a b : Form) : run n a b = true ↔ holds n (.coop a b) none :=
  eval_spec n _ none

/-- For a guarded source, `run` is the truth of the source with the two sources filled in. -/
theorem run_iff_holds_fill {n : ℕ} {a b : Form} (ha : a.isPlaceholder = false)
    (hb : b.isPlaceholder = false) (hg : a.guarded = true) :
    run n a b = true ↔ holds n (a.fill b a) none := by
  rw [run_spec, holds_coop ha hb hg]

end BoundedLob
