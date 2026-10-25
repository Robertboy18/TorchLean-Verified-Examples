/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Agents.Search
public import BoundedLob.Agents.Sources
public import BoundedLob.Checker.Calculus
public import BoundedLob.Theory.Dimcid
import Mathlib.Tactic.Set

/-!
# Outcomes

Theorems about `run` itself, the program that searches for proofs. Two kinds of argument appear:

* **Bounded Löb**, through `calculus`: the abstract theorems `System.pblt` and
  `System.self_fulfilling` give a proof in the checker's system, `calculus_sound` makes it true,
  and `run_spec` makes it what the program does. These hold for every `k` beyond some threshold.
* **The worlds**: a sentence false at world `0` has no proof at all (`not_provable_of_world0`),
  so a search for it fails at every budget. These hold for every `k`.

For DUPOC against itself the threshold is explicit: the kernel builds the certificate the agent
tries first and checks it, and `dupoc_fast` shows that the search accepts it at every larger
budget.

The outcomes are those of Critch, Dennis and Russell's Theorems 3.4, 3.7 and 5.5, the
conjectured `(D, C)` of their Open Problem 3 and the outcome of their Open Problem 5 in this
system, `(D, D)` for their Open Problem 6, and a correction to their Example 5.3(a): DIMCID
defects against CooperateBot.
-/

@[expose] public section

namespace BoundedLob

open Form Agent

/-! ### DUPOC against itself -/

/-- **Theorem 3.7.** DUPOC cooperates with itself at every large budget, from the abstract
bounded Löb theorem. -/
theorem dupoc_self : ∃ K, ∀ k > K, run k dupoc dupoc = true := by
  have hyp := proves_of_check (a := 0) (φ := imp (box .f (coop dupoc dupoc)) (coop dupoc dupoc))
    (.iffR .prog) (by decide)
  obtain ⟨K, hK⟩ := calculus.pblt .f growsFasterThanLog_id hyp
  exact ⟨K, fun k hk => (run_spec k _ _).2 (calculus_sound hK k hk)⟩

/-- The certificate DUPOC builds against itself, of `∀ k > 17879, coop(DUPOC, DUPOC)`. -/
def selfProof : Candidate :=
  (Candidate.lob .f (coop dupoc dupoc) 0 (.iffR .prog)).get (by decide +kernel)

/-- `selfProof` is what the builder returns. -/
theorem selfProof_built : Candidate.lob .f (coop dupoc dupoc) 0 (.iffR .prog) = some selfProof :=
  (Option.some_get _).symm

/-- The guard of `selfProof`: the largest of the guards of its two monotonicity steps and its
inner necessitation (`K₁`, `K₂` and `a₆` in `Candidate.lob`), computed by the builder. It is
where this certificate becomes valid, not the smallest budget at which two DUPOCs cooperate. -/
def selfThreshold : ℕ := 17879

/-- The kernel computes the guard and size of `selfProof` and checks it. -/
theorem selfProof_spec : selfProof.guard = selfThreshold ∧ selfProof.cert.size = 2962 ∧
    check selfProof.cert (.all selfThreshold (coop dupoc dupoc)) = true := by
  decide +kernel

/-- At `k`, the instance of `selfProof` has `lg k + 2993` characters: the certificate (2962),
its instantiation at the guard (`lg 17879 + 3 = 17`), the formula `coop dupoc dupoc` (11) and
the numeral and markers of the sentence (`lg k + 3`). -/
theorem selfProof_len (k : ℕ) :
    len (.inst selfThreshold selfProof.cert) (.inst k (coop dupoc dupoc)) = Nat.log 2 k + 2993 := by
  simp only [len_inst, Cert.size_inst, selfProof_spec.2.1]
  have : Nat.log 2 selfThreshold = 14 := by decide
  rw [this]
  simp [Form.size, Form.text, dupoc, Bound.text]

/-- The proof fits in the budget from the guard on (`12` is a convenient base for the induction
`L + 2993 ≤ 2^L`). -/
private theorem selfProof_fits {k : ℕ} (hk : selfThreshold < k) : Nat.log 2 k + 2993 ≤ k := by
  have big : ∀ L ≥ 12, L + 2993 ≤ 2 ^ L := by
    intro L hL
    induction L, hL using Nat.le_induction with
    | base => decide
    | succ L _ ih => rw [Nat.pow_succ]; omega
  have hpow := Nat.pow_log_le_self 2 (show k ≠ 0 by omega)
  by_cases hL : Nat.log 2 k ≤ 12
  · simp only [selfThreshold] at hk
    omega
  · exact le_trans (big _ (by omega)) hpow

/-- Past the guard, the instance of `selfProof` checks. -/
theorem selfProof_check {k : ℕ} (hk : selfThreshold < k) :
    check (.inst selfThreshold selfProof.cert) (.inst k (coop dupoc dupoc)) = true := by
  simp [check, hk, selfProof_spec.2.2]

/-- Past the guard, the instance of `selfProof` is among DUPOC's candidates. -/
theorem selfProof_mem {k : ℕ} (hk : selfThreshold < k) :
    Cert.inst selfThreshold selfProof.cert ∈ candidates (.inst k (coop dupoc dupoc)) := by
  simp only [candidates, List.mem_filterMap]
  refine ⟨selfProof, ?_, by simp [selfProof_spec.1, hk]⟩
  have hb : Candidate.dupocBudget dupoc = some .f := rfl
  simp only [Candidate.forForm, Candidate.forCooperation, hb, beq_self_eq_true, ↓reduceIte,
    selfProof_built, Option.toList_some, List.cons_append, List.mem_cons, true_or]

/-- **The fast path.** For every `k > 17879`, DUPOC's search against itself is not refuted by the
worlds and accepts the certificate it builds; the complete search over strings is never
reached. -/
theorem dupoc_fast {k : ℕ} (hk : selfThreshold < k) :
    refuted worldDepth (coop dupoc dupoc) = false ∧
      byCandidate k (.inst k (coop dupoc dupoc)) = true := by
  have hp : Provable k (.inst k (coop dupoc dupoc)) :=
    ⟨_, selfProof_check hk, (selfProof_len k).trans_le (selfProof_fits hk)⟩
  refine ⟨?_, List.any_eq_true.2 ⟨_, selfProof_mem hk, affordable_iff.2 ⟨selfProof_check hk,
    (selfProof_len k).trans_le (selfProof_fits hk)⟩⟩⟩
  cases hr : refuted worldDepth (coop dupoc dupoc)
  · rfl
  · exact absurd hp (not_provable_of_refuted (σ := .inst k (coop dupoc dupoc)) hr k)

/-- **Theorem 3.7, with its threshold.** DUPOC cooperates with itself at every `k > 17879`. -/
theorem dupoc_self_of_lt {k : ℕ} (hk : selfThreshold < k) : run k dupoc dupoc = true := by
  obtain ⟨h₁, h₂⟩ := dupoc_fast hk
  have : run k dupoc dupoc = search k (.inst k (coop dupoc dupoc)) := rfl
  rw [this, search]
  simp [Sen.form, h₁, h₂]

/-! ### Self-fulfilling defection -/

/-- **Theorem 3.4.** CUPOD defects against itself at every large budget. -/
theorem cupod_self : ∃ K, ∀ k > K, run k cupod cupod = false := by
  have hyp := proves_of_check (a := 0)
    (φ := imp (box .f (neg (coop cupod cupod))) (neg (coop cupod cupod)))
    (Cert.swap (coop cupod cupod) (box .f (neg (coop cupod cupod))) bot (.iffL .prog))
    (by decide)
  obtain ⟨K, hK⟩ := calculus.pblt .f growsFasterThanLog_id hyp
  refine ⟨K, fun k hk => ?_⟩
  cases h : run k cupod cupod
  · rfl
  · exact absurd ((run_spec k _ _).1 h) (calculus_sound hK k hk)

/-- DIMCID's defection against `b`, proved in the checker's system. -/
theorem dimcid_proves {b : Form} (hb : b.isPlaceholder = false) :
    ∃ K, calculus.Proves (calculus.above K (neg (coop dimcid b))) := by
  set A := coop dimcid b
  set G := imp A (neg (coop b dimcid))
  have exfalso := proves_of_check (a := 0) (φ := imp (neg A) G)
    (Cert.impMono A bot (neg (coop b dimcid)) (Cert.efq (neg (coop b dimcid)))) (by
      simp [A, G, check, Cert.impMono, Cert.efq, Cert.trans, Sen.withForm, Sen.form])
  have code := proves_of_check (a := 0) (φ := imp (box .f G) (neg A))
    (Cert.swap A (box .f G) bot (.iffL .prog)) (by
      simp [A, G, check, Cert.swap, Cert.trans, Sen.withForm, Sen.form, hb, dimcid, guarded,
        fill, fillArg])
  exact calculus.self_fulfilling growsFasterThanLog_id exfalso code

/-- What `calculus` proves about DIMCID's defection, `run` does. -/
private theorem dimcid_run {b : Form} {K : ℕ}
    (h : calculus.Proves (calculus.above K (neg (coop dimcid b)))) :
    ∀ k > K, run k dimcid b = false := by
  intro k hk
  cases hr : run k dimcid b
  · rfl
  · exact absurd ((run_spec k _ _).1 hr) (calculus_sound h k hk)

/-- DIMCID defects against `b` at every large budget, for each source `b`; the threshold depends
on `b`. -/
theorem dimcid_defects {b : Form} (hb : b.isPlaceholder = false) :
    ∃ K, ∀ k > K, run k dimcid b = false :=
  (dimcid_proves hb).imp fun _ => dimcid_run

/-- **Open Problem 6.** DUPOC against DIMCID: DUPOC defects at every budget, DIMCID at every
large budget. -/
theorem dupoc_dimcid :
    (∀ k, run k dupoc dimcid = false) ∧ ∃ K, ∀ k > K, run k dimcid dupoc = false := by
  refine ⟨fun k => ?_, dimcid_defects rfl⟩
  cases h : run k dupoc dimcid
  · rfl
  · rw [run_iff_holds_fill rfl rfl rfl] at h
    refine absurd h (not_provable_of_world0 ?_)
    simp only [Sen.form, Form.within_none]
    decide

/-- **Correction to Example 5.3(a).** DIMCID defects against CooperateBot at every large budget,
where the paper answers `(C, C)`. -/
theorem dimcid_vs_cooperate :
    ∃ K, ∀ k > K, run k dimcid cooperate = false ∧ run k cooperate dimcid = true :=
  (dimcid_defects rfl).imp fun _ h k hk => ⟨h k hk, rfl⟩

/-- **Theorem 5.5(a).** DIMCID defects against itself at every large budget. -/
theorem dimcid_self : ∃ K, ∀ k > K, run k dimcid dimcid = false := dimcid_defects rfl

/-- **Theorem 5.5(b).** CUPOD against DIMCID ends in mutual defection at every large budget:
CUPOD finds the proof of DIMCID's defection. -/
theorem cupod_dimcid :
    ∃ K, ∀ k > K, run k cupod dimcid = false ∧ run k dimcid cupod = false := by
  obtain ⟨K₁, h₁⟩ := dimcid_proves (b := cupod) rfl
  obtain ⟨K₂, h₂⟩ := provable_of_proves h₁
  refine ⟨max K₁ K₂, fun k hk => ⟨?_, dimcid_run h₁ k (by omega)⟩⟩
  cases h : run k cupod dimcid
  · rfl
  · rw [run_iff_holds_fill rfl rfl rfl] at h
    exact absurd (h₂ k (by omega)) h

/-! ### The worlds -/

/-- **Open Problem 3**, in this system: DUPOC defects and CUPOD cooperates, at every budget. -/
theorem dupoc_cupod (k : ℕ) : run k dupoc cupod = false ∧ run k cupod dupoc = true := by
  constructor
  · cases h : run k dupoc cupod
    · rfl
    · rw [run_iff_holds_fill rfl rfl rfl] at h
      refine absurd h (not_provable_of_world0 ?_)
      simp only [Sen.form, Form.within_none]
      decide
  · rw [run_iff_holds_fill rfl rfl rfl]
    exact not_provable_of_world0 (by simp only [Sen.form, Form.within_none]; decide)

/-- **Open Problem 5**, in this system: CUPOD cooperates and CIMCIC defects, at every budget. -/
theorem cupod_cimcic (k : ℕ) : run k cupod cimcic = true ∧ run k cimcic cupod = false := by
  constructor
  · rw [run_iff_holds_fill rfl rfl rfl]
    exact not_provable_of_world0 (by simp only [Sen.form, Form.within_none]; decide)
  · cases h : run k cimcic cupod
    · rfl
    · rw [run_iff_holds_fill rfl rfl rfl] at h
      refine absurd h (not_provable_of_world0 ?_)
      simp only [Sen.form, Form.within_none]
      decide

end BoundedLob
