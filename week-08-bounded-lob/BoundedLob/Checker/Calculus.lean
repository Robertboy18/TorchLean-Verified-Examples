/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Rules
public import BoundedLob.Checker.Truth
public import BoundedLob.Theory.Lob

/-!
# The checker has Critch's properties

`calculus` is the proof system of `check` as a `System`: every field of the structure, which the
bounded Löb theorem takes as given, is here a theorem about the checker. Its proofs within `n`
characters are `Provable n`. The constants are `iffCost = 1`, `distCost = 0` (implication
distribution), `instCost = Cost.inst = 4` (quantifier distribution) and
`expansion = Cost.expansion = 2`. Inner necessitation is given here with the guard `2^|x|`,
which works for every bound; the certificate builders use the often smaller `Bound.fitGuard`.
Monotonicity uses the guard of `Bound.guard`.
-/

@[expose] public section

namespace BoundedLob

open Form

/-- The checker as a proof system with Critch's properties. -/
def calculus : System id where
  Sen := Sen
  Pred := Form
  imp := imp
  iff := iff
  box := box
  above := Sen.all
  ProvesIn := Provable
  mp := by
    rintro a b φ ψ ⟨_, c, hc, -⟩ ⟨_, d, hd, -⟩
    exact ⟨_, .mp φ (.raise a c) (.raise b d), by
      simp [check, Sen.withForm, Sen.form, hc, hd], le_rfl⟩
  trans := by
    rintro a b φ ψ χ ⟨_, c, hc, -⟩ ⟨_, d, hd, -⟩
    exact ⟨_, Cert.trans φ ψ χ (.raise a c) (.raise b d), by
      simp [Cert.trans, check, Sen.withForm, Sen.form, hc, hd], le_rfl⟩
  contract := by
    rintro a b φ ψ χ ⟨_, c, hc, -⟩ ⟨_, d, hd, -⟩
    exact ⟨_, Cert.contract φ ψ χ (.raise a c) (.raise b d), by
      simp [Cert.contract, check, Sen.withForm, Sen.form, hc, hd], le_rfl⟩
  imp_mono := by
    rintro a φ ψ χ ⟨_, c, hc, -⟩
    exact ⟨_, Cert.impMono φ ψ χ c, by
      simp [Cert.impMono, check, Sen.withForm, Sen.form, hc], le_rfl⟩
  iffCost := 1
  iff_forward := by
    rintro n a φ ψ ⟨c, hc, hl⟩
    refine ⟨.iffL c, by simp [check, Sen.withForm, Sen.form, hc], ?_⟩
    simp only [len_all, Form.size_imp, Form.size_iff, Cert.size_iffL] at hl ⊢
    omega
  iff_backward := by
    rintro n a φ ψ ⟨c, hc, -⟩
    exact ⟨_, .iffR c, by simp [check, Sen.withForm, Sen.form, hc], le_rfl⟩
  distCost := 0
  imp_dist x y φ ψ := ⟨_, .dist, by simp [check, Sen.form], le_rfl⟩
  instCost := Cost.inst
  quant_dist := by
    rintro N a φ ⟨c, hc, hl⟩
    refine ⟨_, .qd c, ?_, le_rfl⟩
    have : len c (.all a φ) + Cost.inst ≤ Cost.inst + 2 * N := by omega
    simp [check, hc, this]
  expansion := Cost.expansion
  inner_nec x φ := ⟨2 ^ x.size, _, .four, by simp [check, Bound.fits], le_rfl⟩
  mono := by
    intro x y φ hgap
    obtain ⟨K, hK⟩ := Option.isSome_iff_exists.1 (Bound.guard_isSome hgap)
    exact ⟨K, _, .mono, by simp [check, hK], le_rfl⟩
  diag g p := ⟨diag g p, _, .diag, by simp [check, Sen.form], le_rfl⟩

/-- What `calculus` proves about every `k > a` is true for every `k > a`. -/
theorem calculus_sound {a : ℕ} {φ : Form} (h : calculus.Proves (calculus.above a φ)) :
    ∀ k > a, holds k φ none := by
  obtain ⟨t, ht⟩ := h
  exact Provable.sound (t := t) (σ := .all a φ) ht

/-- A checked certificate for `∀ k > a, φ` makes `calculus` prove it. -/
theorem proves_of_check {a : ℕ} {φ : Form} (c : Cert) (h : check c (.all a φ) = true) :
    calculus.Proves (calculus.above a φ) := ⟨_, c, h, le_rfl⟩

/-- A uniform proof gives an affordable proof at every large instance: instantiating it costs
the numeral for `k` and the two three-character wrappers of `inst`. -/
theorem provable_of_proves {K : ℕ} {φ : Form} (h : calculus.Proves (calculus.above K φ)) :
    ∃ K', ∀ k > K', Provable k (.inst k φ) := by
  obtain ⟨_, c, hc, -⟩ := h
  obtain ⟨Ke, hKe⟩ :=
    growsFasterThanLog_id.eventually_le_div (φ.size + Nat.log 2 K + c.size + 6) 1 (by decide)
  refine ⟨max K Ke, fun k hk => ⟨.inst K c, ?_, ?_⟩⟩
  · simp only [check, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨by omega, hc⟩
  · have := hKe k (by omega)
    simp only [len_inst, Cert.size_inst, id, Nat.div_one] at this ⊢
    omega

end BoundedLob
