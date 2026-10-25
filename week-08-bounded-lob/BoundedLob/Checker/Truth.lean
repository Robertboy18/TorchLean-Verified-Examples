/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Check

/-!
# What the sentences say, and soundness

`holds n φ e` is the truth of `φ` at `k = n`. A box is true when the checker accepts some
certificate within its budget, so truth refers to the checker and never the other way round.
`coop a b` is true when the source `a`, read against `b`, is true; reading a source fills its
placeholders (`e = some (b, a)`), and since every placeholder of a guarded source sits under a
box, reading never needs the truth of another `coop` with a placeholder.

`holds` is defined for every formula, not only guarded sources. A bare placeholder, or a `coop`
outside a box with a placeholder argument, is false; such formulas never occur in the agents'
sources, and the checker's rule for sources (`prog`) only accepts guarded ones.

`sound`: every sentence with a checked certificate is true.

Three readings of a formula appear in turn. `holds` is the intended one, with a box read as
bounded provability. `worlds` (in `Worlds`) forgets the budgets: a sentence false at some world
has no proof, but one true at the worlds an agent tests may still have none. `eval` (in
`Agents/Search`) computes `holds` by proof search (`eval_spec`).
-/

@[expose] public section

namespace BoundedLob

open Form

/-- Truth at `k = n`, while reading the match `e` (`none` for a plain sentence). -/
def holds (n : ℕ) : Form → Option (Form × Form) → Prop
  | .bot, _ => False
  | .imp φ ψ, e => holds n φ e → holds n ψ e
  | .iff φ ψ, e => (holds n φ e ↔ holds n ψ e)
  | .box t φ, e => Provable (t.eval id n) (.inst n (φ.within e))
  | .coop a b, _ => (a.isPlaceholder || b.isPlaceholder) = false ∧ holds n a (some (b, a))
  | .opp, _ => False
  | .me, _ => False
  | .diag g p, e => Provable (g.eval id n) (.inst n ((Form.diag g p).within e)) → holds n p e

/-- A true sentence: a guarded universal is true at every `k` past its guard. -/
def Sen.holds : Sen → Prop
  | .all a φ => ∀ k > a, BoundedLob.holds k φ none
  | .inst n φ => BoundedLob.holds n φ none

/-- Reading a guarded source in a match is the same as reading its filled text. -/
theorem holds_fill {o m : Form} (ho : o.isPlaceholder = false) (hm : m.isPlaceholder = false)
    (n : ℕ) :
    ∀ (a : Form), a.guarded = true → ∀ e, holds n (a.fill o m) e ↔ holds n a (some (o, m))
  | .bot, _, _ => Iff.rfl
  | .imp φ ψ, h, e => by
    simp only [guarded, Bool.and_eq_true] at h
    simp only [fill, holds, holds_fill ho hm n φ h.1 e, holds_fill ho hm n ψ h.2 e]
  | .iff φ ψ, h, e => by
    simp only [guarded, Bool.and_eq_true] at h
    simp only [fill, holds, holds_fill ho hm n φ h.1 e, holds_fill ho hm n ψ h.2 e]
  | .box t φ, _, e => by
    simp only [fill, holds]
    rw [Form.within_fill ho hm]
    rfl
  | .coop a b, h, e => by
    simp only [guarded, Bool.and_eq_true, Bool.not_eq_true'] at h
    simp only [fill, holds, Form.fillArg_self h.1, Form.fillArg_self h.2]
  | .opp, h, _ => by simp [guarded] at h
  | .me, h, _ => by simp [guarded] at h
  | .diag g p, h, e => by
    simp only [guarded] at h
    have hw : (Form.diag g (p.fill o m)).within e = Form.diag g (p.fill o m) :=
      Form.within_fill ho hm (Form.diag g p) e
    simp only [fill, holds]
    rw [hw, holds_fill ho hm n p h e]
    rfl

/-! ### Soundness -/

namespace Sen

/-- A formula true at every `k` makes every sentence about it true. -/
theorem holds_of_valid {σ : Sen} (hv : ∀ n, BoundedLob.holds n σ.form none) : σ.holds := by
  cases σ <;> simp_all [Sen.holds, Sen.form]

/-- Modus ponens for true sentences under the same guard or instance. -/
theorem holds_mp {σ : Sen} {φ : Form} (h₁ : (σ.withForm (imp φ σ.form)).holds)
    (h₂ : (σ.withForm φ).holds) :
    σ.holds := by
  cases σ with
  | all a ψ => exact fun k hk => h₁ k hk (h₂ k hk)
  | inst n ψ => exact h₁ h₂

/-- A pointwise implication between the formulas carries a true sentence over. -/
theorem holds_of_withForm {σ : Sen} {φ : Form} (hσ : ∀ n, BoundedLob.holds n φ none →
    BoundedLob.holds n σ.form none) (h : (σ.withForm φ).holds) : σ.holds := by
  cases σ with
  | all a ψ => exact fun k hk => hσ k (h k hk)
  | inst n ψ => exact hσ n h

end Sen

/-- Implication distribution at one instance. -/
theorem provable_dist {n x y : ℕ} {φ ψ : Form} (h₁ : Provable x (.inst n (imp φ ψ)))
    (h₂ : Provable y (.inst n φ)) : Provable (x + y) (.inst n ψ) := by
  obtain ⟨c, hc, hcl⟩ := h₁
  obtain ⟨d, hd, hdl⟩ := h₂
  refine ⟨.mp φ c d, ?_, ?_⟩
  · simp [check, Sen.withForm, Sen.form, hc, hd]
  · simp only [len_inst, Form.size_imp, Cert.size_mp] at hcl hdl ⊢
    omega

/-- Inner necessitation at one instance, once the text of `x` fits in `x(n)` or in `lg n`.
Boxing a proof of `φ` costs the text of `x` and two characters, the box of the formula and the
`nec` of the certificate, and the hypothesis pays for them out of a second copy of `x(n)`. -/
theorem provable_four {n : ℕ} {x : Bound} {φ : Form}
    (hn : x.size + 2 ≤ x.eval id n ∨ x.size ≤ Nat.log 2 n)
    (h : Provable (x.eval id n) (.inst n φ)) :
    Provable (2 * x.eval id n) (.inst n (box x φ)) := by
  obtain ⟨c, hc, hcl⟩ := h
  refine ⟨.nec c, ?_, ?_⟩
  · simp [check, hc, hcl]
  · have hφ := φ.size_pos
    have hc' := c.size_pos
    simp only [len_inst, Form.size_box, Cert.size_nec] at hcl ⊢
    omega

/-- **Soundness.** A sentence with a checked certificate is true. -/
theorem sound : ∀ (c : Cert) (σ : Sen), check c σ = true → σ.holds
  | .axK, σ, h => by
    simp only [check] at h
    split at h
    · rename_i φ ψ φ' hf
      simp only [beq_iff_eq] at h
      subst h
      exact Sen.holds_of_valid fun n => by rw [hf]; exact fun a _ => a
    · simp at h
  | .axS, σ, h => by
    simp only [check] at h
    split at h
    · rename_i φ ψ χ φ₁ ψ₁ φ₂ χ₁ hf
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩ := h
      subst h₁ h₂ h₃ h₄
      exact Sen.holds_of_valid fun n => by
        rw [hf]; exact fun a b c => a c (b c)
    · simp at h
  | .axDN, σ, h => by
    simp only [check] at h
    split at h
    · rename_i φ φ' hf
      simp only [beq_iff_eq] at h
      subst h
      exact Sen.holds_of_valid fun n => by
        rw [hf]; exact fun a => Classical.byContradiction a
    · simp at h
  | .iffL c, σ, h => by
    simp only [check] at h
    split at h
    · rename_i φ ψ hf
      exact Sen.holds_of_withForm (φ := iff φ ψ) (fun n hv => by rw [hf]; exact hv.1) (sound c _ h)
    · simp at h
  | .iffR c, σ, h => by
    simp only [check] at h
    split at h
    · rename_i ψ φ hf
      exact Sen.holds_of_withForm (φ := iff φ ψ) (fun n hv => by rw [hf]; exact hv.2) (sound c _ h)
    · simp at h
  | .iffI c d, σ, h => by
    simp only [check] at h
    split at h
    · rename_i φ ψ hf
      simp only [Bool.and_eq_true] at h
      have h₁ := sound c _ h.1
      have h₂ := sound d _ h.2
      cases σ with
      | all a χ =>
        simp only [Sen.form] at hf
        subst hf
        exact fun k hk => ⟨h₁ k hk, h₂ k hk⟩
      | inst n χ =>
        simp only [Sen.form] at hf
        subst hf
        exact ⟨h₁, h₂⟩
    · simp at h
  | .mp φ c d, σ, h => by
    simp only [check, Bool.and_eq_true] at h
    exact Sen.holds_mp (sound c _ h.1) (sound d _ h.2)
  | .raise a c, σ, h => by
    cases σ with
    | all b φ =>
      simp only [check, Bool.and_eq_true, decide_eq_true_eq] at h
      exact fun k hk => sound c _ h.2 k (by omega)
    | inst _ _ => simp [check] at h
  | .inst a c, σ, h => by
    cases σ with
    | inst n φ =>
      simp only [check, Bool.and_eq_true, decide_eq_true_eq] at h
      exact sound c _ h.2 n h.1
    | all _ _ => simp [check] at h
  | .qd c, σ, h => by
    simp only [check] at h
    split at h
    · rename_i a M φ
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      intro k hk
      refine ⟨.inst a c, ?_, ?_⟩
      · simp [check, hk, h.2]
      · simp only [len_inst, len_all, Cert.size_inst, Bound.eval, Form.within_none, Cost.inst]
          at h ⊢
        omega
    · simp at h
  | .nec c, σ, h => by
    simp only [check] at h
    split at h
    · rename_i n t φ
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      exact ⟨c, h.2, h.1⟩
    · simp at h
  | .dist, σ, h => by
    simp only [check] at h
    split at h
    · rename_i x φ ψ y φ₁ z ψ₁ hf
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨h₁, h₂⟩, h₃⟩ := h
      subst h₁ h₂ h₃
      exact Sen.holds_of_valid fun n => by
        rw [hf]
        intro hx hy
        have := provable_dist hx hy
        simp only [holds, Form.within_none, Bound.eval, Nat.add_zero]
        exact this
    · simp at h
  | .four, σ, h => by
    simp only [check] at h
    split at h
    · rename_i a x φ y x₁ φ₁
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩ := h
      subst h₁ h₂ h₃
      intro k hk hx
      have := provable_four (Bound.fits_spec h₄ hk) hx
      simp only [holds, Form.within_none, Bound.eval, Cost.expansion]
      exact this
    · rename_i n x φ y x₁ φ₁
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩ := h
      subst h₁ h₂ h₃
      intro hx
      have := provable_four (Or.inl h₄) hx
      simp only [holds, Form.within_none, Bound.eval, Cost.expansion]
      exact this
    · simp at h
  | .mono, σ, h => by
    simp only [check] at h
    split at h
    · rename_i a x φ y φ₁
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨h₁, h₂⟩ := h
      subst h₁
      split at h₂
      · rename_i K hK
        simp only [decide_eq_true_eq] at h₂
        intro k hk hx
        exact hx.mono (Bound.le_of_guard hK k (by omega))
      · simp at h₂
    · rename_i n x φ y φ₁
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      obtain ⟨h₁, h₂⟩ := h
      subst h₁
      exact fun hx => hx.mono h₂
    · simp at h
  | .diag, σ, h => by
    simp only [check] at h
    split at h
    · rename_i g p g₁ g₂ p₁ p₂ hf
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩ := h
      subst h₁ h₂ h₃ h₄
      exact Sen.holds_of_valid fun n => by
        rw [hf]
        exact Iff.rfl
    · simp at h
  | .prog, σ, h => by
    simp only [check] at h
    split at h
    · rename_i a b φ hf
      simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at h
      obtain ⟨⟨⟨ha, hb⟩, hg⟩, hφ⟩ := h
      subst hφ
      exact Sen.holds_of_valid fun n => by
        rw [hf]
        simp only [holds, ha, hb, Bool.or_self, true_and]
        exact (holds_fill hb ha n a hg none).symm
    · simp at h

/-- What has a proof of some length is true. -/
theorem Provable.sound {t : ℕ} {σ : Sen} (h : Provable t σ) : σ.holds :=
  let ⟨c, hc, _⟩ := h
  BoundedLob.sound c σ hc

/-- A match is true exactly when the filled source is. -/
theorem holds_coop {n : ℕ} {a b : Form} (ha : a.isPlaceholder = false)
    (hb : b.isPlaceholder = false) (hg : a.guarded = true) :
    holds n (coop a b) none ↔ holds n (a.fill b a) none := by
  simp only [holds, ha, hb, Bool.or_self, true_and]
  exact (holds_fill hb ha n a hg none).symm

end BoundedLob
