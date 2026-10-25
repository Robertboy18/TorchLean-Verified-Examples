/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Truth

/-!
# The chain of worlds

Forget the budgets and read `□` on the chain `0 < 1 < 2 < …` of provability logic: a box holds at
world `w` when its formula holds at every earlier world, so at world `0` every box holds. Every
rule of the checker is valid at every world (`worlds_sound`), so a sentence that fails at some
world has no proof of any length. An agent tests a few worlds before it searches (`refuted`),
which is how DUPOC defects at once against an opponent it can never prove cooperative.
-/

@[expose] public section

namespace BoundedLob

open Form

/-- Truth at world `w`, budgets ignored. The recursion is on the pair of the world and the
structural size (`sizeOf`, not the printed length). A box, and the boxed self-reference in `diag`,
move to an earlier world, where the formula, filled for the match, may be larger; every other case
moves to a subformula at the same world. -/
def worlds : ℕ → Form → Option (Form × Form) → Bool
  | _, .bot, _ => false
  | w, .imp φ ψ, e => !worlds w φ e || worlds w ψ e
  | w, .iff φ ψ, e => worlds w φ e == worlds w ψ e
  | w, .box _ φ, e => (List.range w).attach.all fun v => worlds v.1 (φ.within e) none
  | w, .coop a b, _ => !(a.isPlaceholder || b.isPlaceholder) && worlds w a (some (b, a))
  | _, .opp, _ => false
  | _, .me, _ => false
  | w, .diag g p, e =>
    !((List.range w).attach.all fun v => worlds v.1 ((Form.diag g p).within e) none) ||
      worlds w p e
termination_by w φ => (w, sizeOf φ)
decreasing_by
  all_goals simp_wf
  all_goals first
    | exact Prod.Lex.left _ _ (List.mem_range.1 v.2)
    | exact Prod.Lex.right _ (by omega)

private theorem imp_iff {a b : Bool} : (!a || b) = true ↔ (a = true → b = true) := by
  cases a <;> simp

private theorem all_range {w : ℕ} {f : ℕ → Bool} :
    ((List.range w).attach.all fun v => f v.1) = true ↔ ∀ v < w, f v = true := by
  simp [List.all_eq_true]

/-! The equations of `worlds`, one per constructor. -/

/-- `worlds` at falsity. -/
theorem worlds_bot (w : ℕ) (e : Option (Form × Form)) : worlds w bot e = false := worlds.eq_1 w e
/-- `worlds` at a bare placeholder. -/
theorem worlds_opp (w : ℕ) (e : Option (Form × Form)) : worlds w opp e = false := worlds.eq_6 w e
/-- `worlds` at a bare placeholder. -/
theorem worlds_me (w : ℕ) (e : Option (Form × Form)) : worlds w me e = false := worlds.eq_7 w e
/-- `worlds` at an implication. -/
theorem worlds_imp (w : ℕ) (e : Option (Form × Form)) (φ ψ : Form) :
    worlds w (imp φ ψ) e = (!worlds w φ e || worlds w ψ e) := worlds.eq_2 w e φ ψ
/-- `worlds` at an equivalence. -/
theorem worlds_iff (w : ℕ) (e : Option (Form × Form)) (φ ψ : Form) :
    worlds w (iff φ ψ) e = (worlds w φ e == worlds w ψ e) := worlds.eq_3 w e φ ψ
/-- `worlds` at a box: every earlier world. -/
theorem worlds_box (w : ℕ) (e : Option (Form × Form)) (t : Bound) (φ : Form) :
    worlds w (box t φ) e = (List.range w).attach.all fun v => worlds v.1 (φ.within e) none :=
  worlds.eq_4 w e t φ
/-- `worlds` at a match: read the source. -/
theorem worlds_coop (w : ℕ) (e : Option (Form × Form)) (a b : Form) :
    worlds w (coop a b) e = (!(a.isPlaceholder || b.isPlaceholder) && worlds w a (some (b, a))) :=
  worlds.eq_5 w e a b
/-- `worlds` at the diagonal formula. -/
theorem worlds_diag (w : ℕ) (e : Option (Form × Form)) (g : Bound) (p : Form) :
    worlds w (diag g p) e =
      (!((List.range w).attach.all fun v => worlds v.1 ((Form.diag g p).within e) none) ||
        worlds w p e) := worlds.eq_8 w e g p

/-- Reading a guarded source in a match is the same as reading its filled text. -/
theorem worlds_fill {o m : Form} (ho : o.isPlaceholder = false) (hm : m.isPlaceholder = false)
    (w : ℕ) :
    ∀ (a : Form), a.guarded = true → ∀ e, worlds w (a.fill o m) e = worlds w a (some (o, m))
  | .bot, _, _ => by simp [fill, worlds]
  | .imp φ ψ, h, e => by
    simp only [guarded, Bool.and_eq_true] at h
    rw [fill, worlds, worlds, worlds_fill ho hm w φ h.1 e, worlds_fill ho hm w ψ h.2 e]
  | .iff φ ψ, h, e => by
    simp only [guarded, Bool.and_eq_true] at h
    rw [fill, worlds, worlds, worlds_fill ho hm w φ h.1 e, worlds_fill ho hm w ψ h.2 e]
  | .box t φ, _, e => by
    rw [fill, worlds, worlds, Form.within_fill ho hm]
    rfl
  | .coop a b, h, e => by
    simp only [guarded, Bool.and_eq_true, Bool.not_eq_true'] at h
    rw [fill, Form.fillArg_self h.1, Form.fillArg_self h.2]
    simp [worlds]
  | .opp, h, _ => by simp [guarded] at h
  | .me, h, _ => by simp [guarded] at h
  | .diag g p, h, e => by
    simp only [guarded] at h
    have hw : (Form.diag g (p.fill o m)).within e = Form.diag g (p.fill o m) :=
      Form.within_fill ho hm (Form.diag g p) e
    rw [fill, worlds, worlds, hw, worlds_fill ho hm w p h e]
    rfl

/-- **World soundness.** A sentence with a checked certificate holds at every world. -/
theorem worlds_sound : ∀ (c : Cert) (σ : Sen), check c σ = true → ∀ w, worlds w σ.form none = true
  | .axK, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i φ ψ φ' hf
      simp only [beq_iff_eq] at h
      subst h
      rw [hf]
      simp only [worlds]
      cases worlds w φ none <;> simp
    · simp at h
  | .axS, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i φ ψ χ φ₁ ψ₁ φ₂ χ₁ hf
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩ := h
      subst h₁ h₂ h₃ h₄
      rw [hf]
      simp only [worlds]
      cases worlds w φ none <;> cases worlds w ψ none <;> cases worlds w χ none <;> simp
    · simp at h
  | .axDN, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i φ φ' hf
      simp only [beq_iff_eq] at h
      subst h
      rw [hf]
      simp only [worlds]
      cases worlds w φ none <;> simp
    · simp at h
  | .iffL c, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i φ ψ hf
      have := worlds_sound c _ h w
      rw [Sen.form_withForm] at this
      rw [hf]
      simp only [worlds, beq_iff_eq] at this ⊢
      rw [this]
      cases worlds w ψ none <;> simp
    · simp at h
  | .iffR c, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i ψ φ hf
      have := worlds_sound c _ h w
      rw [Sen.form_withForm] at this
      rw [hf]
      simp only [worlds, beq_iff_eq] at this ⊢
      rw [this]
      cases worlds w ψ none <;> simp
    · simp at h
  | .iffI c d, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i φ ψ hf
      simp only [Bool.and_eq_true] at h
      have h₁ := worlds_sound c _ h.1 w
      have h₂ := worlds_sound d _ h.2 w
      rw [Sen.form_withForm] at h₁ h₂
      rw [hf]
      simp only [worlds] at h₁ h₂ ⊢
      cases h₃ : worlds w φ none <;> cases h₄ : worlds w ψ none <;> simp_all
    · simp at h
  | .mp φ c d, σ, h, w => by
    simp only [check, Bool.and_eq_true] at h
    have h₁ := worlds_sound c _ h.1 w
    have h₂ := worlds_sound d _ h.2 w
    rw [Sen.form_withForm] at h₁ h₂
    simp only [worlds, h₂, Bool.not_true, Bool.false_or] at h₁
    exact h₁
  | .raise a c, σ, h, w => by
    cases σ with
    | all b φ =>
      simp only [check, Bool.and_eq_true] at h
      exact worlds_sound c _ h.2 w
    | inst _ _ => simp [check] at h
  | .inst a c, σ, h, w => by
    cases σ with
    | inst n φ =>
      simp only [check, Bool.and_eq_true] at h
      exact worlds_sound c _ h.2 w
    | all _ _ => simp [check] at h
  | .qd c, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i a M φ
      simp only [Bool.and_eq_true] at h
      simp only [Sen.form, worlds, Form.within_none]
      exact all_range.2 fun v _ => worlds_sound c _ h.2 v
    · simp at h
  | .nec c, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i n t φ
      simp only [Bool.and_eq_true] at h
      simp only [Sen.form, worlds, Form.within_none]
      exact all_range.2 fun v _ => worlds_sound c _ h.2 v
    · simp at h
  | .dist, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i x φ ψ y φ₁ z ψ₁ hf
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨h₁, h₂⟩, _⟩ := h
      subst h₁ h₂
      rw [hf]
      simp only [worlds, Form.within_none, imp_iff, List.all_eq_true, List.mem_attach,
        forall_const, Subtype.forall, List.mem_range]
      intro h₃ h₄ v hv
      exact h₃ v hv (h₄ v hv)
    · simp at h
  | .four, σ, h, w => by
    -- Valid on the chain: `□φ → □□φ` holds on every transitive frame.
    have four : ∀ (x y : Bound) (φ : Form),
        worlds w (imp (box x φ) (box y (box x φ))) none = true := by
      intro x y φ
      simp only [worlds, Form.within_none, imp_iff, List.all_eq_true, List.mem_attach,
        forall_const, Subtype.forall, List.mem_range]
      intro h₄ v hv u hu
      exact h₄ u (by omega)
    simp only [check] at h
    split at h
    · rename_i x φ y x₁ φ₁
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, -⟩, -⟩ := h
      subst h₁ h₂
      exact four x y φ
    · rename_i x φ y x₁ φ₁
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, -⟩, -⟩ := h
      subst h₁ h₂
      exact four x y φ
    · simp at h
  | .mono, σ, h, w => by
    -- Valid on the chain: the budgets are ignored.
    have mono : ∀ (x y : Bound) (φ : Form), worlds w (imp (box x φ) (box y φ)) none = true := by
      intro x y φ
      simp only [worlds, Form.within_none, imp_iff]
      exact id
    simp only [check] at h
    split at h
    · rename_i x φ y φ₁
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨h₁, -⟩ := h
      subst h₁
      exact mono x y φ
    · rename_i x φ y φ₁
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨h₁, -⟩ := h
      subst h₁
      exact mono x y φ
    · simp at h
  | .diag, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i g p g₁ g₂ p₁ p₂ hf
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩ := h
      subst h₁ h₂ h₃ h₄
      rw [hf, worlds_iff, worlds_diag w none g p, worlds_imp w none (box g (diag g p)) p,
        worlds_box w none g (diag g p)]
      exact beq_self_eq_true _
    · simp at h
  | .prog, σ, h, w => by
    simp only [check] at h
    split at h
    · rename_i a b φ hf
      simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at h
      obtain ⟨⟨⟨ha, hb⟩, hg⟩, hφ⟩ := h
      subst hφ
      rw [hf, worlds_iff, worlds_coop, worlds_fill hb ha w a hg none, ha, hb]
      simp only [Bool.or_self, Bool.not_false, Bool.true_and, beq_self_eq_true]
    · simp at h

/-- `φ` fails at one of the worlds `0, …, W - 1`. -/
def refuted (W : ℕ) (φ : Form) : Bool := (List.range W).any fun w => !worlds w φ none

/-- A refuted sentence has no proof of any length. -/
theorem not_provable_of_refuted {W : ℕ} {σ : Sen} (h : refuted W σ.form = true) (t : ℕ) :
    ¬ Provable t σ := by
  rintro ⟨c, hc, -⟩
  obtain ⟨w, -, hw⟩ := List.any_eq_true.1 h
  simp [worlds_sound c σ hc w] at hw

/-! ### World `0` -/

/-- World `0`, where every box holds, by structural recursion so that the kernel can compute
it. -/
def world0 : Form → Option (Form × Form) → Bool
  | .bot, _ => false
  | .imp φ ψ, e => !world0 φ e || world0 ψ e
  | .iff φ ψ, e => world0 φ e == world0 ψ e
  | .box _ _, _ => true
  | .coop a b, _ => !(a.isPlaceholder || b.isPlaceholder) && world0 a (some (b, a))
  | .opp, _ => false
  | .me, _ => false
  | .diag _ p, e => world0 p e

/-- `world0` is `worlds 0`. -/
theorem worlds_zero : ∀ (φ : Form) (e : Option (Form × Form)), worlds 0 φ e = world0 φ e
  | .bot, e => by rw [worlds_bot]; rfl
  | .imp φ ψ, e => by rw [worlds_imp, worlds_zero φ e, worlds_zero ψ e]; rfl
  | .iff φ ψ, e => by rw [worlds_iff, worlds_zero φ e, worlds_zero ψ e]; rfl
  | .box t φ, e => by rw [worlds_box]; rfl
  | .coop a b, e => by rw [worlds_coop, worlds_zero a]; rfl
  | .opp, e => by rw [worlds_opp]; rfl
  | .me, e => by rw [worlds_me]; rfl
  | .diag g p, e => by rw [worlds_diag, worlds_zero p e]; rfl

/-- Nothing false at world `0` has a proof. -/
theorem not_provable_of_world0 {t : ℕ} {σ : Sen} (h : world0 σ.form none = false) :
    ¬ Provable t σ := by
  rintro ⟨c, hc, -⟩
  have := worlds_sound c σ hc 0
  rw [worlds_zero] at this
  simp [this] at h

end BoundedLob
