/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Theory.System

/-!
# Formulas, sentences and certificates

The concrete proof system has formulas in one numeric parameter `k`:

* `bot`, `imp`, `iff`;
* `box t φ`, "there is a proof of `φ` with at most `t(k)` characters";
* `coop a b`, "the agent with source `a` cooperates with the agent with source `b`". Sources are
  formulas themselves: an agent's source is the formula it tests, written with the placeholders
  `opp` (the opponent's source) and `me` (its own source);
* `diag g p`, a diagonal formula `ψ` with `ψ ↔ (□_g ψ → p)`, the diagonal lemma as a
  constructor.

A sentence is either `all a φ`, "for every `k > a`, `φ`", or `inst n φ`, "`φ` at `k = n`"; the
numeral `n` is written once, which is the abbreviation the bounded Löb theorem needs.

Playing the source `a` against `b` means reading `a` with `opp := b` and `me := a`: `Form.fill`.
Only arguments of `coop` are placeholders, and a source quoted inside another is left alone, since
its placeholders belong to its own match. For DUPOC, `dupoc = □_k coop(opp, me)`, so
`dupoc.fill b dupoc = □_k coop(b, dupoc)`, and the `opp` and `me` inside `b` are untouched.
-/

@[expose] public section

namespace BoundedLob

/-- Formulas in the parameter `k`. -/
inductive Form where
  /-- Falsity. -/
  | bot
  /-- Implication. -/
  | imp (φ ψ : Form)
  /-- Equivalence. -/
  | iff (φ ψ : Form)
  /-- `□_t φ`: `φ` has a proof of at most `t(k)` characters. -/
  | box (t : Bound) (φ : Form)
  /-- The agent with source `a` cooperates with the agent with source `b`. -/
  | coop (a b : Form)
  /-- In a source: the opponent's source. -/
  | opp
  /-- In a source: the agent's own source. -/
  | me
  /-- The diagonal formula `ψ` with `ψ ↔ (□_g ψ → p)`. -/
  | diag (g : Bound) (p : Form)
  deriving DecidableEq, Repr

namespace Form

/-- Negation, `φ → ⊥`. -/
abbrev neg (φ : Form) : Form := imp φ bot

/-- Truth, `⊥ → ⊥`. -/
abbrev top : Form := imp bot bot

/-- Conjunction, `¬(X → ¬Y)`. -/
abbrev conj (X Y : Form) : Form := neg (imp X (neg Y))

/-- `a` is one of the placeholders `opp` and `me`. -/
def isPlaceholder : Form → Bool
  | opp | me => true
  | _ => false

/-- `bot` is not a placeholder. -/
@[simp] theorem isPlaceholder_bot : bot.isPlaceholder = false := rfl
/-- `imp` is not a placeholder. -/
@[simp] theorem isPlaceholder_imp (φ ψ : Form) : (imp φ ψ).isPlaceholder = false := rfl
/-- `iff` is not a placeholder. -/
@[simp] theorem isPlaceholder_iff (φ ψ : Form) : (iff φ ψ).isPlaceholder = false := rfl
/-- `box` is not a placeholder. -/
@[simp] theorem isPlaceholder_box (t : Bound) (φ : Form) :
    (box t φ).isPlaceholder = false := rfl
/-- `coop` is not a placeholder. -/
@[simp] theorem isPlaceholder_coop (a b : Form) : (coop a b).isPlaceholder = false := rfl
/-- `diag` is not a placeholder. -/
@[simp] theorem isPlaceholder_diag (g : Bound) (p : Form) :
    (diag g p).isPlaceholder = false := rfl
/-- `opp` is a placeholder. -/
@[simp] theorem isPlaceholder_opp : opp.isPlaceholder = true := rfl
/-- `me` is a placeholder. -/
@[simp] theorem isPlaceholder_me : me.isPlaceholder = true := rfl

/-- A `coop` argument after filling: `opp` becomes `o`, `me` becomes `m`, anything else stays. -/
def fillArg : Form → Form → Form → Form
  | opp, o, _ => o
  | me, _, m => m
  | a, _, _ => a

/-- Fill the placeholders of a source: `opp` becomes `o` and `me` becomes `m`, in the arguments
of `coop` only; quoted sources are left alone. -/
def fill : Form → Form → Form → Form
  | bot, _, _ => bot
  | imp φ ψ, o, m => imp (φ.fill o m) (ψ.fill o m)
  | iff φ ψ, o, m => iff (φ.fill o m) (ψ.fill o m)
  | box t φ, o, m => box t (φ.fill o m)
  | coop a b, o, m => coop (a.fillArg o m) (b.fillArg o m)
  | opp, _, _ => opp
  | me, _, _ => me
  | diag g p, o, m => diag g (p.fill o m)

/-- Every placeholder sits under a box, so that reading a source never needs to run its
opponent. Outside boxes, `coop` must quote both sources. Quoted sources are not inspected. -/
def guarded : Form → Bool
  | bot => true
  | imp φ ψ => φ.guarded && ψ.guarded
  | iff φ ψ => φ.guarded && ψ.guarded
  | box _ _ => true
  | coop a b => !a.isPlaceholder && !b.isPlaceholder
  | opp | me => false
  | diag _ p => p.guarded

/-- Read `φ` inside a match: with `some (o, m)`, fill `φ` with `opp := o` and `me := m`; with
`none`, `φ` is unchanged. -/
def within : Form → Option (Form × Form) → Form
  | φ, none => φ
  | φ, some (o, m) => φ.fill o m

/-- With no match, the formula is unchanged. -/
@[simp] theorem within_none (φ : Form) : φ.within none = φ := rfl

/-- `fillArg` leaves anything but a placeholder alone. -/
theorem fillArg_self {a : Form} (h : a.isPlaceholder = false) (o m : Form) :
    a.fillArg o m = a := by
  cases a <;> simp_all [isPlaceholder, fillArg]

/-- Filling twice with non-placeholders is filling once. -/
theorem fillArg_fillArg {o m : Form} (ho : o.isPlaceholder = false) (hm : m.isPlaceholder = false)
    (a o' m' : Form) : (a.fillArg o m).fillArg o' m' = a.fillArg o m := by
  cases a with
  | opp => exact fillArg_self ho o' m'
  | me => exact fillArg_self hm o' m'
  | _ => rfl

/-- A filled source has no placeholders left to fill. -/
theorem fill_fill {o m : Form} (ho : o.isPlaceholder = false) (hm : m.isPlaceholder = false)
    (φ o' m' : Form) : (φ.fill o m).fill o' m' = φ.fill o m := by
  induction φ with
  | coop a b => simp [fill, fillArg_fillArg ho hm]
  | _ => simp_all [fill]

/-- Reading a filled source in any match changes nothing. -/
theorem within_fill {o m : Form} (ho : o.isPlaceholder = false) (hm : m.isPlaceholder = false)
    (φ : Form) (e : Option (Form × Form)) : (φ.fill o m).within e = φ.fill o m := by
  cases e with
  | none => rfl
  | some p => exact fill_fill ho hm φ p.1 p.2

end Form

/-- Sentences: `all a φ` is `∀ k > a, φ(k)`, and `inst n φ` is `φ(n)`. -/
inductive Sen where
  /-- `∀ k > a, φ(k)`: `φ` holds at every `k` above the guard `a`. -/
  | all (a : ℕ) (φ : Form)
  /-- `φ(n)`: `φ` holds at the parameter `k = n`. -/
  | inst (n : ℕ) (φ : Form)
  deriving DecidableEq, Repr

namespace Sen

/-- The formula of a sentence. -/
def form : Sen → Form
  | all _ φ | inst _ φ => φ

/-- Replace the formula, keeping the guard of `all` or the parameter of `inst`. -/
def withForm (σ : Sen) (φ : Form) : Sen :=
  match σ with
  | all a _ => all a φ
  | inst n _ => inst n φ

/-- `withForm` sets the formula. -/
@[simp] theorem form_withForm (σ : Sen) (φ : Form) : (σ.withForm φ).form = φ := by
  cases σ <;> rfl

end Sen

/-- Proof certificates. The checker is given the sentence to prove, so a certificate only records
what the sentence does not determine: the cut formula of modus ponens and the guards of `raise`
and `inst`. The side conditions of each rule are in `check`. -/
inductive Cert where
  /-- The axiom `φ → ψ → φ`. -/
  | axK
  /-- The axiom `(φ → ψ → χ) → (φ → ψ) → φ → χ`. -/
  | axS
  /-- Double negation, `¬¬φ → φ`. -/
  | axDN
  /-- From `φ ↔ ψ`, `φ → ψ`. -/
  | iffL (c : Cert)
  /-- From `φ ↔ ψ`, `ψ → φ`. -/
  | iffR (c : Cert)
  /-- From `φ → ψ` and `ψ → φ`, `φ ↔ ψ`. -/
  | iffI (c d : Cert)
  /-- Modus ponens with cut formula `φ`. -/
  | mp (φ : Form) (c d : Cert)
  /-- From `∀ k > a, φ`, `∀ k > b, φ` for `a ≤ b`. -/
  | raise (a : ℕ) (c : Cert)
  /-- From `∀ k > a, φ`, `φ(n)` for `a < n`. -/
  | inst (a : ℕ) (c : Cert)
  /-- Quantifier distribution: from `c` proving `∀ k > a, φ` in `N` characters,
  `∀ k > a, □_{M + lg k} φ` for `N + 4 ≤ M`. -/
  | qd (c : Cert)
  /-- Necessitation at an instance: from `c` proving `φ(n)` within `t(n)`, `(□_t φ)(n)`. -/
  | nec (c : Cert)
  /-- Implication distribution, written exactly as `□_x (φ → ψ) → □_y φ → □_{x + y + 0} ψ`. -/
  | dist
  /-- Inner necessitation `□_x φ → □_{2x} □_x φ`, where the text of `x` can be paid for:
  `x.fits a` for `∀ k > a`, or `|x| + 2 ≤ x(n)` at an instance `n`. -/
  | four
  /-- Monotonicity of the bound: `□_x φ → □_y φ` where `x ≤ y` beyond the guard. -/
  | mono
  /-- The diagonal formula: `diag g p ↔ (□_g (diag g p) → p)`. -/
  | diag
  /-- An agent's source: `coop a b ↔ a` read with `opp := b` and `me := a`, for a guarded source
  `a` and an opponent `b` that are not placeholders. -/
  | prog
  deriving DecidableEq, Repr

end BoundedLob
