/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import Mathlib.Data.Nat.Log

/-!
# A proof system with bounded provability

This is an interface adapted from Critch's proof of the parametric bounded Löb theorem
(Critch, *Parametric Bounded Löb's Theorem and Robust Cooperation of Bounded Agents*,
arXiv:1602.04184v5, Theorem 3; Critch, Dennis and Russell, arXiv:2208.07006, Lemma 3.6): enough
for the proof to go through, with inner necessitation and monotonicity only from some point on,
a linear expansion, and the diagonal lemma as a field.

`ProvesIn n σ` says that the system has a proof of the sentence `σ` with at most `n` characters.
It is a statement about the object system, not about truth: nothing here says that a provable
sentence is true. `box b φ` is the object-language formula `□_{b(k)} φ(k)`, "there is a proof of
`φ(k)` of at most `b(k)` characters", as a formula in the free variable `k`, and `above k₀ φ` is
the sentence `∀ k > k₀, φ(k)`. So `above 0 φ` says `φ(k)` for every positive `k`.

The fields are the properties the proof uses, each stated once. The paper's names for the
constants are `c` (`distCost`), `C` (`instCost`) and `e` (`expansion`).

* the propositional steps it takes under a guarded quantifier (`mp`, `trans`, `contract`,
  `imp_mono`), and separating a proved equivalence into its two implications;
* **implication distribution** (Critch's Property 1), with a constant overhead `distCost`;
* **quantifier distribution** (Property 2): an `N`-character proof of `∀ k > k₀, φ(k)` gives,
  inside the system, a proof of each instance with `instCost + 2N + lg k` characters;
* **bounded inner necessitation** (Property 4) with a linear expansion `expansion`, the assumption
  of Lemma 3.6 (`E(k) = e·k`, Appendix B(d) of Critch, Dennis and Russell), beyond a guard that
  may depend on the bound: the statement `□_x φ` contains the text of `x`, so a fixed expansion
  can only absorb it once `k` is large;
* **monotonicity** of the bound, `□_x φ → □_y φ` beyond some guard, whenever `y` eventually
  exceeds `x` by every constant (all the inequalities the proof needs have such a gap);
* the **parametric diagonal lemma** for the formula `□_{g(k)} ψ(k) → p(k)`.

Critch takes the distribution and necessitation properties for granted and sketches why real
proof systems have them; he proves the diagonal lemma from the representability of computable
functions. Here all of them are hypotheses. `BoundedLob.calculus` packages the concrete checker as
a `System id`, which has them all. Bounded necessitation (Property 3) is not needed: quantifier
distribution already turns a proof of a guarded universal into bounded proofs of its instances.

Bounds are terms of the object language, built from numerals, `lg k`, sums, scaling, division by
a numeral and a fixed function symbol `f`, so that `box` only ever refers to bounds the system
can write.
-/

@[expose] public section

namespace BoundedLob

/-- Bound expressions in the free variable `k`, with a function symbol `f` for the budget. -/
inductive Bound where
  /-- A numeral. -/
  | const (n : ℕ)
  /-- The floor of the base-two logarithm of `k`, `⌊log₂ k⌋`. -/
  | lg
  /-- The budget `f(k)`. -/
  | f
  /-- A sum. -/
  | add (a b : Bound)
  /-- A multiple `c · a`. -/
  | scale (c : ℕ) (a : Bound)
  /-- Floor division `⌊a / d⌋` by a numeral. -/
  | div (a : Bound) (d : ℕ)
  deriving DecidableEq, Repr

namespace Bound

/-- The value of a bound at `k`, given the meaning of `f`. `lg` is `⌊log₂ k⌋`. -/
def eval (f : ℕ → ℕ) : Bound → ℕ → ℕ
  | const n, _ => n
  | lg, k => Nat.log 2 k
  | Bound.f, k => f k
  | add a b, k => a.eval f k + b.eval f k
  | scale c a, k => c * a.eval f k
  | div a d, k => a.eval f k / d

/-- `a + b` is `Bound.add a b`. -/
instance : Add Bound := ⟨add⟩

/-- Sums evaluate to sums. -/
@[simp] theorem eval_add (f : ℕ → ℕ) (a b : Bound) (k : ℕ) :
    (a + b).eval f k = a.eval f k + b.eval f k := rfl

end Bound

/-- A proof system with formulas in one free variable `k` and a bounded provability operator.
`f` is the meaning of the budget symbol `Bound.f`. -/
structure System (f : ℕ → ℕ) where
  /-- Sentences. -/
  Sen : Type
  /-- Formulas with the free variable `k`. -/
  Pred : Type
  /-- Pointwise implication. -/
  imp : Pred → Pred → Pred
  /-- Pointwise equivalence. -/
  iff : Pred → Pred → Pred
  /-- `box b φ` is `□_{b(k)} φ(k)`. -/
  box : Bound → Pred → Pred
  /-- `above k₀ φ` is the sentence `∀ k > k₀, φ(k)`. -/
  above : ℕ → Pred → Sen
  /-- `ProvesIn n σ`: the system proves `σ` with at most `n` characters. -/
  ProvesIn : ℕ → Sen → Prop
  /-- Modus ponens under the guarded quantifier. -/
  mp {a b : ℕ} {φ ψ : Pred} : (∃ n, ProvesIn n (above a (imp φ ψ))) →
    (∃ n, ProvesIn n (above b φ)) → ∃ n, ProvesIn n (above (max a b) ψ)
  /-- Hypothetical syllogism. -/
  trans {a b : ℕ} {φ ψ χ : Pred} : (∃ n, ProvesIn n (above a (imp φ ψ))) →
    (∃ n, ProvesIn n (above b (imp ψ χ))) → ∃ n, ProvesIn n (above (max a b) (imp φ χ))
  /-- `φ → ψ → χ` and `φ → ψ` give `φ → χ`. -/
  contract {a b : ℕ} {φ ψ χ : Pred} : (∃ n, ProvesIn n (above a (imp φ (imp ψ χ)))) →
    (∃ n, ProvesIn n (above b (imp φ ψ))) → ∃ n, ProvesIn n (above (max a b) (imp φ χ))
  /-- Weakening a consequent under a hypothesis. -/
  imp_mono {a : ℕ} {φ ψ χ : Pred} : (∃ n, ProvesIn n (above a (imp ψ χ))) →
    ∃ n, ProvesIn n (above a (imp (imp φ ψ) (imp φ χ)))
  /-- The extra characters for taking the forward half of an equivalence. -/
  iffCost : ℕ
  /-- The forward half of a proved equivalence, at most `iffCost` characters longer. -/
  iff_forward {n a : ℕ} {φ ψ : Pred} : ProvesIn n (above a (iff φ ψ)) →
    ProvesIn (n + iffCost) (above a (imp φ ψ))
  /-- The backward half of a proved equivalence, with no bound on how much longer it is. -/
  iff_backward {n a : ℕ} {φ ψ : Pred} : ProvesIn n (above a (iff φ ψ)) →
    ∃ m, ProvesIn m (above a (imp ψ φ))
  /-- The extra characters for modus ponens inside a box (Critch's `c`). -/
  distCost : ℕ
  /-- Critch's Property 1, implication distribution. -/
  imp_dist (x y : Bound) (φ ψ : Pred) :
    ∃ n, ProvesIn n (above 0
      (imp (box x (imp φ ψ)) (imp (box y φ) (box (x + y + .const distCost) ψ))))
  /-- The extra characters for instantiating a universal statement (Critch's `C`). -/
  instCost : ℕ
  /-- Critch's Property 2, quantifier distribution. -/
  quant_dist {N a : ℕ} {φ : Pred} : ProvesIn N (above a φ) →
    ∃ n, ProvesIn n (above a (box (.const (instCost + 2 * N) + .lg) φ))
  /-- The factor by which checking a proof inside the system lengthens it (Critch's `e`). -/
  expansion : ℕ
  /-- Critch's Property 4, bounded inner necessitation, beyond a guard that may depend on `x`. -/
  inner_nec (x : Bound) (φ : Pred) :
    ∃ a n, ProvesIn n (above a (imp (box x φ) (box (.scale expansion x) (box x φ))))
  /-- The system proves `□_x φ → □_y φ` beyond some guard whenever `y` eventually exceeds `x` by
  every constant. -/
  mono {x y : Bound} (φ : Pred) : (∀ c, ∃ K, ∀ k > K, x.eval f k + c ≤ y.eval f k) →
    ∃ a n, ProvesIn n (above a (imp (box x φ) (box y φ)))
  /-- The parametric diagonal lemma for `□_{g(k)} ψ(k) → p(k)`. -/
  diag (g : Bound) (p : Pred) : ∃ ψ n, ProvesIn n (above 0 (iff ψ (imp (box g ψ) p)))

namespace System

variable {f : ℕ → ℕ} (S : System f)

/-- Provability with some length. -/
def Proves (σ : S.Sen) : Prop := ∃ n, S.ProvesIn n σ

end System

end BoundedLob
