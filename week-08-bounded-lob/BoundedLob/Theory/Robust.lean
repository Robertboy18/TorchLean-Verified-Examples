/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Theory.Lob

/-!
# Robust cooperation

Open Problem 4 of Critch, Dennis and Russell also asks how much two implementations may differ
while still cooperating. Critch's Theorem 4 (arXiv:1602.04184v5) answers it for *G-fair* agents,
which provably cooperate as soon as they have a short enough proof that their opponent
cooperates, each with its own parameter. `System.robust` is that argument for agents sharing the
parameter `k`, in the more general form the implementations need.

Agent `A` provably cooperates (`X`) once it has a proof within `t_A(k)` of some `α`, and agent `B`
(`Y`) once it has one within `t_B(k)` of some `β`. `P` stands for mutual cooperation: the system
proves `X → Y → P`, `P → α` and `P → β` (the interface has no conjunction, so these are
hypotheses). Suppose some budget `u(k)` grows faster than `lg k` and stays below both `t_A` and
`t_B` by `lg k` and any constant. Then the system proves `P` for every large `k`.

The proof applies the bounded Löb theorem to `P` with the budget `u`: a proof of `P` within `u(k)`
gives proofs of `α` and `β` within the two budgets (`System.box_imp_of_gap`), hence `X` and `Y`,
hence `P`.
-/

@[expose] public section

namespace BoundedLob

namespace System

variable {f : ℕ → ℕ} (S : System f)

/-- **Robust cooperation.** Suppose the system proves `X → Y → P`, `P → α`, `P → β`,
`□_{t_A} α → X` and `□_{t_B} β → Y`, and some budget `u` growing faster than `lg k` stays below
`t_A` and `t_B` by `lg k` and any constant. Then it proves `P` beyond some guard. For two fair
agents, `X` and `Y` are their cooperation and `P` is mutual cooperation. -/
theorem robust {P X Y α β : S.Pred} {ta tb u : Bound} (hu : GrowsFasterThanLog (u.eval f))
    (gapA : ∀ c, ∃ K, ∀ k > K, u.eval f k + Nat.log 2 k + c ≤ ta.eval f k)
    (gapB : ∀ c, ∃ K, ∀ k > K, u.eval f k + Nat.log 2 k + c ≤ tb.eval f k)
    (hα : S.Proves (S.above 0 (S.imp P α))) (hβ : S.Proves (S.above 0 (S.imp P β)))
    (hP : S.Proves (S.above 0 (S.imp X (S.imp Y P))))
    (fairA : S.Proves (S.above 0 (S.imp (S.box ta α) X)))
    (fairB : S.Proves (S.above 0 (S.imp (S.box tb β) Y))) :
    ∃ K, S.Proves (S.above K P) := by
  obtain ⟨KX, hX⟩ := S.box_imp_of_gap gapA hα fairA
  obtain ⟨KY, hY⟩ := S.box_imp_of_gap gapB hβ fairB
  exact S.pblt u hu (S.contract (S.trans hX hP) hY)

end System

end BoundedLob
