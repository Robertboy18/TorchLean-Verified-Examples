/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Rules

/-!
# Candidate proofs

Searching every string of `k` characters is hopeless for any interesting `k`. Before it does, an
agent builds a few candidate certificates from the sentence it wants and the two sources, and
lets the checker decide. The builders follow the proofs of the theorems step by step, with the
constants of `calculus`:

* `Candidate.lob b p k₁ hyp`: from `hyp` proving `∀ k > k₁, □_b p → p`, a candidate proof of
  `∀ k > K, p`, following `System.pblt`;
* `Candidate.dimcidDefects a b t`: DIMCID's self-fulfilling defection against `b`, following
  `System.self_fulfilling`;
* `Candidate.robust a b`: mutual cooperation of two fair agents, following `System.robust`;
* `Candidate.forForm φ`: the candidates for `φ`, for the shapes of sentence it recognizes.

Nothing here is trusted. A builder may return a certificate that does not check; the agent asks
the checker, and falls back to the complete search (`search`) when no candidate works.

For DUPOC against itself at `k = n`, the agent wants `coop(DUPOC, DUPOC)` at `n`. `forCooperation`
sees DUPOC's budget `k` and calls `lob` with the program equation
`□_k coop(DUPOC, DUPOC) → coop(DUPOC, DUPOC)` (the certificate `.iffR .prog`). `lob` returns a
guard `K` and a certificate for `∀ k > K, coop(DUPOC, DUPOC)`. `candidates` keeps it when `K < n`
and instantiates it at `n`, and `search` accepts it when the checker agrees and the whole proof,
sentence and certificate, has at most `n` characters (`affordable`). A `Candidate` records neither
the sentence it is meant for nor any evidence that it checks. Its guard is where that certificate
becomes valid, not the smallest budget at which the agents cooperate.
-/

@[expose] public section

namespace BoundedLob

open Form

/-- A candidate certificate for `∀ k > guard, φ`. -/
structure Candidate where
  /-- The guard of the universal sentence. -/
  guard : ℕ
  /-- The certificate. -/
  cert : Cert
  deriving DecidableEq, Repr

namespace Candidate

/-- The divisor of the diagonal budget, `2 (e + 1)` with `e = Cost.expansion`, as in
`System.pblt`: the two uses of the smaller budget then take at most half of the original. -/
def lobDivisor : ℕ := 2 * (Cost.expansion + 1)

/-- The bounded Löb theorem as a certificate builder. From `hyp` proving `∀ k > k₁, □_b p → p`,
a candidate for `∀ k > K, p`, following `System.pblt` with `calculus`'s constants: taking the
forward half of an equivalence costs one character, distribution costs none, and quantifier
distribution allows `Cost.inst`. `none` when a bound comparison fails. -/
def lob (b : Bound) (p : Form) (k₁ : ℕ) (hyp : Cert) : Option Candidate := do
  let g : Bound := .div b lobDivisor
  let expanded : Bound := .scale Cost.expansion g
  let ψ := diag g p
  let G := imp (box g ψ) p
  let n := len .diag (.all 0 (iff ψ G))
  -- Distribute the box over `ψ → G`, then over its consequent.
  let ℓ : Bound := .const (Cost.inst + 2 * (n + 1)) + .lg
  let x₁ : Bound := ℓ + g + .const 0
  let x₂ : Bound := x₁ + expanded + .const 0
  let h₂ := Cert.mp (box ℓ (imp ψ G)) .dist (.qd (.iffL .diag))
  let h₃ := Cert.trans (box g ψ) (box x₁ G) (imp (box expanded (box g ψ)) (box x₂ p)) h₂ .dist
  -- Spend the budget through the hypothesis.
  let K₁ ← x₂.guard b
  let a₄ := max K₁ k₁
  let h₄ := Cert.trans (box x₂ p) (box b p) p (.raise K₁ .mono) (.raise k₁ hyp)
  let h₅ := Cert.trans (box g ψ) (imp (box expanded (box g ψ)) (box x₂ p))
    (imp (box expanded (box g ψ)) p) (.raise 0 h₃)
    (Cert.impMono (box expanded (box g ψ)) (box x₂ p) p h₄)
  -- Inner necessitation, and the diagonal sentence itself.
  let a₆ := g.fitGuard
  let K := max a₄ a₆
  let h₆ := Cert.contract (box g ψ) (box expanded (box g ψ)) p (.raise a₄ h₅) (.raise a₆ .four)
  let hψ := Cert.mp G (.raise 0 (.iffR .diag)) h₆
  -- One more distribution, and `□_g ψ`.
  let N := len hψ (.all K ψ)
  let ℓ' : Bound := .const (Cost.inst + 2 * N) + .lg
  let K₂ ← ℓ'.guard g
  let K' := max K₂ K
  let h₇ := Cert.mp (box ℓ' ψ) (.raise K₂ .mono) (.raise K (.qd hψ))
  return ⟨K', .mp (box g ψ) (.raise K h₆) h₇⟩

/-- DIMCID with source `a` and budget `t` defects against `b`: a candidate for
`∀ k > K, ¬ coop a b`, following `System.self_fulfilling`. The program equation it uses only
checks when `a` really is DIMCID's source. -/
def dimcidDefects (a b : Form) (t : Bound) : Option Candidate := do
  let A := coop a b
  let Y := neg (coop b a)
  let d := neg A
  let G := imp A Y
  let exfalso := Cert.impMono A bot Y (Cert.efq Y)
  let code := Cert.swap A (box t G) bot (.iffL .prog)
  let ℓ : Bound := .const (Cost.inst + 2 * len exfalso (.all 0 (imp d G))) + .lg
  let half : Bound := .div t 2
  let x : Bound := ℓ + half + .const 0
  let h₂ := Cert.mp (box ℓ (imp d G)) .dist (.qd exfalso)
  let K₁ ← x.guard t
  let h₃ := Cert.trans (box half d) (box x G) (box t G) (.raise 0 h₂) .mono
  let h₄ := Cert.trans (box half d) (box t G) d h₃ (.raise 0 code)
  lob half d K₁ h₄

/-- A certificate of `X ∧ Y → α` when `α` is `X`, `Y`, or `ψ → α'` for such an `α'`. -/
def fromBoth (X Y : Form) : Form → Option Cert
  | imp ψ γ =>
    if imp ψ γ == Y then some (Cert.conjRight X Y)
    else if imp ψ γ == X then some (Cert.conjLeft X Y)
    else (fromBoth X Y γ).map (Cert.addHyp (Form.conj X Y) γ ψ)
  | α =>
    if α == Y then some (Cert.conjRight X Y)
    else if α == X then some (Cert.conjLeft X Y)
    else none

/-- The fractions `u = t_A / m` of the first agent's budget that `robust` tries for the Löb
step, in this order. The first one whose two comparisons succeed is used, and no other is tried
if the Löb step then fails, so very lopsided budgets are covered by `System.robust` but not by
this builder. -/
def ratios : List ℕ := [2, 4, 8, 16, 32, 64]

/-- Mutual cooperation of `a` and `b`, when `a` read against `b` is `□_{ta} α`, `b` read against
`a` is `□_{tb} β`, and `α`, `β` follow from `coop a b ∧ coop b a`: a candidate for
`∀ k > K, coop a b ∧ coop b a`, following `System.robust`. -/
def robust (a b : Form) : Option Candidate := do
  let .box ta α := a.fill b a | none
  let .box tb β := b.fill a b | none
  let X := coop a b
  let Y := coop b a
  let P := Form.conj X Y
  let hα ← fromBoth X Y α
  let hβ ← fromBoth X Y β
  let ℓα : Bound := .const (Cost.inst + 2 * len hα (.all 0 (imp P α))) + .lg
  let ℓβ : Bound := .const (Cost.inst + 2 * len hβ (.all 0 (imp P β))) + .lg
  -- A budget for the Löb step that leaves room under both `ta` and `tb`.
  let (u, Kα, Kβ) ← ratios.findSome? fun m =>
    let u : Bound := .div ta m
    match (ℓα + u + .const 0).guard ta, (ℓβ + u + .const 0).guard tb with
    | some Kα, some Kβ => some (u, Kα, Kβ)
    | _, _ => none
  -- `□_u P → ψ`, through a proof of `γ` within `tγ`: `System.box_imp_of_gap`.
  let boxImp (ψ γ : Form) (tγ ℓ : Bound) (Kγ : ℕ) (h : Cert) : Cert :=
    Cert.trans (box u P) (box tγ γ) ψ
      (.raise Kγ (Cert.trans (box u P) (box (ℓ + u + .const 0) γ) (box tγ γ)
        (.raise 0 (.mp (box ℓ (imp P γ)) .dist (.qd h))) .mono))
      (.raise 0 (.iffR .prog))
  let hX := boxImp X α ta ℓα Kα hα
  let hY := boxImp Y β tb ℓβ Kβ hβ
  let hyp := Cert.contract (box u P) Y P
    (Cert.trans (box u P) X (imp Y P) hX (.raise 0 (Cert.conjIntro X Y))) hY
  lob u P (max Kα Kβ) hyp

/-- A candidate for `φ`, derived from the candidate for the mutual cooperation of `a` and `b`. -/
def viaRobust (a b φ : Form) : Option Candidate := do
  let c ← robust a b
  let h ← fromBoth (coop a b) (coop b a) φ
  return ⟨c.guard, .mp (Form.conj (coop a b) (coop b a)) (.raise 0 h) c.cert⟩

/-- The budget of a DUPOC source `□_t coop(opp, me)`. -/
def dupocBudget : Form → Option Bound
  | box t (coop opp me) => some t
  | _ => none

/-- The budget of a CUPOD source `¬□_t ¬coop(opp, me)`. -/
def cupodBudget : Form → Option Bound
  | imp (box t (imp (coop opp me) bot)) bot => some t
  | _ => none

/-- The budget of a DIMCID source `¬□_t (coop(me, opp) → ¬coop(opp, me))`. -/
def dimcidBudget : Form → Option Bound
  | imp (box t (imp (coop me opp) (imp (coop opp me) bot))) bot => some t
  | _ => none

/-- Candidates for `coop a b`: DUPOC against itself, a source that is `⊤` once filled, or two
fair agents. -/
def forCooperation (a b : Form) : List Candidate :=
  let φ := coop a b
  (match dupocBudget a with
    | some t => if a == b then (lob t φ 0 (.iffR .prog)).toList else []
    | none => []) ++
  (if a.fill b a == top then [⟨0, .mp top (.iffR .prog) (Cert.refl bot)⟩] else []) ++
  (viaRobust a b φ).toList

/-- Candidates for `¬coop a b`: CUPOD against itself, DIMCID's defection, or a source that is
`⊥` once filled. -/
def forDefection (a b : Form) : List Candidate :=
  let φ := neg (coop a b)
  (match cupodBudget a with
    | some t =>
      if a == b then (lob t φ 0 (Cert.swap (coop a b) (box t φ) bot (.iffL .prog))).toList else []
    | none => []) ++
  (match dimcidBudget a with
    | some t => (dimcidDefects a b t).toList
    | none => []) ++
  (if a.fill b a == bot then [⟨0, .iffL .prog⟩] else [])

/-- Candidates for DIMCID's guard `coop a b → ¬coop b a`, from its defection. -/
def forDimcidGuard (a b : Form) : List Candidate :=
  match dimcidBudget a with
  | some t =>
    ((dimcidDefects a b t).map fun c =>
      ⟨c.guard, .mp (neg (coop a b)) (.raise 0 (Cert.impMono (coop a b) bot (neg (coop b a))
        (Cert.efq (neg (coop b a))))) c.cert⟩).toList
  | none => []

/-- Candidate proofs of `∀ k > K, φ`, in the order the agent tries them. -/
def forForm : Form → List Candidate
  | coop a b => forCooperation a b
  | φ@(imp ψ χ) =>
    (if ψ == χ then [⟨0, Cert.refl ψ⟩] else []) ++
    -- A provable consequent.
    ((forForm χ).map fun c => ⟨c.guard, .mp χ .axK c.cert⟩) ++
    match ψ, χ with
    | coop a b, bot => forDefection a b
    | coop a b, imp (coop b' a') bot =>
      (if b' == b && a' == a then forDimcidGuard a b else []) ++ (viaRobust a b φ).toList
    | coop a b, _ => (viaRobust a b φ).toList
    | _, _ => []
  | _ => []

end Candidate

/-- The candidate certificates for a sentence at `k = n`: those whose guard is below `n`,
instantiated at `n`. -/
def candidates : Sen → List Cert
  | .inst n φ =>
    (Candidate.forForm φ).filterMap fun c =>
      if c.guard < n then some (.inst c.guard c.cert) else none
  | .all _ _ => []

end BoundedLob
