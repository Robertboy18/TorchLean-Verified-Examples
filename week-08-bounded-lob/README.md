# Löb's theorem with a budget

In the [article](../site/articles/week-08-bounded-lob.md) I follow the bounded agents from Jacob
Tsimerman's MAISI talk: programs that search for proofs about each other with a fixed budget.
Here they run.

The library has three folders, and they read in order. `Theory` proves the theorems for any proof
system with Critch's properties, `Checker` builds one such system as a real proof checker, and
`Agents` writes the agents as programs that use the checker and proves what they do.

## Theory: any system with Critch's properties

- [`System`](BoundedLob/Theory/System.lean) lists the properties the bounded Löb proof uses, as
  fields: implication and quantifier distribution, inner necessitation with a linear expansion,
  monotonicity of the bound and the diagonal lemma. Necessitation and monotonicity only need to
  hold from some `k` on.
- [`Lob`](BoundedLob/Theory/Lob.lean) proves the parametric bounded Löb theorem `System.pblt`
  (Lemma 3.6 of Critch, Dennis and Russell; Theorem 3 of Critch, arXiv:1602.04184) for any budget
  `b` with `b(k) / lg k → ∞`.
- [`Dupoc`](BoundedLob/Theory/Dupoc.lean) describes DUPOC over an arbitrary checker: it can't be
  exploited, a guess in front of a complete search behaves like the search, and a guess alone
  does not.
- [`Dimcid`](BoundedLob/Theory/Dimcid.lean) answers Open Problem 6: DUPOC against DIMCID is mutual
  defection for every large `k`, in any sound system with these properties that proves the
  agents' code and `¬A → (A → B)`.
- [`Robust`](BoundedLob/Theory/Robust.lean) is Critch's robust cooperation of fair agents, for
  agents sharing the parameter `k`.

## Checker: a system in which all of this is true

The abstract names become concrete ones: a formula `Pred` is a `Form`, the sentence `above a φ` is
`Sen.all a φ`, `ProvesIn n` is `Provable n`, and the budget symbol `f` is read as the identity, so
the checker is a [`calculus : System id`](BoundedLob/Checker/Calculus.lean). The `Checker` in
`Theory/Dupoc` is an interface for any proof language; `check` here is one concrete checker.

- [`Form`](BoundedLob/Checker/Form.lean) has the formulas, with agent sources and their
  placeholders, the sentences and the certificates.
- [`Encoding`](BoundedLob/Checker/Encoding.lean) prints them over 35 characters, so every length
  is the length of a text, and [`Parse`](BoundedLob/Checker/Parse.lean) reads certificates back
  and lists every string up to a length.
- [`Compare`](BoundedLob/Checker/Compare.lean) finds a guard beyond which one bound exceeds
  another, when the gap between them grows without bound, which is what the monotonicity rule
  needs. Finding none is no evidence that the inequality fails.
- [`Check`](BoundedLob/Checker/Check.lean) is the checker. Its header summarizes the bounded
  rules, and `check` itself has every rule with its side condition.
  [`Rules`](BoundedLob/Checker/Rules.lean) derives the propositional rules the proofs use.
- [`Truth`](BoundedLob/Checker/Truth.lean) says what sentences mean and proves the checker sound.
  [`Worlds`](BoundedLob/Checker/Worlds.lean) proves every rule valid on the chain of worlds, so a
  sentence false at a world has no proof of any length.
- [`Calculus`](BoundedLob/Checker/Calculus.lean) proves every field of `System` for the checker.

## Agents: programs that search for proofs

- [`Sources`](BoundedLob/Agents/Sources.lean) writes the agents as formulas: CooperateBot,
  DefectBot, DUPOC, CUPOD, CIMCIC, DIMCID and a differently written DUPOC.
- [`Candidates`](BoundedLob/Agents/Candidates.lean) builds the candidate certificates an agent
  tries first, following the proofs in `Theory` step by step. Nothing there is trusted: `Search`
  checks each certificate, and the length of the whole proof, before it accepts one.
- [`Search`](BoundedLob/Agents/Search.lean) defines the search and `run`. It tries the worlds,
  then the candidates, then every string, and `search_spec` and `run_spec` prove that this decides
  exactly bounded provability and the truth of the agents' sources.
- [`Outcomes`](BoundedLob/Agents/Outcomes.lean) proves the outcomes about `run` itself. Some hold
  at every large budget, from the bounded Löb theorem: `dupoc_self` (Theorem 3.7), `cupod_self`
  (Theorem 3.4), `dimcid_self` and `cupod_dimcid` (Theorem 5.5), and `dimcid_vs_cooperate`, the
  correction to Example 5.3(a). For DUPOC against itself the threshold is explicit:
  `dupoc_self_of_lt` holds for every `k > 17879`, and `dupoc_fast` shows that there the search
  accepts the agent's own candidate and never reaches the complete enumeration. Others hold at
  every budget, from world `0`: `dupoc_cupod` and `cupod_cimcic` (Open Problems 3 and 5, in this
  system). `dupoc_dimcid` (Open Problem 6) has one of each: DUPOC defects at every budget, DIMCID
  at every large one.
- [`Fair`](BoundedLob/Agents/Fair.lean) proves that different implementations cooperate: DUPOC
  with any two linear budgets `p·k + r` and `q·k + s` with `p, q ≥ 1`, DUPOC with CIMCIC (Theorem
  5.2), and DUPOC with a DUPOC written with a redundant `⊤ →`. These are theorems about every
  large `k`; the builder only tries a few ratios between the two budgets
  (`Candidate.ratios`), so for a lopsided pair the fast path may find nothing.

This proof system is built for the agents: it has the bounded properties as rules, diagonal
formulas as a primitive, and the agents' sources as its programs. It is weaker than arithmetic
in one way that matters. It cannot prove the true sentence `¬□₀⊤`, because every rule is also valid
at world `0`, where every box is true. So its answers to Open Problems 3 and 5 are about this
system only. The parser reads certificates, which is all the search needs; sentences are never
parsed.

## Running it

`lake build BoundedLob` checks the library, using only Lean and Mathlib, with no `native_decide`.
The executable is [`Play.lean`](Play.lean), built with `lake build play_dupoc`.

`lake exe play_dupoc 500000` plays every ordered pair of nine agents at `k = 500000` and prints
each outcome, how each search was decided and how long the match took. A match is reported and
skipped if it would need the complete search over strings, which is a guarantee and not a
practical computation, or if the preflight cannot read a source. At `500000` none is skipped.
`lake exe play_dupoc 17880 --proof` prints the proof two copies of DUPOC find.
