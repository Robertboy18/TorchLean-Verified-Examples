# LeanCert's benchmarks with `interval`

[LeanCert](https://github.com/alerad/leancert) is a Lean library for certified numerics: interval evaluation,
Taylor models, root finding, integration, and tactics on top of them. It is the closest thing to FloatLib's
`interval` tactic in Lean, so I wanted to see the two side by side on LeanCert's own material: its seven benchmark
suites, its two script benchmarks, and the statements its examples and tests prove.

Three things to keep in mind, stated here once. The bounds of the suite goals are the enclosures LeanCert's own
evaluators report, so those goals sit exactly at LeanCert's edge by construction. LeanCert ran on a port to Lean
v4.36.0 that I made (three edits and one added file, all its tactic tests green), not on the toolchain it is
released for. And the machine was shared and at times saturated: both tools were timed side by side, but single
times carry that noise, and the time of one goal can be off by a factor of two between runs.

## Result

![Horizontal bars in 9 groups of three, one group per suite, on an axis from 0 to 100 percent: FloatLib interval, LeanCert kernel-checked and LeanCert with its compiled checker. Evaluation: 40, 32 and 32 of 40. NumericalRefinement: 2, 2 and 2 of 2. DomainAwareAD: 8, 8 and 8 of 8. bench-showcase: 3, 3 and 3 of 3. bench-trust point: 10, 10 and 10 of 10. bench-trust subdiv: 80, 80 and 80 of 80. bench-trust optimization: 1, 1 and 1 of 1. bench-trust finsum: 0, 2 and 3 of 3. examples and tests: 197, 203 and 210 of 210.](figures/02-goals-per-suite.svg)

**Figure 2.** Share of each suite's inequality goals that FloatLib's `interval` and LeanCert's own tactic prove with one call; the number at the end of a bar is the count of goals. The first eight rows are LeanCert's benchmark suites and script benchmarks, stated as goals; the last row holds the inequalities, with six combinations of comparisons, that LeanCert's examples and tests prove with one tactic call. Goals of other shapes (integrals, root existence and uniqueness, exact identities, list constructions, derivatives) are not in the figure, because `interval` does not address them. Measured on FloatLib and LeanCert 44b323a ported to Lean 4.36.0 and Mathlib v4.36.0, on 2026-11-21 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

LeanCert's seven suites have 119 cases. They time compiled evaluators and checkers, and none of the cases is a
theorem statement, so each case was first stated as a goal (112 goals). Every goal was given to LeanCert's own
tactic and to `interval`, one call each, with the same `lean` binary on the same machine.

- **Inequality goals of the seven suites (50 goals).** `interval` proves all 50: 35 in a plain file, 14 after
  raising Lean's recursion and heartbeat limits, 1 with a larger precision. LeanCert, kernel-checked, proves 42,
  and the same 42 with its compiled checker.
- **Time per goal** on the 42 goals both prove: median 254 ms for `interval` and 4.9 s for kernel-checked LeanCert,
  a median ratio of 8.6. `interval` is the faster one on 38 of them and LeanCert on 4. Against LeanCert's compiled
  checker the median ratio is 4.7.
- **LeanCert proves what `interval` does not address (62 goals of the seven suites).** Bounds on definite
  integrals, existence and uniqueness of roots, derivatives, exact polynomial identities and list constructions.
  `interval` proves real inequalities and refutes sets of real comparisons. It does not integrate, isolate roots or
  differentiate. LeanCert proves 30 of these kernel-checked (37 with its compiled checker), and they are not
  counted for or against `interval`.
- **LeanCert's two script benchmarks** (99 statements, 97 of them inequalities). `interval` proves 94 and
  kernel-checked LeanCert 96. The difference is three finite sums: `interval` proves none of them (100, 1,000 and
  10,000 terms), and LeanCert's `finsum_bound` proves the first two kernel-checked and all three with its compiled
  checker.
- **LeanCert's examples and tests** (210 inequalities or combinations of comparisons). `interval` proves 197 and
  kernel-checked LeanCert 203. What `interval` misses there: finite sums of 100 terms and more, real powers
  `x ^ (1/2)` on a box that touches zero, the Euler-Mascheroni constant, and a conjunction of two quantified bounds.
- **All together** (357 goals in `interval`'s scope): `interval` proves 341 and kernel-checked LeanCert 341. 327 are
  proved by both, **14 only by `interval`, 14 only by LeanCert** and 2 by neither. On the common goals the median
  time per goal is 66 ms for `interval` and 638 ms for LeanCert (median ratio 8.7), and `interval` is the faster one
  on 303 of them.

![Two panels with three rising step curves each on a logarithmic time axis from 0.004 to 300 seconds. In panel (a) the FloatLib curve ends at 144 of 147 goals, the kernel-checked LeanCert curve at 138 and the compiled-checker LeanCert curve at 139; in panel (b) they end at 197, 203 and 210 of 210 goals. The FloatLib curve rises furthest to the left in both panels.](figures/01-time-per-goal.svg)

**Figure 1.** Goals proved against time per goal, by FloatLib's `interval` and by LeanCert's own tactic on the same statements, each with one call. A curve counts the goals proved within the time on the horizontal axis; the time is that of the whole `theorem` command inside Lean (elaboration, the tactic and the kernel check), the median of two repetitions for a run under 20 s and one repetition otherwise, with both tools running side by side. Panel (a) holds the inequality goals of LeanCert's seven benchmark suites and of its two script benchmarks, panel (b) the inequalities, with six combinations of comparisons, that LeanCert's examples and tests prove with one tactic call. The time axis is logarithmic, a curve further left is faster and one that ends higher proves more. Measured on FloatLib and LeanCert 44b323a ported to Lean 4.36.0 and Mathlib v4.36.0, on 2026-11-21 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs), at most 7 Lean processes at once.

In the tables, `n (a, b, c)` means `n` goals proved: `a` in a plain file with default options, `b` only with raised
Lean limits, `c` only with the larger tactic options. Times are per goal, medians over the goals both tools prove.
The ratio is LeanCert's time divided by `interval`'s, per goal, then the median.

**Table 1. The seven suites: inequality goals**

| Suite | Cases | Goals | Inequality goals | `interval` | LeanCert, kernel-checked | LeanCert, compiled checker | Both | Median time `interval` | Median time LeanCert | Median ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evaluation | 69 | 40 | 40 | 40 (26, 14, 0) | 32 (14, 18, 0) | 32 (20, 12, 0) | 32 | 349 ms | 6.3 s | 8.4 |
| Integration | 12 | 26 | 0 |  |  |  |  |  |  |  |
| Krawczyk | 6 | 6 | 0 |  |  |  |  |  |  |  |
| DomainAwareAD | 6 | 17 | 8 | 8 (8, 0, 0) | 8 (8, 0, 0) | 8 (8, 0, 0) | 8 | 553 ms | 3.1 s | 8.3 |
| AlgebraicBezout | 3 | 3 | 0 |  |  |  |  |  |  |  |
| DyadicSubdivision | 18 | 18 | 0 |  |  |  |  |  |  |  |
| NumericalRefinement | 5 | 2 | 2 | 2 (1, 0, 1) | 2 (2, 0, 0) | 2 (2, 0, 0) | 2 | 38 ms | 1.2 s | 29.4 |
| **all** | 119 | 112 | 50 | 50 (35, 14, 1) | 42 (24, 18, 0) | 42 (30, 12, 0) | 42 | 254 ms | 4.9 s | 8.6 |

**Table 2. The seven suites: goals outside `interval`**

| Suite | Goals outside `interval` | Shape | LeanCert, kernel-checked | LeanCert, compiled checker | What `interval` says |
| --- | ---: | --- | ---: | ---: | --- |
| Integration | 14 | integral | 0 | 0 | an atom has no bound (14) |
| Integration | 12 | checker flag | 5 | 12 | not stated: LeanCert's own vocabulary (12) |
| Krawczyk | 6 | root | 4 | 4 | not stated: LeanCert's own vocabulary (4); not an inequality (2) |
| DomainAwareAD | 9 | derivative | 0 | 0 | an atom has no bound (9) |
| AlgebraicBezout | 3 | identity | 3 | 3 | not stated: LeanCert's own vocabulary (3) |
| DyadicSubdivision | 18 | list construction | 18 | 18 | not stated: LeanCert's own vocabulary (18) |

**Table 3. LeanCert's script benchmarks**

| Suite | Cases | Goals | Inequality goals | `interval` | LeanCert, kernel-checked | LeanCert, compiled checker | Both | Median time `interval` | Median time LeanCert | Median ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench-showcase | 5 | 5 | 3 | 3 (3, 0, 0) | 3 (3, 0, 0) | 3 (3, 0, 0) | 3 | 57 ms | 638 ms | 8.3 |
| bench-trust point | 10 | 10 | 10 | 10 (10, 0, 0) | 10 (10, 0, 0) | 10 (10, 0, 0) | 10 | 61 ms | 596 ms | 9.8 |
| bench-trust subdiv | 80 | 80 | 80 | 80 (80, 0, 0) | 80 (80, 0, 0) | 80 (80, 0, 0) | 80 | 66 ms | 749 ms | 11.6 |
| bench-trust optimization | 1 | 1 | 1 | 1 (1, 0, 0) | 1 (1, 0, 0) | 1 (1, 0, 0) | 1 | 56 ms | 313 ms | 5.6 |
| bench-trust finsum | 3 | 3 | 3 | 0 (0, 0, 0) | 2 (1, 1, 0) | 3 (3, 0, 0) | 0 |  |  |  |
| **all** | 99 | 99 | 97 | 94 (94, 0, 0) | 96 (95, 1, 0) | 97 (97, 0, 0) | 94 | 66 ms | 728 ms | 11.4 |

**Table 4. Statements of LeanCert's examples and tests, by shape**

| Shape | Statements | `interval` | LeanCert, kernel-checked | LeanCert as written |
| --- | ---: | ---: | ---: | ---: |
| inequality | 204 | 192 (191, 0, 1) | 197 (194, 3, 0) | 204 (202, 2, 0) |
| existence | 49 | 5 (5, 0, 0) | 49 (49, 0, 0) | 49 (49, 0, 0) |
| inequality in ℕ with a ceiling | 21 | 0 (0, 0, 0) | 21 (21, 0, 0) | 21 (21, 0, 0) |
| integral | 10 | 0 (0, 0, 0) | 10 (10, 0, 0) | 10 (10, 0, 0) |
| conjunction | 4 | 3 (3, 0, 0) | 4 (4, 0, 0) | 4 (4, 0, 0) |
| bound for all large n | 2 | 0 (0, 0, 0) | 2 (2, 0, 0) | 2 (2, 0, 0) |
| disequality | 2 | 2 (2, 0, 0) | 2 (2, 0, 0) | 2 (2, 0, 0) |

## Goals one proves and the other does not

- **`interval` and not kernel-checked LeanCert (14).** Four goals whose bound is the enclosure of LeanCert's own
  rational evaluator for nested sines on `[-1/16, 1/16]`, the 640-map affine chain (two goals) and the 2,048-term
  cancellation (two goals), where LeanCert runs out of the 300 s. And six statements that LeanCert's tests prove
  with `interval_bound_adaptive`, a tactic that has no kernel-checked route.
- **Kernel-checked LeanCert and not `interval` (14).** Finite sums (`∑ k ∈ Finset.Icc 1 100, (k : ℝ) ≤ 5050` takes
  LeanCert's `finsum_bound` 0.4 s, and `interval` does not finish it in 300 s), three bounds on the
  Euler-Mascheroni constant, for which `interval` has no enclosure, four real powers on a box that touches zero
  (`∀ x ∈ Set.Icc 0 4, x ^ (1/2) ≤ 2`), and one conjunction of two quantified bounds.
- **Neither (2).** Finite sums of 5,000 and 10,000 terms. LeanCert's compiled checker proves both in 0.2 s.

## How a case became a goal, and how a goal was run

The rules were fixed before any tactic ran, and they are the same for every case
([`scripts/gen_goals.py`](scripts/gen_goals.py)).

- A case that computes an enclosure `[lo, hi]` of an expression over a box becomes two goals,
  `∀ x ∈ Set.Icc a b, e ≤ hi` and `∀ x ∈ Set.Icc a b, lo ≤ e`, with `lo` and `hi` the exact rationals the case
  reports (up to 55,000 digits for the rational backend). The Lean term mirrors the benchmark's syntax tree node
  for node.
- A goal whose shape is not a real inequality is recorded as outside `interval`.
- The two script benchmarks already consist of statements, and they were taken verbatim.
  [`scripts/harvest_examples.py`](scripts/harvest_examples.py) took every statement that LeanCert's examples and
  tests prove with one LeanCert tactic call, if the statement names nothing of LeanCert.

One Lean process per goal, tool and configuration ([`scripts/run_goals.py`](scripts/run_goals.py)):

- **One call.** FloatLib: `interval`. LeanCert: `leancert`, its front door, or the dedicated tactic that
  LeanCert's own benchmark or test names (`interval_decide`, `finsum_bound`, `certify_bound`, ...).
- **Axioms.** A goal counts only if Lean reports no error and `#print axioms` lists nothing beyond `propext`,
  `Classical.choice` and `Quot.sound`. LeanCert ships with `leancert.trust = "native"`: its certificates are closed
  by `native_decide`, which adds one axiom per proof. So LeanCert was run twice: with `(trust := kernel)`, which is
  the column that counts, and as shipped, reported for reference.
- **A ladder.** A plain file and the default options first. Only if that fails, the same call with
  `maxRecDepth 100000` and `maxHeartbeats 0` ("raised limits"). Only if that fails, one fixed set of larger options,
  the same for every goal. Nothing was tuned to a goal. Every run has 300 s of wall time.
- **Time per goal** is the time of the `theorem` command inside Lean: elaboration of the statement, the tactic and
  the kernel check of the proof, without the import. Importing `LeanCert.Tactic` alone takes a process 3.1 s here
  and importing `FloatLib` 3.2 s. The proving rung of both tools was run again side by side, and the time per goal
  is the median of these timing runs.

## The port of LeanCert

LeanCert pins Lean v4.35.0 and Mathlib v4.35.0. To run both tools with the same `lean` binary I ported it to Lean
v4.36.0 and Mathlib v4.36.0 ([`scripts/leancert-lean-4.36.diff`](scripts/leancert-lean-4.36.diff)): two one-line
source edits, one new file with its import, and the version pins. The new file is the one that matters for a fair comparison.
Mathlib v4.36.0 defines real arithmetic as private definitions that a package outside the module system can unfold,
and LeanCert's metaprograms were written against opaque arithmetic. The file marks ten definitions irreducible
again, so LeanCert behaves as written, and all of its tactic tests build on the port. Three analysis modules that
nothing here needs were not ported, and the compiled benchmark executable was not built: LeanCert's harness was run
with the interpreter, only to read off the enclosure each case computes.

## What was run

| | |
| --- | --- |
| FloatLib | `main` at [`fc2fad1f`](https://github.com/lean-dojo/FloatLib/commit/fc2fad1f9c7de970f0a10a44c58cdf2909a867b4) |
| LeanCert | <https://github.com/alerad/leancert> at `44b323a0d6af00211d5013f6e021432db032087c` (2026-10-28, Apache-2.0), ported as above |
| Lean and Mathlib | `leanprover/lean4:v4.36.0`, the same `lean` binary for both tools, and Mathlib v4.36.0 |
| Date and machine | 2026-11-21, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs. Idle CPU between 0 % and 62 % (median 38 %) during the runs, at most seven Lean processes at once |

**What failed and why.** `interval`, inequality goals not proved: no proof in 300 s (7), an atom has no bound (4), a
partial function rejected the box (4), not an inequality (1). LeanCert kernel-checked, inequality goals not proved:
no proof in 300 s (6), proved only with a compiler axiom (6), no LeanCert strategy proved it (2), maximum recursion
depth (2).

**Not measured here:** LeanCert's compiled harness timings, and LeanCert on its own toolchain. LeanCert publishes
wall times for its `bench-trust` files on an Apple M1 Max with Lean v4.32.2: about 0.1 s per `interval_decide`
lemma on the kernel route.

**From Lean:** every verdict and every time per goal (`IO.monoNanosNow` around the `theorem` command). LeanCert's
enclosures, which are the bounds of the goals, come from LeanCert's own harness. **From scripts:** the goal
statements, the process wall times and the timeouts, the medians, counts and ratios, and the figures.

## Files

- [`results/goals.csv`](results/goals.csv): one row per goal with both tools' verdict, rung and time.
  [`results/suites.csv`](results/suites.csv), [`results/cases.csv`](results/cases.csv) (per harness case),
  [`results/digest.json`](results/digest.json) (the totals quoted above),
  [`results/examples-dropped.csv`](results/examples-dropped.csv) (statements left out, with the reason).
- [`figures/`](figures/CAPTIONS.md): both figures as PNG and SVG, with captions and notes.
- [`scripts/`](scripts/run_goals.py): `fetch.sh` and `leancert-lean-4.36.diff` (LeanCert at its commit, ported),
  `lcbuild.sh`, `lcrun.sh`, `flrun.sh`, `harness_cases.sh`, `gen_goals.py`, `harvest_examples.py`, `run_goals.py`,
  `summarize.py`, `plot.py`.

## Run it again

```bash
lake build                                                  # in the week's directory: FloatLib
bash leancert/scripts/fetch.sh                              # LeanCert at its commit, ported to Lean v4.36.0
bash leancert/scripts/lcbuild.sh LeanCert.Tactic LeanCert.Benchmark.Main
python3 leancert/scripts/run_goals.py --help                # goals, runs and repetitions
python3 leancert/scripts/summarize.py && python3 leancert/scripts/plot.py
```
