# SMT-LIB QF_NRA outside MetiTarski with `interval`

QF_NRA is SMT-LIB's logic of nonlinear real arithmetic. An unsatisfiable benchmark there says that a set of
polynomial constraints has no solution, and FloatLib's `interval` tactic proves exactly such statements when it
can box the variables. So how far does one call of `interval` get on the real thing, next to the solvers of
SMT-COMP 2025? SMT-LIB's `meti-tarski` family (7,006 benchmarks) is left out here, and MetiTarski's own problem
set has [a suite of its own](../metitarski-flyspeck/). This one takes the other 5,148 benchmarks.

## Result

**The set.** 3,133 of the 5,148 benchmarks are unsat candidates: 2,921 by their `:status`, and 212 with status
`unknown` that a solver of SMT-COMP 2025 answered unsat and none answered sat. 3,064 translate to a Lean goal
`¬ ∃ x, constraints`. The other 69 are over a size cap of 200,000 nodes. For 66 of them an interval propagation,
fixed before the runs, bounds every variable. Those 66 are the selection.

| family | benchmarks | `:status unsat` | `unknown`, unsat for a 2025 solver | candidates | translated | selected: every variable bounded | of these in SMT-COMP 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ezsmt | 63 | 0 | 2 | 2 | 2 | 2 | 2 |
| Geogebra | 112 | 0 | 0 | 0 | 0 | 0 | 0 |
| Heizmann | 69 | 0 | 16 | 16 | 16 | 0 | 0 |
| hong | 20 | 20 | 0 | 20 | 20 | 20 | 7 |
| hycomp | 2,752 | 2,227 | 23 | 2,250 | 2,250 | 22 | 0 |
| kissing | 45 | 3 | 0 | 3 | 3 | 3 | 1 |
| LassoRanker | 821 | 137 | 159 | 296 | 231 | 0 | 0 |
| Mulligan | 135 | 42 | 0 | 42 | 42 | 0 | 0 |
| Pine | 245 | 0 | 6 | 6 | 6 | 6 | 6 |
| pPDA | 20 | 2 | 0 | 2 | 2 | 0 | 0 |
| Sturm-MBO | 405 | 285 | 0 | 285 | 281 | 6 | 0 |
| Sturm-MGC | 9 | 7 | 0 | 7 | 7 | 0 | 0 |
| UltimateAutomizer | 61 | 13 | 0 | 13 | 13 | 0 | 0 |
| Uncu | 225 | 155 | 0 | 155 | 155 | 3 | 1 |
| zankl | 166 | 30 | 6 | 36 | 36 | 4 | 0 |
| all | 5,148 | 2,921 | 212 | 3,133 | 3,064 | 66 | 17 |

**On the selection `interval` proves 56 of 66**, one call each, Lean's default limits, 60 seconds per goal. Every
proof is accepted by Lean's kernel with `propext`, `Classical.choice` and `Quot.sound` only.

| family | translated | proved | refused | refused: by the tactic | refused: at `maxHeartbeats` | refused: statement at `maxRecDepth` | refused: tactic at `maxRecDepth` | timed out (60 s) | not run | median (max) wall of the proofs (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ezsmt (status unknown) | 2 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | n/a |
| hong | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7.7 (28.9) |
| hycomp | 22 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8.6 (14.2) |
| kissing | 3 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | 0 | n/a |
| Pine (status unknown) | 6 | 1 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 22.5 (22.5) |
| Sturm-MBO | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3.4 (3.7) |
| Uncu | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4.4 (5.5) |
| zankl | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3.8 (4.8) |
| all | 66 | 56 | 10 | 5 | 3 | 2 | 0 | 0 | 0 | 7.3 (28.9) |

![Two panels of horizontal bars on an axis from 0 to 100 percent. Above, 8 bars for the families with bounded variables, of which 5 are entirely blue, the colour of proved goals, one is mostly grey and two are hatched or dotted. Below, 10 bars for the families of the other benchmarks, which are almost entirely grey, hatched or dotted, with a thin blue part at the left of a few. Each bar ends in its count.](figures/01-qfnra-families.svg)

**Figure 1.** What FloatLib's interval tactic proves on the unsatisfiable SMT-LIB 2025 QF_NRA benchmarks outside MetiTarski, by benchmark family. A bar is one family, split into the goals that Lean accepted as proved, those the tactic refused, and those that stopped at Lean's default limit of heartbeats or of recursion depth, with the number proved at its end. Panel (a) has the 66 benchmarks whose variables are all bounded by their constraints (the selection), and panel (b), for information, all 2,998 other translated benchmarks, where the selection analysis finds no finite bound for some variable. One call of interval per goal with Lean's default limits and 60 seconds of wall time, run on 2026-11-21 with FloatLib 4.1.0 and Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

One of the proofs settles a benchmark whose status in the release is `unknown`:
`20200911-Pine/1599121980156583000.smt2`, which Z3-alpha, cvc5, Yices2 and SMT-RAT all answered unsat in the
competition.

**Against the solvers, on the same instances.** A solver's `unsat` is the answer of a program that is trusted as a
whole. A goal counted for FloatLib is a theorem that Lean's kernel checked. The tables count both in one column
each, and that difference is the reason the comparison is worth making.

SMT-COMP 2025 ran 17 of the 66. From its published results (1,200 seconds per benchmark on the competition's
cluster):

| family | instances | FloatLib `interval` proved | Z3-alpha unsat | cvc5 unsat | Yices2 unsat | SMT-RAT unsat | proved, and unsat for none of the four | not proved, and unsat for all four |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ezsmt | 2 | 0 | 2 | 2 | 2 | 2 | 0 | 2 |
| hong | 7 | 7 | 2 | 7 | 7 | 7 | 0 | 0 |
| kissing | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pine | 6 | 1 | 6 | 5 | 5 | 4 | 0 | 2 |
| Uncu | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| all | 17 | 9 | 11 | 15 | 15 | 14 | 0 | 4 |

`interval` proves 9 and the four solvers answer 11 to 15. It proves all 7 hong instances, of which Z3-alpha
answered 2. It misses the 2 ezsmt instances and 5 of the 6 Pine instances, which most solvers answer.

On this machine, all 66 with 60 seconds each and every tool timed the same way, from the start of its process to
its exit:

| family | instances | FloatLib `interval` proved | Z3 5.3.3 unsat | cvc5 1.4.2 unsat | Yices 2.8.0 unsat |
|---|---:|---:|---:|---:|---:|
| ezsmt | 2 | 0 | 2 | 2 | 2 |
| hong | 20 | 20 | 9 | 20 | 20 |
| hycomp | 22 | 22 | 22 | 22 | 22 |
| kissing | 3 | 0 | 0 | 0 | 0 |
| Pine | 6 | 1 | 5 | 3 | 5 |
| Sturm-MBO | 6 | 6 | 6 | 6 | 6 |
| Uncu | 3 | 3 | 3 | 3 | 3 |
| zankl | 4 | 4 | 4 | 4 | 4 |
| all | 66 | 56 | 51 | 60 | 62 |

| on the 66 instances | FloatLib `interval` | Z3 5.3.3 | cvc5 1.4.2 | Yices 2.8.0 |
|---|---:|---:|---:|---:|
| proved / answered unsat | 56 | 51 | 60 | 62 |
| refused (Lean reported an error within 60 s) | 10 |  |  |  |
| timeout at 60 s | 0 | 15 | 6 | 4 |
| unknown |  | 0 | 0 | 0 |
| sat |  | 0 | 0 | 0 |
| error |  | 0 | 0 | 0 |
| median wall time of these (s) | 7.30 | 0.03 | 0.02 | 0.02 |
| largest wall time of these (s) | 28.85 | 35.03 | 3.32 | 1.40 |
| proved by FloatLib, not answered unsat by the solver | | 11 | 0 | 0 |
| answered unsat by the solver, not proved by FloatLib | | 6 | 4 | 6 |

Z3 5.3.3 answers none of hong_10 to hong_20 within 60 s, and `interval` proves all eleven in 7.6 to 28.9 s. cvc5
and Yices answer every instance that `interval` proves. No tool answers the three kissing instances. **On what
they answer, the solvers are faster by two orders of magnitude**: 0.02 to 0.03 s in the median against 7.3 s, about
2 of them to start Lean and load the library. These are the current releases with their default options, not the
versions or configurations that their authors entered in the competition.

![Three panels. In the first and the third, one bar per tool: FloatLib interval in blue, then Z3-alpha, cvc5, Yices2 and SMT-RAT in dark grey. In the first the bars reach between half and nine tenths of the width; in the third the bar of interval is a thin sliver and the solvers' bars cover almost all of the width. In the middle panel four rising step curves: the three solvers' curves climb at a few hundredths of a second to between 40 and 60 instances, and the blue curve of interval starts near 3 seconds and passes the curve of Z3 before 30 seconds.](figures/02-qfnra-solvers.svg)

**Figure 2.** FloatLib's interval tactic next to SMT solvers on the same unsatisfiable QF_NRA instances outside MetiTarski. Panels (a) and (c) give, for interval and for four solvers of SMT-COMP 2025, the share of the instances that it proved or that the solver answered unsat in the published results of the competition, with the count at the end of the bar: on the 17 selected instances (every variable bounded) that the competition ran, and on the 1,065 other candidates that the competition ran. Panel (b) gives, for all 66 selected instances, the number proved by interval or answered unsat by a current release of three solvers within a wall time, in seconds on a logarithmic axis, when all four ran on one machine with 60 seconds per instance. Run on 2026-11-21 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs); the competition allowed 1,200 seconds per instance on its own cluster.

**Outside the selection the picture is the opposite.** The other 2,998 translated candidates are the ones where
the selection analysis finds no finite bound for some variable. `interval` proves 208 of them (6.9 percent): all
150 hycomp instances at unrolling depth 0 that are not in the selection, and 58 small polynomial goals.

| family | translated | proved | refused | refused: by the tactic | refused: at `maxHeartbeats` | refused: statement at `maxRecDepth` | refused: tactic at `maxRecDepth` | timed out (60 s) | not run | median (max) wall of the proofs (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heizmann | 16 | 0 | 16 | 0 | 8 | 8 | 0 | 0 | 0 | n/a |
| hycomp | 2,228 | 150 | 2,078 | 24 | 1,336 | 718 | 0 | 0 | 0 | 4.4 (9.5) |
| LassoRanker | 231 | 0 | 231 | 0 | 7 | 224 | 0 | 0 | 0 | n/a |
| Mulligan | 42 | 6 | 36 | 32 | 4 | 0 | 0 | 0 | 0 | 4.4 (6.4) |
| pPDA | 2 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | n/a |
| Sturm-MBO | 275 | 23 | 252 | 0 | 179 | 0 | 73 | 0 | 0 | 7.6 (24.0) |
| Sturm-MGC | 7 | 0 | 7 | 7 | 0 | 0 | 0 | 0 | 0 | n/a |
| UltimateAutomizer | 13 | 0 | 13 | 0 | 10 | 3 | 0 | 0 | 0 | n/a |
| Uncu | 152 | 26 | 126 | 119 | 7 | 0 | 0 | 0 | 0 | 4.0 (12.8) |
| zankl | 32 | 3 | 29 | 6 | 22 | 1 | 0 | 0 | 0 | 10.5 (13.8) |
| all | 2,998 | 208 | 2,790 | 189 | 1,574 | 954 | 73 | 0 | 0 | 4.5 (24.0) |

On the 1,065 of them that the competition ran, `interval` proves 17 and the four solvers answer 948 to 1,039. The
size of a goal says most of it: of the 694 goals with at most 20 variables 263 are proved, and of the 2,370 with
more, one.

| variables | goals run | proved | of them in the selection: run | proved |
|---|---:|---:|---:|---:|
| 0 to 4 | 152 | 53 | 20 | 18 |
| 5 to 9 | 323 | 28 | 12 | 9 |
| 10 to 14 | 188 | 170 | 18 | 17 |
| 15 to 20 | 31 | 12 | 12 | 12 |
| 21 to 49 | 375 | 1 | 1 | 0 |
| 50 to 140 | 1,036 | 0 | 1 | 0 |
| 141 and more | 959 | 0 | 2 | 0 |
| all | 3,064 | 264 | 66 | 56 |

Over all 3,064 translated candidates that is 264 proofs. A trace run shows how they were found:

| the goal is closed by | proofs in the selection | proofs outside it | all | families |
|---|---:|---:|---:|---|
| the ranges of the atoms alone, without a box | 24 | 171 | 195 | hycomp 172, Sturm-MBO 20, Uncu 2, hong 1 |
| a certificate of one box | 26 | 3 | 29 | hong 19, Sturm-MBO 9, zankl 1 |
| certificates of more than one box | 3 | 0 | 3 | Pine 1, Uncu 1, zankl 1 |
| linarith or nlinarith, without a box | 1 | 31 | 32 | Uncu 25, Mulligan 6, zankl 1 |
| a definite form, without a box | 2 | 3 | 5 | zankl 4, Uncu 1 |
| all | 56 | 208 | 264 | |

So on this set the branch and bound search itself decides 3 goals of 264, and the goals that would need many boxes
(Pine, kissing) are the ones that are refused.

## What failed and why

- **Pine, 5 of 6.** Each goal checks that a loop invariant is inductive, in 4 or 6 real variables. `interval`
  stops at its own limits (40 cuts along a branch, 20,000 boxes) after 19 to 31 s. Larger limits don't help within
  60 s. Fewer variables do: with the two variables of the next state replaced by the expressions that define them,
  an equivalent goal outside the protocol, the default call proves 3 of the 6.
- **kissing, 3 of 3.** 14, 39 and 100 real variables: points on a sphere pairwise at distance at least 1, which
  the kissing numbers 6, 12 and 24 exclude. Lean's heartbeat limit is reached. None of Z3, cvc5 and Yices answers
  these in 60 s here either.
- **ezsmt, 2 of 2.** 1,985 and 4,829 variables, almost all Boolean. Lean stops with "maximum recursion depth has
  been reached" in the binder list of the statement, so the tactic never runs. This is Lean, not FloatLib: more
  than about 140 variables under one `∃` do not elaborate at the default `maxRecDepth`.
- **Outside the selection.** 1,574 goals stop at Lean's heartbeat limit, 954 at Lean's recursion limit in the
  statement, 189 because the tactic reports that a variable has no bound, and 73 at the recursion limit inside
  the tactic. hycomp is bounded model checking of hybrid automata: every depth-0 goal is proved and nothing from
  depth 1 on, where the route spends its budget on the Boolean structure before it searches.
- No goal timed out. The slowest refusal took 59.6 s.

## The translation, and how it was checked

[`scripts/smt2lean.py`](scripts/smt2lean.py) parses a benchmark into a term graph and writes one Lean goal. One
rule per SMT-LIB construct, the same for every benchmark:

- The real variables that occur become `x0 x1 ... : ℝ` under one `∃`.
- A Boolean variable becomes a real `b` with the conjunct `b = 0 ∨ b = 1`, and the literal `b` reads `1 / 2 < b`.
  `interval` reads only comparisons of reals as constraints, and `∃ b : Bool, φ b` is equivalent to
  `∃ r : ℝ, (r = 0 ∨ r = 1) ∧ φ (1/2 < r)`.
- `let` and `define-fun` are substituted. `+ - * /`, numerals and decimals become the same operations on `ℝ` with
  exact rational literals. Terms are kept as written: nothing is expanded, factored or reordered.
- The operands of an n-ary `+`, `*`, `and`, `or`, and the assertions themselves, are joined as balanced binary
  trees.
- `< <= > >= = distinct` become `< ≤ > ≥ = ≠`, and `not`, `=>`, `xor`, Boolean `=` and Boolean `ite` become `¬`,
  `→`, `¬ (· ↔ ·)`, `↔` and `(c → a) ∧ (¬ c → b)`.

A goal file is

```lean
module

public import FloatLib.Tactic.Interval.Constraints
meta import FloatLib.Tactic.Interval.Constraints

theorem goal : ¬ ∃ (x0 x1 x2 : ℝ),
    (((((x0 * x0) + ((x1 * x1) + (x2 * x2))) : ℝ) < 1) ∧
    (((x0 * (x1 * x2)) : ℝ) > 1)) := by
  interval

#print axioms goal
```

(this one is `hong/hong_3.smt2`). The translation was tested, not proved:

- **A differential test** ([`scripts/crosscheck.py`](scripts/crosscheck.py)). At random rational points, an
  interpreter of the SMT-LIB source written for this test and an evaluator of the Lean text on disk must agree on
  the truth of the whole goal, the values of every comparison, the truth of every assertion and the truth of every
  top-level conjunct. It finds no mismatch on any of the 3,064 translations at 12 points each, and it reports all
  20 goals that [`scripts/mutation_test.py`](scripts/mutation_test.py) alters on purpose.
- **Controls.** 200 benchmarks with `:status sat` were translated and run in the same way. Their goals are false
  statements, and Lean proved none of them.

## What was run

| | |
| --- | --- |
| FloatLib | 4.1.0 ([release](https://github.com/lean-dojo/FloatLib/releases/tag/v4.1.0)), Lean 4.36.0 (`611c117b`), Mathlib v4.36.0 (`fc4f59e5`) |
| SMT-LIB 2025, QF_NRA | release 2025.08.04, `QF_NRA.tar.zst`, 210,801,606 bytes, MD5 `85a04fcea6dd10fb4c505703cac46e2c`, CC BY 4.0, <https://zenodo.org/records/16740866/files/QF_NRA.tar.zst>. 12,154 benchmarks, 7,006 of them in `meti-tarski` |
| SMT-COMP 2025 results | `data/results-sq-2025.json.gz` of <https://github.com/SMT-COMP/smt-comp.github.io> at `cd15cbedd4ef312d615448100797a80e922cf575`. The single-query track ran 3,104 QF_NRA benchmarks, 1,955 of them outside MetiTarski, with 1,200 seconds each |
| Z3 | 5.3.3, the binary of `z3_solver-5.3.3.0-py3-none-manylinux_2_27_x86_64.whl` (<https://github.com/Z3Prover/z3/releases/tag/z3-5.3.3>) |
| cvc5 | 1.4.2, `cvc5-Linux-x86_64-static.zip` (<https://github.com/cvc5/cvc5/releases/tag/cvc5-1.4.2>) |
| Yices | 2.8.0, `yices-2.8.0-x86_64-pc-linux-gnu-static-gmp.tar.gz` (<https://github.com/SRI-CSL/yices2/releases/tag/yices-2.8.0>) |
| Date and machine | 2026-11-21, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs |

**One process per goal.** `lean -M 8192 <file>`, 8 processes at once, one thread each. The limit is 60 seconds of
wall time from the start of the process to its exit, so it includes the 2 seconds or so that Lean needs to start
and load the library. A goal is *proved* only when Lean exits with status 0, prints no error, and
`#print axioms goal` lists nothing beyond the three standard axioms. The tactic's own defaults are 40 cuts along a
branch and 20,000 boxes, and Lean's are `maxHeartbeats` 200,000 and `maxRecDepth` 512. Peak memory is 1.9 GiB in
the median and 3.23 GiB at most.

**How busy the machine was.** The load average was between 50 and 226 during the runs on 96 hardware threads, with
a median of 144 during the reported run of the selection. The same 66 goals took 1.36 times longer in the median
in that run than in an earlier one, with the same outcome for every goal. So the times are those of a loaded shared
machine, good to a factor of about 1.5. The outcomes do not depend on the load: Lean's heartbeats count work and
not time.

**From Lean:** every count of proofs. **From scripts:** the selection, the translation, the timing and the tables.
**From the competition:** its published per-instance results.

## Files

- [`results/per-instance.csv`](results/per-instance.csv): one row for each of the 3,133 candidate benchmarks
  (selection, translation, Lean outcome and time, the published answers of the eight SMT-COMP 2025 entries, the
  answers of the three solvers run here). [`results/how-closed.csv`](results/how-closed.csv): the step of the route
  that closed each of the 264 proofs. [`results/run-facts.csv`](results/run-facts.csv): versions and load.
- [`figures/`](figures/CAPTIONS.md): both figures as PNG and SVG, with captions and notes.
- [`scripts/`](scripts/smt2lean.py): `fetch.sh`, `scan_status.py`, the translator and selection `smt2lean.py` and
  `analyze.py`, `run_lean.py` and `launch_lean.sh`, `run_solvers.py`, `crosscheck.py`, `mutation_test.py`,
  `make_trace_goals.py`, `trace_summary.py`, `substitute_defined.py`, `compare.py`, `plot.py`.

## Run it again

```bash
lake build                                                  # in the week's directory: FloatLib 4.1.0
bash smtlib-qf-nra/scripts/fetch.sh                         # SMT-LIB QF_NRA (3.3 GB extracted), results, solvers
R=smtlib-qf-nra/run; Q=$R/dl/non-incremental/QF_NRA; C=$R/dl/results-sq-2025.json.gz; P=smtlib-qf-nra/scripts
python3 -I $P/scan_status.py $Q $R/scan.tsv
python3 -I $P/analyze.py $Q $C $R/final --workers 8 --limit 300 --bool-literal half
bash $P/launch_lean.sh $R/final/lean $R/main-final.list $R/runs/main 60 --workers 8 --threads 1 --memory-mb 8192
python3 -I $P/run_solvers.py $Q $R/selected.paths $R/runs/solvers --limit 60 --workers 8 \
    --solver "z3=$PWD/$R/tools/z3/z3 -smt2" --solver "cvc5=$PWD/$R/tools/cvc5/cvc5" \
    --solver "yices2=$PWD/$R/tools/yices/yices-smt2"
python3 -I $P/crosscheck.py $Q $R/final/candidates.jsonl $R/final $R/crosscheck.jsonl
```

`analyze.py` writes the goal files and the lists of their names, and `compare.py --help` shows how the join of the
Lean runs, the solver runs and the competition's results is made.
