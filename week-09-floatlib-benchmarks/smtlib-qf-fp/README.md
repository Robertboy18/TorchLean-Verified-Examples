# SMT-LIB QF_FP: 39,994 single-operation tests as Lean theorems

The floating-point benchmarks of [SMT-LIB](https://smt-lib.org/) (logic QF_FP) hold 40,406 files. 39,994 of them
are the Wintersteiger family: each fixes the operands and the result of one binary64 operation and asks whether
the operation gives that result. The results were produced on hardware and by Z3's arithmetic, so the family is a
conformance test. Here every one of them becomes a Lean statement about FloatLib's operation, and `float_decide`
proves the benchmark's status in Lean's kernel.

## Result

| | count |
| --- | ---: |
| instances in the family | 39,994 |
| translated | 39,994 |
| proved by `float_decide` with the three standard axioms | 39,994 |
| disagree with the expected status | 0 |

All 39,994 are translated and proved with `propext`, `Classical.choice` and `Quot.sound` only, and none disagrees
with the benchmark. Two full runs gave the same count. By operation, in the run that counts (kernel times by the
wall clock, eight Lean processes at once on a busy machine):

| operation | FloatLib operation | instances | sat | unsat | proved | disagree | kernel total (s) | kernel median (ms) | of which decision (ms) | 95th percentile (ms) | slowest (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp.add` | `add` | 3,024 | 1,512 | 1,512 | 3,024 | 0 | 21.66 | 6.59 | 0.57 | 9.84 | 115.8 |
| `fp.sub` | `sub` | 2,790 | 1,395 | 1,395 | 2,790 | 0 | 17.32 | 5.36 | 0.49 | 9.21 | 113.8 |
| `fp.mul` | `mul` | 2,938 | 1,469 | 1,469 | 2,938 | 0 | 21.34 | 6.28 | 0.66 | 20.50 | 56.5 |
| `fp.div` | `div` | 2,792 | 1,396 | 1,396 | 2,792 | 0 | 19.71 | 6.62 | 0.71 | 12.23 | 63.5 |
| `fp.fma` | `fma` | 2,864 | 1,432 | 1,432 | 2,864 | 0 | 22.16 | 7.05 | 0.96 | 12.81 | 73.4 |
| `fp.sqrt` | `sqrt` | 2,828 | 1,414 | 1,414 | 2,828 | 0 | 18.96 | 6.41 | 0.75 | 11.36 | 74.4 |
| `fp.rem` | `remainder` | 2,804 | 1,402 | 1,402 | 2,804 | 0 | 23.71 | 7.28 | 2.84 | 17.37 | 71.2 |
| `fp.roundToIntegral` | `roundToIntegral` | 2,864 | 1,432 | 1,432 | 2,864 | 0 | 14.13 | 3.56 | 0.57 | 13.62 | 68.6 |
| `fp.min` | `minimumNumber` | 2,852 | 1,426 | 1,426 | 2,852 | 0 | 23.13 | 6.75 | 2.60 | 10.46 | 55.9 |
| `fp.max` | `maximumNumber` | 2,854 | 1,427 | 1,427 | 2,854 | 0 | 29.25 | 9.34 | 3.53 | 28.16 | 119.7 |
| `fp.abs` | `abs` | 2,856 | 1,428 | 1,428 | 2,856 | 0 | 12.26 | 4.54 | 0.52 | 5.15 | 55.6 |
| `fp.eq` | `compareEqual` | 2,776 | 1,388 | 1,388 | 2,776 | 0 | 10.53 | 3.29 | 0.51 | 4.81 | 37.1 |
| `fp.lt` | `compareLess` | 2,848 | 1,424 | 1,424 | 2,848 | 0 | 12.67 | 4.08 | 0.64 | 11.67 | 83.4 |
| `fp.gt` | `compareGreater` | 2,904 | 1,452 | 1,452 | 2,904 | 0 | 15.96 | 4.18 | 0.65 | 17.37 | 94.7 |
| all | | 39,994 | 19,997 | 19,997 | 39,994 | 0 | 262.78 | 5.84 | 0.68 | 11.77 | 119.7 |

The kernel takes 262.8 s for the 39,994 proofs: 52.2 s is the decision by evaluation and 208.7 s the check of the
rewrite of the public operations to their kernel operations. The whole family takes 7:28 of wall time.

![Two panels of markers with whiskers on a logarithmic time axis from 2 to 100 milliseconds, seven operations each: add, sub, mul, div, fma, sqrt and rem above, and rounding to an integer, min, max, abs, eq, lt and gt below. In every group the open FloatLib marker for the whole Lean run is highest, between 47 and 73 milliseconds, the Z3 square lies between 21 and 24, the cvc5 triangle is near 6, and the filled FloatLib marker for the kernel check is lowest, between 2.7 and 4.8, except for min and max, where it is just above the cvc5 triangle at about 6.4.](figures/01-time-by-operation.svg)

**Figure 1.** Time per instance on SMT-LIB's single-operation floating-point benchmarks (the Wintersteiger family, binary64), by operation: FloatLib's proof by `float_decide` in Lean against the SMT solvers Z3 5.4.1 and cvc5 1.4.2. A marker is the median over the 100 sampled instances of an operation of the time per instance, in milliseconds, with a whisker from the 5th to the 95th percentile: for FloatLib the time Lean's kernel takes to check the instance's proof, and the whole Lean run, which adds parsing, elaboration, the axiom report and an equal share of loading the library with 250 instances per process; for a solver the wall time of one process per instance. The time axis is logarithmic, and lower is faster. Measured on FloatLib 4.1.0 on 2026-11-21 with Lean 4.36.0, the fastest of three runs of each instance, one process at a time on one core of an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

**The solvers are faster at this task, and that is fine.** Z3 and cvc5 fold the constants, so their time is the
cost of starting a process and reading a file. On a sample of 1,400 instances, one process at a time on a core of
its own, the fastest of three runs:

| per instance, 1,400 instances | median (ms) | 5th to 95th percentile (ms) |
| --- | ---: | ---: |
| FloatLib: Lean's kernel checks the proof | 4.11 | 2.64 to 6.37 |
| of which the rewrite to kernel operations | 3.55 | 2.18 to 4.38 |
| of which the decision by evaluation | 0.48 | 0.36 to 2.55 |
| FloatLib: the whole Lean run, 250 instances per process | 62.3 | 47.2 to 75.8 |
| FloatLib: one instance per Lean process (40 instances) | 3,360 | 3,059 to 3,584 (least to greatest) |
| Z3 5.4.1, one process per instance | 23.5 (19.7 of CPU) | 17.2 to 25.9 |
| Z3 5.4.1, 250 instances per process | 7.9 to 9.5 over three repetitions | |
| cvc5 1.4.2, one process per instance | 5.75 (4.24 of CPU) | 5.39 to 6.22 |
| cvc5 1.4.2, 250 instances per process | 1.8 to 2.6 over three repetitions | |

Lean's kernel check alone costs about as much as a cvc5 process and a sixth of a Z3 process. The whole Lean run
costs about 3 times Z3's time and 11 times cvc5's per instance, and about 8 and 35 times when the solvers answer a
batch in one process as Lean does. Z3 and cvc5 gave the benchmark's status in all 4,218 runs on the 1,406 sampled
instances that have one. What the Lean run adds is something else: each status is a theorem about FloatLib's proved
operation, checked by Lean's kernel with three axioms.

![Two panels of horizontal bars on an axis from 0 to 100 percent. Above, six bars for the families: the wintersteiger bar is entirely blue and ends in 39,994 of 39,994, and the bars of griggio, schanda, Vector, ramalho and UltimateAutomizer are almost or entirely light, ending in 2 of 214, 4 of 44, 6 of 91, 0 of 36 and 0 of 27. Below, five dark bars for Bitwuzla, COLIBRI, cvc5, colibri2 and Z3-Owl that fill between about two thirds and nine tenths of the axis, ending in 255 of 275 for Bitwuzla, and a light bar for FloatLib that ends in 0 of 275.](figures/02-reach-and-competition.svg)

**Figure 2.** What FloatLib's `float_decide` reaches in the QF_FP benchmarks of SMT-LIB 2025, and what SMT solvers solved in that logic at SMT-COMP 2025. In panel (a) a bar is one benchmark family, with the share of its benchmarks whose inputs are all fixed and whose status Lean proved, and the count at its end; the rest of a bar are search problems, which `float_decide` does not attempt. In panel (b) a bar is one solver of the single-query track on the competition's 275 selected benchmarks, from the published results, with FloatLib's count of benchmarks in reach among the same 275 below them. FloatLib 4.1.0 was run on 2026-11-21 with Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs); the competition ran in 2025 on its own cluster with 1,200 seconds per benchmark.

**The other families are search problems.** `float_decide` evaluates a closed statement and does not search.

| family | benchmarks | fixed inputs | translated | proved by `float_decide` | search problems |
| --- | ---: | ---: | ---: | ---: | ---: |
| wintersteiger | 39,994 | 39,994 | 39,994 | 39,994 | 0 |
| griggio | 214 | 2 | 2 | 2 | 212 |
| schanda | 44 | 4 | 4 | 4 | 40 |
| Vector | 91 | 6 | 6 | 6 | 85 |
| ramalho | 36 | 0 | 0 | 0 | 36 |
| UltimateAutomizer | 27 | 0 | 0 | 0 | 27 |
| all | 40,406 | 40,006 | 40,006 | 40,006 | 400 |

Twelve files outside the Wintersteiger family fix their inputs, and Lean proves all twelve: four of schanda
(`10 + 0.4`, `10 - 0.4`, `10 * 0.4` and `10 / 0.4` in binary32 lie in a stated interval), two of griggio that
contain no assertion, so they say nothing about floating point, and six of Vector without a declared constant. The
benchmark gives those six no status. Lean proves `sat`, and Z3 and cvc5 answer `sat`. The other 400 files ask for
inputs that satisfy a formula, or for a proof that none exists, and they are out of scope.

**SMT-COMP 2025 has no benchmark in common with what `float_decide` reaches.** Its QF_FP division ran 275
selected benchmarks with 1,200 seconds each, all of them search problems. Its rules remove every benchmark that
all solvers answered within a second in earlier years, which takes out the whole Wintersteiger family. So the
competition and this suite answer different questions: search for inputs there, evaluation at given inputs with a
kernel-checked proof here. From the published results:

| solver | benchmarks | solved | sat | unsat | not solved | CPU time on the solved (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bitwuzla | 275 | 255 | 140 | 115 | 20 | 12,530 |
| COLIBRI | 275 | 232 | 133 | 99 | 43 | 1,682 |
| cvc5 | 275 | 230 | 139 | 91 | 45 | 18,499 |
| colibri2 | 275 | 199 | 117 | 82 | 76 | 5,740 |
| Z3-Owl | 275 | 173 | 104 | 69 | 102 | 23,773 |
| Z3-Owl-base | 275 | 139 | 82 | 57 | 136 | 27,556 |
| FloatLib `float_decide` | 275 | 0 in reach | | | | |

The last competition that ran the family was SMT-COMP 2018, where each solver answered all 39,994 correctly (CPU
times on StarExec, not comparable with the times above):

| solver, SMT-COMP 2018 | benchmarks | correct | wrong or unknown | CPU time, median (ms) | slowest (ms) | total (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CVC4 2018-06-10 | 39,994 | 39,994 | 0 | 10.1 | 29 | 412 |
| Z3 4.7.1 | 39,994 | 39,994 | 0 | 35.3 | 45 | 1,410 |
| COLIBRI v2038 | 39,994 | 39,994 | 0 | 286.8 | 303 | 11,487 |

## One instance depends on a choice that SMT-LIB leaves open

`fp.min(-0, +0)` may be either zero in SMT-LIB. One instance, `min-has-solution-13472`, expects `+0`, which is what
the benchmark's generator got from the hardware. IEEE 754-2019 `minimumNumber`, and so FloatLib, gives `-0`. The
translation quantifies over the choice, as a model of the benchmark may, and the statement is proved. With `fp.min`
and `fp.max` read as `minimumNumber` and `maximumNumber` everywhere, 5,705 of the 5,706 min and max instances are
proved and this one is not. This is a difference between two specifications. It is no bug in FloatLib and no
error in the benchmark.

## The translation

An instance is in reach when every declared constant that its assertions use is fixed by the instance: some
assertion is an equation `(= c t)` whose other side mentions only constants that are already fixed. Those
equations are the stated model, the other assertions are the constraint, and the Lean statement is the constraint
evaluated at the model: `constraint = true` for status sat, `constraint = false` for unsat. The translator
([`scripts/qffp2lean.py`](scripts/qffp2lean.py)) reads the syntax and the sorts of a term only: never the file's
name, its expected result or the value of an operand.

| SMT-LIB | Lean, on FloatLib 4.1.0 |
| --- | --- |
| `(_ FloatingPoint eb sb)` | `ExecFloat.Binary (exponentBits := eb) (fractionBits := sb - 1)`; at 11 and 53 this is `Binary64` |
| `(fp s e f)`, `(_ +oo ..)`, `(_ -zero ..)`, `(_ NaN ..)`, `((_ to_fp eb sb) bits)` | `ExecFloat.Binary.ofNatBits` of the encoding; the NaN of SMT-LIB as the quiet NaN with an empty payload |
| `=` on floats | `same a b := Nat.beq (toNatBits a) (toNatBits b) \|\| (isNaN a && isNaN b)`: one NaN, two zeros |
| `fp.add`, `fp.sub`, `fp.mul`, `fp.div`, `fp.fma`, `fp.sqrt` with a mode | `ExecFloat.Binary.add x y (mode := m)` and the like |
| `fp.rem`, `fp.roundToIntegral`, `fp.abs`, `fp.neg` | `ExecFloat.Binary.remainder`, `.roundToIntegral x m`, `.abs`, `-x` |
| `fp.min`, `fp.max` | `minimumNumber`, `maximumNumber`, with the model's choice on zeros of opposite signs (below) |
| `fp.eq`, `fp.lt`, `fp.leq`, `fp.gt`, `fp.geq` | `ExecFloat.compareEqual`, `compareLess`, `compareLessEqual`, `compareGreater`, `compareGreaterEqual` |
| `fp.isNaN` and the other classes | `ExecFloat.Binary.isNaN`, `isInfinite`, `isZero`, `isSubnormal`, `isNormal`; `isNegative`, `isPositive` from `isNaN` and `signBit` |
| RNE, RNA, RTP, RTN, RTZ | `.nearest`, `.nearestAway`, `.up`, `.down`, `.towardZero` |
| `not`, `and`, `or`, `=>`, `xor`, `ite`, `let`, `distinct`, bit-vector literals, defined functions | Boolean operations, `bif`, substitution |

For example:

```lean
theorem «wintersteiger/add/add-has-solution-10020.smt2» :
    (F11_53.same (ExecFloat.Binary.add (ExecFloat.Binary.ofNatBits 0xe9021f4d5f5c6903 : F11_53)
      (ExecFloat.Binary.ofNatBits 0x87dab3681267e43b : F11_53) (mode := .nearest))
      (ExecFloat.Binary.ofNatBits 0xe9021f4d5f5c6903 : F11_53)) = true := by
  float_decide [F11_53.same]
```

Each generated file imports FloatLib, defines the format's `same`, `min` and `max` once and holds 250 theorems.
After each theorem a command of the file checks the theorem's proof term and the auxiliary lemma of `float_decide`
again with `Lean.Kernel.check`, times that with the monotonic clock, and prints the theorem's axioms. A theorem
counts as proved when that line shows a kernel time and no axiom beyond the three, and Lean reported no error.

**Controls.** A statement that is trivially true would also be "proved", so three controls ran. The claim of the
opposite status for 286 instances: Lean rejected all 286. The same claim with the lowest bit of the result flipped
for 1,100 instances: Lean rejected 1,045 and accepted 55, and all 55 have a NaN as expected result, which the flip
leaves a NaN. And the rounding modes check themselves: with about 5,000 instances per mode, a wrong mode in the
translation would fail on most inexact results.

**A self-test beyond the benchmark.** The benchmark has one format and four rounding modes. So there are 4,158 more
instances of the same shape for seven formats from 4 to 128 bits, all five rounding modes and the 24 functions
the translator knows, with Z3's value as the expected result. Lean proves all 4,158 through the same translator
(kernel 23.5 s, median 4.5 ms). cvc5 answers sat on the 1,188 instances of binary32 and binary64, and on the other
2,970 with `--fp-exp`. These are not part of SMT-LIB and are counted nowhere above.

## What was run

| | |
| --- | --- |
| FloatLib | 4.1.0 ([release](https://github.com/lean-dojo/FloatLib/releases/tag/v4.1.0)), Lean 4.36.0, Mathlib v4.36.0 |
| SMT-LIB 2025, QF_FP | release 2025.08.04, `QF_FP.tar.zst`, 7,524,054 bytes, MD5 `b7f9786b5317d7be8a185c861088f504`, CC BY 4.0, Zenodo record 16740866 (<https://zenodo.org/records/16740866>, DOI 10.5281/zenodo.16740866). 40,406 files: 39,994 wintersteiger, 214 griggio, 91 Vector, 44 schanda, 36 ramalho, 27 UltimateAutomizer |
| SMT-COMP results | `data/results-sq-2025.json.gz`, `benchmarks-2025.json.gz` and `results-sq-2018.json.gz` of <https://github.com/SMT-COMP/smt-comp.github.io>, and the [division's page](https://smt-comp.github.io/2025/results/qf_fp-single-query) |
| Z3 | 5.4.1, the binary of the PyPI wheel `z3_solver-5.4.1.0-py3-none-manylinux_2_27_x86_64.whl` |
| cvc5 | 1.4.2 (`282d5e0`), `cvc5-Linux-x86_64-static.zip` of <https://github.com/cvc5/cvc5/releases/tag/cvc5-1.4.2> |
| Date and machine | 2026-11-21, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs |

**Timing.** The main run took its kernel times with eight Lean processes at once while 58 to 77 percent of the
machine's CPUs were busy, so they include moments in which a process was not running. The per-instance comparison
with the solvers ran one process at a time, pinned to the least busy physical core with a core-scheduling cookie,
each tool three times in rotated order, with the machine at a load average of 97 to 131. The time of a process is
the monotonic clock from its start to its exit, the same for the three tools. One instance per Lean process costs
3.36 s, nearly all of it `import FloatLib`, which is why the instances are batched.

**What failed, and limits.** Nothing failed among the instances with fixed inputs: 40,006 translated, 40,006
proved. The 400 search problems are not translated, by design. The translator does not translate the conversions
between formats, to and from bit-vectors as numbers and to reals, nor bit-vector arithmetic: no instance with
fixed inputs uses them. A run of the two solvers on all 39,994 instances was not made: the solvers ran on the
sample.

**From Lean:** every proof, its axioms, the kernel time of each theorem and the clock between consecutive
theorems. **From scripts:** the classification and the translation, the counts and sums, the process times of
Lean, Z3 and cvc5, and the competition's numbers, read from its JSON files. Nothing is estimated.

## Files

- [`results/`](results/facts.json): `fig-time.csv`, `fig-coverage.csv` and `fig-smtcomp2025.csv` (what the figures
  draw), `facts.json`, `wintersteiger-summary.json` (per operation), `side-summary.json` (other families and
  controls), `ieee-minmax-summary.json`, `selftest-summary.json`, `zero-pairs.csv`,
  `smtcomp2018-wintersteiger-summary.csv` and the batch timings. The per-instance files are large and are made
  again by the scripts.
- [`figures/`](figures/CAPTIONS.md): both figures as PNG and SVG, with captions and notes.
- [`scripts/`](scripts/qffp2lean.py): `fetch.sh`, the translator `qffp2lean.py`, `run_batches.sh` and `run_one.sh`,
  `collect.py`, `check_controls.py`, `time_solvers.py` and `run_timing.sh`, `make_smt_batches.py`,
  `selftest_gen.py` and `run_selftest.sh`, `smtcomp_qffp.py`, `smtcomp2018_wintersteiger.py`, `zero_pairs.py`,
  `survey.py`, `make_results.py`, `plot.py`.

## Run it again

```bash
lake build                                                  # in the week's directory: FloatLib 4.1.0
bash smtlib-qf-fp/scripts/fetch.sh                          # SMT-LIB QF_FP, SMT-COMP results, Z3, cvc5
R=smtlib-qf-fp/run; Q=$R/dl/non-incremental/QF_FP
python3 -I smtlib-qf-fp/scripts/qffp2lean.py classify $Q $R/meta/classify.jsonl
python3 -I smtlib-qf-fp/scripts/qffp2lean.py emit $Q $R/meta/classify.jsonl $R/lean/wintersteiger \
    --only-family wintersteiger --batch 250 --prefix W
bash smtlib-qf-fp/scripts/run_batches.sh $R/lean/wintersteiger 8        # about 8 minutes
python3 -I smtlib-qf-fp/scripts/collect.py $R/lean/wintersteiger \
    $R/results/wintersteiger-instances.csv $R/results/wintersteiger-summary.json
```

`qffp2lean.py --help` lists the options for the other families, the controls and the min and max variant, and
`make_results.py` and `plot.py` make the small result files and the figures from the raw results.
