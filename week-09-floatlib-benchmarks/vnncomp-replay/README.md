# VNN-COMP 2025: exact binary32 replay of the published counterexamples

When a neural-network verifier answers "sat" at the International Verification of Neural Networks Competition, it
publishes a counterexample: an input point, in decimals, and the outputs it computed there. That is a concrete
floating-point claim, and a concrete claim can be checked exactly. Here FloatLib evaluates each network in IEEE
binary32 at each published point, in four orders of summation and in exact arithmetic, and decides the property on
exact rationals.

This is exact checking of counterexamples. **No network is verified**: an "unsat" result can be refuted here by a
point, and it is never confirmed. The results are reported by benchmark family. The competition's own scoring is
the place for per-tool numbers.

## Result

Ten benchmark families were replayed: 5,737 counterexample files with 9,165 points.

![10 horizontal bars on an axis from 0 to 100 percent, one per benchmark family. Most bars are dark blue, the reported point is a counterexample, and light blue, the point had to be moved into the box, in varying shares. The SoundnessBench bar is light blue with a short orange end, depends on rounding; the SafeNLP bar ends in a red part of about a sixth, rejected; the Collins RUL CNN bar is mostly green, rounded box. Each bar ends in its count.](figures/01-counterexample-verdicts.svg)

**Figure 1.** What exact evaluation says about the counterexamples that the verifiers of VNN-COMP 2025 published for ACAS Xu, SoundnessBench, SafeNLP, SAT ReLU, Collins RUL CNN, the CORA benchmark, Malbeware, ReLUSplitter, TLL Verify Bench and the test family. A bar is the counterexample files of one benchmark family, split into the files whose reported point is a counterexample, those whose point becomes one after each coordinate outside the input box is moved to the nearest binary32 number inside, those that are counterexamples only for the box with its bounds rounded to binary32, those whose verdict depends on rounding, and those rejected; the number at its end is the files confirmed in one of the first three ways. Each network was evaluated with FloatLib's compiled binary32 operations in four orders of summation and in exact arithmetic, and the property was decided on exact rationals. Evaluated on 2026-11-21 with FloatLib 4.1.0 and Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

| family | instances | with a "sat" result | with an exact counterexample | for the rounded box only | files replayed | as reported | moved into the box | rounded box | depends on rounding | rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACAS Xu | 186 | 47 | 47 | 0 | 362 | 273 | 69 | 0 | 1 | 19 |
| SoundnessBench | 50 | 50 | 50 | 0 | 80 | 0 | 75 | 0 | 5 | 0 |
| SafeNLP | 1,080 | 647 | 647 | 0 | 4,116 | 819 | 2,498 | 0 | 83 | 716 |
| SAT ReLU | 98 | 49 | 49 | 0 | 256 | 248 | 1 | 0 | 7 | 0 |
| Collins RUL CNN | 62 | 23 | 5 | 18 | 153 | 27 | 8 | 117 | 1 | 0 |
| CORA benchmark | 180 | 131 | 131 | 0 | 594 | 190 | 399 | 0 | 0 | 5 |
| Malbeware | 150 | 19 | 19 | 0 | 85 | 19 | 66 | 0 | 0 | 0 |
| ReLUSplitter | 220 | 20 | 20 | 0 | 48 | 8 | 38 | 0 | 0 | 2 |
| TLL Verify Bench | 32 | 17 | 5 | 0 | 40 | 40 | 0 | 0 | 0 | 0 |
| test | 5 | 1 | 1 | 0 | 3 | 1 | 2 | 0 | 0 | 0 |
| all ten | 2,063 | 1,004 | 974 | 18 | 5,737 | 1,625 | 3,156 | 117 | 97 | 742 |

- **Most published counterexamples are real, but not at the point printed.** 1,625 files are counterexamples as
  reported. 3,156 more become one only after the point is moved into the box: the printed point, read to binary32,
  lies outside the box, by at most 6.9e-7 in a coordinate and by 7e-9 in the median, and the nearest binary32 point
  inside violates the property. Verifiers print points on the faces of the box, and the binary32 number nearest to
  a decimal bound is often just outside it.
- **117 files (Collins RUL CNN) are counterexamples only for the box with rounded bounds.** Their properties fix
  inputs to decimals that are not binary32 numbers, so the box holds no binary32 point at all.
- **For 97 files the verdict depends on rounding**: the four binary32 orders and exact arithmetic do not agree on
  whether the property is violated.
- **742 files are rejected**: at the deciding point no order of summation and no exact evaluation violates the
  property. They are of two kinds. 474 print outputs that violate the property but that the network does not
  produce at the printed input. The other 268 are boundary points on SafeNLP: the printed point is outside the box
  by about 7e-9 with a margin of the property below 7e-7, and at the nearest point inside the box the exact output
  is on the safe side by 2e-8 to 1e-6.
- **No "sat" result is wrong, only some witnesses.** In every one of the 742 instances another file is an exact
  counterexample. Of the 1,004 instances with a "sat" result in these families, 974 have an exact counterexample
  and 18 one for the rounded box. The other 12 are TLL Verify Bench instances whose networks were too large to
  replay.

![A scatter plot on two logarithmic axes with a dashed diagonal where the margin equals the rounding error. The confirmed files of ACAS Xu and of the other families lie far above the diagonal, by a factor of more than a hundred for all but two of them; those of SoundnessBench lie on both sides of it, with margins between a hundredth of the rounding error and ten times it. The orange diamonds, files whose verdict depends on rounding, lie along the diagonal, on both sides. The red crosses, rejected files, form two groups: one far above the diagonal at margins from a thousandth to ten, and one just above it at margins below one millionth.](figures/02-margin-against-rounding.svg)

**Figure 2.** How decisive each published counterexample is: the margin by which the network's exact output violates, or fails to violate, the property, against the rounding error of the binary32 evaluation. A marker is one counterexample file; the margin is the absolute slack of the output property at the exact rational output, and the rounding error is the largest absolute difference between a binary32 output and that exact output, both in the units of the network's output. Both axes are logarithmic. Evaluated on 2026-11-21 with FloatLib 4.1.0 and Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

Figure 2 shows why the borderline files are borderline: their margin is the size of the network's own rounding
error.

**"Unsat" results on instances that have an exact counterexample.** 476 "unsat" results are on instances for which
another file is a counterexample inside the box: a binary32 point at which the binary32 network, in each of the
four orders, and the real-valued network all violate the property.

| family | "unsat" results | refuted by an exact counterexample | of which with a kernel-accepted witness | penalised by the competition, tolerance 1e-4 | penalised, tolerance 0 | given points, tolerance 0 | not scored by the competition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACAS Xu | 1,010 | 1 | 1 | 1 | 1 | 0 | 0 |
| SoundnessBench | 30 | 30 | 0 | 30 | 0 | 30 | 0 |
| SafeNLP | 3,118 | 403 | 403 | 403 | 21 | 382 | 0 |
| SAT ReLU | 206 | 42 | 42 | 32 | 32 | 0 | 10 |
| all | 4,364 | 476 | 446 | 466 | 54 | 412 | 10 |

The competition scores the same results twice, with tolerance 1e-4 and with tolerance 0. **Its own scoring with
tolerance 1e-4 already penalises 466 of the 476.** The other 10 are results of a tool that the competition did not
score. With tolerance 0 it penalises 54 and gives 10 points each to the other 412: a counterexample that is only
"correct up to tolerance" leaves the instance undecided, and an "unsat" on an undecided instance earns points too.
Over the 486 instances that the zero-tolerance scoring leaves undecided, exact replay finds a counterexample in the
box for 470 and one for the rounded box for the other 16. So exact replay decides what the two scorings disagree on.

On SoundnessBench all 50 instances have an exact counterexample, although every one of the 80 published points lies
outside the box (by 3.5e-9 to 1.2e-7). Moved into the box, all 80 violate the property in exact arithmetic, 75 of
them also in all four binary32 orders and 5 in some of them, by an exact margin of only 5e-8 to 9e-5, the size of
the rounding error of the network itself.

**In Lean's kernel.** The verdicts above are computed by compiled Lean code. For the small networks they are also
theorems: Lean's kernel replayed all of ACAS Xu and SAT ReLU, one witness per instance of SafeNLP and of the test
family, and the boundary files on SafeNLP. That is 2,576 points and 18,414 theorems in 193 Lean files, every one
accepted, and every audited theorem rests on `propext`, `Classical.choice` and `Quot.sound` only.

| family | Lean files | points | points accepted | theorems | audited with #print axioms | files with exit 0 | errors | instances with a kernel counterexample | refuted unsat results with a kernel witness | sum of wall time (s) | time inside the kernel (s) | median file (s) | slowest file (s) | peak memory of a file (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACAS Xu | 42 | 563 | 563 | 6384 | 2128 | 42 | 0 | 47 of 47 | 1 of 1 | 9073 | 7909 | 163 | 563 | 6.2 |
| SafeNLP | 52 | 1755 | 1755 | 10530 | 7020 | 52 | 0 | 647 of 647 | 403 of 403 | 7784 | 6101 | 155 | 182 | 5.7 |
| SAT ReLU | 98 | 257 | 257 | 1488 | 992 | 98 | 0 | 49 of 49 | 42 of 42 | 4410 | 2154 | 31 | 182 | 9.1 |
| test | 1 | 1 | 1 | 12 | 4 | 1 | 0 | 1 of 1 | 0 of 0 | 31 | 15 | 31 | 31 | 4.9 |

744 of the 974 instances with an exact counterexample have one accepted by the kernel, and 446 of the 476 refuted
"unsat" results have a kernel-accepted witness. The 30 on SoundnessBench do not: one multiply-add costs about 1.1 ms
in the kernel in the form the generated files use, against about 0.15 microseconds compiled, and one evaluation of
that model (24 million multiply-adds) would take the kernel about seven hours.

**Cross-checks** (none feeds a result). numpy float32, in the same order of operations, gives the same binary32
outputs as FloatLib's compiled `Binary32` at all 9,165 points, and FloatLib's decimal parser agrees with exact
rational rounding on all 2.4 million input values. onnxruntime 1.32.0, the runtime of the competition's checker,
sums in another order and differs in the last bits at most points. Its verdict on the output differs from the
reference order's at 114 of 14,432 evaluations, all at points whose margin is below the rounding error. An exact
integer evaluation in Python agrees with the Lean replay on all 1,424 cases it checked.

## What is checked

A counterexample file lists decimal values for the inputs and for the outputs. The replay
([`scripts/Replay.lean`](scripts/Replay.lean), a compiled Lean program on FloatLib) does the following for every
point of every file.

1. **Input.** Each decimal is rounded once to binary32, to nearest with ties to even, by FloatLib's
   `ExecFloat.Binary.parse`, and checked against rounding the exact fraction (`ExecFloat.Binary.ofRat`).
2. **Network.** The ONNX graph, lowered to a chain of layers with every weight as its binary32 word, is evaluated
   on FloatLib's `Binary32` with its compiled `*`, `+`, `-` and `<`. The reference order accumulates every dot
   product from `+0`, left to right, with one rounded multiplication and one rounded addition per term, and adds
   the bias last. The same network is also evaluated in three other orders (right to left, pairwise, one fused
   multiply-add per term) and in exact arithmetic (dyadic numbers, no rounding at all).
3. **Property.** The VNNLIB property is decided on exact rationals.
4. **Moving into the box.** A coordinate outside its interval is moved to the nearest binary32 number inside, and
   the network is evaluated again at the moved point. Where the interval of a coordinate holds no binary32 number
   at all, the coordinate is moved to its bounds rounded to nearest instead: the "rounded box".
5. **Verdict.** ONNX fixes the element type of `MatMul`, `Gemm` and `Conv` but not the order of their additions, so
   "the binary32 output" is one number only up to that order. A file is confirmed only when the four orders and
   exact arithmetic all violate the property, rejected only when none does, and reported as depending on rounding
   otherwise.

The competition's checker works differently, and on purpose: it reads the decimals as binary64, tests the box on
those values, runs onnxruntime on their binary32 rounding, compares with the outputs printed in the file and tests
the property on the printed outputs, all with a tolerance. Exact replay and a tolerance answer different questions.

## What was run

| | |
| --- | --- |
| FloatLib | 4.1.0 ([release](https://github.com/lean-dojo/FloatLib/releases/tag/v4.1.0)), Lean 4.36.0, Mathlib v4.36.0 |
| Benchmarks | <https://github.com/VNN-COMP/vnncomp2025_benchmarks> at `8b7b811b`, 14 families fetched sparse (927 MB) |
| Published results | <https://github.com/VNN-COMP/vnncomp2025_results> at `ea89fbc2` (81 MB): `results.csv`, the counterexample files of the 8 participating tools (alpha-beta-CROWN, CORA, NeuralSAT, nnenum, NNV, PyRAT, Rover, SobolBox) and the logs of the competition's two scorings |
| Python | 3.12.13 with numpy 2.5.0, onnx 1.22.0, onnxruntime 1.32.0 and matplotlib 3.11.0 |
| Date and machine | 2026-11-21, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs. The load average was between 60 and 220 during the whole run |

**Time.** One binary32 evaluation with FloatLib's compiled operations takes a median of 2 ms on ACAS Xu (13,000
multiply-adds) and 3.4 s on the SoundnessBench model (24 million), about 0.15 microseconds per multiply-add in
both. Exact arithmetic takes up to 7 times as long. The whole replay of the nine families other than SoundnessBench
took 19 min 48 s with 5 processes, SoundnessBench alone 14 min 19 s, and the kernel files 5.9 hours of Lean summed
over the files. These are wall-clock times on a loaded machine. They say what the check costs here, not how fast
FloatLib is.

## Limits

- **No network is verified.** An "unsat" that no published point refutes is neither confirmed nor refuted.
- **Families.** The competition's result repository covers 26 families. 14 with small files were fetched and 10 of
  them translate. Cersyve, Dist Shift, LinearizeNN and LSNC ReLU are not translated, because their ONNX graphs are
  not chains of layers, and 16 of the 32 TLL Verify Bench networks have 10 to 67 million weights and were left out
  by a size limit of 5 million. That leaves 273 counterexample files unreplayed. The other 12 families (CIFAR,
  ImageNet, YOLO, ViT and other large networks) were not fetched.
- **Order of summation.** Four orders are evaluated. onnxruntime's own order (vectorised and blocked) is not one of
  them.
- **The translation is trusted.** The ONNX reader, the VNNLIB reader and the counterexample reader are Python
  scripts without proofs. They are checked by agreement with numpy, onnxruntime and an exact integer evaluation.
  The kernel theorems are about the generated Lean definitions.
- **SoundnessBench.** What is replayed is the VNN-COMP 2025 family `soundnessbench` (one model, 50 instances) with
  the counterexamples published in the competition. The hidden counterexamples of the separate SoundnessBench
  benchmark are not published, so there is nothing to replay there.

**From Lean, compiled:** every binary32 and exact evaluation, the rounding of the inputs, the moved points and
every comparison behind a verdict. **From Lean's kernel:** the reference-order verdicts at the points of the kernel
table. **From Python scripts:** the translation of ONNX, VNNLIB and counterexample files, the cross-checks, the
reading of the competition's logs, the counts, the tables and the figures.

## Files

- [`results/families.csv`](results/families.csv): one row per family, the numbers of the tables above.
  [`results/margins.csv`](results/margins.csv): one row per replayed file, its family, its verdict, the rounding
  error of the binary32 output and the margin of the property in exact arithmetic.
- [`figures/`](figures/CAPTIONS.md): both figures as PNG and SVG, with captions and notes.
- [`scripts/`](scripts/Replay.lean): `fetch.sh`, the replay `Replay.lean` (the `replay` executable of the week's
  Lake project), the readers `common.py` and `prepare.py`, `pipeline.sh` and `run_chunks.sh`, the cross-checks
  `crosscheck.py`, `ort_compare.py`, `spotcheck.py` and `handcheck.py`, `competition.py`, `analyze.py`,
  `family_summary.py`, the kernel files `gen_kernel.py`, `witness_ids.py`, `run_kernel.sh` and `kernel_table.py`,
  and `plot.py`.

## Run it again

```bash
lake build replay                                           # in the week's directory
bash vnncomp-replay/scripts/fetch.sh                        # the benchmarks and the published results, pinned
R=vnncomp-replay/run; P=vnncomp-replay/scripts
F="acasxu_2023 soundnessbench safenlp_2024 sat_relu collins_rul_cnn_2022 cora_2024 malbeware relusplitter tllverifybench_2023 test"
python3 -I $P/prepare.py $R/dl/vnncomp2025_benchmarks/benchmarks $R/dl/vnncomp2025_results $R/work $F
bash $P/pipeline.sh 5 $F                                    # replay, numpy and onnxruntime comparisons
python3 -I $P/competition.py $R/dl/vnncomp2025_results $R/results $F     # the competition's two scorings
python3 -I $P/analyze.py $R/work $R/results $F
python3 $P/family_summary.py $R/results
python3 $P/plot.py
```

The kernel files are written by `gen_kernel.py` and checked with `lake env bash vnncomp-replay/scripts/run_kernel.sh`.
