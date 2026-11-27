# Exact replay of VNN-COMP 2025 counterexamples: figure captions

Written by `scripts/plot.py`; do not edit by hand. Every figure is `NAME.png` (9 inches wide at 200 dpi) and `NAME.svg`. An image holds only the plot, its axes with units, a legend with short series names and panel letters; the caption says what the figure shows, what was measured, in which unit, on which scales, where and when. The notes say how, which numbers come from Lean and which from scripts, and give the numbers behind the marks.

## Figure 1: `01-counterexample-verdicts`

![10 horizontal bars on an axis from 0 to 100 percent, one per benchmark family. Most bars are dark blue, the reported point is a counterexample, and light blue, the point had to be moved into the box, in varying shares. The SoundnessBench bar is light blue with a short orange end, depends on rounding; the SafeNLP bar ends in a red part of about a sixth, rejected; the Collins RUL CNN bar is mostly green, rounded box. Each bar ends in its count.](01-counterexample-verdicts.svg "What exact evaluation says about the counterexamples that the verifiers of VNN-COMP 2025 published for ACAS Xu, SoundnessBench, SafeNLP, SAT ReLU, Collins RUL CNN, the CORA benchmark, Malbeware, ReLUSplitter, TLL Verify Bench and the test family. A bar is the counterexample files of one benchmark family, split into the files whose reported point is a counterexample, those whose point becomes one after each coordinate outside the input box is moved to the nearest binary32 number inside, those that are counterexamples only for the box with its bounds rounded to binary32, those whose verdict depends on rounding, and those rejected; the number at its end is the files confirmed in one of the first three ways. Each network was evaluated with FloatLib's compiled binary32 operations in four orders of summation and in exact arithmetic, and the property was decided on exact rationals. Evaluated on 2026-11-21 with FloatLib and Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).")

**Figure 1.** What exact evaluation says about the counterexamples that the verifiers of VNN-COMP 2025 published for ACAS Xu, SoundnessBench, SafeNLP, SAT ReLU, Collins RUL CNN, the CORA benchmark, Malbeware, ReLUSplitter, TLL Verify Bench and the test family. A bar is the counterexample files of one benchmark family, split into the files whose reported point is a counterexample, those whose point becomes one after each coordinate outside the input box is moved to the nearest binary32 number inside, those that are counterexamples only for the box with its bounds rounded to binary32, those whose verdict depends on rounding, and those rejected; the number at its end is the files confirmed in one of the first three ways. Each network was evaluated with FloatLib's compiled binary32 operations in four orders of summation and in exact arithmetic, and the property was decided on exact rationals. Evaluated on 2026-11-21 with FloatLib and Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

Alt text: 10 horizontal bars on an axis from 0 to 100 percent, one per benchmark family. Most bars are dark blue, the reported point is a counterexample, and light blue, the point had to be moved into the box, in varying shares. The SoundnessBench bar is light blue with a short orange end, depends on rounding; the SafeNLP bar ends in a red part of about a sixth, rejected; the Collins RUL CNN bar is mostly green, rounded box. Each bar ends in its count.

Notes: 5,737 counterexample files with 9,165 points: 1,625 confirmed as reported, 3,156 after moving into the box, 117 for the rounded box, 97 depend on rounding, 742 rejected.

| family | files | as reported | after moving into the box | for the rounded box | depends on rounding | rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACAS Xu | 362 | 273 | 69 | 0 | 1 | 19 |
| SoundnessBench | 80 | 0 | 75 | 0 | 5 | 0 |
| SafeNLP | 4116 | 819 | 2498 | 0 | 83 | 716 |
| SAT ReLU | 256 | 248 | 1 | 0 | 7 | 0 |
| Collins RUL CNN | 153 | 27 | 8 | 117 | 1 | 0 |
| CORA benchmark | 594 | 190 | 399 | 0 | 0 | 5 |
| Malbeware | 85 | 19 | 66 | 0 | 0 | 0 |
| ReLUSplitter | 48 | 8 | 38 | 0 | 0 | 2 |
| TLL Verify Bench | 40 | 40 | 0 | 0 | 0 | 0 |
| test | 3 | 1 | 2 | 0 | 0 | 0 |

A point is judged at the reported point when that lies in the box and otherwise at the moved point. It is confirmed when the four binary32 orders (left to right, right to left, pairwise, fused multiply-add) and exact arithmetic all violate the property there, rejected when none does, and depends on rounding when they disagree. The rounded box is the box with every bound rounded to the nearest binary32 number; it is used only in a coordinate where the box itself holds no binary32 number. A file has the best verdict of its points. From Lean: every evaluation and every comparison (the compiled replay). From scripts: the translation of the ONNX and VNNLIB files, the counts and the drawing. Source: `results/families.csv`.

Files: `01-counterexample-verdicts.png`, `01-counterexample-verdicts.svg`.

## Figure 2: `02-margin-against-rounding`

![A scatter plot on two logarithmic axes with a dashed diagonal where the margin equals the rounding error. The confirmed files of ACAS Xu and of the other families lie far above the diagonal, by a factor of more than a hundred for all but two of them; those of SoundnessBench lie on both sides of it, with margins between a hundredth of the rounding error and ten times it. The orange diamonds, files whose verdict depends on rounding, lie along the diagonal, on both sides. The red crosses, rejected files, form two groups: one far above the diagonal at margins from a thousandth to ten, and one just above it at margins below one millionth.](02-margin-against-rounding.svg "How decisive each published counterexample is: the margin by which the network's exact output violates, or fails to violate, the property, against the rounding error of the binary32 evaluation. A marker is one counterexample file; the margin is the absolute slack of the output property at the exact rational output, and the rounding error is the largest absolute difference between a binary32 output and that exact output, both in the units of the network's output. Both axes are logarithmic. Evaluated on 2026-11-21 with FloatLib and Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).")

**Figure 2.** How decisive each published counterexample is: the margin by which the network's exact output violates, or fails to violate, the property, against the rounding error of the binary32 evaluation. A marker is one counterexample file; the margin is the absolute slack of the output property at the exact rational output, and the rounding error is the largest absolute difference between a binary32 output and that exact output, both in the units of the network's output. Both axes are logarithmic. Evaluated on 2026-11-21 with FloatLib and Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

Alt text: A scatter plot on two logarithmic axes with a dashed diagonal where the margin equals the rounding error. The confirmed files of ACAS Xu and of the other families lie far above the diagonal, by a factor of more than a hundred for all but two of them; those of SoundnessBench lie on both sides of it, with margins between a hundredth of the rounding error and ten times it. The orange diamonds, files whose verdict depends on rounding, lie along the diagonal, on both sides. The red crosses, rejected files, form two groups: one far above the diagonal at margins from a thousandth to ten, and one just above it at margins below one millionth.

Notes: 5,450 files drawn, 96 of them below the diagonal. Not drawn: 283 files whose rounding error or margin is exactly zero, and 4 files with a rounding error below 1e-10 or a margin below 1e-9 (ties to within 1e-13). A file is drawn at the point that decides its verdict (the reported point, or the point moved into the box). A verdict depends on rounding when the four binary32 orders (left to right, right to left, pairwise, fused multiply-add) and exact arithmetic do not all agree on whether the output violates the property; for a rejected file the margin is the distance to a violation.

| group | files | rounding error | margin | median of margin / rounding error | below the diagonal |
| --- | ---: | --- | --- | ---: | ---: |
| ACAS Xu, confirmed | 342 | 4.5e-10 to 3.8e-07 | 1.8e-06 to 5.7e-02 | 3.03e+05 | 0 |
| SoundnessBench, confirmed | 75 | 2.5e-06 to 1.4e-05 | 9.8e-08 to 9.0e-05 | 0.805 | 43 |
| other families, confirmed | 4,210 | 7.2e-10 to 3.1e-04 | 3.6e-08 to 1.7e+02 | 3.81e+06 | 1 |
| rejected | 730 | 6.9e-10 to 4.3e-05 | 2.3e-08 to 2.2e+01 | 1.15e+07 | 8 |
| depends on rounding | 93 | 5.3e-09 to 1.2e-04 | 3.2e-09 to 7.2e-05 | 1 | 44 |

From Lean: the binary32 outputs, the exact outputs (dyadic arithmetic without rounding) and the slacks. From scripts: the absolute values, the grouping and the drawing. Source: `results/margins.csv`.

Files: `02-margin-against-rounding.png`, `02-margin-against-rounding.svg`.
