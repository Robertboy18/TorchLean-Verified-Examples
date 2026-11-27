# decide +kernel against native_decide: figure captions

Written by `scripts/plot.py`; do not edit by hand. Every figure is `NAME.png` (9 inches wide at 200 dpi) and `NAME.svg`. An image holds only the plot, its axes with units, a legend with short series names and panel letters; the caption says what the figure shows, what was measured, in which unit, on which scales, where and when. The notes say how, which numbers come from Lean and which from scripts, and give the numbers behind the marks.

## Figure 1: `01-kernel-native-decide`

![Six rows, one per closed statement, each with its name above a linear time axis in milliseconds: a blue circle for decide +kernel and a grey cross for native_decide joined by a bar, all between 3 and 30 milliseconds; one row has no cross.](01-kernel-native-decide.svg "Time to prove one closed statement by decide +kernel on FloatLib's kernel layer and by native_decide. A mark is the time Lean takes to elaborate one example, in milliseconds, elaboration included; native_decide compiles the statement and trusts the compiler, and a row without its mark is a statement on which it fails. The time axis is linear, and lower is faster. Measured on FloatLib on 2026-11-20 (the median of 5 runs) with Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).")

**Figure 1.** Time to prove one closed statement by decide +kernel on FloatLib's kernel layer and by native_decide. A mark is the time Lean takes to elaborate one example, in milliseconds, elaboration included; native_decide compiles the statement and trusts the compiler, and a row without its mark is a statement on which it fails. The time axis is linear, and lower is faster. Measured on FloatLib on 2026-11-20 (the median of 5 runs) with Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

Alt text: Six rows, one per closed statement, each with its name above a linear time axis in milliseconds: a blue circle for decide +kernel and a grey cross for native_decide joined by a bar, all between 3 and 30 milliseconds; one row has no cross.

Notes: The statements are closed Boolean statements on the functions of `Numerics.Kernel` (`benchmarks/exact/native.py`); each time is the median of 5 runs of FloatLib's runner, one sample per run. FloatLib never uses `native_decide`. Over the 5 statements that both prove, `decide +kernel` takes 0.90 to 2.04 times the time of `native_decide`, with no compiler in the trusted base; `native_decide` fails on packed row product of 256 lanes with its guards and norm (it cannot compile a definition that the statement depends on).

| statement | `decide +kernel` | `native_decide` |
| --- | ---: | ---: |
| 3 ^ (p - 1) % p = 1 for p = 2 ^ 255 - 19 | 13.1 ms | 13.6 ms |
| 3 ^ (p - 1) % p = 1 for p = 2 ^ 1279 - 1 | 30.0 ms | 17.6 ms |
| the square root of 2 ^ 2001 + 12345, checked | 11.1 ms | 12.4 ms |
| sum of 1 / k ^ 2 for k ≤ 40 below 1.63, on fractions | 4.1 ms | 3.4 ms |
| exp (7/5) ≤ 4.0553 on the grid, degree 20, 64 bits | 8.9 ms | 4.3 ms |
| packed row product of 256 lanes with its guards and norm | 6.2 ms | fails |

From Lean: every time. From scripts: nothing but the drawing. Source: `results/kernel-native.csv`.

Files: `01-kernel-native-decide.png`, `01-kernel-native-decide.svg`.
