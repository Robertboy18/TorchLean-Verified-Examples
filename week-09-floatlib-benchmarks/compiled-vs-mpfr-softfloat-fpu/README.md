# Compiled code: FloatLib next to the host FPU, SoftFloat and MPFR

How fast is a proved software float when you just run it? Here FloatLib's public `Binary32` and `Binary64`
operations, the ones you get when you write `x + y`, run in a compiled loop next to the host processor's own
arithmetic, Berkeley SoftFloat 3e and MPFR 4.1.0. Then the same operations go up to 4,096 bits against MPFR.

## Result

![Two panels, binary32 above binary64, each with six groups of four markers, FloatLib, MPFR, SoftFloat and the host FPU from left to right, on a logarithmic time axis: in every group the host FPU is lowest, between 8 and 13 nanoseconds, SoftFloat lies between 18 and 33, MPFR between 32 and 59 and FloatLib between 24 and 118.](figures/01-public-binary32-64.svg)

**Figure 1.** Time of FloatLib's public binary32 and binary64 operations in compiled code, next to the host processor's own arithmetic, Berkeley SoftFloat 3e and MPFR 4.1.0. A marker is the median of 9 trials of the time per operation, in nanoseconds, of a loop in which each result selects the next operands, and its whisker runs from the fastest to the slowest trial; the operations are addition, subtraction, multiplication, division, square root and fused multiply-add, rounded to nearest even. The time axis is logarithmic, and lower is faster. Measured on FloatLib on 2026-11-20 in one run on one pinned core of an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs) with Lean 4.36.0.

Median of 9 trials (fastest to slowest trial), nanoseconds per operation:

| format | operation | FloatLib | MPFR | SoftFloat | host FPU |
| --- | --- | ---: | ---: | ---: | ---: |
| binary32 | add | 40.6 (40.3 to 42.4) | 33.7 (33.5 to 36.2) | 22.8 (22.8 to 23.3) | 8.4 (8.4 to 10.3) |
| binary32 | sub | 40.9 (40.8 to 41.9) | 35.5 (35.4 to 37.5) | 22.7 (22.6 to 24.3) | 8.4 (8.4 to 8.7) |
| binary32 | mul | 23.8 (23.7 to 24.6) | 32.6 (32.5 to 33.6) | 17.6 (17.5 to 21.1) | 8.5 (8.4 to 10.1) |
| binary32 | div | 57.8 (57.5 to 59.7) | 36.1 (36.0 to 37.7) | 25.0 (24.9 to 25.9) | 10.4 (10.3 to 10.5) |
| binary32 | sqrt | 29.9 (29.8 to 30.1) | 40.9 (40.6 to 42.9) | 25.5 (25.5 to 26.0) | 10.5 (10.5 to 10.8) |
| binary32 | fma | 39.1 (38.8 to 43.2) | 55.5 (54.9 to 59.4) | 26.9 (26.7 to 30.6) | 8.8 (8.7 to 8.9) |
| binary64 | add | 31.3 (31.2 to 32.5) | 37.0 (36.4 to 39.6) | 21.8 (21.6 to 22.8) | 8.3 (8.3 to 8.9) |
| binary64 | sub | 30.8 (30.6 to 31.8) | 36.0 (35.7 to 37.3) | 21.5 (21.4 to 23.1) | 8.3 (8.3 to 12.1) |
| binary64 | mul | 23.7 (23.6 to 24.9) | 32.2 (31.9 to 35.5) | 17.7 (17.6 to 21.6) | 8.4 (8.3 to 8.7) |
| binary64 | div | 67.9 (67.8 to 68.2) | 34.3 (34.1 to 41.1) | 33.4 (33.2 to 34.1) | 11.2 (11.2 to 11.8) |
| binary64 | sqrt | 102.3 (102.0 to 103.0) | 44.6 (44.1 to 47.1) | 27.6 (27.5 to 28.9) | 12.7 (12.4 to 13.6) |
| binary64 | fma | 118.0 (117.8 to 129.0) | 58.6 (58.0 to 68.0) | 27.0 (26.9 to 28.5) | 8.6 (8.6 to 8.8) |

- **binary32.** FloatLib takes 23.8 to 57.8 ns, MPFR 32.6 to 55.5, SoftFloat 17.6 to 26.9 and the host FPU 8.4 to
  10.5. FloatLib is faster than MPFR at multiplication, square root and fused multiply-add (0.70 to 0.73 of MPFR's
  time) and slower at addition, subtraction and division (1.15 to 1.60).
- **binary64.** FloatLib takes 23.7 to 118.0 ns, MPFR 32.2 to 58.6, SoftFloat 17.7 to 33.4 and the host FPU 8.3 to
  12.7. FloatLib is faster than MPFR at addition, subtraction and multiplication (0.74 to 0.86) and slower at
  division, square root and fused multiply-add (1.98 to 2.29).
- Over the 12 cells FloatLib takes 0.70 to 2.29 times MPFR's time (faster in 6, median 1.0), 1.17 to 4.38 times
  SoftFloat's (median 1.6) and 2.8 to 13.7 times the host FPU's. SoftFloat is faster than FloatLib in every cell,
  and the hardware is faster than everything, as it should be.

![Six panels, one per operation, each with a FloatLib line and an MPFR line against format width from 32 to 4,096 bits on logarithmic axes: the two lines are close at 32 and 64 bits, the FloatLib line is above at 128 bits, steps up to about two microseconds or more at 256 bits and is nearly flat from there, while the MPFR line rises steadily, so that the gap narrows towards 4,096 bits in every panel, least for addition and subtraction.](figures/02-floatlib-mpfr-widths.svg)

**Figure 2.** Compiled time of FloatLib's binary operations against MPFR 4.1.0 at the same precision, exponent range and rounding, from 32 to 4,096 bits. A marker is the median of 9 trials of the time per operation, in nanoseconds, of a loop in which each result selects the next operands, with a whisker from the fastest to the slowest trial. Both axes are logarithmic, and lower is faster. Measured on FloatLib on 2026-11-20 in one run on one pinned core of an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs) with Lean 4.36.0.

FloatLib's time over MPFR's (ratio of medians, below 1 FloatLib is faster):

| width | add | sub | mul | div | sqrt | fma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.20 | 1.15 | 0.73 | 1.60 | 0.73 | 0.70 |
| 64 | 0.85 | 0.86 | 0.74 | 1.98 | 2.29 | 2.01 |
| 128 | 1.96 | 1.93 | 1.21 | 4.17 | 3.96 | 1.50 |
| 256 | 35 | 35 | 30 | 14 | 26 | 23 |
| 512 | 32 | 34 | 19 | 11 | 17 | 18 |
| 1,024 | 29 | 32 | 11 | 6.69 | 13 | 9.74 |
| 2,048 | 25 | 27 | 5.74 | 3.89 | 9.67 | 4.65 |
| 4,096 | 20 | 22 | 3.27 | 2.11 | 7.51 | 2.78 |

- **128 bits** (FloatLib's fixed-limb kernel): 1.21 to 4.17 times MPFR.
- **256 to 4,096 bits.** MPFR wins clearly. FloatLib has one kernel per operation there, the exact baseline on
  Lean's arbitrary-precision naturals: addition and subtraction take 1.8 to 2.5 µs, 20 to 35 times MPFR.
  Multiplication takes 1.7 to 4.3 µs (30 times MPFR at 256 bits, 3.3 at 4,096), division 1.9 to 4.5 µs (14 down
  to 2.1), square root 4.5 to 14.1 µs (26 down to 7.5) and fused multiply-add 2.3 to 5.0 µs (23 down to 2.8).
  FloatLib's time is nearly flat in the width and MPFR's grows, so the ratio falls with the width.

These are times of the loop: the lookup of the operands, the fingerprint of the result and the checksum are in every
implementation's time, and at 8 ns the host FPU's row is mostly that.

## What was run

| | |
| --- | --- |
| FloatLib | `main` at [`fc2fad1f`](https://github.com/lean-dojo/FloatLib/commit/fc2fad1f9c7de970f0a10a44c58cdf2909a867b4), Lean 4.36.0, Mathlib `fc4f59e5` |
| MPFR | 4.1.0 (patch level 13, the system's library), with precision, exponent range, nearest-even and gradual underflow matched to the format |
| Berkeley SoftFloat | 3e at `a0c6494cdc11865811dec815d5c0049fba9d82a8` (<https://github.com/ucb-bar/berkeley-softfloat-3>), built by its own makefile |
| Host FPU | C `float` and `double` |
| C side | each behind FloatLib's C adapter, compiled by `cc (GCC) 11.5.0 20240719 (Red Hat 11.5.0-5)` with `-O3 -march=native -std=c11 -Wall -Wextra -Werror` |
| Date and machine | 2026-11-20, 2026-11-20T20:10:04Z to 2026-11-20T20:43:18Z, one pinned core of an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs |

**Method.** One run of FloatLib's own comparison harness, `benchmarks/scripts/format-comparison.sh`, with
9 trials.

- *The loop.* Each result selects, through a fingerprint of its sign, exponent and significand, one of 16 prepared
  operand sets for the next operation, so an operation cannot start before the previous one has finished. 256
  warm-up operations, then one timed interval of about 200 ms, whose iteration count a calibration pass chose.
- *The same work everywhere.* Every row also runs an untimed 256-step prefix whose checksum and operand trace must
  be the same in every implementation. They are, in all 144 cells of the run and all 9 trials
  ([`results/compiled-checks.txt`](results/compiled-checks.txt)). The run also timed FloatLib's fixed-word kernels
  called by name at 32 and 64 bits, which is why it has 144 cells. The files here keep the four series of the
  figures.
- *Order.* The harness shuffles the rows once and starts each trial one ninth further into that order, so no
  implementation is always first or last.
- *One core.* Every measured process was pinned to one CPU, and the harness ran with a core-scheduling cookie of
  its own, so nothing else ran on the second hardware thread of that core during a measurement.
- *The public operations.* The harness times FloatLib's kernels by name at 32 and 64 bits and has no series for the
  operations a user calls. [`scripts/make_public.py`](scripts/make_public.py) writes a copy of FloatLib's benchmark
  file whose rows are the public operations on `Binary32` and `Binary64` with the default planning policy, and
  [`scripts/make_harness.py`](scripts/make_harness.py) adds those rows to a copy of the harness. Lean checks by
  `rfl` that `x + y = ExecFloat.add x y`, and the same for `-`, `*` and `/`, on both types
  ([`scripts/PublicIsOperator.lean`](scripts/PublicIsOperator.lean)). From 128 bits on the harness's own FloatLib
  series is the public operation.

**How busy the machine was.** It was 67 percent idle in the median
sample (33 to 82), with a
one-minute load average from 24 to 103 on 96 threads. The
pinned thread spent 96 percent of the time in the benchmark. A repeat of
the 32- and 64-bit cells 46 minutes later (5 trials, same core) gave FloatLib's public operations within -0.2 to
+1.2 percent of the main run.

**What is not here.** This is the harness's development mode with series and widths selected, not its publication
mode, which needs its whole matrix. Nothing below 32 bits, no posits, no decimal formats and no directed rounding:
the six operations of the binary formats from 32 to 4,096 bits, to nearest even.

**From Lean:** FloatLib's times (`IO.monoNanosNow` around the loop, in the benchmark executable). **From C programs
in the same run:** MPFR's, SoftFloat's and the host FPU's (`clock_gettime(CLOCK_MONOTONIC)`). **From scripts:** the
iteration counts (the harness's calibration), the medians, the spread and the ratios. Nothing is timed in Python.

## Files

- [`results/compiled.csv`](results/compiled.csv) (one row per cell), [`results/compiled-trials.csv`](results/compiled-trials.csv)
  (one row per trial), [`results/compiled-ratios.csv`](results/compiled-ratios.csv),
  [`results/compiled-run.csv`](results/compiled-run.csv) (the facts of the run).
- [`figures/`](figures/CAPTIONS.md): both figures as PNG and SVG, with captions and notes.
- [`scripts/`](scripts/run_comparison.py): `make_public.py`, `build_public.sh`, `PublicIsOperator.lean`,
  `make_harness.py`, `run_comparison.py`, `summarize_comparison.py`, `collect.py`, `plot.py`.

## Run it again

The harness lives in FloatLib's repository, so this needs a FloatLib checkout at `fc2fad1f` with its benchmark
workspace built (`FLOATLIB_DIR`), MPFR's headers and a C compiler.

```bash
export FLOATLIB_DIR=/path/to/FloatLib            # at fc2fad1f
bash compiled-vs-mpfr-softfloat-fpu/scripts/build_public.sh
python3 compiled-vs-mpfr-softfloat-fpu/scripts/run_comparison.py 9 compiled-vs-mpfr-softfloat-fpu/work/run
python3 compiled-vs-mpfr-softfloat-fpu/scripts/summarize_comparison.py \
    compiled-vs-mpfr-softfloat-fpu/work/run compiled-vs-mpfr-softfloat-fpu/work/summary
python3 compiled-vs-mpfr-softfloat-fpu/scripts/collect.py \
    compiled-vs-mpfr-softfloat-fpu/work/run compiled-vs-mpfr-softfloat-fpu/work/summary
python3 compiled-vs-mpfr-softfloat-fpu/scripts/plot.py
```
