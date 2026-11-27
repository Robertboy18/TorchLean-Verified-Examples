# Compiled code against the host FPU, SoftFloat and MPFR: figure captions

Written by `scripts/plot.py`; do not edit by hand. Every figure is `NAME.png` (9 inches wide at 200 dpi) and `NAME.svg`. An image holds only the plot, its axes with units, a legend with short series names and panel letters; the caption says what the figure shows, what was measured, in which unit, on which scales, where and when. The notes say how, which numbers come from Lean and which from scripts, and give the numbers behind the marks.

## Figure 1: `01-public-binary32-64`

![Two panels, binary32 above binary64, each with six groups of four markers, FloatLib, MPFR, SoftFloat and the host FPU from left to right, on a logarithmic time axis: in every group the host FPU is lowest, between 8 and 13 nanoseconds, SoftFloat lies between 18 and 33, MPFR between 32 and 59 and FloatLib between 24 and 118.](01-public-binary32-64.svg "Time of FloatLib's public binary32 and binary64 operations in compiled code, next to the host processor's own arithmetic, Berkeley SoftFloat 3e and MPFR 4.1.0. A marker is the median of 9 trials of the time per operation, in nanoseconds, of a loop in which each result selects the next operands, and its whisker runs from the fastest to the slowest trial; the operations are addition, subtraction, multiplication, division, square root and fused multiply-add, rounded to nearest even. The time axis is logarithmic, and lower is faster. Measured on FloatLib on 2026-11-20 in one run on one pinned core of an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs) with Lean 4.36.0.")

**Figure 1.** Time of FloatLib's public binary32 and binary64 operations in compiled code, next to the host processor's own arithmetic, Berkeley SoftFloat 3e and MPFR 4.1.0. A marker is the median of 9 trials of the time per operation, in nanoseconds, of a loop in which each result selects the next operands, and its whisker runs from the fastest to the slowest trial; the operations are addition, subtraction, multiplication, division, square root and fused multiply-add, rounded to nearest even. The time axis is logarithmic, and lower is faster. Measured on FloatLib on 2026-11-20 in one run on one pinned core of an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs) with Lean 4.36.0.

Alt text: Two panels, binary32 above binary64, each with six groups of four markers, FloatLib, MPFR, SoftFloat and the host FPU from left to right, on a logarithmic time axis: in every group the host FPU is lowest, between 8 and 13 nanoseconds, SoftFloat lies between 18 and 33, MPFR between 32 and 59 and FloatLib between 24 and 118.

Notes: FloatLib's series is the public operation as a user writes it (`+`, `-`, `*`, `/`, `sqrt`, `fma` on `Binary32` and `Binary64`, default planning policy), proved and run in software; the host FPU is the unproved reference. Over the 12 cells FloatLib takes 0.70 to 2.29 times MPFR's time (median 1.00, faster than MPFR in 6 cells), 1.17 to 4.38 times SoftFloat's (median 1.61) and 2.8 to 13.7 times the host FPU's (median 4.6). The times include the loop: the lookup of the operands, the fingerprint of the result and the checksum. Median (fastest to slowest trial), nanoseconds:

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

From Lean: FloatLib's times (`IO.monoNanosNow` in the benchmark executable). From C programs in the same run: MPFR, SoftFloat and the host FPU (`clock_gettime`). From scripts: the medians and the drawing. Source: `results/compiled.csv`.

Files: `01-public-binary32-64.png`, `01-public-binary32-64.svg`.

## Figure 2: `02-floatlib-mpfr-widths`

![Six panels, one per operation, each with a FloatLib line and an MPFR line against format width from 32 to 4,096 bits on logarithmic axes: the two lines are close at 32 and 64 bits, the FloatLib line is above at 128 bits, steps up to about two microseconds or more at 256 bits and is nearly flat from there, while the MPFR line rises steadily, so that the gap narrows towards 4,096 bits in every panel, least for addition and subtraction.](02-floatlib-mpfr-widths.svg "Compiled time of FloatLib's binary operations against MPFR 4.1.0 at the same precision, exponent range and rounding, from 32 to 4,096 bits. A marker is the median of 9 trials of the time per operation, in nanoseconds, of a loop in which each result selects the next operands, with a whisker from the fastest to the slowest trial. Both axes are logarithmic, and lower is faster. Measured on FloatLib on 2026-11-20 in one run on one pinned core of an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs) with Lean 4.36.0.")

**Figure 2.** Compiled time of FloatLib's binary operations against MPFR 4.1.0 at the same precision, exponent range and rounding, from 32 to 4,096 bits. A marker is the median of 9 trials of the time per operation, in nanoseconds, of a loop in which each result selects the next operands, with a whisker from the fastest to the slowest trial. Both axes are logarithmic, and lower is faster. Measured on FloatLib on 2026-11-20 in one run on one pinned core of an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs) with Lean 4.36.0.

Alt text: Six panels, one per operation, each with a FloatLib line and an MPFR line against format width from 32 to 4,096 bits on logarithmic axes: the two lines are close at 32 and 64 bits, the FloatLib line is above at 128 bits, steps up to about two microseconds or more at 256 bits and is nearly flat from there, while the MPFR line rises steadily, so that the gap narrows towards 4,096 bits in every panel, least for addition and subtraction.

Notes: FloatLib's series is the public operation on the configured binary type (`ExecFloat.add`, ...): `Binary32` and `Binary64` at 32 and 64 bits, binary128 at 128, and layouts with 19 exponent bits from 256 bits on, where FloatLib has one kernel per operation, the exact baseline on Lean's arbitrary-precision naturals. FloatLib's median is below MPFR's in 6 of 48 cells: fma at 32, mul at 32, sqrt at 32, add at 64, mul at 64, sub at 64. FloatLib's time over MPFR's (medians):

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

From Lean: FloatLib's times. From a C program in the same run: MPFR's. From scripts: the medians, the ratios and the drawing. Source: `results/compiled.csv`.

Files: `02-floatlib-mpfr-widths.png`, `02-floatlib-mpfr-widths.svg`.
