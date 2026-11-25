# One operation inside the proof checker: Lean's kernel with FloatLib, Coq with Flocq

`float_decide` proves a closed floating-point statement by running the arithmetic in Lean's kernel. So what does
one operation cost there? This suite times FloatLib's kernel path against
[Flocq](https://flocq.gitlabpages.inria.fr/) 4.2.2 in Coq 8.20.1, under `vm_compute` and under `lazy`, on the same
operands, from 32 to 4,096 bits.

## Result

![Five panels of lines, one per operation, against format width from 32 to 4,096 bits on logarithmic axes: the FloatLib line is nearly flat near the bottom of each panel, the vm_compute line starts just above it, climbs steeply and ends at 256 or 512 bits, the lazy line starts higher, climbs more slowly and reaches 4,096 bits, and crosses along the top edge mark the widths where Coq gave no result.](figures/01-kernel-coq-flocq.svg)

**Figure 1.** Time of one floating-point operation inside the proof checker: FloatLib in Lean's kernel against Flocq 4.2.2 in Coq 8.20.1 under vm_compute and under lazy, from 32 to 4,096 bits. A point is the time per operation, in milliseconds, in a loop of 100 operations (10 from 512 bits on) on the same operands; a cross at the top marks a width where Coq gave no result within 8 GB and 300 seconds. Both axes are logarithmic, and lower is faster. The Lean times are the median of 5 runs of FloatLib 4.1.0 on 2026-11-20 with Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs); the Coq times are the recorded run of 2026-11-15 on the same machine, in a container with 8 GB and two CPUs, and do not depend on FloatLib.

| operation | width (bits) | FloatLib in Lean's kernel (µs) | Coq + Flocq, `vm_compute` (ms) | Coq + Flocq, `lazy` (ms) |
| --- | ---: | ---: | ---: | ---: |
| add | 32 | 39.9 | 0.10 | 1.45 |
| mul | 32 | 45.5 | 0.11 | 0.43 |
| div | 32 | 57.5 | 0.19 | 1.28 |
| sqrt | 32 | 64.5 | 0.25 | 1.82 |
| fma | 32 | 87.2 | 0.12 | 0.93 |
| add | 64 | 39.2 | 0.28 | 1.58 |
| mul | 64 | 47.3 | 0.38 | 1.31 |
| div | 64 | 59.0 | 0.67 | 7.18 |
| sqrt | 64 | 68.5 | 1.46 | 2.73 |
| fma | 64 | 106.9 | 0.26 | 3.36 |
| add | 128 | 44.3 | 3.31 | 2.07 |
| mul | 128 | 50.9 | 3.35 | 6.58 |
| div | 128 | 62.7 | 3.63 | 23.20 |
| sqrt | 128 | 77.6 | 1.72 | 8.80 |
| fma | 128 | 153.5 | 3.95 | 9.46 |
| add | 256 | 46.1 | 45.38 | 2.69 |
| mul | 256 | 53.0 | 84.32 | 23.25 |
| div | 256 | 65.4 | 97.71 | 85.21 |
| sqrt | 256 | 82.4 | 6.60 | 34.22 |
| fma | 256 | 176.1 | 56.80 | 50.77 |
| add | 512 | 62.3 | 1,673.80 | 16.00 |
| mul | 512 | 70.7 | 2,213.40 | 122.70 |
| div | 512 | 83.2 | no result | 410.50 |
| sqrt | 512 | 106.3 | no result | 160.00 |
| fma | 512 | 197.6 | no result | 175.40 |
| add | 1,024 | 63.2 | no result | 21.60 |
| mul | 1,024 | 71.7 | no result | 531.00 |
| div | 1,024 | 84.2 | no result | 1,550.10 |
| sqrt | 1,024 | 130.4 | no result | 671.90 |
| fma | 1,024 | 209.2 | no result | 788.90 |
| add | 4,096 | 64.0 | no result | 72.60 |
| mul | 4,096 | 75.3 | no result | 10,535.20 |
| div | 4,096 | 87.7 | no result | 26,733.50 |
| sqrt | 4,096 | 276.0 | no result | 11,169.20 |
| fma | 4,096 | 237.3 | no result | 14,328.80 |

- Lean's kernel takes 39 to 276 µs per operation over all widths and operations, and the time barely moves with
  the width.
- Where Coq gave a result, `vm_compute` takes 1.4 to 31,329 times Lean's time (median 24, 22 cells) and `lazy` 9 to
  304,898 times (median 342, 35 cells). At binary32 the two checkers are close: 100 µs against 40 µs for an
  addition under `vm_compute`.
- Coq gave no result in 13 of the 70 cells within 8 GB and 300 seconds, all of them `vm_compute` from 512 bits on.
- In the recorded run Coq printed the value that Lean computed in every row.

Flocq is a general formalization that people use to *prove* things about floating point, and it was not written
to be run at 4,096 bits inside `vm_compute`. FloatLib's kernel layer was written for exactly that. So read this as
what each library costs when you use it for proof by computation, not as a ranking of the two.

## What was run

| | |
| --- | --- |
| FloatLib | 4.1.0 ([release](https://github.com/lean-dojo/FloatLib/releases/tag/v4.1.0)), Lean 4.36.0 |
| Coq and Flocq | Coq 8.20.1 with Flocq 4.2.2: `Bplus`, `Bmult`, `Bdiv`, `Bsqrt` and `Bfma` of `BinarySingleNaN` with `mode_NE` |
| Date and machine | Lean: 2026-11-20, 5 runs between 20:50Z and 21:18Z on an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs, 43 to 66 percent idle. Coq: the recorded run of 2026-11-15 on the same machine, in a container with 8 GB and two CPUs. The Coq times do not depend on FloatLib |

**Method.** Both sides run FloatLib's own matched loops (`benchmarks/suite/matched.py` in its repository): a loop of
100 operations (10 from 512 bits on) on the same operands, and the time per operation is the loop's time divided by
its length. On the Lean side the timed term is the one that `float_decide` checks (`#kernel_cost rewritten`,
measured with `Lean.Kernel.whnf`), run through FloatLib's timing mode with a core-scheduling cookie. A cell of that
runner is one sample, so I ran it five times and the table has the median per cell. Over the 35 Lean cells the five
runs differ by 5.8 percent of the median in the median cell and 13 percent at the 90th percentile. One cell is an
outlier: binary32 addition took 75.5 µs in the first run, while the load of the machine was rising, and 39.1 to
40.3 µs in the other four.

**What failed, and what was not run.** Coq was not run again for this week: every Coq cell gets 300 seconds, and
13 cells of the recorded run had no result within that and 8 GB. The Coq numbers are the recorded ones
(`benchmarks/results/figures/recorded/coq-flocq.csv` in FloatLib's repository).

**From Lean:** the Lean times. **From scripts:** the Coq times (printed by Coq in a container), the medians and
the ratios.

## Files

- [`results/kernel-coq.csv`](results/kernel-coq.csv): per operation and width, Lean's median with the fastest and
  the slowest run, and Coq's two times. [`results/run.csv`](results/run.csv): the facts of the five runs.
- [`figures/`](figures/CAPTIONS.md): the figure as PNG and SVG, with its caption and notes.
- [`scripts/`](scripts/run_kernel_sections.py): `run_kernel_sections.py`, `collect.py`, `plot.py`.

## Run it again

```bash
export FLOATLIB_DIR=/path/to/FloatLib            # at v4.1.0, built, with its benchmark workspace
for n in 1 2 3 4 5; do python3 kernel-vs-coq/scripts/run_kernel_sections.py kernel-vs-coq/work/timing$n; done
python3 kernel-vs-coq/scripts/collect.py kernel-vs-coq/work/timing{1,2,3,4,5}
python3 kernel-vs-coq/scripts/plot.py
```

The Coq side is `benchmarks/suite/run.sh matched` in FloatLib's repository, which needs a container with Coq
8.20.1 and Flocq 4.2.2.
