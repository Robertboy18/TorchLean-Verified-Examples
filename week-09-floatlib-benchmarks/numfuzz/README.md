# The NumFuzz benchmarks with `roundoff`

[NumFuzz](https://github.com/ak-2485/NumFuzz) (Kellison and Hsu, "Numerical Fuzz: A Type System for Rounding Error
Analysis", PLDI 2024, [arXiv:2405.04612](https://arxiv.org/abs/2405.04612)) is a type system that infers a bound on
the relative rounding error of a program. Its paper compares it with FPTaylor and Gappa on a set of benchmarks. Here
FloatLib's `roundoff` tactic runs on the same programs. The difference in kind: NumFuzz, FPTaylor and Gappa print a
bound, and `roundoff` produces a Lean theorem that the kernel checks.

No competitor was run here. NumFuzz, FPTaylor and Gappa appear with their published per-program numbers only, so
the times of the tools are not measured on one machine.

## Result

- **Translated.** The artifact holds 110 NumFuzz files (`.fz`). 80 of them are programs that round something, and
  all 80 were translated to Lean. The other 30 are 23 files that are not programs (library folds and maps, broken
  files) and 7 programs that round nothing.
- **Proved, in the model the three tools analyze** (real inputs, each operation rounded toward +∞, no overflow):
  `roundoff mixed theorem` proves the forward theorem of 75 of the 80 programs, both for all positive inputs and on
  the box: 71 at Lean's default limits and 4 more with the limits lifted. For 73 of the 75 Lean also accepts a
  relative error bound as a number on the box [0.1, 1000], the interval the paper gave FPTaylor and Gappa.
- **The 17 small benchmarks** (the paper's Table 3): all 17 proved. FloatLib's bound equals NumFuzz's in 13 and is
  smaller in 4. **Gappa's bound is smaller than FloatLib's in 7** and equal in 10 (FloatLib is up to 2.0 times
  Gappa's). FPTaylor has a bound for 16: FloatLib's is smaller in 13, equal in 1 and larger in 2.
- **The 9 large benchmarks** (the paper's Table 4): every one has a FloatLib bound that Lean checked. Horner50,
  Horner75, Horner100 and the serial sum of 1024 values equal NumFuzz's bound and the textbook bound to the three
  digits the paper prints. The four matrix products have the textbook bound `n u`, 0.50 to 0.57 times NumFuzz's
  `(2n - 1) u`, and poly50 has 2.2205e-14 (`100 u`) where NumFuzz has 2.94e-13.
- **The 4 conditional benchmarks** (the paper's Table 5): all 4 proved, with NumFuzz's bounds.
- **Long sums, Horner forms and dot products are proved once for every length.** Written as loops over IEEE 754
  binary64 operations rounded toward +∞, `roundoff` proves the forward, closed and backward theorems of the serial
  sum, of Horner's rule with fused multiply-adds and of the dot product in 0.9 to 1.8 s each, for a symbolic
  length. Lean checks by `rfl` that the loop at the size of each of 21 benchmark programs is that program.
- **Time.** NumFuzz is much faster: the paper reports 0.002 to 0.008 s for NumFuzz, 0.034 to 2.2 s for Gappa and
  3.2 to 109 s for FPTaylor, on another machine. For the 17 small benchmarks Lean elaborates the forward theorem
  in 0.39 to 3.1 s and the forward theorem with the relative bound in 1.7 to 68 s (median 2.8 s).

![Two panels sharing 17 rows, one per program, each row holding up to four markers one above the other. In the left panel the FloatLib circles lie between 2 and 20 units; they sit on the NumFuzz squares in 13 rows and left of them in 4, and the Gappa diamonds sit on them in 10 rows and left of them in 7. The FPTaylor triangles lie right of the circles in 13 rows, on them in 1 and left of them in 2; in 7 of those rows they are open triangles at the right edge, and 1 more row has a cross there. In the right panel the NumFuzz squares stand between 0.002 and 0.008 seconds, the Gappa diamonds between 0.034 and 2.2, the FPTaylor triangles between 3.2 and 109 and the FloatLib circles between 1.7 and 68.](figures/01-bounds-and-time.svg)

**Figure 1.** Relative error bounds and analysis times of the 17 small benchmarks of the NumFuzz paper (its Table 3): FloatLib's `roundoff` against the published numbers of NumFuzz, Gappa and FPTaylor. Panel (a) gives each tool's bound on the relative error in units of 2^-52, the unit roundoff of binary64 rounded toward plus infinity, and panel (b) the time of each analysis in seconds; both axes are logarithmic, and left is tighter in (a) and faster in (b). FloatLib's bound is the one Lean accepted on the box [0.1, 1000] that the paper gave Gappa and FPTaylor, and its time is Lean's elaboration of the forward theorem and of that bound, measured on 2026-11-21 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs) with Lean 4.36.0 on FloatLib 4.1.0. The other tools' numbers are the paper's, measured on a MacBook with a 1.4 GHz processor.

**Table 1. The 17 small benchmarks (the paper's Table 3).** "yes" and "no": Lean proved the case or not. "f c b":
`roundoff mixed theorem` (forward), `roundoff closed theorem`, `roundoff (backward := true) theorem`. Times are
Lean's elaboration time in seconds inside one process, and each process adds 3 to 5 s to start and import FloatLib
and Mathlib. The published columns are the paper's.

| program | ops | FloatLib, box | NumFuzz | FPTaylor | Gappa | FloatLib / NumFuzz | FloatLib / best other | forward (s) | forward and relative (s) | NumFuzz (s) | FPTaylor (s) | Gappa (s) | IEEE route: up f c b, nearest f c b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hypot | 4 | 4.4409e-16 | 5.55e-16 | 5.17e-16 | 4.46e-16 | 0.80 | 1.00 | 0.752 | 2.726 | 0.002 | 3.55 | 0.069 | no no no yes yes yes |
| x_by_xy | 3 | 4.4409e-16 | 4.44e-16 | fail | 2.22e-16 | 1.00 | 2.00 | 0.872 | 2.559 | 0.002 | - | 0.034 | no no no yes yes no |
| one_by_sqrtxx | 3 | 5.5512e-16 | 5.55e-16 | 5.09e-13 | 3.33e-16 | 1.00 | 1.67 | 0.556 | 2.128 | 0.002 | 3.34 | 0.047 | no no no yes yes no |
| sqrt_add | 5 | 7.7716e-16 | 9.99e-16 | 6.66e-16 | 5.54e-16 | 0.78 | 1.40 | 0.882 | 3.153 | 0.003 | 3.28 | 0.092 | no no no yes yes no |
| test02_sum8 | 8 | 1.5544e-15 | 1.55e-15 | 9.32e-14 | 1.55e-15 | 1.00 | 1.00 | 0.577 | 3.968 | 0.002 | 14.61 | 0.244 | yes yes yes yes yes yes |
| nonlin1 | 2 | 4.4409e-16 | 4.44e-16 | 4.49e-16 | 2.22e-16 | 1.00 | 2.00 | 1.245 | 1.733 | 0.003 | 3.24 | 0.040 | no no no yes yes no |
| test05_nonlin1 | 2 | 4.4409e-16 | 4.44e-16 | 4.46e-16 | 2.22e-16 | 1.00 | 2.00 | 1.055 | 2.339 | 0.008 | 3.27 | 0.042 | no no no yes yes no |
| verhulst | 4 | 8.8818e-16 | 8.88e-16 | 7.38e-16 | 4.44e-16 | 1.00 | 2.00 | 0.708 | 2.756 | 0.002 | 3.25 | 0.069 | no no no yes yes no |
| predatorPrey | 7 | 1.5544e-15 | 1.55e-15 | 4.21e-11 | 8.88e-16 | 1.00 | 1.75 | 0.652 | 2.964 | 0.002 | 3.28 | 0.114 | no no no yes yes no |
| test06_sums4_sum1 | 4 | 6.6614e-16 | 6.66e-16 | 6.71e-16 | 6.66e-16 | 1.00 | 1.00 | 0.477 | 2.05 | 0.003 | 3.84 | 0.069 | yes yes yes yes yes yes |
| test06_sums4_sum2 | 4 | 4.4409e-16 | 6.66e-16 | 1.78e-14 | 4.44e-16 | 0.67 | 1.00 | 0.392 | 1.907 | 0.002 | 11.02 | 0.055 | yes yes yes yes yes yes |
| i4 | 4 | 4.4409e-16 | 4.44e-16 | 4.50e-16 | 4.44e-16 | 1.00 | 1.00 | 0.702 | 3.357 | 0.002 | 3.30 | 0.055 | no no no yes yes yes |
| Horner2 | 4 | 4.4409e-16 | 4.44e-16 | 6.49e-11 | 4.44e-16 | 1.00 | 1.00 | 0.446 | 2.729 | 0.002 | 11.72 | 0.052 | yes yes yes yes yes yes |
| Horner2_with_er | 4 | 1.1103e-15 | 1.55e-15 | 1.61e-10 | 1.11e-15 | 0.72 | 1.00 | 0.502 | 3.161 | 0.002 | 19.56 | 0.119 | - - - - - - |
| Horner5 | 10 | 1.1103e-15 | 1.11e-15 | 1.62e-01 | 1.11e-15 | 1.00 | 1.00 | 1.017 | 5.531 | 0.003 | 22.08 | 0.209 | yes yes yes yes yes yes |
| Horner10 | 20 | 2.2205e-15 | 2.22e-15 | 1.14e+13 | 2.22e-15 | 1.00 | 1.00 | 1.042 | 10.652 | 0.003 | 40.68 | 0.650 | yes yes yes yes yes yes |
| Horner20 | 40 | 4.4409e-15 | 4.44e-15 | 2.53e+43 | 4.44e-15 | 1.00 | 1.00 | 3.089 | 67.945 | 0.003 | 109.42 | 2.246 | yes yes yes yes yes yes |

In the 4 programs where FloatLib is below NumFuzz (`hypot`, `sqrt_add`, `test06_sums4_sum2`, `Horner2_with_er`), a
sum of two inexact terms keeps the larger of their relative errors where NumFuzz adds them. In the 7 programs where
Gappa is below FloatLib (`x_by_xy`, `one_by_sqrtxx`, `sqrt_add`, `nonlin1`, `test05_nonlin1`, `verhulst`,
`predatorPrey`), a quotient or a square root has an operand that was rounded. A rounding toward +∞ never lowers a
value, so the error of a rounded divisor and the error of the rounding of the quotient have opposite signs, and
Gappa follows the direction of each rounding. FloatLib's rule for a directed mode bounds each rounding by `u` on
both sides, so Gappa's bound is 1.4 to 2.0 times smaller there. FPTaylor is below FloatLib in `sqrt_add` and
`verhulst`. NumFuzz's bound holds for all positive inputs and assumes that nothing underflows or overflows.
FloatLib's holds on the box and accounts for underflow.

**Table 2. The 9 large benchmarks (the paper's Table 4).** `*`: with Lean's limits lifted
(`maxHeartbeats 0`, `maxRecDepth 20000`), tried only after a case stopped at a resource limit. "FloatLib, box" is
the relative bound from the straight-line theorem on rounded reals, and "FloatLib, loop form" the weight of the
loop theorem at the program's size.

| program | ops | straight line: reals, positive | reals, box | IEEE up f c b | IEEE nearest f c b | FloatLib, box | FloatLib, loop form | NumFuzz | literature | loop = program | straight line (s) | NumFuzz (s) | Satire (s) |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Horner50 | 100 | yes | yes | yes* no no | yes* no no | - | 1.1103e-14 | 1.11e-14 | 1.11e-14 | proved | 7.393 | 9e-03 | 5 |
| mat_mul_4 | 112 | yes | yes | yes yes yes | yes yes yes | 8.8818e-16 | 8.8818e-16 | 1.55e-15 | 8.88e-16 | proved | 0.685 | 3e-03 | - |
| Horner75 | 150 | yes* | yes* | no no no | no no no | - | 1.6654e-14 | 1.66e-14 | 1.66e-14 | proved | 12.625 | 2e-02 | - |
| Horner100 | 200 | no | no | no* no* no* | no no no | - | 2.2205e-14 | 2.22e-14 | 2.22e-14 | proved | 5.37 | 4e-02 | - |
| serial_sum | 1023 | no* | no* | no* no* no* | no* no* no* | - | 2.2716e-13 | 2.27e-13 | 2.27e-13 | proved | 25.094 | 5 | 5407 |
| poly50 | 1325 | yes* | yes* | no* no no | no* no* no* | 2.2205e-14 | - | 2.94e-13 | - | - | 34.91 | 2.12 | 3 |
| mat_mul_16 | 7936 | yes | yes | yes yes* yes* | yes yes yes | 3.5528e-15 | 3.5528e-15 | 6.88e-15 | 3.55e-15 | proved | 1.811 | 4e-02 | - |
| mat_mul_64 | 520192 | yes* | yes* | no no no | yes* yes* yes* | 1.4211e-14 | 1.4211e-14 | 2.82e-14 | 1.42e-14 | proved | 12.167 | 10 | 65 |
| mat_mul_128 | 4177920 | no* | no* | no* no no | no* no no | - | 2.8422e-14 | 5.66e-14 | 2.84e-14 | proved | 8.209 | 1080 | 763 |

**Table 3. The 4 conditional benchmarks (the paper's Table 5).**

| program | FloatLib, box [0.1, 1000] | FloatLib, box [1e-9, 1e-6] | NumFuzz | FloatLib / NumFuzz | reals, positive | reals, box | forward (s) | forward and relative (s) | second box: forward and relative (s) | NumFuzz (s) | IEEE route: up f c b, nearest f c b |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| pythagorean_sum | 8.8818e-16 | 8.8818e-16 | 8.88e-16 | 1.00 | yes | yes | 0.616 | 7.343 | 9.395 | 0.002 | no no no yes yes no |
| hammarling_distance | 1.1103e-15 | 1.1103e-15 | 1.11e-15 | 1.00 | yes | yes | 0.86 | 4.643 | 5.392 | 0.002 | no no no yes yes no |
| squareRoot3 | 4.4409e-16 | 4.4409e-16 | 4.44e-16 | 1.00 | yes | yes | 0.863 | 3.348 | 2.233 | 0.002 | no no no yes yes no |
| squareRoot3Invalid | 4.4409e-16 | 4.4409e-16 | 4.44e-16 | 1.00 | yes | yes | 0.743 | 2.306 | 2.141 | 0.002 | no no no yes yes no |

![Three panels with logarithmic axes and one blue line each, rising one for one with the size: from 2 to 100 fused multiply-adds, from 4 to 128 pairs and from 7 to 1,023 additions. In the first and third panels the circles on the line sit inside the NumFuzz squares at every size that has one, inside Gappa diamonds at the small sizes and under textbook plus signs at the large ones; open triangles at the top edge of the first panel mark FPTaylor's bounds, and one FPTaylor triangle stands far above the line at the small sum. In the middle panel the plus signs sit on the circles and the NumFuzz squares lie about a factor two above the line.](figures/02-sums-horner-dot.svg)

**Figure 2.** Relative error bounds of the three families of long programs in the NumFuzz benchmarks, against their size: (a) Horner's rule with one fused multiply-add per coefficient, (b) the dot product behind each entry of a matrix product, (c) the serial sum. The line is FloatLib's theorem for every size, proved once by `roundoff` on the program written as a loop, and its circles are the sizes of the benchmarks; the other marks are the published bounds of NumFuzz, Gappa and FPTaylor and the textbook bounds that the paper quotes. Bounds are in units of 2^-52, both axes are logarithmic, and lower is tighter. FloatLib's bounds hold for nonnegative binary64 inputs when no operation overflows, with the underflow of the products and fused operations kept as a separate absolute term; they come from Lean 4.36.0 on FloatLib 4.1.0, run on 2026-11-21 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs).

**Table 4. Loop forms at the sizes of the benchmarks.**

| program | loop | weight | FloatLib (Lean) | NumFuzz | FPTaylor | Gappa | literature | loop = program (`rfl`) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| test02_sum8 | sum | (1+u)^7 - 1 | 1.5544e-15 | 1.55e-15 | 9.32e-14 | 1.55e-15 | - | proved |
| large_benchmarks/serial_sum | sum | (1+u)^1023 - 1 | 2.2716e-13 | 2.27e-13 | - | - | 2.27e-13 | proved |
| Horner2 | horner | (1+u)^2 - 1 | 4.4409e-16 | 4.44e-16 | 6.49e-11 | 4.44e-16 | - | proved |
| Horner5 | horner | (1+u)^5 - 1 | 1.1103e-15 | 1.11e-15 | 1.62e-01 | 1.11e-15 | - | proved |
| Horner10 | horner | (1+u)^10 - 1 | 2.2205e-15 | 2.22e-15 | 1.14e+13 | 2.22e-15 | - | proved |
| Horner20 | horner | (1+u)^20 - 1 | 4.4409e-15 | 4.44e-15 | 2.53e+43 | 4.44e-15 | - | proved |
| large_benchmarks/Horner10 | horner | (1+u)^10 - 1 | 2.2205e-15 | - | - | - | - | proved |
| large_benchmarks/Horner20 | horner | (1+u)^20 - 1 | 4.4409e-15 | - | - | - | - | proved |
| large_benchmarks/Horner50 | horner | (1+u)^50 - 1 | 1.1103e-14 | 1.11e-14 | - | - | 1.11e-14 | proved |
| large_benchmarks/Horner75 | horner | (1+u)^75 - 1 | 1.6654e-14 | 1.66e-14 | - | - | 1.66e-14 | proved |
| large_benchmarks/Horner100 | horner | (1+u)^100 - 1 | 2.2205e-14 | 2.22e-14 | - | - | 2.22e-14 | proved |
| large_benchmarks/dotprods/dotprod4 | dot | (1+u)^4 - 1 | 8.8818e-16 | - | - | - | - | proved |
| large_benchmarks/dotprods/dotprod8 | dot | (1+u)^8 - 1 | 1.7764e-15 | - | - | - | - | proved |
| large_benchmarks/dotprods/dotprod16 | dot | (1+u)^16 - 1 | 3.5528e-15 | - | - | - | - | proved |
| large_benchmarks/dotprods/dotprod64 | dot | (1+u)^64 - 1 | 1.4211e-14 | - | - | - | - | proved |
| large_benchmarks/dotprods/dotprod128 | dot | (1+u)^128 - 1 | 2.8422e-14 | - | - | - | - | proved |
| large_benchmarks/mat_mul_4 | dot | (1+u)^4 - 1 | 8.8818e-16 | 1.55e-15 | - | - | 8.88e-16 | proved |
| large_benchmarks/mat_mul_8 | dot | (1+u)^8 - 1 | 1.7764e-15 | - | - | - | - | proved |
| large_benchmarks/mat_mul_16 | dot | (1+u)^16 - 1 | 3.5528e-15 | 6.88e-15 | - | - | 3.55e-15 | proved |
| large_benchmarks/mat_mul_64 | dot | (1+u)^64 - 1 | 1.4211e-14 | 2.82e-14 | - | - | 1.42e-14 | proved |
| large_benchmarks/mat_mul_128 | dot | (1+u)^128 - 1 | 2.8422e-14 | 5.66e-14 | - | - | 2.84e-14 | proved |

The closed theorem of the sum reads

    |toReal (nfSum x0 xs) - (x0 + Σ xs)| ≤ |x0| ((1 + u)^n - 1) + Σ_k |xs[k]| ((1 + u)^(n - k) - 1),   n = xs.length,

under the condition that no step overflows, and the backward theorem gives perturbations
`|δ_k| ≤ (1 + u)^(n - k) - 1` of the inputs for which the computed sum is the exact sum.

## What failed and why

| program | operations | what stops |
| --- | ---: | --- |
| `large_benchmarks/Horner100` | 200 | a fixed traversal limit of `roundoff` (256 nested operations; the program has about 300) |
| `large_benchmarks/serial_sum` | 1023 | at default limits Lean's heartbeat limit after 13 s; with the limits lifted the traversal limit after 25 s |
| `large_benchmarks/dotprods/dotprod128`, `large_benchmarks/mat_mul_128` | 255 per output | refused after 6 to 12 s, with and without the limits lifted: the term is too large |
| `large_benchmarks/reduction` | 1023 | a pairwise sum of 1024 values: Lean's heartbeat limit after 18 s, and no result in 300 s with the limits lifted |
| `large_benchmarks/Horner50`, `Horner75` (relative number only) | 100, 150 | the closure of the bound to a number, not `roundoff`: 300 s on every rung |

So as straight-line programs, Horner100, the serial sum of 1024 values, the pairwise sum of 1024 values and the two
programs with 128-pair dot products are not proved. The loop forms cover all of them but `reduction`, the one long
sum of the artifact without a FloatLib bound.

Two more limits, on the route through IEEE operations on stored values (FloatLib's FPBench pipeline) instead of
rounded reals: in the rounding mode `toPositive` it proves 35 of 65 programs, because division, square root and
literals have no rule in a directed mode there, and in the default mode 58 of 65.

## How a bound becomes a number

`roundoff mixed theorem` states `|computed - exact| ≤ R |exact| + A`, where `R` and `A` are expressions in the
inputs: `R` composes one unit roundoff per rounding and `A` is the allowance for underflow. NumFuzz, FPTaylor and
Gappa print one number. To compare, a second theorem states `|computed - exact| ≤ B * |exact|` for all inputs of
the box, with a rational `B`, proved from the generated theorem. `B` is proposed by a script and never trusted:
Lean accepts the theorem or refuses it. The first proposal is the worst case of the textbook propagation over all
positive inputs. A program that Lean refuses is tried on up to 8 coarser rungs, and the reported bound is the
lowest rung accepted. No number below the first proposal was tried, so where the box would give less than the worst
case over all positive inputs, the reported bound is not the smallest that the theorem implies.

## What was run

| | |
| --- | --- |
| FloatLib | 4.1.0 ([release](https://github.com/lean-dojo/FloatLib/releases/tag/v4.1.0)), Lean 4.36.0, Mathlib v4.36.0 |
| NumFuzz artifact | <https://github.com/ak-2485/NumFuzz>, `main` at `80c54f35348c60b8a1e95f1881adda8d2e9c1dfb` (2024-12-05), GPL-3.0. Read as data, not vendored here |
| Published numbers | the paper's Tables 3, 4 and 5 (arXiv:2405.04612v2), transcribed in [`scripts/gen.py`](scripts/gen.py) and checked against the PDF text. The paper ran everything on "a MacBook with a 1.4 GHz processor and 8 GB of memory", with FPTaylor 0.9.4 and Gappa 1.4.2 |
| Date and machine | 2026-11-21, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs. Between 28 and 94 percent of the 96 threads were busy (median 66) |

**Method.** [`scripts/nfz.py`](scripts/nfz.py) reads a `.fz` file with its includes and evaluates it symbolically
to an expression in which every `rnd` is a rounding node. Its count of operations agrees with the paper's "Ops"
column on all 9 large benchmarks and on 11 of the 17 small ones (on the other 6 the paper counts one more than the
program has operations). From that expression, two models: rounded reals, the model of NumFuzz, FPTaylor and Gappa
(`Model.roundAt FloatFormat.binary64 t .up`), and IEEE operations on stored values through FloatLib's FPBench
pipeline. Every Lean case is its own Lean process with a wall-clock limit of 300 s. A case counts as proved only
when Lean elaborates it without error and the theorem depends on no axiom other than `propext`, `Classical.choice`
and `Quot.sound`. Over all 1,198 cases the axioms found are exactly these three. The runner is FloatLib's own
`benchmarks/suites/Proofs.lean`, and [`scripts/ProofsNF.lean`](scripts/ProofsNF.lean) is the same runner with this
suite's lemmas added, for the relative bounds and the loop corollaries.

**From Lean:** whether a case is proved, its statement and axioms, FloatLib's bounds (each is the statement of a
theorem), the `rfl` links, and FloatLib's times. **From scripts:** the reading of the `.fz` files, the Lean text of
each program, the proposal of every numeric bound, the tables and figures. **From the paper:** every number of
NumFuzz, FPTaylor, Gappa and Satire and the textbook bounds. Nothing was tuned to one program.

## Files

- [`results/programs.csv`](results/programs.csv): one row per `.fz` file with every route's outcome, time and
  bound. [`results/cases.csv`](results/cases.csv): one row per Lean case (1,198), with Lean's message.
  [`results/summary.json`](results/summary.json): the counts.
- [`figures/`](figures/CAPTIONS.md): both figures as PNG and SVG, with captions and notes.
- [`scripts/`](scripts/gen.py): `fetch.sh`, the readers `nfz.py` and `fpt.py`, `emit.py`, `gen.py` (inventory, plan
  and the published tables), `lifted.py`, `closure.py`, `loops.py`, `branches.py`, `direct.py` (plans), `prove.py`,
  `passes.sh` and `final.sh` (run a plan), `ProofsNF.lean`, `collect.py`, `plot.py`.

## Run it again

```bash
lake build                                                  # in the week's directory: FloatLib 4.1.0
bash numfuzz/scripts/fetch.sh                               # the NumFuzz artifact, pinned
F=$PWD/.lake/packages/floatlib; N=numfuzz/run; P=numfuzz/scripts
python3 -I $P/gen.py $N/dl/NumFuzz/examples $F $N/work
python3 -I $P/lifted.py $N/work && python3 -I $P/loops.py $N/work
WORK=$N/work lake env bash $P/final.sh $N/work/run 6 300    # about an hour of Lean
python3 -I $P/collect.py $N/work numfuzz/results && python3 $P/plot.py
```
