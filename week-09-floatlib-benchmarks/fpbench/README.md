# FPBench with `roundoff`

[FPBench](https://fpbench.org/) is the standard collection of floating-point programs for error analysis tools.
FloatLib's `roundoff` tactic takes a Lean program and proves a bound on its rounding error. Here every FPBench
program is translated to a Lean program and given to `roundoff`, once per kind of theorem.

## Result

![Five horizontal bars on an axis from 0 to 100 percent, three for conditional theorems and two for numerical bounds, with a blue part for the proved cases that is about two thirds of the bar for the forward and closed conditional theorems and small for the others. Each bar ends in its count.](figures/01-fpbench-roundoff.svg)

**Figure 1.** What FloatLib's roundoff tactic proves on FPBench. A bar is one kind of theorem over the whole suite, split into the cases proved, not proved and not translated to Lean, with the number proved at its end: 272 cases, the 136 programs of twelve FPBench suite files in binary32 and in binary64. Run on 2026-11-20 with Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs); these are the suite runs recorded in the repository of FloatLib 4.1.0.

| theorem | proved | not proved | not translated | cases |
| --- | ---: | ---: | ---: | ---: |
| forward, conditional theorem | 184 | 4 | 84 | 272 |
| closed, conditional theorem | 180 | 8 | 84 | 272 |
| backward, conditional theorem | 22 | 166 | 84 | 272 |
| forward, numerical bound | 33 | 75 | 164 | 272 |
| closed, numerical bound | 18 | 90 | 164 | 272 |

There are 272 cases: the 136 programs of the twelve FPBench suite files, in binary32 and in binary64.

- A **conditional theorem** keeps its premises (for example, that no operation overflows). `roundoff` proves 184
  forward, 180 closed-form and 22 backward ones. Forward, closed and backward are its three methods.
- A **numerical bound** is a second Lean proof that discharges those premises on the program's input box and
  proves one proposed number. That is the kind of result FPTaylor or Gappa print, and here it is proved for 33
  cases forward and 18 closed, 51 in all. This is the weak spot today: most cases that get a conditional
  theorem don't get their number yet.
- **Not translated** (84 cases) means that the translator refuses the program: loops, tensors and operators it
  has no translation for. It refuses instead of truncating a loop or changing a program.
- **Backward** analysis is proved for 22 cases only. `roundoff` gives a backward theorem when the program has a
  backward-error form on its inputs, and it refuses the rest with the reason.

A count says what the tactic did on this suite, not what it does on all programs.

## What was run

| | |
| --- | --- |
| FloatLib | 4.1.0 ([release](https://github.com/lean-dojo/FloatLib/releases/tag/v4.1.0)), Lean 4.36.0 |
| FPBench | <https://github.com/FPBench/FPBench> at `8ac4ff6cf0c3be4f9669a562ef2bbbbc3b15182d`, MIT: all 12 `benchmarks/*.fpcore` suite files |
| Limits | 60 seconds and 4 GB per Lean process, for a conditional theorem and for a numerical bound |
| Date and machine | 2026-11-20T11:24:20Z to 2026-11-20T11:32:47Z, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs |

**Method.** This is FloatLib's own suite run, the one recorded in its repository under
`benchmarks/results/suites/`. Its driver fetches FPBench at the pinned commit with checksums, translates each
FPCore program (`benchmarks/fpbench/translate.py`), and elaborates each proof in a fresh Lean environment that also
checks the transitive axioms: a case counts only with `propext`, `Classical.choice` and `Quot.sound`. The suite was
not run again for this week. The library sources of the recorded run differ from the 4.1.0 tag by one added
corollary and the version number, and no tactic differs.

**From Lean:** every proof and its axioms. **From scripts:** the translation and the counts.

## Files

- [`results/counts.csv`](results/counts.csv) and [`results/run.csv`](results/run.csv).
- [`figures/`](figures/CAPTIONS.md): the figure as PNG and SVG, with its caption and notes.
- [`scripts/`](scripts/collect.py): `collect.py` (reads the counts from a FloatLib checkout) and `plot.py`.

## Run it again

The harness is FloatLib's. In a FloatLib checkout at `v4.1.0`:

```bash
bash benchmarks/suites/run.sh selftest
FLOATLIB_SUITE_RESULTS=$PWD/suite-results bash benchmarks/suites/run.sh fpbench
python3 benchmarks/suites/summarize.py $PWD/suite-results
```

and then, here, `FLOATLIB_DIR=/path/to/FloatLib python3 fpbench/scripts/collect.py fpbench` and
`python3 fpbench/scripts/plot.py`.
