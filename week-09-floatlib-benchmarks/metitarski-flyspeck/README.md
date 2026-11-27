# MetiTarski and Flyspeck with `interval`

Two classic collections of real inequalities. [MetiTarski](https://www.cl.cam.ac.uk/~lp15/papers/Arith/)'s problem
set has inequalities over `exp`, `log`, `sin`, `cos`, `sqrt` and friends, and
[Flyspeck](https://github.com/flyspeck/flyspeck) has the nonlinear inequalities of the formal proof of the Kepler
conjecture. Each goal is translated to one Lean goal and given to FloatLib's `interval` tactic, one call each.

## Result

![Four horizontal bars on an axis from 0 to 100 percent, for MetiTarski and Flyspeck with interval alone and with the constraint engine. The blue part, the proved goals, is largest, about a third, for MetiTarski with the constraint engine; the Flyspeck bars are mostly hatched, not translated. Each bar ends in its count.](figures/01-metitarski-flyspeck-interval.svg)

**Figure 1.** What FloatLib's interval tactic proves on the MetiTarski problems and on Flyspeck's nonlinear inequalities. A bar is one route over a whole suite, split into the goals proved, not proved and not translated to Lean, with the number proved at its end: 948 MetiTarski goals and 164 Flyspeck records, one tactic call each. Run on 2026-11-20 with Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs); these are the suite runs recorded in the repository of FloatLib.

| suite and route | proved | not proved | not translated | goals |
| --- | ---: | ---: | ---: | ---: |
| MetiTarski, interval alone | 124 | 674 | 150 | 948 |
| MetiTarski, with the constraint engine | 340 | 458 | 150 | 948 |
| Flyspeck, interval alone | 3 | 17 | 144 | 164 |
| Flyspeck, with the constraint engine | 6 | 14 | 144 | 164 |

- **MetiTarski.** `interval` proves 124 of the 948 goals with `FloatLib.Tactic.Interval` alone and 340 with the
  constraint engine imported as well, which is `interval` as `import FloatLib` provides it. That is 36 percent of
  the suite and 43 percent of the 798 goals that translate.
- **Flyspeck.** 3 and 6 of 164 records (153 nonlinear inequalities and eleven examples). Only 20 records translate
  at all: the reader extracts the quoted inequalities and unfolds the definitions it supports, and it refuses the
  rest instead of guessing a definition.
- 20 seconds per goal is short. MetiTarski itself and the Flyspeck project's own tools prove far more of their own
  problems than this, with more time and with methods built for them. This suite measures one general tactic
  with one call and a small budget.

A count says what the tactic did on these suites, not what it does on all goals.

## What was run

| | |
| --- | --- |
| FloatLib | `main` at [`fc2fad1f`](https://github.com/lean-dojo/FloatLib/commit/fc2fad1f9c7de970f0a10a44c58cdf2909a867b4), Lean 4.36.0 |
| MetiTarski | <https://bitbucket.org/lcpaulson/metitarski-git/> at `5a63f3ecb8c8edf721f0051431db3e0f689d5c17`, MIT |
| Flyspeck | <https://github.com/flyspeck/flyspeck> at `32948714cdf06d700af2c993c9a92b7b551d7af5`, MIT |
| Limits | 20 seconds and 4 GB per goal, one goal per Lean process |
| Date and machine | 2026-11-20T11:44:50Z to 2026-11-20T12:03:31Z, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs |

**Method.** This is FloatLib's own suite run, the one recorded in its repository under
`benchmarks/results/suites/`. Its driver fetches both sources at the pinned commits with checksums. The TPTP
translator reads arithmetic and transcendental symbols over the reals and proves the conjecture directly. It does
not add the prover's analytic axiom files as Lean axioms. No sampled evaluation is counted as a proof: a goal counts
only when Lean accepts it with `propext`, `Classical.choice` and `Quot.sound`. Not proved means refused or over the
limits, and not translated means the translator refuses the goal. The suite was not run again for this week. The
library sources of the recorded run differ from the commit `fc2fad1f` by one added corollary and the version number, and no
tactic differs.

**From Lean:** every proof and its axioms. **From scripts:** the translation and the counts.

## Files

- [`results/counts.csv`](results/counts.csv) and [`results/run.csv`](results/run.csv).
- [`figures/`](figures/CAPTIONS.md): the figure as PNG and SVG, with its caption and notes.
- [`scripts/`](scripts/plot.py): `plot.py`. The collector is [`../fpbench/scripts/collect.py`](../fpbench/scripts/collect.py).

## Run it again

The harness is FloatLib's. In a FloatLib checkout at `fc2fad1f`:

```bash
FLOATLIB_SUITE_RESULTS=$PWD/suite-results bash benchmarks/suites/run.sh inequalities
python3 benchmarks/suites/summarize.py $PWD/suite-results
```

and then, here, `FLOATLIB_DIR=/path/to/FloatLib python3 fpbench/scripts/collect.py inequalities` and
`python3 metitarski-flyspeck/scripts/plot.py`.
