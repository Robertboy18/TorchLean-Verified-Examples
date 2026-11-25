"""FPBench with `roundoff`: the counts of FloatLib's recorded suite run.

    python3 scripts/plot.py [RESULTS [FIGURES]]

Reads the files of `results/` and writes `figures/NN-NAME.png`, `.svg` and `figures/CAPTIONS.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.patches import Patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plots"))
from benchplot import MACHINE, Suite, fs, plt, stacked_rows  # noqa: E402

suite = Suite(__file__)
read, done = suite.read, suite.done


def fig_fpbench() -> None:
    data = read("counts.csv")
    facts = suite.facts()
    colours = [(fs.BLUE, None, fs.BLUE), (fs.LINE, None, fs.LINE), ("white", "////", fs.GREY_LIGHT)]
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.86 * len(data) + 1.9))
    stacked_rows(ax, [(row["label"], [int(row["proved"]), int(row["not_proved"]), int(row["not_translated"])])
                      for row in data], colours, "{done} of {total}")
    ax.set_xlabel("share of the suite (%)")
    handles = [Patch(facecolor=fs.BLUE, edgecolor=fs.BLUE, label="proved"),
               Patch(facecolor=fs.LINE, edgecolor=fs.LINE, label="not proved"),
               Patch(facecolor="white", edgecolor=fs.GREY_LIGHT, hatch="////", label="not translated")]
    fs.legend(fig, handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.45, 0.985), handlelength=1.6)
    fig.subplots_adjust(left=0.03, right=0.80, top=0.985, bottom=0.17)
    by = {row["key"]: row for row in data}
    done(fig, "01-fpbench-roundoff",
         "What FloatLib's roundoff tactic proves on FPBench. A bar is one kind of theorem over the whole suite, "
         "split into the cases proved, not proved and not translated to Lean, with the number proved at its "
         "end: 272 cases, the 136 programs of twelve FPBench suite files in binary32 and in binary64. Run on "
         f"{facts['date']} with Lean {facts['lean']} on {MACHINE}; these are the suite runs recorded in the "
         "repository of FloatLib 4.1.0.",
         "Five horizontal bars on an axis from 0 to 100 percent, three for conditional theorems and two for "
         "numerical bounds, with a blue part for the proved cases that is about two thirds of the bar for the "
         "forward and closed conditional theorems and small for the others. Each bar ends in its count.",
         "Each program is translated to a Lean program and given to `roundoff` once per kind of theorem. A "
         "conditional theorem keeps its premises; a numerical bound is a second Lean proof that discharges "
         "them on the program's input box and proves one proposed bound. Proved: "
         f"{by['fpbench-forward-conditional']['proved']} forward, {by['fpbench-closed-conditional']['proved']} "
         f"closed and {by['fpbench-backward-conditional']['proved']} backward conditional theorems, "
         f"{by['fpbench-forward-numerical']['proved']} forward and {by['fpbench-closed-numerical']['proved']} closed "
         "numerical bounds (forward, closed and backward are the three methods of `roundoff`); 60 seconds and "
         "4 GB per Lean process. Not proved means refused, over the limits or, for a numerical bound, not "
         "attempted; not translated means the translator refuses the case. A count says what the tactic did "
         "on this suite, not what it does on all programs. From Lean: every proof and its axioms. From "
         "scripts: the translation and the counts. The counts are `benchmarks/results/suites/counts.csv` of "
         f"FloatLib's repository ({facts['measured_utc']}); the library sources of that run differ from the "
         "4.1.0 tag by one added corollary and the version number, and no tactic differs. "
         "Source: `results/counts.csv`.")


if __name__ == "__main__":
    fig_fpbench()
    suite.write_captions("FPBench with roundoff")
