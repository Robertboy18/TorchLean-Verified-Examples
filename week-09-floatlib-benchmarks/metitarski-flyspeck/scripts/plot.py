"""MetiTarski and Flyspeck with `interval`: the counts of FloatLib's recorded suite run.

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


def fig_inequalities() -> None:
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
    fig.subplots_adjust(left=0.03, right=0.80, top=0.985, bottom=0.20)
    by = {row["key"]: row for row in data}
    done(fig, "01-metitarski-flyspeck-interval",
         "What FloatLib's interval tactic proves on the MetiTarski problems and on Flyspeck's nonlinear "
         "inequalities. A bar is one route over a whole suite, split into the goals proved, not proved and not "
         "translated to Lean, with the number proved at its end: 948 MetiTarski goals and 164 Flyspeck "
         f"records, one tactic call each. Run on {facts['date']} with Lean {facts['lean']} on {MACHINE}; these "
         "are the suite runs recorded in the repository of FloatLib.",
         "Four horizontal bars on an axis from 0 to 100 percent, for MetiTarski and Flyspeck with interval "
         "alone and with the constraint engine. The blue part, the proved goals, is largest, about a third, "
         "for MetiTarski with the constraint engine; the Flyspeck bars are mostly hatched, not translated. "
         "Each bar ends in its count.",
         "Each goal is translated to one Lean goal and given to `interval` once with "
         "`FloatLib.Tactic.Interval` alone and once with the constraint engine imported as well, which is "
         "`interval` as `import FloatLib` provides it; 20 seconds and 4 GB per goal. MetiTarski: "
         f"{by['metitarski-interval']['proved']} and {by['metitarski-constraint']['proved']} of 948; Flyspeck: "
         f"{by['flyspeck-interval']['proved']} and {by['flyspeck-constraint']['proved']} of 164 (153 nonlinear "
         "inequalities and eleven examples). Not proved means refused or over the limits; not translated "
         "means the translator refuses the goal. A count says what the tactic did on these suites, not what "
         "it does on all goals. From Lean: every proof and its axioms. From scripts: the translation and the "
         "counts. The counts are `benchmarks/results/suites/counts.csv` of FloatLib's repository "
         f"({facts['measured_utc']}); the library sources of that run differ from the commit `fc2fad1f` by one added "
         "corollary and the version number, and no tactic differs. Source: `results/counts.csv`.")


if __name__ == "__main__":
    fig_inequalities()
    suite.write_captions("MetiTarski and Flyspeck with interval")
