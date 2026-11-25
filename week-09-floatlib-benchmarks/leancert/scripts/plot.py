"""LeanCert's benchmarks with FloatLib's `interval`: the two figures.

    python3 scripts/plot.py

Draws `figures/01-time-per-goal.png` and `figures/02-goals-per-suite.png` (9 inches wide at 200 dpi)
with their `.svg`, and writes `figures/CAPTIONS.md`. The look is `../plots/figstyle.py` with the
larger type of the week's figures. Nothing is measured here: every number drawn is read from
`results/goals.csv`, which `scripts/summarize.py` wrote from the run records of `scripts/run_goals.py`.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(SUITE.parent / "plots"))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

fs.SIZE_LABEL = fs.SIZE_PANEL = 17.0
fs.SIZE_TICK = fs.SIZE_LEGEND = 15.0
fs.SIZE_SMALL = 14.0
fs.LW_DATA, fs.LW_REF, fs.LW_AXIS, fs.LW_GRID, fs.MARKER = 2.6, 1.5, 1.1, 0.8, 9.0
fs.setup()

FIGURES = SUITE / "figures"
MACHINE = "an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs)"
CAPTIONS: list[tuple[str, str, str, str]] = []

# LeanCert is outside FloatLib's fixed series map: grey, labelled, with its own marker and line style.
LEANCERT_KERNEL = dict(color=fs.GREY_DARK, linestyle="--", marker="P")
LEANCERT_NATIVE = dict(color=fs.GREY_LIGHT, linestyle=":", marker="X")
LABEL = {"fl": "FloatLib interval", "lck": "LeanCert, kernel-checked",
         "lcn": "LeanCert, compiled checker"}
BENCH_SUITES = ["evaluation", "numerical-refinement", "ad", "showcase", "trust-point",
                "trust-subdiv", "trust-optimization", "trust-finsum"]
SUITE_LABEL = {
    "evaluation": "Evaluation", "numerical-refinement": "NumericalRefinement",
    "ad": "DomainAwareAD", "showcase": "bench-showcase", "trust-point": "bench-trust point",
    "trust-subdiv": "bench-trust subdiv", "trust-optimization": "bench-trust optimization",
    "trust-finsum": "bench-trust finsum", "examples": "examples and tests",
}


def goals() -> list[dict]:
    with (SUITE / "results/goals.csv").open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["in_scope"] == "1"]


def proved(row: dict, ladder: str) -> bool:
    status = row[f"{ladder}_status"]
    return status == "proved" or (ladder == "lcn" and status == "proved-extra-axioms")


def plain(value, _=None) -> str:
    return f"{value:g}"


def log_axis(ax, ticks) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def style(ladder: str) -> dict:
    if ladder == "fl":
        return fs.series("FloatLib", label=LABEL["fl"])
    look = LEANCERT_KERNEL if ladder == "lck" else LEANCERT_NATIVE
    return dict(look, label=LABEL[ladder], linewidth=fs.LW_DATA, markersize=fs.MARKER)


def done(fig, stem: str, caption: str, alt: str, notes: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    target = fs.save(fig, f"{stem}.png", out=FIGURES / f"{stem}.png")
    CAPTIONS.append((stem, caption, alt, notes))
    print(target)


def fig_time() -> None:
    rows = goals()
    panels = [("benchmark suites", [r for r in rows if r["suite"] in BENCH_SUITES]),
              ("examples and tests", [r for r in rows if r["suite"] == "examples"])]
    fig, axes = plt.subplots(2, 1, figsize=(fs.WIDTH, 10.6), sharex=True)
    facts = []
    for index, (ax, (title, group)) in enumerate(zip(axes, panels)):
        for ladder in ("lcn", "lck", "fl"):
            times = sorted(float(r[f"{ladder}_time_s"]) for r in group if proved(r, ladder))
            xs = [0.004] + times + [300.0]
            ys = [0] + list(range(1, len(times) + 1)) + [len(times)]
            look = style(ladder)
            every = max(1, len(times) // 9)
            ax.step(xs, ys, where="post", color=look["color"], linestyle=look["linestyle"],
                    linewidth=look["linewidth"], zorder=3 if ladder == "fl" else 2)
            marks = list(range(every - 1, len(times), every))
            ax.plot([times[i] for i in marks], [i + 1 for i in marks], linestyle="none",
                    marker=look["marker"], color=look["color"], markersize=look["markersize"],
                    markerfacecolor=look.get("markerfacecolor", look["color"]), zorder=4)
            facts.append((title, ladder, len(times), len(group),
                          statistics.median(times) if times else float("nan"),
                          max(times) if times else float("nan")))
        log_axis(ax, [0.01, 0.1, 1, 10, 100])
        ax.set_xlim(0.004, 300)
        ax.set_ylim(0, len(group) * 1.04)
        ax.set_ylabel("goals proved")
        fs.value_grid(ax, "both")
        fs.panel_title(ax, index, f"{title}, {len(group)} goals")
    axes[-1].set_xlabel("time per goal (s)")
    handles = [Line2D([], [], **style(ladder)) for ladder in ("fl", "lck", "lcn")]
    fs.legend(fig, handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.53, 0.985),
              columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.945, bottom=0.075, hspace=0.2)
    by = {(title, ladder): (count, total, median, worst) for title, ladder, count, total, median, worst in facts}
    a, b = panels[0][0], panels[1][0]
    caption = (
        "Goals proved against time per goal, by FloatLib's `interval` and by LeanCert's own tactic on the same "
        "statements, each with one call. A curve counts the goals proved within the time on the horizontal axis; "
        "the time is that of the whole `theorem` command inside Lean (elaboration, the tactic and the kernel check), "
        "the median of two repetitions for a run under 20 s and one repetition otherwise, with both tools running "
        "side by side. Panel (a) holds the inequality goals of LeanCert's seven "
        "benchmark suites and of its two script benchmarks, panel (b) the inequalities, with six combinations of "
        "comparisons, that LeanCert's examples and tests prove with one tactic call. The time axis is logarithmic, a curve further left is faster and one that "
        "ends higher proves more. Measured on FloatLib 4.1.0 and LeanCert 44b323a ported to Lean 4.36.0 and "
        f"Mathlib v4.36.0, on 2026-11-21 on {MACHINE}, at most 7 Lean processes at once.")
    alt = (
        "Two panels with three rising step curves each on a logarithmic time axis from 0.004 to 300 seconds. "
        f"In panel (a) the FloatLib curve ends at {by[(a, 'fl')][0]} of {by[(a, 'fl')][1]} goals, the kernel-checked "
        f"LeanCert curve at {by[(a, 'lck')][0]} and the compiled-checker LeanCert curve at {by[(a, 'lcn')][0]}; "
        f"in panel (b) they end at {by[(b, 'fl')][0]}, {by[(b, 'lck')][0]} and {by[(b, 'lcn')][0]} of "
        f"{by[(b, 'fl')][1]} goals. The FloatLib curve rises furthest to the left in both panels.")
    notes = (
        "A goal counts for a curve only when Lean accepted the proof; for FloatLib and for kernel-checked LeanCert "
        "`#print axioms` lists only propext, Classical.choice and Quot.sound, while LeanCert's compiled checker "
        "(`native_decide`, the route LeanCert ships as its default) adds one compiler axiom per proof and is drawn "
        "for reference only. A goal is counted at the first rung of its ladder that proves it (plain file and default "
        "options, then raised `maxRecDepth` and `maxHeartbeats`, then one fixed set of larger tactic options); "
        "the suite's README gives the counts per rung. Goals proved, median and longest time per goal: " +
        "; ".join(f"panel ({'a' if title == a else 'b'}) {LABEL[ladder]} {count} of {total}, "
                  f"{median:.2f} s, {worst:.0f} s" for title, ladder, count, total, median, worst in facts) +
        ". Five LeanCert times of panel (b), of three goals that take more than 20 s, are from the first run and not "
        "from a repetition. From Lean: every time (`IO.monoNanosNow` around the `theorem` command) and every "
        "verdict. From scripts: the medians, the counts and the drawing. Source: `results/goals.csv`.")
    done(fig, "01-time-per-goal", caption, alt, notes)


def fig_suites() -> None:
    rows = goals()
    suites = BENCH_SUITES + ["examples"]
    groups = [(suite, [r for r in rows if r["suite"] == suite]) for suite in suites]
    groups = [(suite, group) for suite, group in groups if group]
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.95 * len(groups) + 1.6))
    height = 0.26
    order = [("fl", fs.BLUE, ""), ("lck", fs.GREY_DARK, "//"), ("lcn", fs.GREY_LIGHT, "..")]
    table = []
    for position, (suite, group) in enumerate(groups):
        counts = []
        for offset, (ladder, colour, hatch) in zip((-height, 0.0, height), order):
            count = sum(proved(r, ladder) for r in group)
            share = 100.0 * count / len(group)
            ax.barh(position + offset, share, height=height * 0.92, color=colour, hatch=hatch,
                    edgecolor="white", linewidth=0.0, zorder=3)
            ax.text(share + 1.2, position + offset, f"{count}", va="center", ha="left",
                    fontsize=fs.SIZE_SMALL, color=fs.INK)
            counts.append(count)
        table.append((suite, len(group), *counts))
    ax.set_yticks(range(len(groups)),
                  [f"{SUITE_LABEL[suite]}\n{len(group)} goal{'s' if len(group) != 1 else ''}"
                   for suite, group in groups])
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("goals proved (% of the suite's inequality goals)")
    ax.tick_params(axis="y", length=0)
    fs.value_grid(ax, "x")
    handles = [Patch(facecolor=colour, hatch=hatch, edgecolor="white", label=LABEL[ladder])
               for ladder, colour, hatch in order]
    fs.legend(fig, handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.965),
              columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.3, right=0.975, top=0.955, bottom=0.11)
    total = sum(n for _, n, *_ in table)
    caption = (
        "Share of each suite's inequality goals that FloatLib's `interval` and LeanCert's own tactic prove with one "
        "call; the number at the end of a bar is the count of goals. The first eight rows are LeanCert's benchmark "
        "suites and script benchmarks, stated as goals; the last row holds the inequalities, with six combinations of "
        "comparisons, that LeanCert's examples and tests prove with one tactic call. Goals of other shapes (integrals, root existence and uniqueness, exact "
        "identities, list constructions, derivatives) are not in the figure, because `interval` does not address "
        "them. Measured on FloatLib 4.1.0 and LeanCert 44b323a ported to Lean 4.36.0 and Mathlib v4.36.0, on "
        f"2026-11-21 on {MACHINE}.")
    alt = (
        f"Horizontal bars in {len(table)} groups of three, one group per suite, on an axis from 0 to 100 percent: "
        "FloatLib interval, LeanCert kernel-checked and LeanCert with its compiled checker. " +
        " ".join(f"{SUITE_LABEL[suite]}: {fl}, {lck} and {lcn} of {n}." for suite, n, fl, lck, lcn in table))
    notes = (
        "A bar counts a goal at any rung of its ladder (plain file and default options, raised Lean limits, one "
        "fixed set of larger options), within 300 s per run. FloatLib's and kernel-checked LeanCert's proofs depend "
        "only on propext, Classical.choice and Quot.sound; the compiled checker adds a compiler axiom and is shown "
        "for reference. The examples row exists because LeanCert's call proves each of its goals, so its LeanCert "
        f"compiled-checker bar is full by construction. {total} goals in all. Counts (suite, goals, FloatLib, "
        "LeanCert kernel-checked, LeanCert compiled checker): " +
        "; ".join(f"{SUITE_LABEL[suite]}, {n}, {fl}, {lck}, {lcn}" for suite, n, fl, lck, lcn in table) +
        ". From Lean: every verdict. From scripts: the counts and the drawing. Source: `results/goals.csv`.")
    done(fig, "02-goals-per-suite", caption, alt, notes)


def write_captions() -> None:
    lines = ["# LeanCert's benchmarks with interval: figure captions", "",
             "Written by `scripts/plot.py`; do not edit by hand. Every figure is `NAME.png` (9 inches wide at "
             "200 dpi) and `NAME.svg`. An image holds only the plot, its axes with units, a legend with short "
             "series names and panel letters; the caption says what the figure shows, what was measured, in "
             "which unit, on which scales, where and when. The notes say how, which numbers come from Lean and "
             "which from scripts, and give the numbers behind the marks.", ""]
    for number, (stem, caption, alt, notes) in enumerate(CAPTIONS, start=1):
        lines += [f"## Figure {number}: `{stem}`", "",
                  f"![{alt}]({stem}.svg \"{caption}\")", "",
                  f"**Figure {number}.** {caption}", "", f"Alt text: {alt}", "", f"Notes: {notes}", "",
                  f"Files: `{stem}.png`, `{stem}.svg`.", ""]
    (FIGURES / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    fig_time()
    fig_suites()
    write_captions()


if __name__ == "__main__":
    main()
