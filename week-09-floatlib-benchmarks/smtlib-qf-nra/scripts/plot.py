"""SMT-LIB 2025 QF_NRA outside MetiTarski with FloatLib's `interval`: the two figures.

    python3 scripts/plot.py [families] [solvers]

Draws `figures/NN-NAME.png` (9 inches wide at 200 dpi) and `.svg` from `results/per-instance.csv` and
`results/run-facts.csv`, and writes `figures/CAPTIONS.md`. The look is `../plots/figstyle.py` with the larger
type of the week's figures. Nothing is measured here: every number drawn is read from the CSV files that
`scripts/compare.py` wrote from the logs of the Lean runs, from the logs of the solver runs and from the
published SMT-COMP 2025 results.
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

RESULTS = SUITE / "results"
FIGURES = SUITE / "figures"
SOLVERS = ["Z3-alpha", "cvc5", "Yices2", "SMT-RAT"]
CAPTIONS: list[tuple[str, str, str, str]] = []      # (file stem, caption, alt text, notes)


def read(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def facts() -> dict:
    return {row["key"]: row["value"] for row in read("run-facts.csv")}


def rows_axis(ax, labels) -> None:
    """Rows named on the left axis, a share axis from 0 to 100 percent below."""
    n = len(labels)
    ax.set_ylim(-0.62, n - 0.38)
    ax.set_yticks([n - 1 - i for i in range(n)], labels, fontsize=fs.SIZE_SMALL)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines["left"].set_visible(False)


def share_axis(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    fs.value_grid(ax, "x")


def done(fig, stem: str, caption: str, alt: str, notes: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    target = fs.save(fig, f"{stem}.png", out=FIGURES / f"{stem}.png")
    CAPTIONS.append((stem, caption, alt, notes))
    print(target)


# (kinds of per-instance.csv, legend label, (face, hatch, edge))
PARTS = [(("proved",), "proved", (fs.BLUE, None, fs.BLUE)),
         (("tactic",), "refused by the tactic", (fs.LINE, None, fs.LINE)),
         (("heartbeats",), "heartbeat limit", ("white", "////", fs.GREY_LIGHT)),
         (("recursion-statement", "recursion-tactic"), "recursion limit", ("white", "....", fs.GREY_LIGHT)),
         (("timeout",), "timed out", (fs.ORANGE, None, fs.ORANGE)),
         (("memory", "other"), "other", (fs.VERMILION, None, fs.VERMILION))]


def family_rows(rows):
    """[(family, [count per part])] sorted by size, largest first."""
    families: dict = {}
    for row in rows:
        counts = families.setdefault(row["family"], [0] * len(PARTS))
        for index, (kinds, _, _) in enumerate(PARTS):
            if row["kind"] in kinds:
                counts[index] += 1
    order = sorted(families, key=lambda f: (-sum(families[f]), f.lower()))
    return [(family, families[family]) for family in order]


def fig_families() -> None:
    data = [row for row in read("per-instance.csv") if row["kind"]]
    every = [row for row in read("per-instance.csv") if row["translated"] == "True"]
    fact = facts()
    groups = [("variables bounded by the constraints", family_rows([r for r in data if r["selected"] == "True"])),
              ("the other unsatisfiable benchmarks", family_rows([r for r in data if r["selected"] == "False"]))]
    used = [i for i in range(len(PARTS)) if any(parts[i] for _, rows in groups for _, parts in rows)]
    heights = [len(rows) for _, rows in groups]
    fig, axes = plt.subplots(2, 1, figsize=(fs.WIDTH, 0.5 * sum(heights) + 3.0),
                             gridspec_kw={"height_ratios": heights})
    for index, (ax, (title, rows)) in enumerate(zip(axes, groups)):
        for position, (family, parts) in enumerate(rows):
            y = len(rows) - 1 - position
            total, left = sum(parts), 0.0
            for i in used:
                face, hatch, edge = PARTS[i][2]
                share = 100.0 * parts[i] / total
                if parts[i]:
                    ax.barh(y, share, left=left, height=0.56, color=face, hatch=hatch, edgecolor=edge,
                            linewidth=fs.LW_AXIS, zorder=2)
                left += share
            ax.text(101.5, y, f"{parts[0]:,} of {total:,}", ha="left", va="center", fontsize=fs.SIZE_SMALL,
                    color=fs.INK)
        rows_axis(ax, [family for family, _ in rows])
        share_axis(ax)
        fs.panel_title(ax, index, title)
    axes[-1].set_xlabel("share of the family's goals (%)")
    axes[0].tick_params(labelbottom=False)
    handles = [Patch(facecolor=PARTS[i][2][0], edgecolor=PARTS[i][2][2], hatch=PARTS[i][2][1], label=PARTS[i][1])
               for i in used]
    fs.legend(fig, handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.945),
              handlelength=1.6, columnspacing=2.4)
    fig.subplots_adjust(left=0.235, right=0.80, top=0.905, bottom=0.07, hspace=0.2)

    def totals(rows):
        return [sum(parts[i] for _, parts in rows) for i in range(len(PARTS))]

    def listing(rows):
        return ", ".join(f"{family} {parts[0]:,} of {sum(parts):,}" for family, parts in rows)

    sel, rest = totals(groups[0][1]), totals(groups[1][1])
    n_sel, n_rest = sum(sel), sum(rest)
    all_rest = sum(r["selected"] == "False" for r in every)
    kinds = collections_counter(data)
    timed = ", and those killed at the time limit" if sel[4] + rest[4] else ""
    measured = (f"all {all_rest:,}" if n_rest == all_rest else
                f"the {n_rest:,} that were run, in a seeded random order, of {all_rest:,}")
    done(fig, "01-qfnra-families",
         "What FloatLib's interval tactic proves on the unsatisfiable SMT-LIB 2025 QF_NRA benchmarks outside "
         "MetiTarski, by benchmark family. A bar is one family, split into the goals that Lean accepted as "
         "proved, those the tactic refused, and those that stopped at Lean's default limit of heartbeats or of "
         f"recursion depth{timed}, with the number proved at its end. "
         f"Panel (a) has the {n_sel} benchmarks whose variables are all bounded by their constraints (the "
         f"selection), and panel (b), for information, {measured} other translated benchmarks, where the "
         "selection analysis finds no finite bound for some variable. "
         f"One call of interval per goal with Lean's default limits and {fact['limit_s']} seconds of wall time, "
         f"run on {fact['date']} with FloatLib 4.1.0 and Lean {fact['lean']} on {fact['machine']}.",
         f"Two panels of horizontal bars on an axis from 0 to 100 percent. Above, {len(groups[0][1])} bars for "
         f"the families with bounded variables, of which "
         f"{sum(parts[0] == sum(parts) for _, parts in groups[0][1])} are entirely blue, the colour of proved "
         "goals, one is mostly grey and two are hatched or dotted. "
         f"Below, {len(groups[1][1])} bars for the families of the other benchmarks, which are almost "
         "entirely grey, hatched or dotted, with a thin blue part at the left of a few. Each bar ends in its "
         "count.",
         f"Panel (a): {sel[0]} of {n_sel} proved ({listing(groups[0][1])}); {sel[1]} refused by the tactic, "
         f"{sel[2]} stopped at the heartbeat limit, {sel[3]} at the recursion limit, {sel[4]} timed out. "
         f"Panel (b): {rest[0]:,} of {n_rest:,} proved ({listing(groups[1][1])}); {rest[1]:,} refused by the "
         f"tactic, {rest[2]:,} stopped at the heartbeat limit, {rest[3]:,} at the recursion limit "
         f"({kinds[('False', 'recursion-statement')]:,} of them while Lean elaborated the statement, before "
         f"the tactic ran), {rest[4]:,} timed out. "
         "A goal is `¬ ∃ x, constraints`, translated from the benchmark by one rule per SMT-LIB construct; it "
         "counts as proved only when Lean exits without an error and `#print axioms` lists nothing beyond "
         "`propext`, `Classical.choice` and `Quot.sound`. Refused by the tactic: `interval` reports that its "
         "search could not refute the constraints within its own default limits (40 cuts along a branch, "
         "20,000 boxes) or that a variable has no bound. Heartbeat limit: `maxHeartbeats` (200,000) was "
         "reached. Recursion limit: `maxRecDepth` (512) was reached, by the tactic or already by the "
         "elaboration of a statement with more than about 140 variables. "
         "From Lean: whether each goal was proved, and its axioms. From scripts: the selection, the "
         "translation, the wall time of each Lean process and the counts. Source: `results/per-instance.csv`.")


def collections_counter(data) -> dict:
    out: dict = {}
    for row in data:
        key = (row["selected"], row["kind"])
        out[key] = out.get(key, 0) + 1

    class Zero(dict):
        def __missing__(self, key):
            return 0

    return Zero(out)


def fig_solvers() -> None:
    table = read("per-instance.csv")
    fact = facts()
    comp = [row for row in table if row["kind"] and row["in_smtcomp_2025"] == "True"]
    groups = [("selected instances that SMT-COMP 2025 ran", [r for r in comp if r["selected"] == "True"]),
              ("other instances that SMT-COMP 2025 ran", [r for r in comp if r["selected"] == "False"])]
    here = [row for row in table if row["selected"] == "True" and row["kind"] and row["here_z3"]]
    # three panels with their own gaps: the first gap holds an axis label, a panel title and a legend
    fig = plt.figure(figsize=(fs.WIDTH, 14.1))
    grid = fig.add_gridspec(5, 1, height_ratios=[5, 4.75, 8.2, 2.4, 5], hspace=0.0)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[2]), fig.add_subplot(grid[4])]
    tools = [("FloatLib interval", None)] + [(name, name) for name in SOLVERS]
    counts = []
    for index, ax, (title, rows) in ((0, axes[0], groups[0]), (2, axes[2], groups[1])):
        bars = []
        for label, solver in tools:
            solved = sum((r["kind"] == "proved") if solver is None else (r[solver] == "unsat") for r in rows)
            bars.append((label, solved, len(rows)))
        counts.append(bars)
        for position, (label, solved, total) in enumerate(bars):
            y = len(bars) - 1 - position
            colour = fs.BLUE if position == 0 else fs.GREY_DARK
            share = 100.0 * solved / total
            if solved:
                ax.barh(y, share, height=0.56, color=colour, edgecolor=colour, linewidth=fs.LW_AXIS, zorder=2)
            if solved < total:
                ax.barh(y, 100.0 - share, left=share, height=0.56, color=fs.LINE, edgecolor=fs.LINE,
                        linewidth=fs.LW_AXIS, zorder=2)
            ax.text(101.5, y, f"{solved:,} of {total:,}", ha="left", va="center", fontsize=fs.SIZE_SMALL,
                    color=fs.INK)
        rows_axis(ax, [label for label, _, _ in bars])
        share_axis(ax)
        ax.set_xlabel("instances proved or answered unsat (%)")
        fs.panel_title(ax, index, title)

    # panel (b): instances proved or answered within a time, on this machine
    ax = axes[1]
    limit = float(fact["limit_s"])
    series = [("FloatLib interval", [float(r["wall_s"]) for r in here if r["kind"] == "proved"],
               dict(color=fs.BLUE, linestyle="-", marker="o")),
              (f"Z3 {fact['z3']}", [float(r["here_z3_wall_s"]) for r in here if r["here_z3"] == "unsat"],
               dict(color=fs.GREY_DARK, linestyle="--", marker="s")),
              (f"cvc5 {fact['cvc5']}", [float(r["here_cvc5_wall_s"]) for r in here if r["here_cvc5"] == "unsat"],
               dict(color=fs.GREY_DARK, linestyle=":", marker="^")),
              (f"Yices {fact['yices']}", [float(r["here_yices2_wall_s"]) for r in here if r["here_yices2"] == "unsat"],
               dict(color=fs.GREY_DARK, linestyle="-.", marker="D"))]
    handles, summary = [], []
    for label, times, look in series:
        times = sorted(times)
        xs = [times[0]] + times + [limit]
        ys = [0] + list(range(1, len(times) + 1)) + [len(times)]
        ax.step(xs, ys, where="post", color=look["color"], linestyle=look["linestyle"], linewidth=fs.LW_DATA,
                zorder=3)
        ax.plot([times[-1]], [len(times)], linestyle="none", marker=look["marker"], color=look["color"],
                markerfacecolor="white", markersize=fs.MARKER, markeredgewidth=1.9, zorder=4, clip_on=False)
        handles.append(Line2D([], [], color=look["color"], linestyle=look["linestyle"], marker=look["marker"],
                              markerfacecolor="white", markersize=fs.MARKER, markeredgewidth=1.9,
                              linewidth=fs.LW_DATA, label=label))
        summary.append((label, len(times), statistics.median(times), times[0], times[-1]))
    ax.set_xscale("log")
    ax.set_xlim(0.008, limit)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, limit]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, len(here))
    ax.set_yticks([0, 20, 40, 60, len(here)] if len(here) - 60 >= 5 else [0, 20, 40, 60])
    fs.value_grid(ax, "both")
    ax.set_xlabel("wall time on an instance (s)")
    ax.set_ylabel("instances proved or answered unsat")
    ax.set_title("(b) all selected instances, same time limit", loc="left", fontsize=fs.SIZE_PANEL,
                 color=fs.INK, pad=84)
    fs.legend(ax, handles=handles, loc="lower left", ncols=2, bbox_to_anchor=(0.0, 1.04), columnspacing=2.0,
              handlelength=2.6)
    fig.subplots_adjust(left=0.235, right=0.80, top=0.972, bottom=0.058)

    def listing(bars):
        return ", ".join(f"{label} {solved:,}" for label, solved, _ in bars)

    n_sel, n_rest = len(groups[0][1]), len(groups[1][1])
    n_all_rest = sum(r["selected"] == "False" and r["translated"] == "True" and r["in_smtcomp_2025"] == "True"
                     for r in table)
    rest_text = (f"the {n_rest:,} other candidates that the competition ran" if n_rest == n_all_rest else
                 f"{n_rest:,} of the {n_all_rest:,} other candidates that the competition ran (those reached "
                 "here in a seeded random order)")
    done(fig, "02-qfnra-solvers",
         "FloatLib's interval tactic next to SMT solvers on the same unsatisfiable QF_NRA instances outside "
         "MetiTarski. Panels (a) and (c) give, for interval and for four solvers of SMT-COMP 2025, the share of "
         "the instances that it proved or that the solver answered unsat in the published results of the "
         f"competition, with the count at the end of the bar: on the {n_sel} selected instances (every variable "
         f"bounded) that the competition ran, and on {rest_text}. Panel (b) gives, for all {len(here)} selected "
         "instances, the number proved by interval or answered unsat by a current release of three solvers "
         "within a wall time, in seconds on a logarithmic axis, when all four ran on one machine with "
         f"{fact['limit_s']} seconds per instance. "
         f"Run on {fact['date']} on {fact['machine']}; the competition allowed 1,200 seconds per instance on "
         "its own cluster.",
         "Three panels. In the first and the third, one bar per tool: FloatLib interval in blue, then Z3-alpha, "
         "cvc5, Yices2 and SMT-RAT in dark grey. In the first the bars reach between half and nine tenths of "
         "the width; in the third the bar of interval is a thin sliver and the solvers' bars cover almost all "
         "of the width. In the middle panel four rising step curves: the three solvers' curves climb at a few "
         "hundredths of a second to between 40 and 60 instances, and the blue curve of interval starts near 3 "
         "seconds and passes the curve of Z3 before 30 seconds.",
         f"Panel (a), {n_sel} instances: {listing(counts[0])}. Panel (c), {n_rest:,} instances: "
         f"{listing(counts[1])}. Panel (b), {len(here)} instances, proved or answered unsat within "
         f"{fact['limit_s']} s, with the median (fastest to slowest) wall time in seconds: "
         + "; ".join(f"{label} {n} ({m:.2f}, {lo:.2f} to {hi:.2f})" for label, n, m, lo, hi in summary) + ". "
         "A solver's unsat answer is not a proof: it is the output of a program that is trusted as a whole, "
         "while a goal counted for interval was checked by Lean's kernel. "
         f"In panel (b) every process was timed in the same way, from its start to its exit; the time of "
         f"interval includes about {fact['startup_s']} seconds in which Lean loads the library, and the solvers "
         "are the release binaries with their default options, not the competition's configurations. "
         "In panels (a) and (c) the two sides did not run on the same machine or with the same limit. "
         "From Lean: whether each goal was proved. From the SMT-COMP 2025 results: the answers of Z3-alpha, "
         "cvc5, Yices2 and SMT-RAT. From scripts: the join, the solver runs of panel (b), the wall time of each "
         "process and the counts. Source: `results/per-instance.csv`.")


FIGURES_BY_NAME = {"families": fig_families, "solvers": fig_solvers}


def write_captions() -> None:
    lines = ["# SMT-LIB QF_NRA with interval: figure captions", "",
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
    names = sys.argv[1:] or list(FIGURES_BY_NAME)
    for name in names:
        FIGURES_BY_NAME[name]()
    if set(names) == set(FIGURES_BY_NAME):
        write_captions()


if __name__ == "__main__":
    main()
