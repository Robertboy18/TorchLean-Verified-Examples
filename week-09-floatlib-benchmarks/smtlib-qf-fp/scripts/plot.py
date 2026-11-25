"""FloatLib on the QF_FP benchmarks of SMT-LIB 2025: the two figures.

    python3 scripts/plot.py [time] [reach]

Draws `figures/NN-NAME.png` (9 inches wide at 200 dpi) and `.svg` from the files under `results/`, and writes
`figures/CAPTIONS.md`. The look is `../plots/figstyle.py` (colours, markers, grid, the text checks of `save`)
with the larger type of the week's figures. Nothing is measured here: every number drawn is read from a
file that `scripts/make_results.py` wrote from the logs of the runs.
"""
from __future__ import annotations

import csv
import json
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

# Larger type than the figures of FloatLib's guide (11, 10 and 9 points), heavier lines and marks.
fs.SIZE_LABEL = fs.SIZE_PANEL = 17.0
fs.SIZE_TICK = fs.SIZE_LEGEND = 15.0
fs.SIZE_SMALL = 14.0
fs.LW_DATA, fs.LW_REF, fs.LW_AXIS, fs.LW_GRID, fs.MARKER = 2.6, 1.5, 1.1, 0.8, 9.0
fs.setup()

RESULTS = SUITE / "results"
FIGURES = SUITE / "figures"
MACHINE = "an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs)"
CAPTIONS: list[tuple[str, str, str, str]] = []      # (file stem, caption, alt text, notes)
FACTS = json.loads((RESULTS / "facts.json").read_text(encoding="utf-8"))
TICK = {"toIntegral": "roundTo\nIntegral"}


def read(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plain(value, _=None) -> str:
    return f"{value:,.0f}" if value >= 1000 else f"{value:g}"


def log_axis(ax, ticks) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def done(fig, stem: str, caption: str, alt: str, notes: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    target = fs.save(fig, f"{stem}.png", out=FIGURES / f"{stem}.png")
    CAPTIONS.append((stem, caption, alt, notes))
    print(target)


# ---------------------------------------------------------------------------------------------
# Figure 1: time per instance by operation (results/fig-time.csv)
# ---------------------------------------------------------------------------------------------

def fig_time() -> None:
    table = {(row["series"], row["operation"]): row for row in read("fig-time.csv")}
    panels = [["add", "sub", "mul", "div", "fma", "sqrt", "rem"],
              ["toIntegral", "min", "max", "abs", "eq", "lt", "gt"]]
    grey = dict(linewidth=fs.LW_DATA, markersize=fs.MARKER, linestyle="none")
    order = [
        ("lean kernel", fs.series("FloatLib", label="FloatLib, kernel check", linestyle="none")),
        ("lean batch whole", fs.series("FloatLib", label="FloatLib, whole Lean run", linestyle="none",
                                       markerfacecolor="white", markeredgewidth=2.2)),
        ("z3 process wall", dict(color=fs.GREY_DARK, marker="s", label="Z3 5.4.1", **grey)),
        ("cvc5 process wall", dict(color=fs.GREY_LIGHT, marker="^", label="cvc5 1.4.2", **grey)),
    ]
    offsets = [-0.285, -0.095, 0.095, 0.285]
    fig, axes = plt.subplots(2, 1, figsize=(fs.WIDTH, 9.4), sharey=True)
    lows = [float(table[(name, op)]["p05_ms"]) for name, _ in order for ops in panels for op in ops]
    highs = [float(table[(name, op)]["p95_ms"]) for name, _ in order for ops in panels for op in ops]
    handles = []
    for index, (ax, operations) in enumerate(zip(axes, panels)):
        for (name, look), offset in zip(order, offsets):
            rows = [table[(name, op)] for op in operations]
            xs = [position + offset for position in range(len(operations))]
            median = [float(row["median_ms"]) for row in rows]
            low = [float(row["median_ms"]) - float(row["p05_ms"]) for row in rows]
            high = [float(row["p95_ms"]) - float(row["median_ms"]) for row in rows]
            ax.errorbar(xs, median, yerr=[low, high], ecolor=look["color"], elinewidth=fs.LW_REF,
                        capsize=3.5, capthick=fs.LW_REF, zorder=3, **look)
            if index == 0:
                handles.append(Line2D([], [], **look))
        for boundary in range(1, len(operations)):
            ax.axvline(boundary - 0.5, color=fs.GRID, linewidth=fs.LW_GRID, zorder=1)
        log_axis(ax, [2, 5, 10, 20, 50, 100])
        ax.set_xlim(-0.5, len(operations) - 0.5)
        ax.set_xticks(range(len(operations)), [TICK.get(op, op) for op in operations])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("time per instance (ms)")
        fs.value_grid(ax)
        fs.panel_title(ax, index, "arithmetic" if index == 0 else "rounding to an integer, selection, comparison")
    axes[0].set_ylim(min(lows) / 1.35, max(highs) * 1.35)
    axes[-1].set_xlabel("operation")
    fs.legend(fig, handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.53, 0.972),
              columnspacing=1.6, handletextpad=0.3, labelspacing=0.5)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.915, bottom=0.085, hspace=0.3)

    def cell(name, op="all", column="median_ms"):
        return float(table[(name, op)][column])

    def span(name, operations, digits=0):
        values = [cell(name, op) for op in operations]
        return f"{min(values):.{digits}f} and {max(values):.{digits}f}"

    everything = panels[0] + panels[1]
    sample = FACTS["sample"]
    batch = sample["lean_batch"]
    per_theorem = sorted(v["ms_per_theorem"] for v in batch.values())
    process = FACTS.get("process", {})
    solver_batch = FACTS.get("smt_batch", {})
    lines = ["| operation | FloatLib, kernel check | of which the decision | FloatLib, whole Lean run | Z3 5.4.1 | "
             "cvc5 1.4.2 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for op in everything + ["all"]:
        cells = [f"{cell(name, op):.1f} ({cell(name, op, 'p05_ms'):.1f} to {cell(name, op, 'p95_ms'):.1f})"
                 for name in ("lean kernel", "lean batch whole", "z3 process wall", "cvc5 process wall")]
        cells.insert(1, f"{cell('lean kernel decision', op):.2f}")
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    single = ""
    if "lean" in process:
        single = (f" With one instance per Lean process the wall time is {process['lean']['median_ms'] / 1e3:.1f} "
                  f"seconds per instance (median of {process['lean']['instances']} instances, "
                  f"{process['lean']['min_ms'] / 1e3:.1f} to {process['lean']['max_ms'] / 1e3:.1f}), nearly all of "
                  "it loading the library, which is why the instances are run in batches (a separate job on "
                  "another core, at nice 19).")
    batched = ""
    if solver_batch:
        batched = (" A solver also answers a batch in one process (the same 250 instances with `(reset)` between "
                   f"them): Z3 then takes {solver_batch['z3']['ms_per_instance_min']:.1f} to "
                   f"{solver_batch['z3']['ms_per_instance_max']:.1f} ms per instance and cvc5 "
                   f"{solver_batch['cvc5']['ms_per_instance_min']:.1f} to "
                   f"{solver_batch['cvc5']['ms_per_instance_max']:.1f} ms over the three repetitions.")
    load = FACTS.get("sample_load", {})
    done(fig, "01-time-by-operation",
         "Time per instance on SMT-LIB's single-operation floating-point benchmarks (the Wintersteiger family, "
         "binary64), by operation: FloatLib's proof by `float_decide` in Lean against the SMT solvers Z3 5.4.1 "
         "and cvc5 1.4.2. A marker is the median over the 100 sampled instances of an operation of the time per "
         "instance, in milliseconds, with a whisker from the 5th to the 95th percentile: for FloatLib the time "
         "Lean's kernel takes to check the instance's proof, and the whole Lean run, which adds parsing, "
         "elaboration, the axiom report and an equal share of loading the library with 250 instances per "
         "process; for a solver the wall time of one process per instance. The time axis is logarithmic, and "
         "lower is faster. Measured on FloatLib 4.1.0 on 2026-11-21 with Lean 4.36.0, the fastest of three runs "
         f"of each instance, one process at a time on one core of {MACHINE}.",
         "Two panels of markers with whiskers on a logarithmic time axis from 2 to 100 milliseconds, seven "
         "operations each: add, sub, mul, div, fma, sqrt and rem above, and rounding to an integer, min, max, "
         "abs, eq, lt and gt below. In every group the open FloatLib marker for the whole Lean run is highest, "
         f"between {span('lean batch whole', everything)} milliseconds, the Z3 square lies between "
         f"{span('z3 process wall', everything)}, the cvc5 triangle is near "
         f"{cell('cvc5 process wall'):.0f}, and the filled FloatLib marker for the kernel check is lowest, between "
         f"{span('lean kernel', [op for op in everything if op not in ('min', 'max')], 1)}, except for min and "
         f"max, where it is just above the cvc5 triangle at about {cell('lean kernel', 'min'):.1f}.",
         f"The sample is {sample['instances'] - 12:,} instances drawn with a fixed seed from the 39,994 of the "
         "family, 50 of each status for each of the 14 operations. Every tool ran every instance three times, "
         "and the time of an instance is the fastest of the three. Lean proved the benchmark's status of every "
         "sampled instance with the three standard axioms, and Z3 and cvc5 answered it in every run. Over the "
         f"whole sample the medians are {cell('lean kernel'):.1f} ms for the kernel check (of which "
         f"{cell('lean kernel decision'):.2f} ms is the decision by evaluation and the rest the check of the "
         "rewrite of the public operations to their kernel operations), "
         f"{cell('lean batch whole'):.1f} ms for the whole Lean run, {cell('z3 process wall'):.1f} ms for Z3 "
         f"({cell('z3 process cpu'):.1f} ms of CPU time) and {cell('cvc5 process wall'):.1f} ms for cvc5 "
         f"({cell('cvc5 process cpu'):.1f} ms of CPU time). A Lean process took {per_theorem[0]:.0f} to "
         f"{per_theorem[-1]:.0f} ms of wall time per theorem over the three repetitions of the six batch files."
         + single + batched +
         " The solvers fold the constants, so their time is starting a process and reading the file. Lean's "
         "answer is a theorem whose proof the kernel has checked. Median (5th to 95th percentile), "
         "milliseconds:\n\n" + "\n".join(lines) + "\n\n"
         "The machine was busy: " + load.get("sentence", "the load is in `results/timing-sample.csv.load.csv`.")
         + " From Lean: the kernel time (each proof term checked again with `Lean.Kernel.check`, timed with "
         "`IO.monoNanosNow`) and the clock between consecutive instances. From scripts: the process times "
         "(`scripts/time_solvers.py`, the monotonic clock around each process), the medians and the drawing. "
         "The whole Lean run of an instance is the clock from the end of the previous instance's report to the "
         "end of its own, plus an equal share of the rest of the process (loading the library, the prelude, "
         "exit). Source: `results/fig-time.csv`.")


# ---------------------------------------------------------------------------------------------
# Figure 2: what is in reach, and the competition (results/fig-coverage.csv, fig-smtcomp2025.csv)
# ---------------------------------------------------------------------------------------------

def stacked_rows(ax, rows, total_label: str) -> None:
    """Rows of a 100 % stacked bar, each with its name above and its count at the right end."""
    for position, (label, parts, colours) in enumerate(rows):
        y = len(rows) - 1 - position
        total = sum(parts)
        left = 0.0
        for value, colour in zip(parts, colours):
            share = 100.0 * value / total
            ax.barh(y, share, left=left, height=0.42, color=colour, edgecolor=colour, linewidth=fs.LW_AXIS,
                    zorder=2)
            left += share
        ax.text(0.0, y + 0.29, label, ha="left", va="bottom", fontsize=fs.SIZE_SMALL, color=fs.INK)
        ax.text(101.5, y, total_label.format(done=parts[0], total=total), ha="left", va="center",
                fontsize=fs.SIZE_SMALL, color=fs.INK)
    ax.set_ylim(-0.45, len(rows) - 0.15)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    fs.value_grid(ax, "x")


def fig_reach() -> None:
    coverage = read("fig-coverage.csv")
    solvers = [row for row in read("fig-smtcomp2025.csv") if row["solver"] != "Z3-Owl-base"]
    base = [row for row in read("fig-smtcomp2025.csv") if row["solver"] == "Z3-Owl-base"]
    track = int(solvers[0]["benchmarks"])
    in_reach = int(solvers[0]["benchmarks_with_fixed_inputs"])
    top = [(row["family"], [int(row["proved"]), int(row["benchmarks"]) - int(row["proved"])], [fs.BLUE, fs.LINE])
           for row in coverage]
    bottom = [(row["solver"], [int(row["solved"]), track - int(row["solved"])], [fs.GREY_DARK, fs.LINE])
              for row in solvers]
    bottom.append(("FloatLib", [in_reach, track - in_reach], [fs.BLUE, fs.LINE]))
    heights = [len(top), len(bottom)]
    fig, axes = plt.subplots(2, 1, figsize=(fs.WIDTH, 0.86 * sum(heights) + 2.6),
                             gridspec_kw={"height_ratios": heights})
    stacked_rows(axes[0], top, "{done:,} of {total:,}")
    fs.panel_title(axes[0], 0, "SMT-LIB 2025, by family")
    stacked_rows(axes[1], bottom, "{done} of {total}")
    fs.panel_title(axes[1], 1, "SMT-COMP 2025 selection")
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("share of the benchmarks (%)")
    handles = [Patch(facecolor=fs.BLUE, edgecolor=fs.BLUE, label="proved by FloatLib"),
               Patch(facecolor=fs.GREY_DARK, edgecolor=fs.GREY_DARK, label="solved by the solver"),
               Patch(facecolor=fs.LINE, edgecolor=fs.LINE, label="not solved")]
    fs.legend(fig, handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.45, 0.985), handlelength=1.6,
              columnspacing=1.4)
    fig.subplots_adjust(left=0.03, right=0.76, top=0.935, bottom=0.085, hspace=0.2)

    total = sum(int(row["benchmarks"]) for row in coverage)
    fixed = sum(int(row["fixed_inputs"]) for row in coverage)
    proved = sum(int(row["proved"]) for row in coverage)
    by = {row["family"]: row for row in coverage}
    best = solvers[0]
    done(fig, "02-reach-and-competition",
         "What FloatLib's `float_decide` reaches in the QF_FP benchmarks of SMT-LIB 2025, and what SMT solvers "
         "solved in that logic at SMT-COMP 2025. In panel (a) a bar is one benchmark family, with the share of "
         "its benchmarks whose inputs are all fixed and whose status Lean proved, and the count at its end; the "
         "rest of a bar are search problems, which `float_decide` does not attempt. In panel (b) a bar is one "
         f"solver of the single-query track on the competition's {track} selected benchmarks, from the published "
         "results, with FloatLib's count of benchmarks in reach among the same "
         f"{track} below them. FloatLib 4.1.0 was run on 2026-11-21 with Lean 4.36.0 on {MACHINE}; the "
         "competition ran in 2025 on its own cluster with 1,200 seconds per benchmark.",
         "Two panels of horizontal bars on an axis from 0 to 100 percent. Above, six bars for the families: the "
         "wintersteiger bar is entirely blue and ends in 39,994 of 39,994, and the bars of griggio, schanda, "
         "Vector, ramalho and UltimateAutomizer are almost or entirely light, ending in 2 of 214, 4 of 44, "
         "6 of 91, 0 of 36 and 0 of 27. Below, five dark bars for Bitwuzla, COLIBRI, cvc5, colibri2 and Z3-Owl "
         f"that fill between about two thirds and nine tenths of the axis, ending in {best['solved']} of {track} "
         f"for {best['solver']}, and a light bar for FloatLib that ends in {in_reach} of {track}.",
         f"Panel (a): of the {total:,} benchmarks, {fixed:,} fix every input (every declared constant that the "
         f"assertions use is equated to a literal or to a term over fixed constants), and Lean proved the "
         f"status of all {proved:,} with `propext`, `Classical.choice` and `Quot.sound` only: "
         f"{int(by['wintersteiger']['proved']):,} single-operation tests of the wintersteiger family, "
         f"{by['schanda']['proved']} of schanda (unsat, binary32), {by['griggio']['proved']} of griggio (they "
         f"assert nothing, so they are trivially sat) and {by['Vector']['proved']} of Vector (no declared "
         "constant; the benchmarks give no status, and Lean proves sat, as Z3 and cvc5 answer). The other "
         f"{total - fixed} are search problems. Panel (b): the competition leaves out benchmarks that every "
         "solver answered within a second the year before, which removes the whole wintersteiger family, and "
         f"none of its {track} selected benchmarks fixes its inputs: "
         + ", ".join(f"{row['solver']} {row['solved']} ({row['sat']} sat, {row['unsat']} unsat)" for row in solvers)
         + (f"; the non-competing {base[0]['solver']} solved {base[0]['solved']}" if base else "")
         + ". The two panels answer different questions: (a) is evaluation at given inputs with a proof that "
         "Lean's kernel checks, (b) is search for inputs. From Lean: every proof and its axioms. From scripts: "
         "the classification and translation (`scripts/qffp2lean.py`), the counts, and the competition's "
         "numbers (`results-sq-2025.json.gz` of the competition's repository). Sources: "
         "`results/fig-coverage.csv`, `results/fig-smtcomp2025.csv`.")


FIGURES_BY_NAME = {"time": fig_time, "reach": fig_reach}


def write_captions() -> None:
    lines = ["# SMT-LIB QF_FP with float_decide: figure captions", "",
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
