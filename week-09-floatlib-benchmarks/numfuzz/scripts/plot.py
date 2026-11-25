"""The NumFuzz benchmarks with FloatLib's `roundoff`: the two figures.

    python3 scripts/plot.py [RESULTS [FIGURES]]

Draws `figures/01-bounds-and-time.png` and `figures/02-sums-horner-dot.png` (9 inches wide at
200 dpi) with their `.svg`, and writes `figures/CAPTIONS.md`. The look is `../plots/figstyle.py`
with the larger type of the week's figures. Nothing is measured or proved here: every number
drawn is read from `results/programs.csv` and `results/summary.json`, which `scripts/collect.py`
wrote from the Lean results and from the published tables transcribed in `scripts/gen.py`. Every
count, range and ratio in a caption, an alt text or a note is computed here from the same file.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(SUITE.parent / "plots"))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

fs.SIZE_LABEL = fs.SIZE_PANEL = 17.0
fs.SIZE_TICK = fs.SIZE_LEGEND = 15.0
fs.SIZE_SMALL = 14.0
fs.LW_DATA, fs.LW_REF, fs.LW_AXIS, fs.LW_GRID, fs.MARKER = 2.6, 1.5, 1.1, 0.8, 9.0
fs.setup()

# Optional arguments: the directory of the collected results and the directory to draw into.
RESULTS = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SUITE / "results"
FIGURES = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else SUITE / "figures"
MACHINE = "an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs)"
CAPTIONS: list[tuple[str, str, str, str]] = []
U = 2.0 ** -52

# NumFuzz, Gappa and FPTaylor are outside FloatLib's fixed series map: grey, labelled, each with
# its own marker and line style. The textbook bound is a plain mark.
LOOK = {
    "FloatLib": dict(fs.SERIES["FloatLib"], label="FloatLib"),
    "NumFuzz": dict(color=fs.GREY_DARK, linestyle="--", marker="s", markerfacecolor="white",
                    label="NumFuzz"),
    "Gappa": dict(color=fs.GREY_DARK, linestyle=":", marker="D", label="Gappa"),
    "FPTaylor": dict(color=fs.GREY_LIGHT, linestyle="-.", marker="^", label="FPTaylor"),
    "literature": dict(color=fs.INK, linestyle="none", marker="+", label="textbook bound"),
}
TOOLS = ["FloatLib", "NumFuzz", "Gappa", "FPTaylor"]
BOUND = {"FloatLib": "box_relative_bound", "NumFuzz": "numfuzz_bound", "Gappa": "gappa_bound",
         "FPTaylor": "fptaylor_bound"}
TIME = {"FloatLib": "box_relative_seconds", "NumFuzz": "numfuzz_seconds", "Gappa": "gappa_seconds",
        "FPTaylor": "fptaylor_seconds"}


def look(name: str, **overrides) -> dict:
    style = dict(LOOK[name], linewidth=fs.LW_DATA, markersize=fs.MARKER, markeredgewidth=2.0)
    style.update(overrides)
    return style


def handle(name: str, **overrides) -> Line2D:
    return Line2D([], [], **look(name, **overrides))


def programs() -> dict:
    with (RESULTS / "programs.csv").open(newline="", encoding="utf-8") as stream:
        return {row["program"]: row for row in csv.DictReader(stream)}


def number(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def plain(value, _=None) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def log_axis(ax, axis: str, ticks, unlabelled=()) -> None:
    """A logarithmic axis with plain numbers at `ticks`; `unlabelled` adds grid lines without labels."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    target.set_major_locator(FixedLocator(list(ticks)))
    target.set_major_formatter(FuncFormatter(plain))
    target.set_minor_locator(FixedLocator(list(unlabelled)) if unlabelled else NullLocator())
    target.set_minor_formatter(NullFormatter())
    if unlabelled:
        ax.grid(True, axis=axis, which="minor", color=fs.GRID, linewidth=fs.LW_GRID)
        ax.tick_params(axis=axis, which="minor", length=0)


def done(fig, stem: str, caption: str, alt: str, notes: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    target = fs.save(fig, f"{stem}.png", out=FIGURES / f"{stem}.png")
    CAPTIONS.append((stem, caption, alt, notes))
    print(target)


def spread(values: list[float], digits: int = 2) -> str:
    return f"{min(values):.{digits}g} to {max(values):.{digits}g}"


def compare(mine: list[float], theirs: list[float]) -> tuple[int, int, int]:
    """How often `mine` is smaller, equal within 0.5 percent, larger."""
    ratios = [a / b for a, b in zip(mine, theirs)]
    return (sum(r < 0.995 for r in ratios), sum(0.995 <= r <= 1.005 for r in ratios),
            sum(r > 1.005 for r in ratios))


# ---------------------------------------------------------------------------------------------
# Figure 1: the 17 small benchmarks, bound and time
# ---------------------------------------------------------------------------------------------

def fig_small(summary: dict) -> None:
    table = programs()
    rows = sorted((row for row in table.values() if row["paper_table"] == "3"),
                  key=lambda row: int(row["paper_row"]))
    names = [row["program"] for row in rows]
    offsets = dict(zip(TOOLS, [0.33, 0.11, -0.11, -0.33]))
    edge = 1000.0
    size = fs.MARKER - 1.0          # four marks share the height of one row
    fig, (left, right) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.58 * len(rows) + 1.9), sharey=True,
                                      gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    beyond, failed = [], []
    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        for name in TOOLS:
            value = number(row[BOUND[name]])
            if value is None:
                if row[BOUND[name]] == "fail":
                    failed.append((name, row["program"]))
                    left.plot([edge], [y + offsets[name]], linestyle="none", marker="x",
                              color=LOOK[name]["color"], markersize=size, markeredgewidth=2.0,
                              clip_on=False, zorder=4)
                continue
            units = value / U
            if units > edge:
                beyond.append((name, row["program"], value))
                left.plot([edge], [y + offsets[name]], clip_on=False, zorder=4,
                          **look(name, linestyle="none", marker=">", markerfacecolor="white", label=None,
                                 markersize=size))
            else:
                left.plot([units], [y + offsets[name]], zorder=5 if name == "FloatLib" else 4,
                          **look(name, linestyle="none", label=None, markersize=size))
            seconds = number(row[TIME[name]])
            if seconds is not None:
                right.plot([seconds], [y + offsets[name]], zorder=5 if name == "FloatLib" else 4,
                           **look(name, linestyle="none", label=None, markersize=size))
    for ax in (left, right):
        for boundary in range(1, len(rows)):
            ax.axhline(boundary - 0.5, color=fs.GRID, linewidth=fs.LW_GRID, zorder=1)
        fs.value_grid(ax, "x")
        ax.tick_params(axis="y", length=0)
    left.set_ylim(-0.62, len(rows) - 0.5)
    left.set_yticks(range(len(rows)), list(reversed(names)), fontsize=fs.SIZE_SMALL)
    log_axis(left, "x", [1, 2, 5, 10, 20, 100])
    left.set_xlim(0.7, edge)
    left.set_xlabel("relative error bound (units of $2^{-52}$)")
    log_axis(right, "x", [0.001, 0.1, 10], unlabelled=[0.01, 1, 100])
    right.set_xlim(0.0008, 300)
    right.set_xlabel("time (s)")
    fs.panel_title(left, 0, "bound")
    fs.panel_title(right, 1, "time")
    handles = [handle(name, linestyle="none") for name in TOOLS]
    handles.append(Line2D([], [], linestyle="none", marker=">", color=fs.GREY_LIGHT, markerfacecolor="white",
                          markersize=fs.MARKER, markeredgewidth=2.0, label="beyond the axis"))
    handles.append(Line2D([], [], linestyle="none", marker="x", color=fs.GREY_LIGHT, markersize=fs.MARKER,
                          markeredgewidth=2.0, label="no bound"))
    fig.subplots_adjust(left=0.265, right=0.975, top=0.91, bottom=0.075, wspace=0.09)
    fs.legend(fig, handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.55, 0.95),
              columnspacing=1.4, handletextpad=0.2)

    # ---- the facts of the caption, the alt text and the notes, all from the table
    mine = {row["program"]: number(row[BOUND["FloatLib"]]) for row in rows}
    proved = [name for name in names if mine[name] is not None]
    assert len(proved) == len(rows), "a row of the figure has no FloatLib bound; the texts assume all"
    assert all(table[name]["box_relative_rung"] == "0" for name in proved), "a bound is not the first proposal"
    versus = {}
    for other in ("NumFuzz", "Gappa", "FPTaylor"):
        pairs = [(mine[name], number(table[name][BOUND[other]])) for name in proved]
        pairs = [(a, b) for a, b in pairs if b is not None]
        ratios = [a / b for a, b in pairs]
        versus[other] = (len(pairs), *compare([a for a, _ in pairs], [b for _, b in pairs]),
                         min(ratios), max(ratios))
    total = [number(table[name]["box_relative_seconds"]) for name in proved]
    wall = [number(table[name]["box_relative_process_seconds"]) for name in proved]
    forward = [number(table[name]["B_box_forward_seconds"]) for name in proved]
    published = {other: [number(table[name][TIME[other]]) for name in names
                         if number(table[name][TIME[other]]) is not None]
                 for other in ("NumFuzz", "Gappa", "FPTaylor")}
    units = [mine[name] / U for name in proved]
    n_nf, less_nf, same_nf, more_nf, low_nf, high_nf = versus["NumFuzz"]
    n_ga, less_ga, same_ga, more_ga, low_ga, high_ga = versus["Gappa"]
    n_ft, less_ft, same_ft, more_ft, low_ft, high_ft = versus["FPTaylor"]
    beyond_ft = [item for item in beyond if item[0] == "FPTaylor"]
    assert all(item[0] == "FPTaylor" for item in beyond)
    assert more_nf == 0 and less_ga == 0, "the alt text assumes these orders"
    listing = "; ".join(f"{program} {value:.3g}" for _, program, value in beyond)
    caption = (
        f"Relative error bounds and analysis times of the {len(rows)} small benchmarks of the NumFuzz paper "
        "(its Table 3): FloatLib's `roundoff` against the published numbers of NumFuzz, Gappa and FPTaylor. "
        "Panel (a) gives each tool's bound on the relative error in units of 2^-52, the unit roundoff of binary64 "
        "rounded toward plus infinity, and panel (b) the time of each analysis in seconds; both axes are "
        "logarithmic, and left is tighter in (a) and faster in (b). "
        "FloatLib's bound is the one Lean accepted on the box [0.1, 1000] that the paper gave Gappa and FPTaylor, "
        "and its time is Lean's elaboration of the forward theorem and of that bound, measured on 2026-11-21 on "
        f"{MACHINE} with Lean 4.36.0 on FloatLib 4.1.0. The other tools' numbers are the paper's, measured on a "
        "MacBook with a 1.4 GHz processor.")
    alt = (
        f"Two panels sharing {len(rows)} rows, one per program, each row holding up to four markers one above "
        f"the other. In the left panel the FloatLib circles lie between {min(units):.2g} and {max(units):.2g} "
        f"units; they sit on the NumFuzz squares in {same_nf} rows and left of them in {less_nf}, "
        f"and the Gappa diamonds sit on them in {same_ga} rows and left of them in {more_ga}. "
        f"The FPTaylor triangles lie right of the circles in {less_ft} rows, on them in "
        f"{same_ft} and left of them in {more_ft}; in {len(beyond_ft)} of those rows they are "
        f"open triangles at the right edge, and {len(failed)} more row has a cross there. In the right "
        "panel the NumFuzz squares "
        f"stand between {min(published['NumFuzz']):g} and {max(published['NumFuzz']):g} seconds, the Gappa "
        f"diamonds between {min(published['Gappa']):.2g} and {max(published['Gappa']):.2g}, the FPTaylor "
        f"triangles between {min(published['FPTaylor']):.2g} and {max(published['FPTaylor']):.3g} and the "
        f"FloatLib circles between {min(total):.2g} and {max(total):.2g}.")
    lines = ["| program | FloatLib | NumFuzz | Gappa | FPTaylor | FloatLib / NumFuzz | FloatLib / Gappa | "
             "FloatLib (s) | of it forward (s) | process (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        row = table[name]
        lines.append(
            f"| {name} | {row['box_relative_bound']} | {row['numfuzz_bound']} | {row['gappa_bound']} | "
            f"{row['fptaylor_bound']} | {mine[name] / float(row['numfuzz_bound']):.2f} | "
            f"{mine[name] / float(row['gappa_bound']):.2f} | {float(row['box_relative_seconds']):.1f} | "
            f"{float(row['B_box_forward_seconds']):.1f} | {float(row['box_relative_process_seconds']):.1f} |")
    notes = (
        "FloatLib's model here is the one the three tools analyze: real inputs, and each operation of the "
        "program rounded toward plus infinity to binary64 (`Model.roundAt FloatFormat.binary64 t .up`), with "
        "underflow kept as an absolute term and no overflow. `roundoff mixed theorem` proves a mixed bound for "
        "all inputs of the box; a second theorem, proved from it with FloatLib's `interval` tactic and the "
        "suite's lemmas, states |computed - exact| <= B |exact| with a number B that a script proposes and Lean "
        f"accepts or refuses. Lean accepted the first proposal for all {len(proved)} programs. "
        f"Against NumFuzz, FloatLib's bound is smaller in {less_nf} programs, equal (within 0.5 percent) in "
        f"{same_nf} and larger in {more_nf}; the ratio FloatLib / NumFuzz runs from {low_nf:.2f} to "
        f"{high_nf:.2f}. Against Gappa it is smaller in {less_ga}, equal in {same_ga} and larger in {more_ga}; "
        f"the ratio runs from {low_ga:.2f} to {high_ga:.2f}. Against FPTaylor ({n_ft} programs with a bound) it "
        f"is smaller in {less_ft}, equal in {same_ft} and larger in {more_ft}. NumFuzz's bound holds for all "
        "positive inputs and assumes no underflow or overflow; FloatLib's, Gappa's and FPTaylor's hold on the "
        f"box. FPTaylor's bounds beyond the axis (open triangles at the right edge, as relative errors): "
        f"{listing}. The cross marks the program for which the paper reports that FPTaylor failed "
        f"({', '.join(program for _, program in failed)}). "
        f"FloatLib's time is {spread(total)} seconds per program (median {statistics.median(total):.1f}) for "
        f"the forward theorem and the relative bound together, of which the forward theorem alone takes "
        f"{spread(forward)} seconds; it is Lean's elaboration time inside one process, as NumFuzz's and "
        "FPTaylor's times are the ones those tools print, while Gappa's is the wall-clock time of its process. "
        f"The whole Lean process, with start-up and the import of FloatLib and Mathlib, takes {spread(wall)} "
        "seconds. The numbers behind the marks (bounds as relative errors):\n\n" + "\n".join(lines) + "\n\n"
        "From Lean: FloatLib's bounds (each is the statement of a theorem that depends only on propext, "
        "Classical.choice and Quot.sound) and its times. From scripts: the proposal of each bound, the "
        "ratios and the transcription of the paper's Table 3 (arXiv:2405.04612v2). Source: "
        "`results/programs.csv`.")
    done(fig, "01-bounds-and-time", caption, alt, notes)


# ---------------------------------------------------------------------------------------------
# Figure 2: long sums, Horner forms and dot products
# ---------------------------------------------------------------------------------------------

def weight(exponent: int) -> float:
    """`((1 + 2^-52)^exponent - 1) / 2^-52`, exactly, then as a float."""
    u = Fraction(1, 2 ** 52)
    return float(((1 + u) ** exponent - 1) / u)


def fig_loops(summary: dict) -> None:
    table = programs()
    panels = [
        ("Horner's rule", "fused multiply-adds", "horner",
         ["Horner2", "Horner5", "Horner10", "Horner20", "large_benchmarks/Horner50",
          "large_benchmarks/Horner75", "large_benchmarks/Horner100"], [2, 5, 10, 20, 100]),
        ("dot product", "pairs", "dot",
         ["large_benchmarks/mat_mul_4", "large_benchmarks/mat_mul_8", "large_benchmarks/mat_mul_16",
          "large_benchmarks/mat_mul_64", "large_benchmarks/mat_mul_128"], [4, 8, 16, 64, 128]),
        ("serial sum", "additions", "sum",
         ["test02_sum8", "large_benchmarks/serial_sum"], [7, 100, 1023]),
    ]
    top = 3000.0
    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 5.8), sharey=True)
    facts, beyond, missing = [], [], []
    for index, (ax, (title, unit, kind, members, ticks)) in enumerate(zip(axes, panels)):
        missing += [name for name in members if not table[name].get("loop_bound")]
        rows = [table[name] for name in members if table[name].get("loop_bound")]
        sizes = [int(row["loop_exponent"]) for row in rows]
        low, high = min(sizes), max(sizes)
        grid = sorted({low, high, *range(low, high + 1, max(1, (high - low) // 200))})
        ax.plot(grid, [weight(n) for n in grid], zorder=3, **look("FloatLib", marker="none", label=None))
        ax.plot(sizes, [float(row["loop_bound"]) / U for row in rows], zorder=6,
                **look("FloatLib", linestyle="none", label=None))
        for name, column, zorder, extra in (("NumFuzz", "numfuzz_bound", 5, dict(markersize=fs.MARKER + 6)),
                                            ("Gappa", "gappa_bound", 4, dict(markersize=fs.MARKER + 9,
                                                                             markerfacecolor="none")),
                                            ("FPTaylor", "fptaylor_bound", 4, {}),
                                            ("literature", "literature_bound", 7, dict(markersize=fs.MARKER + 3))):
            points = [(n, float(row[column]) / U) for n, row in zip(sizes, rows) if number(row[column]) is not None]
            inside = [(n, v) for n, v in points if v <= top]
            outside = [(n, v) for n, v in points if v > top]
            if inside:
                ax.plot([n for n, _ in inside], [v for _, v in inside], zorder=zorder,
                        **look(name, linestyle="none", label=None, **extra))
            if outside:
                beyond += [(name, title, n, v * U) for n, v in outside]
                ax.plot([n for n, _ in outside], [top] * len(outside), clip_on=False, zorder=zorder,
                        **look(name, linestyle="none", marker="^", markerfacecolor="white", label=None))
        for n, row in zip(sizes, rows):
            facts.append((title, row["program"], n, float(row["loop_bound"]), row["numfuzz_bound"],
                          row["literature_bound"], row["gappa_bound"], row["fptaylor_bound"], row["loop_link"]))
        log_axis(ax, "x", ticks)
        ax.set_xlim(low / 1.5, high * 1.5)
        ax.set_xlabel(unit)
        fs.value_grid(ax, "y")
        fs.panel_title(ax, index, title)
        ax.title.set_position((0.0, 1.0))
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=fs.SIZE_PANEL, color=fs.INK, pad=20)
    assert not missing, f"no loop bound for {missing}; the texts assume every size"
    assert all(name == "FPTaylor" for name, *_ in beyond)
    log_axis(axes[0], "y", [1, 10, 100, 1000])
    axes[0].set_ylim(1.2, top)
    axes[0].set_ylabel("relative error bound\n(units of $2^{-52}$)")
    handles = [handle("FloatLib"), handle("NumFuzz", linestyle="none"), handle("Gappa", linestyle="none",
                                                                             markerfacecolor="none"),
               handle("FPTaylor", linestyle="none"), handle("literature"),
               Line2D([], [], linestyle="none", marker="^", color=fs.GREY_LIGHT, markerfacecolor="white",
                      markersize=fs.MARKER, markeredgewidth=2.0, label="beyond the axis")]
    fig.subplots_adjust(left=0.135, right=0.97, top=0.75, bottom=0.14, wspace=0.1)
    fs.legend(fig, handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.55, 0.87),
              columnspacing=1.4)

    # ---- the facts of the caption, the alt text and the notes
    def agree(mine: float, text: str) -> bool:
        """Whether `mine` is the published number to the three digits the paper prints (the paper
        truncates or rounds, so either is accepted)."""
        return bool(text) and abs(mine / float(text) - 1) < 0.004

    by_panel = {title: [fact for fact in facts if fact[0] == title] for title, *_ in panels}
    horner, dot, serial = (by_panel[title] for title, *_ in panels)
    same_nf = {title: sum(agree(fact[3], fact[4]) for fact in items if fact[4])
               for title, items in by_panel.items()}
    with_nf = {title: sum(bool(fact[4]) for fact in items) for title, items in by_panel.items()}
    same_std = {title: sum(agree(fact[3], fact[5]) for fact in items if fact[5])
                for title, items in by_panel.items()}
    with_std = {title: sum(bool(fact[5]) for fact in items) for title, items in by_panel.items()}
    dot_ratio = [float(fact[4]) / fact[3] for fact in dot if fact[4]]
    links = sum(fact[8] == "proved" for fact in facts)
    assert same_nf["Horner's rule"] == with_nf["Horner's rule"] and same_nf["serial sum"] == with_nf["serial sum"]
    assert same_std["dot product"] == with_std["dot product"]
    listing = "; ".join(f"{title} of size {n}: {value:.3g}" for _, title, n, value in beyond)
    lines = ["| family | program | size | FloatLib | NumFuzz | textbook | Gappa | FPTaylor | loop = program |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for title, program, n, bound, nf, std, ga, ft, link in facts:
        lines.append(f"| {title} | {program} | {n:,} | {bound:.4e} | {nf or '-'} | {std or '-'} | {ga or '-'} | "
                     f"{ft or '-'} | {link or '-'} |")
    caption = (
        "Relative error bounds of the three families of long programs in the NumFuzz benchmarks, against their "
        "size: (a) Horner's rule with one fused multiply-add per coefficient, (b) the dot product behind each "
        "entry of a matrix product, (c) the serial sum. The line is FloatLib's theorem for every size, proved once "
        "by `roundoff` on the program written as a loop, and its circles are the sizes of the benchmarks; the other "
        "marks are the published bounds of NumFuzz, Gappa and FPTaylor and the textbook bounds that the paper "
        "quotes. Bounds are in units of 2^-52, both axes are logarithmic, and lower is tighter. "
        "FloatLib's bounds hold for nonnegative binary64 inputs when no operation overflows, with the underflow of "
        "the products and fused operations kept as a separate absolute term; they come from Lean 4.36.0 on "
        "FloatLib 4.1.0, run on "
        f"2026-11-21 on {MACHINE}.")
    alt = (
        "Three panels with logarithmic axes and one blue line each, rising one for one with the size: from "
        f"{horner[0][2]} to {horner[-1][2]} fused multiply-adds, from {dot[0][2]} to {dot[-1][2]} pairs and from "
        f"{serial[0][2]} to {serial[-1][2]:,} additions. In the first and third panels the circles on the line "
        "sit inside the NumFuzz squares at every size that has one, inside Gappa diamonds at the small sizes and "
        "under textbook plus signs at the large ones; open triangles at the top edge of the first panel mark "
        "FPTaylor's bounds, and one FPTaylor triangle stands far above the line at the small sum. In the middle "
        "panel the plus signs sit on the circles and the NumFuzz squares lie about a factor two above the line.")
    notes = (
        "FloatLib: `roundoff closed theorem` proves a closed bound for the loop at every length (a `List.foldl` "
        "of fused multiply-adds, a `Fin.foldl` of products and additions, a `List.foldl` of additions, each "
        "operation the IEEE 754 operation of `Binary64` rounded toward plus infinity); the suite's corollaries, "
        "proved in Lean from that theorem, give the relative form with the weight (1+u)^n - 1 for n fused "
        "operations, n additions or n pairs, and Lean checks the decimal value of the weight at each benchmark "
        f"size. For {'each' if links == len(facts) else links} of the {len(facts)} programs drawn Lean also "
        "checks by `rfl` that the loop at the benchmark's size is the straight-line program that FloatLib's "
        "FPBench pipeline makes of the artifact's file. `roundoff closed` does not follow a loop that starts from a computed value, so the dot product "
        "is proved as a loop from an input value and the corollary puts the first product in its place. "
        f"Marks that coincide are drawn one inside the other. FloatLib's weight is NumFuzz's bound to the "
        f"three digits the paper prints at {same_nf[panels[0][0]]} of {with_nf[panels[0][0]]} sizes of Horner's "
        f"rule and {same_nf[panels[2][0]]} of {with_nf[panels[2][0]]} of the serial sum. For the dot product "
        f"FloatLib's weight is the textbook bound n u at {same_std[panels[1][0]]} of {with_std[panels[1][0]]} "
        f"sizes, and NumFuzz's bound (2n - 1) u is {min(dot_ratio):.2f} to {max(dot_ratio):.2f} times it. "
        f"Beyond the axis (as relative errors): {listing}. NumFuzz, Gappa and FPTaylor analyze real inputs and "
        "rounded real operations without overflow; FloatLib's loop theorems are about the IEEE 754 operations "
        "on binary64 values and carry the condition that no step overflows. The numbers behind the marks, as "
        "relative errors:\n\n" + "\n".join(lines) + "\n\n"
        "From Lean: FloatLib's theorems, the decimal weights and the `rfl` checks. From scripts: the proposal "
        "of each decimal weight (rounded up to five digits) and the transcription of the paper's Tables 3 and 4. "
        "Source: `results/programs.csv`.")
    done(fig, "02-sums-horner-dot", caption, alt, notes)


def write_captions() -> None:
    lines = ["# The NumFuzz benchmarks with roundoff: figure captions", "",
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
    summary = json.loads((RESULTS / "summary.json").read_text())
    fig_small(summary)
    fig_loops(summary)
    write_captions()


if __name__ == "__main__":
    main()
