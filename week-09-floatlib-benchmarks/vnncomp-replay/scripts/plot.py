"""The two figures of the VNN-COMP 2025 counterexample replay.

    python3 scripts/plot.py [RESULTS [FIGURES]]

Reads `results/families.csv` and `results/margins.csv` (written by `scripts/family_summary.py`) and writes
`figures/NN-NAME.png`, `.svg` and `figures/CAPTIONS.md`.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plots"))
from benchplot import MACHINE, Suite, fs, plt  # noqa: E402

suite = Suite(__file__)
read, done = suite.read, suite.done

# one colour per verdict in both figures
EXACT, PROJECTED, ROUNDED, DEPENDS, REJECTED = fs.BLUE, fs.SKY, fs.GREEN, fs.ORANGE, fs.VERMILION
CLASSES = [("as_reported", EXACT, "counterexample as reported"),
           ("moved_into_the_box", PROJECTED, "after moving into the box"),
           ("rounded_box", ROUNDED, "for the rounded box"),
           ("depends_on_rounding", DEPENDS, "depends on rounding"),
           ("rejected", REJECTED, "rejected")]
MAIN = ["acasxu_2023", "soundnessbench"]


def listing(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def in_prose(row: dict) -> str:
    """The family's name inside a sentence."""
    return {"cora_2024": "the CORA benchmark", "test": "the test family"}.get(row["family"], row["title"])


# ---------------------------------------------------------------------------------------------
# Figure 1: verdicts per benchmark family
# ---------------------------------------------------------------------------------------------

def fig_verdicts() -> None:
    families = [row for row in read("families.csv") if int(row["files_replayed"]) > 0]
    rows = [(row["title"], [int(row[column]) for column, _, _ in CLASSES]) for row in families]
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.76 * (len(rows) + 0.3) + 2.35))
    for position, (label, parts) in enumerate(rows):
        y = len(rows) - 1 - position
        total = sum(parts)
        left = 0.0
        for value, (_, colour, _) in zip(parts, CLASSES):
            share = 100.0 * value / total
            if value:
                ax.barh(y, share, left=left, height=0.42, color=colour, edgecolor=colour,
                        linewidth=fs.LW_AXIS, zorder=2)
            left += share
        ax.text(0.0, y + 0.29, label, ha="left", va="bottom", fontsize=fs.SIZE_SMALL, color=fs.INK)
        ax.text(101.5, y, f"{sum(parts[:3]):,} of {total:,}", ha="left", va="center",
                fontsize=fs.SIZE_SMALL, color=fs.INK)
    ax.set_ylim(-0.45, len(rows) - 0.15)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    fs.value_grid(ax, "x")
    ax.set_xlabel("share of the counterexample files (%)")
    handles = [Patch(facecolor=colour, edgecolor=colour, label=label) for _, colour, label in CLASSES]
    height = fig.get_figheight()
    fs.legend(fig, handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.45, 1.0 - 0.95 / height),
              handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.03, right=0.79, top=1.0 - 1.0 / height, bottom=0.9 / height)

    def total(column: str) -> int:
        return sum(int(row[column]) for row in families)

    lines = ["| family | files | as reported | after moving into the box | for the rounded box | "
             "depends on rounding | rejected |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in families:
        lines.append(f"| {row['title']} | {row['files_replayed']} | {row['as_reported']} | "
                     f"{row['moved_into_the_box']} | {row['rounded_box']} | {row['depends_on_rounding']} | "
                     f"{row['rejected']} |")
    done(fig, "01-counterexample-verdicts",
         "What exact evaluation says about the counterexamples that the verifiers of VNN-COMP 2025 published "
         f"for {listing([in_prose(row) for row in families])}. A bar is the counterexample files of one benchmark "
         "family, split into the files whose reported point is a counterexample, those whose point becomes one "
         "after each coordinate outside the input box is moved to the nearest binary32 number inside, those "
         "that are counterexamples only for the box with its bounds rounded to binary32, those whose verdict "
         "depends on rounding, and those rejected; the number at its end is the files confirmed in one of the "
         "first three ways. Each network was evaluated with FloatLib's compiled binary32 operations in four "
         "orders of summation and in exact arithmetic, and the property was decided on exact rationals. "
         f"Evaluated on 2026-11-21 with FloatLib and Lean 4.36.0 on {MACHINE}.",
         f"{len(rows)} horizontal bars on an axis from 0 to 100 percent, one per benchmark family. Most bars are "
         "dark blue, the reported point is a counterexample, and light blue, the point had to be moved into "
         "the box, in varying shares. The SoundnessBench bar is light blue with a short orange end, depends on "
         "rounding; the SafeNLP bar ends in a red part of about a sixth, rejected; the Collins RUL CNN bar is "
         "mostly green, rounded box. Each bar ends in its count.",
         f"{total('files_replayed'):,} counterexample files with {total('points_replayed'):,} points: "
         f"{total('as_reported'):,} confirmed as reported, {total('moved_into_the_box'):,} after moving into the "
         f"box, {total('rounded_box'):,} for the rounded box, {total('depends_on_rounding'):,} depend on "
         f"rounding, {total('rejected'):,} rejected.\n\n" + "\n".join(lines) + "\n\n"
         "A point is judged at the reported point when that lies in the box and otherwise at the moved point. "
         "It is confirmed when the four binary32 orders (left to right, right to left, pairwise, fused "
         "multiply-add) and exact arithmetic all violate the property there, rejected when none does, and "
         "depends on rounding when they disagree. The rounded box is the box with every bound rounded to the "
         "nearest binary32 number; it is used only in a coordinate where the box itself holds no binary32 "
         "number. A file has the best verdict of its points. From Lean: every evaluation and every comparison "
         "(the compiled replay). From scripts: the translation of the ONNX and VNNLIB files, the counts and "
         "the drawing. Source: `results/families.csv`.")


# ---------------------------------------------------------------------------------------------
# Figure 2: margin of the violation against the rounding error of the evaluation
# ---------------------------------------------------------------------------------------------

def plain_power(value, _=None) -> str:
    return f"$10^{{{round(math.log10(value))}}}$"


FLOOR_X, FLOOR_Y = 1e-10, 1e-9      # the axes of figure 2 start here


def fig_margins() -> None:
    groups: dict[str, list] = {}
    skipped = tiny = 0
    for row in read("margins.csv"):
        if not row["margin"]:
            continue
        error, margin = float(row["rounding_error"]), float(row["margin"])
        if margin == 0 or error == 0:
            skipped += 1
            continue
        if error < FLOOR_X or margin < FLOOR_Y:
            tiny += 1
            continue
        if row["verdict"] in ("rejected", "outside"):
            kind = "rejected"
        elif row["verdict"] == "rounding_dependent":
            kind = "depends"
        else:
            kind = row["family"] if row["family"] in MAIN else "others"
        groups.setdefault(kind, []).append((error, margin))
    look = {"others": dict(marker="o", markersize=3.2, markerfacecolor=fs.LINE, markeredgecolor=fs.LINE,
                           markeredgewidth=0.0, zorder=2),
            "acasxu_2023": dict(marker="o", markersize=6.5, markerfacecolor="none", markeredgecolor=fs.BLUE,
                                markeredgewidth=1.3, zorder=3),
            "soundnessbench": dict(marker="s", markersize=6.5, markerfacecolor="none", markeredgecolor=fs.PURPLE,
                                   markeredgewidth=1.3, zorder=3),
            "rejected": dict(marker="x", markersize=6.5, markeredgecolor=REJECTED, markeredgewidth=1.5, zorder=4),
            "depends": dict(marker="D", markersize=6.0, markerfacecolor=DEPENDS, markeredgecolor=fs.INK,
                            markeredgewidth=0.8, zorder=5)}
    labels = {"others": "other families, confirmed", "acasxu_2023": "ACAS Xu, confirmed",
              "soundnessbench": "SoundnessBench, confirmed", "rejected": "rejected",
              "depends": "depends on rounding"}
    order = [k for k in ("acasxu_2023", "soundnessbench", "others", "rejected", "depends") if k in groups]
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 8.2))
    handles = []
    for kind in ("others", "acasxu_2023", "soundnessbench", "rejected", "depends"):
        if kind in groups:
            xs, ys = zip(*groups[kind])
            ax.plot(xs, ys, linestyle="none", **look[kind])
    for kind in order:
        style = dict(look[kind])
        style["markersize"] = max(style["markersize"], 8.0)
        handles.append(Line2D([], [], linestyle="none", label=labels[kind], **style))
    every = [point for points in groups.values() for point in points]
    low_x, low_y = FLOOR_X, FLOOR_Y
    high_x = 10.0 ** math.ceil(math.log10(max(p[0] for p in every)))
    high_y = 10.0 ** math.ceil(math.log10(max(p[1] for p in every)))
    low, high = min(low_x, low_y), max(high_x, high_y)
    ax.plot([low, high], [low, high], color=fs.GREY_DARK, linewidth=fs.LW_REF, linestyle="--", zorder=1)
    handles.append(Line2D([], [], color=fs.GREY_DARK, linewidth=fs.LW_REF, linestyle="--",
                          label="margin = rounding error"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low_x, high_x)
    ax.set_ylim(low_y, high_y)
    for axis, (a, b) in ((ax.xaxis, (low_x, high_x)), (ax.yaxis, (low_y, high_y))):
        exponents = list(range(round(math.log10(a)), round(math.log10(b)) + 1))
        step = 1 if len(exponents) <= 8 else 2
        axis.set_major_locator(FixedLocator([10.0 ** e for e in exponents][::step]))
        axis.set_major_formatter(FuncFormatter(plain_power))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("rounding error of the binary32 output")
    ax.set_ylabel("margin of the property in exact arithmetic")
    fs.value_grid(ax, "both")
    fs.legend(fig, handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.53, 0.895),
              columnspacing=1.4, handletextpad=0.3)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.875, bottom=0.105)

    def span(points):
        return (f"{min(p[0] for p in points):.1e} to {max(p[0] for p in points):.1e}",
                f"{min(p[1] for p in points):.1e} to {max(p[1] for p in points):.1e}",
                f"{statistics.median(p[1] / p[0] for p in points):.3g}",
                sum(1 for p in points if p[1] < p[0]))

    lines = ["| group | files | rounding error | margin | median of margin / rounding error | below the diagonal |",
             "| --- | ---: | --- | --- | ---: | ---: |"]
    for kind in order:
        a, b, c, d = span(groups[kind])
        lines.append(f"| {labels[kind]} | {len(groups[kind]):,} | {a} | {b} | {c} | {d} |")
    below = sum(1 for p in every if p[1] < p[0])
    done(fig, "02-margin-against-rounding",
         "How decisive each published counterexample is: the margin by which the network's exact output "
         "violates, or fails to violate, the property, against the rounding error of the binary32 evaluation. "
         "A marker is one counterexample file; the margin is the absolute slack of the output property at the "
         "exact rational output, and the rounding error is the largest absolute difference between a binary32 "
         "output and that exact output, both in the units of the network's output. Both axes are logarithmic. "
         f"Evaluated on 2026-11-21 with FloatLib and Lean 4.36.0 on {MACHINE}.",
         "A scatter plot on two logarithmic axes with a dashed diagonal where the margin equals the rounding "
         "error. The confirmed files of ACAS Xu and of the other families lie far above the diagonal, by a "
         "factor of more than a hundred for all but two of them; those of SoundnessBench lie on both sides of "
         "it, with margins between a hundredth of the rounding error and ten times it. The orange diamonds, "
         "files whose verdict depends on rounding, lie along the diagonal, on both sides. "
         "The red crosses, rejected files, form two groups: one far above the diagonal at margins from a "
         "thousandth to ten, and one just above it at margins below one millionth.",
         f"{len(every):,} files drawn, {below:,} of them below the diagonal. Not drawn: {skipped:,} files whose "
         f"rounding error or margin is exactly zero, and {tiny:,} files with a rounding error below 1e-10 or a "
         "margin below 1e-9 (ties to within 1e-13). A file is drawn at the point "
         "that decides its verdict (the reported point, or the point moved into the box). A verdict depends on "
         "rounding when the four binary32 orders (left to right, right to left, pairwise, fused multiply-add) "
         "and exact arithmetic do not all agree on whether the output violates the property; for a rejected "
         "file the margin is the distance to a violation.\n\n" + "\n".join(lines) + "\n\n"
         "From Lean: the binary32 outputs, the exact outputs (dyadic arithmetic without rounding) and the "
         "slacks. From scripts: the absolute values, the grouping and the drawing. Source: "
         "`results/margins.csv`.")


if __name__ == "__main__":
    fig_verdicts()
    fig_margins()
    suite.write_captions("Exact replay of VNN-COMP 2025 counterexamples")
