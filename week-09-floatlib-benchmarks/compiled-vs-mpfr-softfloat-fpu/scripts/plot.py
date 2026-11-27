"""Compiled code: FloatLib's public operations next to the host FPU, SoftFloat and MPFR.

    python3 scripts/plot.py [RESULTS [FIGURES]]

Reads the files of `results/` and writes `figures/NN-NAME.png`, `.svg` and `figures/CAPTIONS.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path
import statistics

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plots"))
from benchplot import MACHINE, OPERATIONS, Suite, fs, log_axis, plt, width_axis  # noqa: E402

suite = Suite(__file__)
read, done = suite.read, suite.done


def compiled() -> dict:
    table = {}
    for row in read("compiled.csv"):
        table[(row["series"], int(row["bits"]), row["operation"])] = row
    return table


def run_facts() -> dict:
    """The facts of the comparison run that the captions quote (results/compiled-run.csv)."""
    return suite.facts("compiled-run.csv")


def fig_public() -> None:
    table, facts = compiled(), run_facts()
    order = [("FloatLib public", "FloatLib", "FloatLib"),
             ("MPFR", "MPFR", f"MPFR {facts['mpfr']}"),
             ("SoftFloat", "SoftFloat", "Berkeley SoftFloat 3e"),
             ("native FPU", "native C", "host FPU")]
    offsets = [-0.285, -0.095, 0.095, 0.285]
    fig, axes = plt.subplots(2, 1, figsize=(fs.WIDTH, 9.4), sharex=True, sharey=True)
    drawn = [float(table[(name, bits, op)]["max_ns"]) for name, _, _ in order for bits in (32, 64)
             for op in OPERATIONS]
    assert min(float(table[(name, bits, op)]["min_ns"]) for name, _, _ in order for bits in (32, 64)
               for op in OPERATIONS) > 5.5
    top = max(drawn) * 1.3
    handles = []
    for index, (ax, bits) in enumerate(zip(axes, (32, 64))):
        for (name, style, label), offset in zip(order, offsets):
            rows = [table[(name, bits, op)] for op in OPERATIONS]
            xs = [position + offset for position in range(len(OPERATIONS))]
            median = [float(row["median_ns"]) for row in rows]
            low = [float(row["median_ns"]) - float(row["min_ns"]) for row in rows]
            high = [float(row["max_ns"]) - float(row["median_ns"]) for row in rows]
            look = fs.series(style, label=label, linestyle="none")
            ax.errorbar(xs, median, yerr=[low, high], ecolor=look["color"], elinewidth=fs.LW_REF,
                        capsize=3.5, capthick=fs.LW_REF, zorder=3, markeredgewidth=2.0, **look)
            if index == 0:
                handles.append(fs.series_handle(style, label=label, linestyle="none", markeredgewidth=2.0))
        for boundary in range(1, len(OPERATIONS)):
            ax.axvline(boundary - 0.5, color=fs.GRID, linewidth=fs.LW_GRID, zorder=1)
        log_axis(ax, "y", [5, 10, 20, 50, 100, 200])
        ax.set_ylim(5.5, top)
        ax.set_xlim(-0.5, len(OPERATIONS) - 0.5)
        ax.set_xticks(range(len(OPERATIONS)), OPERATIONS)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("time per operation (ns)")
        fs.value_grid(ax)
        fs.panel_title(ax, index, f"binary{bits}")
    axes[-1].set_xlabel("operation")
    fs.legend(fig, handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.53, 0.985),
              columnspacing=1.4, handletextpad=0.2)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.93, bottom=0.085, hspace=0.2)

    def cell(name, bits, op):
        return float(table[(name, bits, op)]["median_ns"])

    def span(name):
        values = [cell(name, bits, op) for bits in (32, 64) for op in OPERATIONS]
        return f"{min(values):.0f} and {max(values):.0f}"

    ratios_mpfr = [cell("FloatLib public", bits, op) / cell("MPFR", bits, op) for bits in (32, 64)
                   for op in OPERATIONS]
    ratios_soft = [cell("FloatLib public", bits, op) / cell("SoftFloat", bits, op) for bits in (32, 64)
                   for op in OPERATIONS]
    ratios_fpu = [cell("FloatLib public", bits, op) / cell("native FPU", bits, op) for bits in (32, 64)
                  for op in OPERATIONS]
    faster = sum(ratio < 1 for ratio in ratios_mpfr)
    lines = ["| format | operation | FloatLib | MPFR | SoftFloat | host FPU |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for bits in (32, 64):
        for op in OPERATIONS:
            cells = []
            for name in ("FloatLib public", "MPFR", "SoftFloat", "native FPU"):
                row = table[(name, bits, op)]
                cells.append(f"{float(row['median_ns']):.1f} ({float(row['min_ns']):.1f} to {float(row['max_ns']):.1f})")
            lines.append(f"| binary{bits} | {op} | " + " | ".join(cells) + " |")
    done(fig, "01-public-binary32-64",
         "Time of FloatLib's public binary32 and binary64 operations in compiled code, next to the host "
         f"processor's own arithmetic, Berkeley SoftFloat 3e and MPFR {facts['mpfr']}. A marker is the median "
         f"of {facts['trials']} trials of the time per operation, in nanoseconds, of a loop in which each result selects "
         "the next operands, and its whisker runs from the fastest to the slowest trial; the operations are "
         "addition, subtraction, multiplication, division, square root and fused multiply-add, rounded to "
         "nearest even. The time axis is logarithmic, and lower is faster. "
         f"Measured on FloatLib on {facts['date']} in one run on one pinned core of {MACHINE} with Lean "
         f"{facts['lean']}.",
         "Two panels, binary32 above binary64, each with six groups of four markers, FloatLib, MPFR, "
         f"SoftFloat and the host FPU from left to right, on a logarithmic time axis: in every group the host "
         f"FPU is lowest, between {span('native FPU')} nanoseconds, SoftFloat "
         f"lies between {span('SoftFloat')}, MPFR between {span('MPFR')} and FloatLib between "
         f"{span('FloatLib public')}.",
         "FloatLib's series is the public operation as a user writes it (`+`, `-`, `*`, `/`, `sqrt`, `fma` on "
         "`Binary32` and `Binary64`, default planning policy), proved and run in software; the host FPU is the "
         "unproved reference. "
         f"Over the 12 cells FloatLib takes {min(ratios_mpfr):.2f} to {max(ratios_mpfr):.2f} times MPFR's time "
         f"(median {statistics.median(ratios_mpfr):.2f}, faster than MPFR in {faster} cells), "
         f"{min(ratios_soft):.2f} to {max(ratios_soft):.2f} times SoftFloat's (median "
         f"{statistics.median(ratios_soft):.2f}) and {min(ratios_fpu):.1f} to {max(ratios_fpu):.1f} times the host "
         f"FPU's (median {statistics.median(ratios_fpu):.1f}). The times include the loop: the lookup of the "
         "operands, the fingerprint of the result and the checksum. Median (fastest to slowest trial), "
         "nanoseconds:\n\n" + "\n".join(lines) + "\n\n"
         "From Lean: FloatLib's times (`IO.monoNanosNow` in the benchmark executable). From C programs in the "
         "same run: MPFR, SoftFloat and the host FPU (`clock_gettime`). From scripts: the medians and the "
         "drawing. Source: `results/compiled.csv`.")


def fig_widths() -> None:
    table, facts = compiled(), run_facts()
    widths = sorted({key[1] for key in table if key[0] == "MPFR"})
    fig, axes = plt.subplots(3, 2, figsize=(fs.WIDTH, 10.6), sharex=True, sharey=True)
    for index, (ax, op) in enumerate(zip(axes.flat, OPERATIONS)):
        for name, style, label in (("MPFR", "MPFR", f"MPFR {facts['mpfr']}"),
                                   ("FloatLib public", "FloatLib", "FloatLib")):
            rows = [table[(name, bits, op)] for bits in widths]
            median = [float(row["median_ns"]) for row in rows]
            low = [float(row["median_ns"]) - float(row["min_ns"]) for row in rows]
            high = [float(row["max_ns"]) - float(row["median_ns"]) for row in rows]
            look = fs.series(style, label=label)
            ax.errorbar(widths, median, yerr=[low, high], ecolor=look["color"], elinewidth=fs.LW_REF,
                        capsize=3, capthick=fs.LW_REF, zorder=3 if name != "MPFR" else 2, **look)
        width_axis(ax, widths)
        log_axis(ax, "y", [10, 100, 1000, 10000, 100000, 1000000])
        fs.value_grid(ax)
        fs.panel_title(ax, index, fs.OPERATION_LABEL[op])
    low_all = min(float(row["min_ns"]) for key, row in table.items() if key[0] in ("MPFR", "FloatLib public"))
    high_all = max(float(row["max_ns"]) for key, row in table.items() if key[0] in ("MPFR", "FloatLib public"))
    axes.flat[0].set_ylim(low_all / 1.6, high_all * 1.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("time per operation (ns)")
    for ax in axes[-1, :]:
        ax.set_xlabel("format width (bits)")
    handles = [fs.series_handle("FloatLib"), fs.series_handle("MPFR", label=f"MPFR {facts['mpfr']}")]
    fs.legend(fig, handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.54, 0.985))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.94, bottom=0.075, hspace=0.3, wspace=0.09)

    ratio = {(bits, op): float(table[("FloatLib public", bits, op)]["median_ns"])
             / float(table[("MPFR", bits, op)]["median_ns"]) for bits in widths for op in OPERATIONS}
    lines = ["| width | " + " | ".join(OPERATIONS) + " |", "| ---: |" + " ---: |" * len(OPERATIONS)]
    for bits in widths:
        lines.append(f"| {bits:,} | " + " | ".join(f"{ratio[(bits, op)]:.2f}" if ratio[(bits, op)] < 10
                                                   else f"{ratio[(bits, op)]:,.0f}" for op in OPERATIONS) + " |")
    wins = sorted((bits, op) for (bits, op), value in ratio.items() if value < 1)
    done(fig, "02-floatlib-mpfr-widths",
         f"Compiled time of FloatLib's binary operations against MPFR {facts['mpfr']} at the same precision, "
         "exponent range and rounding, from 32 to 4,096 bits. A marker is the median of "
         f"{facts['trials']} trials of the time per operation, in nanoseconds, of a loop in which each result selects "
         "the next operands, with a whisker from the fastest to the slowest trial. Both axes are logarithmic, "
         f"and lower is faster. Measured on FloatLib on {facts['date']} in one run on one pinned core of "
         f"{MACHINE} with Lean {facts['lean']}.",
         "Six panels, one per operation, each with a FloatLib line and an MPFR line against format width "
         "from 32 to 4,096 bits on logarithmic axes: the two lines are close at 32 and 64 bits, the FloatLib "
         "line is above at 128 bits, steps up to about two microseconds or more at 256 bits and is nearly flat "
         "from there, while the MPFR line rises steadily, so that the gap narrows towards 4,096 bits in every "
         "panel, least for addition and subtraction.",
         "FloatLib's series is the public operation on the configured binary type (`ExecFloat.add`, ...): "
         "`Binary32` and `Binary64` at 32 and 64 bits, binary128 at 128, and layouts with 19 exponent bits "
         "from 256 bits on, where FloatLib has one kernel per operation, the exact baseline on Lean's "
         "arbitrary-precision naturals. "
         f"FloatLib's median is below MPFR's in {len(wins)} of {len(ratio)} cells"
         + (": " + ", ".join(f"{op} at {bits}" for bits, op in wins) if wins else "")
         + ". FloatLib's time over MPFR's (medians):\n\n" + "\n".join(lines) + "\n\n"
         "From Lean: FloatLib's times. From a C program in the same run: MPFR's. From scripts: the medians, "
         "the ratios and the drawing. Source: `results/compiled.csv`.")


if __name__ == "__main__":
    fig_public()
    fig_widths()
    suite.write_captions("Compiled code against the host FPU, SoftFloat and MPFR")
