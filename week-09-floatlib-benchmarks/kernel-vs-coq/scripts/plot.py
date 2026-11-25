"""Lean's kernel with FloatLib against Coq with Flocq, per operation and width.

    python3 scripts/plot.py [RESULTS [FIGURES]]

Reads the files of `results/` and writes `figures/NN-NAME.png`, `.svg` and `figures/CAPTIONS.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path
import statistics

from matplotlib.lines import Line2D

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plots"))
from benchplot import MACHINE, OPERATIONS, Suite, fs, log_axis, plt, width_axis  # noqa: E402

suite = Suite(__file__)
read, done = suite.read, suite.done


def fig_coq() -> None:
    data = read("kernel-coq.csv")
    facts = suite.facts()
    operations = [op for op in OPERATIONS if any(row["operation"] == op for row in data)]
    widths = sorted({int(row["bits"]) for row in data})
    series = (("lean_kernel_us", "FloatLib", "FloatLib, Lean's kernel"),
              ("flocq_vm_us", "Flocq", "Coq + Flocq, vm_compute"),
              ("flocq_lazy_us", "Flocq lazy", "Coq + Flocq, lazy"))
    milli = lambda text: float(text) / 1000.0          # the files hold microseconds
    values = [milli(row[column]) for row in data for column, _, _ in series if row[column]]
    top = max(values) * 2.4
    fig, axes = plt.subplots(3, 2, figsize=(fs.WIDTH, 10.6), sharex=True, sharey=True)
    absent = 0
    for index, (ax, op) in enumerate(zip(axes.flat, operations)):
        selected = sorted((row for row in data if row["operation"] == op), key=lambda row: int(row["bits"]))
        for column, style, label in series:
            points = [(int(row["bits"]), milli(row[column]) if row[column] else None) for row in selected]
            ax.plot([x for x, y in points if y], [y for _, y in points if y],
                    zorder=3 if style == "FloatLib" else 2, **fs.series(style, label=label))
            gone = [x for x, y in points if not y]
            if gone:
                absent += len(gone)
                ax.plot(gone, [top] * len(gone), linestyle="none", marker="x", color=fs.SERIES[style]["color"],
                        markersize=fs.MARKER, markeredgewidth=2.0, clip_on=False, zorder=4)
        width_axis(ax, widths)
        log_axis(ax, "y", [0.1, 1, 10, 100, 1000, 10000])
        fs.value_grid(ax)
        fs.panel_title(ax, index, fs.OPERATION_LABEL[op])
        ax.tick_params(labelbottom=True)
    axes.flat[0].set_ylim(min(values) / 2.2, top * 1.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("time per operation (ms)")
    spare = list(axes.flat)[len(operations):]
    for ax in spare:
        ax.axis("off")
    for ax in list(axes.flat)[len(operations) - 2:len(operations)]:
        ax.set_xlabel("format width (bits)")
    handles = [fs.series_handle(style, label=label) for _, style, label in series]
    if absent:
        handles.append(Line2D([], [], linestyle="none", marker="x", color=fs.MUTED, markersize=fs.MARKER,
                              markeredgewidth=2.0, label="Coq: no result"))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.965, bottom=0.075, hspace=0.3, wspace=0.09)
    box = spare[0].get_position()
    fs.legend(fig, handles=handles, loc="center left", bbox_to_anchor=(box.x0 + 0.02, box.y0 + box.height * 0.5),
              labelspacing=0.9)

    lean = [float(row["lean_kernel_us"]) for row in data if row["lean_kernel_us"]]
    vm = [float(row["flocq_vm_us"]) / float(row["lean_kernel_us"]) for row in data
          if row["flocq_vm_us"] and row["lean_kernel_us"]]
    lazy = [float(row["flocq_lazy_us"]) / float(row["lean_kernel_us"]) for row in data
            if row["flocq_lazy_us"] and row["lean_kernel_us"]]
    assert len(operations) == 5
    done(fig, "01-kernel-coq-flocq",
         "Time of one floating-point operation inside the proof checker: FloatLib in Lean's kernel against "
         "Flocq 4.2.2 in Coq 8.20.1 under vm_compute and under lazy, from 32 to 4,096 bits. A point is the time "
         "per operation, in milliseconds, in a loop of 100 operations (10 from 512 bits on) on the same operands; "
         "a cross at the top marks a width where Coq gave no result within 8 GB and 300 seconds. Both axes are "
         f"logarithmic, and lower is faster. The Lean times are the median of {facts['runs']} runs of FloatLib "
         f"4.1.0 on {facts['date']} with Lean {facts['lean']} on {MACHINE}; the Coq times are the recorded run of "
         "2026-11-15 on the same machine, in a container with 8 GB and two CPUs, and do not depend on FloatLib.",
         "Five panels of lines, one per operation, against format width from 32 to 4,096 bits on logarithmic "
         "axes: the FloatLib line is nearly flat near the bottom of each panel, the vm_compute line starts just "
         "above it, climbs steeply and ends at 256 or 512 bits, the lazy line starts higher, climbs more slowly "
         "and reaches 4,096 bits, and crosses along the top edge mark the widths where Coq gave no result.",
         f"Lean: FloatLib's kernel path, the term that `float_decide` checks (`#kernel_cost rewritten`, "
         f"`benchmarks/suite/matched.py`); it takes {min(lean):.0f} to {max(lean):.0f} microseconds per operation "
         "over all widths and operations. Coq: Flocq's `Bplus`, `Bmult`, `Bdiv`, `Bsqrt` and `Bfma` "
         "(`BinarySingleNaN`, `mode_NE`); in the recorded run Coq printed the value that Lean computed in every "
         f"row. Where Coq gave a result, vm_compute takes {min(vm):.1f} to {max(vm):,.0f} times Lean's time "
         f"(median {statistics.median(vm):.0f}, {len(vm)} cells) and lazy {min(lazy):.0f} to {max(lazy):,.0f} times "
         f"(median {statistics.median(lazy):.0f}, {len(lazy)} cells); Coq gave no result in {absent} of "
         f"{2 * len(data)} cells. From Lean: the Lean times (`Lean.Kernel.whnf`). From scripts: the Coq times "
         "(printed by Coq in a container, `benchmarks/results/figures/recorded/coq-flocq.csv` of "
         "FloatLib's repository) and the ratios. Source: `results/kernel-coq.csv`.")


if __name__ == "__main__":
    fig_coq()
    suite.write_captions("Lean's kernel against Coq with Flocq")
