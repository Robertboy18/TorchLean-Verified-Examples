"""`decide +kernel` on FloatLib's kernel layer against `native_decide`.

    python3 scripts/plot.py [RESULTS [FIGURES]]

Reads the files of `results/` and writes `figures/NN-NAME.png`, `.svg` and `figures/CAPTIONS.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.lines import Line2D

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plots"))
from benchplot import MACHINE, NUMBER, Suite, fs, plt  # noqa: E402

suite = Suite(__file__)
read, done = suite.read, suite.done


def fig_native() -> None:
    data = read("kernel-native.csv")
    facts = suite.facts()
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.98 * len(data) + 1.5))
    grey = fs.GREY_DARK
    high = max(float(row[column]) for row in data for column in ("kernel_ms", "native_decide_ms") if row[column])
    for position, row in enumerate(data):
        y = len(data) - 1 - position
        kernel = float(row["kernel_ms"])
        native = float(row["native_decide_ms"]) if row["native_decide_ms"] else None
        if native is not None:
            ax.plot([kernel, native], [y, y], color=fs.LINE, linewidth=fs.LW_DATA + 1.5, zorder=1,
                    solid_capstyle="round")
            ax.plot([native], [y], linestyle="none", marker="X", color=grey, markersize=fs.MARKER + 3,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.plot([kernel], [y], linestyle="none", marker="o", color=fs.BLUE, markersize=fs.MARKER + 2, zorder=4)
        ax.text(0.0, y + 0.3, row["statement"], ha="left", va="bottom", fontsize=fs.SIZE_SMALL, color=fs.INK)
    ax.set_ylim(-0.5, len(data) - 0.05)
    ax.set_xlim(0, high * 1.06)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time to prove the statement (ms)")
    fs.value_grid(ax, "x")
    handles = [Line2D([], [], linestyle="none", marker="o", color=fs.BLUE, markersize=fs.MARKER + 2,
                      label="decide +kernel, FloatLib"),
               Line2D([], [], linestyle="none", marker="X", color=grey, markersize=fs.MARKER + 3,
                      markeredgecolor="white", markeredgewidth=0.8, label="native_decide")]
    fs.legend(fig, handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.985), handletextpad=0.2)
    fig.subplots_adjust(left=0.03, right=0.975, top=0.94, bottom=0.14)
    both = [(float(row["kernel_ms"]), float(row["native_decide_ms"])) for row in data if row["native_decide_ms"]]
    failed = [row["statement"] for row in data if not row["native_decide_ms"]]
    lines = ["| statement | `decide +kernel` | `native_decide` |", "| --- | ---: | ---: |"]
    for row in data:
        native = f"{float(row['native_decide_ms']):.1f} ms" if row["native_decide_ms"] else "fails"
        lines.append(f"| {row['statement']} | {float(row['kernel_ms']):.1f} ms | {native} |")
    done(fig, "01-kernel-native-decide",
         "Time to prove one closed statement by decide +kernel on FloatLib's kernel layer and by "
         "native_decide. A mark is the time Lean takes to elaborate one example, in milliseconds, elaboration "
         "included; native_decide compiles the statement and trusts the compiler, and a row without its mark "
         "is a statement on which it fails. The time axis is linear, and lower is faster. "
         f"Measured on FloatLib 4.1.0 on {facts['date']} (the median of {facts['runs']} runs) with Lean "
         f"{facts['lean']} on {MACHINE}.",
         f"{NUMBER[len(data)]} rows, one per closed statement, each with its name above a linear time axis in "
         "milliseconds: a blue circle for decide +kernel and a grey cross for native_decide joined by a bar, "
         f"all between 3 and {high:.0f} milliseconds; {NUMBER[len(failed)].lower()} row has no cross.",
         "The statements are closed Boolean statements on the functions of `Numerics.Kernel` "
         f"(`benchmarks/exact/native.py`); each time is the median of {facts['runs']} runs of FloatLib's "
         "runner, one sample per run. FloatLib never uses "
         f"`native_decide`. Over the {len(both)} statements that both prove, `decide +kernel` takes "
         f"{min(k / n for k, n in both):.2f} to {max(k / n for k, n in both):.2f} times the time of "
         f"`native_decide`, with no compiler in the trusted base; `native_decide` fails on "
         f"{'; '.join(failed) if failed else 'none'} (it cannot compile a definition that the statement "
         "depends on).\n\n" + "\n".join(lines) + "\n\n"
         "From Lean: every time. From scripts: nothing but the drawing. Source: `results/kernel-native.csv`.")


if __name__ == "__main__":
    fig_native()
    suite.write_captions("decide +kernel against native_decide")
