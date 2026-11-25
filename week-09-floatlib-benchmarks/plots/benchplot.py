"""What the plot scripts of this week share: the type sizes, CSV reading, axes and captions.

Every figure of the week is drawn by `SUITE/scripts/plot.py` from the files in `SUITE/results/` and written to
`SUITE/figures/` as PNG (9 inches wide at 200 dpi) and SVG, with `SUITE/figures/CAPTIONS.md` beside them. The look
is `figstyle.py` in this directory, with larger type and heavier marks because the figures are read on phones.
Nothing is measured in a plot script: every number drawn is read from a result file.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

# Larger type than the figures of FloatLib's guide (11, 10 and 9 points), heavier lines and marks.
fs.SIZE_LABEL = fs.SIZE_PANEL = 17.0
fs.SIZE_TICK = fs.SIZE_LEGEND = 15.0
fs.SIZE_SMALL = 14.0
fs.LW_DATA, fs.LW_REF, fs.LW_AXIS, fs.LW_GRID, fs.MARKER = 2.6, 1.5, 1.1, 0.8, 9.0
fs.setup()

OPERATIONS = ["add", "sub", "mul", "div", "sqrt", "fma"]
MACHINE = "an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs)"
NUMBER = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


class Suite:
    """The directories of one suite, seen from its `scripts/plot.py`, and the captions of its figures."""

    def __init__(self, script: str):
        self.root = Path(script).resolve().parents[1]
        self.results = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else self.root / "results"
        self.figures = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else self.root / "figures"
        self.captions: list[tuple[str, str, str, str]] = []      # (file stem, caption, alt text, notes)

    def read(self, name: str) -> list[dict]:
        with (self.results / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def facts(self, name: str = "run.csv") -> dict:
        return {row["key"]: row["value"] for row in self.read(name)}

    def done(self, fig, stem: str, caption: str, alt: str, notes: str) -> None:
        self.figures.mkdir(parents=True, exist_ok=True)
        target = fs.save(fig, f"{stem}.png", out=self.figures / f"{stem}.png")
        self.captions.append((stem, caption, alt, notes))
        print(target)

    def write_captions(self, title: str) -> None:
        lines = [f"# {title}: figure captions", "",
                 "Written by `scripts/plot.py`; do not edit by hand. Every figure is `NAME.png` (9 inches wide at "
                 "200 dpi) and `NAME.svg`. An image holds only the plot, its axes with units, a legend with "
                 "short series names and panel letters; the caption says what the figure shows, what was "
                 "measured, in which unit, on which scales, where and when. The notes say how, which numbers "
                 "come from Lean and which from scripts, and give the numbers behind the marks.", ""]
        for number, (stem, caption, alt, notes) in enumerate(self.captions, start=1):
            lines += [f"## Figure {number}: `{stem}`", "",
                      f"![{alt}]({stem}.svg \"{caption}\")", "",
                      f"**Figure {number}.** {caption}", "", f"Alt text: {alt}", "", f"Notes: {notes}", "",
                      f"Files: `{stem}.png`, `{stem}.svg`.", ""]
        (self.figures / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def plain(value, _=None) -> str:
    """A tick label without exponent notation."""
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:g}"


def log_axis(ax, axis: str, ticks) -> None:
    """A logarithmic axis with the given labelled ticks and no minor labels."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    target.set_major_locator(FixedLocator(list(ticks)))
    target.set_major_formatter(FuncFormatter(plain))
    target.set_minor_locator(NullLocator())
    target.set_minor_formatter(NullFormatter())


def width_axis(ax, widths) -> None:
    """Format widths on a logarithmic axis: every width has a tick; 32, 128, 512 and 4,096 a label."""
    ax.set_xscale("log", base=2)
    labelled = [w for w in widths if w in (32, 128, 512, 4096)]
    ax.xaxis.set_major_locator(FixedLocator(labelled))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_locator(FixedLocator([w for w in widths if w not in labelled]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(widths) / 1.35, max(widths) * 1.35)


def stacked_rows(ax, rows, colours, total_label: str) -> None:
    """Rows of a 100 % stacked bar, each with its name above and its count at the right end."""
    for position, (label, parts) in enumerate(rows):
        y = len(rows) - 1 - position
        total = sum(parts)
        left = 0.0
        for value, (face, hatch, edge) in zip(parts, colours):
            share = 100.0 * value / total
            ax.barh(y, share, left=left, height=0.42, color=face, hatch=hatch, edgecolor=edge,
                    linewidth=fs.LW_AXIS, zorder=2)
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
