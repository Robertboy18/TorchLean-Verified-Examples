#!/usr/bin/env python3
"""Render Week 3 loss and LeanProfiler summaries as dependency-free SVG."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Iterable

WIDTH = 1000
HEIGHT = 520
INK = "#111827"
MUTED = "#667085"
GRID = "#d9dee7"
TEAL = "#0f8b8d"
BLUE = "#2463a8"
ORANGE = "#d66a2c"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render Week 3 run artifacts.")
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top", type=int, default=10)
    return p


def svg_document(content: str, title: str, height: int = HEIGHT) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}"
 role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title>
<desc id="desc">Generated from the checked Week 3 run artifacts.</desc>
<rect width="{WIDTH}" height="{height}" fill="white"/>
<style>
  text {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; fill: {INK}; }}
  .muted {{ fill: {MUTED}; }}
  .grid {{ stroke: {GRID}; stroke-width: 1; }}
  .label {{ font-size: 14px; }}
  .small {{ font-size: 12px; }}
</style>
{content}
</svg>
"""


def polyline(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def loss_svg(metrics: dict) -> str:
    points = metrics.get("points", [])
    finite = [
        p for p in points if isinstance(p.get("loss"), (int, float))
    ]
    if not finite:
        raise SystemExit("metrics artifact has no finite loss points")
    left, right, top, bottom = 90, 960, 70, 445
    max_step = max(1, max(int(p["step"]) for p in finite))
    losses = [float(p["loss"]) for p in finite]
    low, high = min(losses), max(losses)
    if high == low:
        high = low + 1.0

    def x_of(step: int) -> float:
        return left + (right - left) * step / max_step

    def y_of(loss: float) -> float:
        return bottom - (bottom - top) * (loss - low) / (high - low)

    content = [
        '<text x="48" y="40" font-size="24" font-weight="700">Training and validation loss</text>',
        f'<line class="grid" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<line class="grid" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<text class="small muted" x="{left}" y="{bottom + 28}">0</text>',
        f'<text class="small muted" x="{right - 18}" y="{bottom + 28}">{max_step}</text>',
        f'<text class="small muted" x="18" y="{top + 5}">{high:.4g}</text>',
        f'<text class="small muted" x="18" y="{bottom + 5}">{low:.4g}</text>',
        f'<text class="label muted" x="475" y="492">optimizer step</text>',
    ]
    for phase, color in (("train", BLUE), ("validation", ORANGE)):
        phase_points = [
            (x_of(int(p["step"])), y_of(float(p["loss"])))
            for p in finite
            if p.get("phase") == phase
        ]
        if not phase_points:
            continue
        content.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'points="{polyline(phase_points)}"/>'
        )
        for x, y in phase_points:
            content.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
            )
    content.extend(
        [
            f'<line x1="720" y1="34" x2="750" y2="34" stroke="{BLUE}" stroke-width="3"/>',
            '<text class="label" x="758" y="39">train</text>',
            f'<line x1="825" y1="34" x2="855" y2="34" stroke="{ORANGE}" stroke-width="3"/>',
            '<text class="label" x="863" y="39">validation</text>',
        ]
    )
    return svg_document("\n".join(content), "Training and validation loss")


def row_label(row: dict) -> str:
    key = row.get("key", {})
    name = key.get("name", "unnamed")
    phase = key.get("phase")
    return f"{phase}: {name}" if phase else name


def profile_svg(summary: dict, top_n: int) -> str:
    rows = [
        row
        for row in summary.get("rows", [])
        if isinstance(row.get("total_ns"), int) and row["total_ns"] > 0
    ]
    rows.sort(key=lambda row: row["total_ns"], reverse=True)
    rows = rows[: max(1, top_n)]
    if not rows:
        raise SystemExit("LeanProfiler summary has no nonzero timing rows")
    height = max(420, 125 + 42 * len(rows))
    chart_left, chart_right = 390, 950
    max_ns = max(row["total_ns"] for row in rows)
    content = [
        '<text x="48" y="42" font-size="24" font-weight="700">LeanProfiler phase totals</text>',
        '<text class="label muted" x="48" y="68">'
        "Inclusive host time grouped by recorded span metadata</text>",
    ]
    for index, row in enumerate(rows):
        y = 105 + 42 * index
        width = (chart_right - chart_left) * row["total_ns"] / max_ns
        milliseconds = row["total_ns"] / 1_000_000
        label = row_label(row)
        if len(label) > 46:
            label = label[:43] + "..."
        content.extend(
            [
                f'<text class="label" x="48" y="{y + 16}">{escape(label)}</text>',
                f'<rect x="{chart_left}" y="{y}" width="{width:.2f}" height="22" '
                f'rx="3" fill="{TEAL}"><title>{milliseconds:.3f} ms, '
                f'{row.get("calls", 0)} call(s)</title></rect>',
                f'<text class="small muted" x="{min(chart_left + width + 8, 915):.2f}" '
                f'y="{y + 16}">{milliseconds:.2f} ms</text>',
            ]
        )
    process = summary.get("process_resources") or {}
    peak_mb = process.get("peak_resident_set_size_kb", 0) / 1024
    footer = (
        f"events={summary.get('event_count', 0)}  "
        f"threads={summary.get('thread_count', 0)}  "
        f"peak RSS={peak_mb:.1f} MiB"
    )
    content.append(
        f'<text class="small muted" x="48" y="{height - 28}">{escape(footer)}</text>'
    )
    return svg_document("\n".join(content), "LeanProfiler phase totals", height)


def main() -> None:
    args = parser().parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    loss_path = args.output / "loss.svg"
    profile_path = args.output / "profile.svg"
    loss_path.write_text(loss_svg(metrics), encoding="utf-8")
    profile_path.write_text(profile_svg(summary, args.top), encoding="utf-8")
    print(f"wrote {loss_path}")
    print(f"wrote {profile_path}")


if __name__ == "__main__":
    main()
