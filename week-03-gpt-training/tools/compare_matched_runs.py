#!/usr/bin/env python3
"""Compare two matched Week 3 training runs and render their loss curves."""

from __future__ import annotations

import argparse
import base64
from html import escape
import json
import math
from pathlib import Path
import statistics


WIDTH = 720
HEIGHT = 690
INK = "#252b32"
MUTED = "#69737d"
GRID = "#e0e5e8"
TORCHLEAN = "#286493"
PYTORCH = "#b16943"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torchlean", type=Path, required=True)
    p.add_argument("--pytorch", type=Path, required=True)
    p.add_argument("--tokens-per-update", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def load_points(path: Path) -> tuple[list[dict], str | None]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    points = artifact.get("points")
    if not isinstance(points, list):
        raise SystemExit(f"{path} has no metric-point array")
    return points, artifact.get("dtype")


def by_phase(points: list[dict], phase: str) -> dict[int, dict]:
    selected = {
        int(point["step"]): point
        for point in points
        if point.get("phase") == phase
    }
    if not selected:
        raise SystemExit(f"run has no {phase} points")
    return selected


def matched(
    left: dict[int, dict], right: dict[int, dict], phase: str
) -> list[tuple[int, dict, dict]]:
    if left.keys() != right.keys():
        raise SystemExit(
            f"{phase} step sets differ: {sorted(left)} versus {sorted(right)}"
        )
    return [(step, left[step], right[step]) for step in sorted(left)]


def finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"{label} is not finite")
    return float(value)


def line(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_svg(
    train: list[tuple[int, dict, dict]],
    validation: list[tuple[int, dict, dict]],
    tokens_per_update: int,
    summary: dict,
    width: int = WIDTH,
) -> str:
    compact = width < 600
    height = 790 if compact else HEIGHT
    left, right, top, bottom = (52 if compact else 66), width - 26, 149, 388
    all_losses = [
        finite_number(point[side]["loss"], "loss")
        for point in train + validation
        for side in (1, 2)
    ]
    low = min(all_losses)
    high = max(all_losses)
    padding = max(0.03, (high - low) * 0.08)
    low -= padding
    high += padding
    max_step = max(step for step, _, _ in train)

    def x_of(step: int) -> float:
        return left + (right - left) * step / max_step

    def y_of(loss: float) -> float:
        return bottom - (bottom - top) * (loss - low) / (high - low)

    fonts = Path(__file__).resolve().parents[2] / "site" / "assets" / "fonts"
    font = base64.b64encode((fonts / "ibm-plex-sans-latin-400-normal.woff2").read_bytes()).decode()
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">Matched TorchLean and PyTorch GPT-2 training</title>',
        f'<desc id="desc">Loss over {len(train)} matched updates from the same checkpoint and batches. '
        'Lines are training loss. Diamonds and circles are validation loss. '
        'The lower bars show median update throughput, starting at zero.</desc>',
        f'<metadata>{escape(json.dumps({"train": train, "validation": validation, "summary": summary}))}</metadata>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<style>'
        f'@font-face{{font-family:Plex;src:url(data:font/woff2;base64,{font}) format("woff2");}}'
        'text { font-family: Plex, Arial, sans-serif; font-size: 15px; '
        f'fill: {INK}; }} .muted {{ fill: {MUTED}; }} .grid {{ stroke: {GRID}; '
        'stroke-width: 1; }} .small { font-size: 13px; }</style>',
        f'<text x="24" y="33" style="font-size:{18 if compact else 21}px;font-weight:600">'
        f'GPT-2 Small: {len(train)} matched updates</text>',
        ('<text class="muted" x="24" y="59">Same checkpoint, batches, optimizer,</text>'
         '<text class="muted" x="24" y="81">and learning-rate schedule</text>') if compact else
        '<text class="muted" x="24" y="59">Same checkpoint, batches, optimizer, and learning-rate schedule</text>',
        '<text class="muted small" x="24" y="133">Cross-entropy loss' + (' · lower is better' if compact else '') + '</text>',
        '' if compact else f'<text class="muted small" x="{right}" y="133" text-anchor="end">Lower is better</text>',
    ]
    for tick in range(5):
        fraction = tick / 4
        y = bottom - (bottom - top) * fraction
        value = low + (high - low) * fraction
        svg.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text class="muted" text-anchor="end" x="{left-12}" y="{y + 5:.2f}">{value:.2f}</text>',
            ]
        )
    for tick in range(6):
        step = round(max_step * tick / 5)
        x = x_of(step)
        tokens = step * tokens_per_update
        svg.extend(
            [
                f'<text class="muted" text-anchor="middle" x="{x:.2f}" '
                f'y="{bottom + 25}">{tokens / 1000:.0f}k</text>',
            ]
        )

    for side, color in ((1, TORCHLEAN), (2, PYTORCH)):
        train_points = [
            (
                x_of(step),
                y_of(
                    finite_number(
                        (left_point, right_point)[side - 1]["loss"],
                        "train loss",
                    )
                ),
            )
            for step, left_point, right_point in train
        ]
        dash = "" if side == 1 else ' stroke-dasharray="9 6"'
        svg.append(
            f'<polyline data-series="train-{side}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'points="{line(train_points)}"{dash}/>'
        )
        for step, left_point, right_point in validation:
            point = (left_point, right_point)[side - 1]
            x = x_of(step)
            y = y_of(finite_number(point["loss"], "validation loss"))
            svg.append(f'<g data-series="validation-{side}" data-step="{step}" data-loss="{point["loss"]}">')
            if side == 1:
                svg.append(f'<path d="M{x:.2f} {y-7:.2f}l7 7-7 7-7-7Z" fill="white" stroke="{color}" stroke-width="2"/>')
            else:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>')
            svg.append('</g>')

    svg.extend(
        [
            f'<line x1="24" y1="{103 if compact else 91}" x2="56" y2="{103 if compact else 91}" stroke="{TORCHLEAN}" '
            'stroke-width="3"/>',
            f'<text x="66" y="{108 if compact else 96}">TorchLean</text>',
            f'<line x1="190" y1="{103 if compact else 91}" x2="222" y2="{103 if compact else 91}" stroke="{PYTORCH}" '
            'stroke-width="3" stroke-dasharray="9 6"/>',
            f'<text x="232" y="{108 if compact else 96}">PyTorch</text>',
            '' if compact else '<text class="small muted" x="410" y="96">Lines: train · markers: validation</text>',
            f'<text class="muted small" text-anchor="middle" x="{(left+right)/2}" y="440">Scheduled tokens</text>',
        ]
    )
    if compact:
        svg.extend([
            '<text class="small muted" x="24" y="469">Lines: train · markers: validation</text>',
            '<text x="24" y="505">Final validation loss</text>',
            f'<text class="muted" x="24" y="531">TorchLean: {summary["final_validation_loss"]["torchlean"]:.6f}</text>',
            f'<text class="muted" x="24" y="555">PyTorch: {summary["final_validation_loss"]["pytorch"]:.6f}</text>',
            f'<line x1="24" y1="579" x2="{right}" y2="579" class="grid"/>',
            '<text x="24" y="609" style="font-size:18px;font-weight:600">Median update throughput</text>',
            '<text class="small muted" x="24" y="632">Tokens / second · higher is better</text>',
        ])
    else:
        svg.extend([
            f'<text x="24" y="477">Final validation loss: '
            f'{summary["final_validation_loss"]["torchlean"]:.6f} vs. '
            f'{summary["final_validation_loss"]["pytorch"]:.6f}</text>',
            f'<line x1="24" y1="502" x2="{right}" y2="502" class="grid"/>',
            '<text x="24" y="533" style="font-size:18px;font-weight:600">Median update throughput</text>',
            f'<text class="small muted" x="{right}" y="533" text-anchor="end">Tokens / second · higher is better</text>',
        ])
    rates = summary["median_tokens_per_second"]
    peak_rate = max(rates.values())
    for name, key, color, y in [("TorchLean", "torchlean", TORCHLEAN, 570), ("PyTorch", "pytorch", PYTORCH, 612)]:
        y += 102 if compact else 0
        bar_left = 114 if compact else 132
        length = (146 if compact else 414) * rates[key] / peak_rate
        svg.extend([
            f'<text x="24" y="{y+5}">{name}</text>',
            f'<rect data-series="throughput-{key}" data-value="{rates[key]}" x="{bar_left}" y="{y-10}" '
            f'width="{length:.3f}" height="20" fill="{color}"/>',
            f'<text x="{bar_left+length+12:.3f}" y="{y+5}">{rates[key]:,.0f}</text>',
        ])
    svg.extend([
        f'<text class="small muted" x="24" y="{756 if compact else 667}">PyTorch / TorchLean: '
        f'{summary["pytorch_throughput_ratio"]:.2f}×' + ('</text><text class="small muted" x="24" y="777">throughput in this run</text>' if compact else ' throughput in this run</text>'),
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    args = parser().parse_args()
    torchlean, torchlean_dtype = load_points(args.torchlean)
    pytorch, pytorch_dtype = load_points(args.pytorch)
    if torchlean_dtype != pytorch_dtype:
        raise SystemExit(
            f"run precisions differ: {torchlean_dtype!r} versus {pytorch_dtype!r}"
        )
    if torchlean_dtype is None:
        print("These older artifacts do not record precision; check their run notes before comparing.")
    train = matched(by_phase(torchlean, "train"), by_phase(pytorch, "train"), "train")
    validation = matched(
        by_phase(torchlean, "validation"),
        by_phase(pytorch, "validation"),
        "validation",
    )

    train_differences = [
        abs(finite_number(a["loss"], "TorchLean train loss") -
            finite_number(b["loss"], "PyTorch train loss"))
        for _, a, b in train
    ]
    validation_differences = [
        abs(finite_number(a["loss"], "TorchLean validation loss") -
            finite_number(b["loss"], "PyTorch validation loss"))
        for _, a, b in validation
    ]
    torchlean_rates = [
        finite_number(a["tokens_per_second"], "TorchLean throughput")
        for _, a, _ in train
    ]
    pytorch_rates = [
        finite_number(b["tokens_per_second"], "PyTorch throughput")
        for _, _, b in train
    ]
    torchlean_median = statistics.median(torchlean_rates)
    pytorch_median = statistics.median(pytorch_rates)
    _, final_torchlean, final_pytorch = validation[-1]
    summary = {
        "schema": "torchlean.gpt-training.matched-comparison.v1",
        "dtype": torchlean_dtype,
        "updates": len(train),
        "tokens_per_update": args.tokens_per_update,
        "scheduled_tokens": len(train) * args.tokens_per_update,
        "max_absolute_train_loss_difference": max(train_differences),
        "max_absolute_validation_loss_difference": max(validation_differences),
        "final_validation_loss": {
            "torchlean": finite_number(final_torchlean["loss"], "final TorchLean loss"),
            "pytorch": finite_number(final_pytorch["loss"], "final PyTorch loss"),
        },
        "median_tokens_per_second": {
            "torchlean": torchlean_median,
            "pytorch": pytorch_median,
        },
        "pytorch_throughput_ratio": pytorch_median / torchlean_median,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "comparison.json"
    figure_path = args.output / "loss-and-throughput.svg"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure_path.write_text(
        render_svg(train, validation, args.tokens_per_update, summary),
        encoding="utf-8",
    )
    figure_path.with_stem(figure_path.stem + "-mobile").write_text(
        render_svg(train, validation, args.tokens_per_update, summary, width=360),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
