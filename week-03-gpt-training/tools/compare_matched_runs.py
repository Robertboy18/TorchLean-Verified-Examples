#!/usr/bin/env python3
"""Compare two matched Week 3 training runs and render their loss curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


WIDTH = 1040
HEIGHT = 600
INK = "#172033"
MUTED = "#667085"
GRID = "#d9dee7"
TORCHLEAN = "#0f8b8d"
PYTORCH = "#d65f2e"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torchlean", type=Path, required=True)
    p.add_argument("--pytorch", type=Path, required=True)
    p.add_argument("--tokens-per-update", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def load_points(path: Path) -> list[dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    points = artifact.get("points")
    if not isinstance(points, list):
        raise SystemExit(f"{path} has no metric-point array")
    return points


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
) -> str:
    left, right, top, bottom = 92, 994, 92, 462
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

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">Matched TorchLean and PyTorch GPT-2 training</title>',
        '<desc id="desc">Loss over fifty matched updates from the same checkpoint and batches.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        '<style>text { font-family: Inter, ui-sans-serif, system-ui, sans-serif; '
        f'fill: {INK}; }} .muted {{ fill: {MUTED}; }} .grid {{ stroke: {GRID}; '
        'stroke-width: 1; }} .small { font-size: 13px; }</style>',
        '<text x="48" y="42" font-size="25" font-weight="700">'
        'Matched GPT-2 Small training</text>',
        '<text class="muted" x="48" y="68" font-size="14">'
        'Same checkpoint, dialogue records, target weights, optimizer, and schedule</text>',
    ]
    for tick in range(5):
        fraction = tick / 4
        y = bottom - (bottom - top) * fraction
        value = low + (high - low) * fraction
        svg.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text class="small muted" x="35" y="{y + 5:.2f}">{value:.2f}</text>',
            ]
        )
    for tick in range(6):
        step = round(max_step * tick / 5)
        x = x_of(step)
        tokens = step * tokens_per_update
        svg.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top}" '
                f'x2="{x:.2f}" y2="{bottom}"/>',
                f'<text class="small muted" text-anchor="middle" x="{x:.2f}" '
                f'y="{bottom + 25}">{tokens // 1000}k</text>',
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
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'points="{line(train_points)}"{dash}/>'
        )
        for step, left_point, right_point in validation:
            point = (left_point, right_point)[side - 1]
            svg.append(
                f'<circle cx="{x_of(step):.2f}" '
                f'cy="{y_of(finite_number(point["loss"], "validation loss")):.2f}" '
                f'r="5" fill="white" stroke="{color}" stroke-width="3"/>'
            )

    svg.extend(
        [
            f'<line x1="700" y1="39" x2="734" y2="39" stroke="{TORCHLEAN}" '
            'stroke-width="3"/>',
            '<text x="742" y="44" font-size="14">TorchLean</text>',
            f'<line x1="850" y1="39" x2="884" y2="39" stroke="{PYTORCH}" '
            'stroke-width="3" stroke-dasharray="9 6"/>',
            '<text x="892" y="44" font-size="14">PyTorch</text>',
            '<text class="small muted" text-anchor="middle" x="543" y="515">'
            'scheduled tokens</text>',
            '<text class="small muted" transform="translate(18 292) rotate(-90)">loss</text>',
            f'<text x="48" y="557" font-size="14">Final validation: '
            f'TorchLean {summary["final_validation_loss"]["torchlean"]:.6f}, '
            f'PyTorch {summary["final_validation_loss"]["pytorch"]:.6f}</text>',
            f'<text class="muted" x="48" y="582" font-size="14">Median update throughput: '
            f'{summary["median_tokens_per_second"]["torchlean"]:.0f} vs '
            f'{summary["median_tokens_per_second"]["pytorch"]:.0f} tokens/s '
            f'({summary["pytorch_throughput_ratio"]:.2f}x)</text>',
            '</svg>',
        ]
    )
    return "\n".join(svg) + "\n"


def main() -> None:
    args = parser().parse_args()
    torchlean = load_points(args.torchlean)
    pytorch = load_points(args.pytorch)
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
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
