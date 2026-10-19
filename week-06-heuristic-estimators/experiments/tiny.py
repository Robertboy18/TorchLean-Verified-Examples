#!/usr/bin/env python3
"""Summarize the sparsemax transformer's Lean replay without refitting its coordinate masks.

The input preserves binary64 words because Lean's displayed JSON float values are rounded.
Calibration and evaluation results stay separate; the complete finite task is also reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def decode(words: list[str | int]) -> list[float]:
    return [struct.unpack("<d", struct.pack("<Q", int(word)))[0] for word in words]


def summarize(rows: list[dict]) -> dict:
    changes = 0
    passes = 0
    baseline_passes = 0
    baseline_top1 = 0
    top1 = 0
    top1_changes = 0
    squared_error = 0.0
    squared_baseline = 0.0
    maximum_error = 0.0
    margins: list[float] = []
    for row in rows:
        reference = decode(row["baseline_bits"])
        actual = decode(row["output_bits"])
        baseline_margin = decode([row["baseline_margin_bits"]])[0]
        margin = decode([row["margin_bits"]])[0]
        assert all(math.isfinite(value) for value in reference + actual)
        assert len(reference) == len(actual) == 32
        baseline_passes += baseline_margin > 0
        passes += margin > 0
        changes += (baseline_margin > 0) != (margin > 0)
        reference_token = max(range(32), key=reference.__getitem__)
        actual_token = max(range(32), key=actual.__getitem__)
        baseline_top1 += reference_token == row["target"]
        top1 += actual_token == row["target"]
        top1_changes += reference_token != actual_token
        margins.append(margin)
        for original, reduced in zip(reference, actual):
            difference = reduced - original
            squared_error += difference * difference
            squared_baseline += original * original
            maximum_error = max(maximum_error, abs(difference))
    return {
        "prompts": len(rows),
        "baseline_projected_correct": baseline_passes,
        "projected_correct": passes,
        "changed_projected_decisions": changes,
        "baseline_full_vocabulary_top1_correct": baseline_top1,
        "full_vocabulary_top1_correct": top1,
        "changed_full_vocabulary_top1": top1_changes,
        "minimum_target_margin": min(margins),
        "mean_target_margin": sum(margins) / len(margins),
        "full_vocabulary_logit_relative_l2_error": math.sqrt(squared_error / squared_baseline),
        "maximum_logit_error": maximum_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/tmp/torchlean-multimodel/tiny"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    metadata = json.loads((arguments.directory / "metadata.json").read_text())
    observations = (arguments.directory / "observations.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in observations]
    calibration = set(metadata["calibration_indices"])
    evaluation = set(metadata["evaluation_indices"])
    assert len(calibration) == len(evaluation) == 128
    assert calibration.isdisjoint(evaluation) and calibration | evaluation == set(range(256))
    assert sorted(metadata["ordering"]) == list(range(16))
    for task in range(2):
        for label in range(2):
            assert sum(index // 128 == task and index % 2 == label for index in calibration) == 32
            assert sum(index // 128 == task and index % 2 == label for index in evaluation) == 32
    ranks = sorted({row["rank"] for row in rows}, reverse=True)
    assert ranks == [16, 12, 8, 4]
    results = []
    for rank in ranks:
        matching = [row for row in rows if row["rank"] == rank]
        assert sorted(row["index"] for row in matching) == list(range(256))
        assert all(row["calibration"] == (row["index"] in calibration) for row in matching)
        partitions = {
            "calibration": [row for row in matching if row["index"] in calibration],
            "evaluation": [row for row in matching if row["index"] in evaluation],
            "complete_finite_domain": matching,
        }
        record = {"retained_coordinates": rank, "selected": sorted(metadata["ordering"][:rank])}
        for name, partition in partitions.items():
            record[name] = summarize(partition)
            record[name]["tasks"] = {
                task: summarize([row for row in partition if row["task"] == task])
                for task in ("quote", "bracket")
            }
        results.append(record)
    assert results[0]["complete_finite_domain"]["maximum_logit_error"] == 0
    assert results[0]["complete_finite_domain"]["baseline_projected_correct"] == 256
    artifact = {
        "model": "trained upstream two-layer sparsemax GPT; width 16; sequence length 6",
        "scope": metadata["scope"],
        "selection": "greedy calibration score-error quadratic, first attention block only",
        "split": "content-token parity, balanced by task and candidate target",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "calibration_indices": sorted(calibration),
        "evaluation_indices": sorted(evaluation),
        "fit_seconds": metadata["fit_ns"] / 1e9,
        "rank_16_control_exact_on_all_256_prompts": True,
        "provenance": {
            "lean_source_sha256": hashlib.sha256(
                Path(__file__).with_name("Tiny.lean").read_bytes()
            ).hexdigest(),
            "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "observations_sha256": hashlib.sha256(
                (arguments.directory / "observations.jsonl").read_bytes()
            ).hexdigest(),
            "metadata_sha256": hashlib.sha256(
                (arguments.directory / "metadata.json").read_bytes()
            ).hexdigest(),
        },
        "results": results,
        "limitations": [
            "Evaluation prompts were excluded from this coordinate search, "
            "not necessarily original training.",
            "The projected decision compares its two task candidates, not all vocabulary tokens.",
            "These are executable floating-point observations, not emitted kernel proof terms.",
            "Coordinate removal changes only the first block's QK scores; "
            "other computations remain.",
            "No end-to-end speedup is claimed.",
        ],
    }
    arguments.output.write_text(json.dumps(artifact, indent=2) + "\n")
    for record in results:
        held_out = record["evaluation"]
        print(
            f"rank {record['retained_coordinates']:2d}: "
            f"held-out projected decisions {held_out['projected_correct']}/128; "
            f"min margin {held_out['minimum_target_margin']:.9g}; "
            f"logit relative L2 {held_out['full_vocabulary_logit_relative_l2_error']:.3%}"
        )


if __name__ == "__main__":
    main()
