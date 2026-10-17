#!/usr/bin/env python3
"""Fit coordinate summaries on trained attention, then compare a TorchLean replay.

The checkpoint and token shard are read-only. Python prepares a selected block's
queries, keys and values; Attention.lean evaluates the exported attention cases.
Only calibration windows influence the masks. Evaluation windows stay separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "week-03-gpt-training/artifacts/gpt2-small-2.5b-continuation-20260806/final-parameters.tlf32"
TOKENS = ROOT / "week-03-gpt-training/data/fineweb-edu-continuation-1.5b/val.bin"
WIDTH, HEADS, DIM, CONTEXT = 768, 12, 64, 64
RANKS = (48, 32, 16, 8)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_checkpoint(path: Path, all_parameters: bool = False) -> list[np.ndarray]:
    """Validate the historical layout and map the requested parameter records.

    This archived checkpoint predates the current StateIO header. Its tensor
    records have the same little-endian rank, dimensions, count and FP32 payload.
    We recognize that format explicitly rather than modifying the original file.
    """
    block = [(768,), (768,)] + [(768, 768)] * 4 + [
        (768,), (768,), (768,), (3072, 768), (3072,), (768, 3072), (768,)
    ]
    expected = [(50257, 768), (1024, 768)] + block * 12 + [(768,), (768,)]
    selected = []
    with path.open("rb") as handle:
        def u64() -> int:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError("truncated checkpoint metadata")
            return struct.unpack("<Q", raw)[0]

        if handle.read(7) != b"TLPF32B" or u64() != 2:
            raise ValueError("expected the archived parameter-stream v2 checkpoint")
        count = u64()
        if count != len(expected):
            raise ValueError(f"expected {len(expected)} tensors, found {count}")
        for index, shape in enumerate(expected):
            rank = u64()
            if rank != len(shape):
                raise ValueError(f"tensor {index}: unexpected rank {rank}")
            actual = tuple(u64() for _ in range(rank))
            size = u64()
            if actual != shape or size != math.prod(shape):
                raise ValueError(f"tensor {index}: unexpected shape {actual}/{size}")
            offset = handle.tell()
            if all_parameters or index < 8:
                selected.append(np.memmap(path, mode="r", dtype="<f4", offset=offset, shape=shape))
            handle.seek(size * 4, 1)
        if handle.tell() != path.stat().st_size or handle.read(1):
            raise ValueError("checkpoint length does not match its tensor records")
    return selected


def extract(arrays: list[np.ndarray], ids: np.ndarray, device: torch.device,
            layer: int, context: int, position_anchor: int) -> list[torch.Tensor]:
    """Use the existing matched-reference blocks for the unmodified prefix.

    Probe Q/K/V before the selected block's output projection. This reconstruction
    is not asserted bit-identical to the original training runtime's GPU kernels.
    """
    embeddings = torch.from_numpy(np.array(arrays[0])).to(device)
    positions = torch.from_numpy(np.array(arrays[1])).to(device)
    hidden = embeddings[torch.from_numpy(ids).to(device)]
    hidden += positions[position_anchor - context:position_anchor]
    if layer > 1:
        path = ROOT / "week-03-gpt-training/tools/matched_pytorch_run.py"
        spec = importlib.util.spec_from_file_location("matched_gpt_reference", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the existing GPT reference")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        remaining = iter(arrays[2:])

        def tensor(expected: tuple[int, ...]) -> torch.Tensor:
            array = next(remaining)
            if array.shape != expected:
                raise ValueError(f"reference requested {expected}, got {array.shape}")
            return torch.from_numpy(np.array(array)).to(device)

        reader = SimpleNamespace(tensor=tensor)
        for _ in range(layer - 1):
            block = reference.Block(reader, WIDTH, 4 * WIDTH).eval()
            hidden = block(hidden, HEADS, 1e-6)
    offset = 2 + (layer - 1) * 13
    parameters = [torch.from_numpy(np.array(a)).to(device) for a in arrays[offset:offset + 5]]
    normalized = F.layer_norm(hidden, (WIDTH,), parameters[0], parameters[1], eps=1e-6)
    return [(normalized @ weight).view(len(ids), context, HEADS, DIM).transpose(1, 2).double()
            for weight in parameters[2:5]]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Retaining fewer coordinates must not change the original 1/sqrt(64) scale.
    return torch.softmax(torch.einsum("bhd,bhtd->bht", q, k) / 8, dim=-1).unsqueeze(-2).matmul(v).squeeze(-2)


def summary(full: torch.Tensor, approximate: torch.Tensor) -> dict:
    delta = approximate - full
    per_head = torch.sqrt(delta.square().sum((0, 2)) / full.square().sum((0, 2)))
    per_window = torch.sqrt(delta.square().sum((1, 2)) / full.square().sum((1, 2)))
    return {
        "relative_l2": math.sqrt(float(delta.square().sum() / full.square().sum())),
        "absolute_rmse": math.sqrt(float(delta.square().mean())),
        "maximum_absolute_error": float(delta.abs().max()),
        "per_head_relative_l2": per_head.tolist(),
        "window_relative_l2_median": float(per_window.median()),
        "window_relative_l2_p95": float(torch.quantile(per_window, 0.95)),
        "window_relative_l2_max": float(per_window.max()),
    }


def greedy_masks(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[int, torch.Tensor]:
    """Discard the coordinate with the least calibration-output error at each step.

    Candidates use the actual nonlinear attention output, including the values.
    This is a small exhaustive greedy search, not a trained predictor of its error.
    """
    full = attention(q, k, v)
    masks = {r: torch.zeros((HEADS, DIM), device=q.device, dtype=torch.float64) for r in RANKS}
    for head in range(HEADS):
        contributions = q[:, head, None, :] * k[:, head] / 8
        logits = contributions.sum(-1)
        active = list(range(DIM))
        for retained in range(DIM - 1, min(RANKS) - 1, -1):
            # candidate × calibration window × token, then output coordinate.
            trial = logits.unsqueeze(0) - contributions[:, :, active].permute(2, 0, 1)
            outputs = torch.softmax(trial, dim=-1).unsqueeze(-2).matmul(
                v[:, head].unsqueeze(0)).squeeze(-2)
            losses = (outputs - full[:, head].unsqueeze(0)).square().mean((1, 2))
            removed = active.pop(int(losses.argmin()))
            logits = logits - contributions[:, :, removed]
            if retained in masks:
                masks[retained][head, active] = 1
        print(f"fitted head {head + 1}/{HEADS}", flush=True)
    return masks


def prepare(output: Path, calibration: int, evaluation: int, *, layer: int = 1,
            context: int = CONTEXT, window_span: int | None = None,
            position_anchor: int | None = None) -> None:
    if calibration < 1 or evaluation < 1:
        raise ValueError("both window counts must be positive")
    window_span = context if window_span is None else window_span
    position_anchor = context if position_anchor is None else position_anchor
    if not 1 <= layer <= 12 or not 1 <= context <= position_anchor <= 1024:
        raise ValueError("invalid checkpoint layer, context or position anchor")
    if window_span < context:
        raise ValueError("window span must cover the complete context")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    arrays = read_checkpoint(CHECKPOINT, all_parameters=layer > 1)
    tokens = np.memmap(TOKENS, dtype="<u2", mode="r")
    # Fixed, disjoint windows spread over the validation shard. No target or
    # evaluation output participates in choosing their positions or fitting masks.
    count = calibration + evaluation
    offsets = np.linspace(0, len(tokens) - window_span, count, dtype=np.int64) + window_span - context
    if np.any(np.diff(offsets) < window_span):
        raise ValueError("requested windows overlap")
    ids = np.stack([tokens[offset:offset + context] for offset in offsets]).astype(np.int64)
    with torch.inference_mode():
        projected = extract(arrays, ids, device, layer, context, position_anchor)
        q = projected[0][:, :, -1]
        k, v = projected[1:]
        if not all(bool(tensor.isfinite().all()) for tensor in (q, k, v)):
            raise ValueError("nonfinite checkpoint activations")
        if device.type == "cuda":
            torch.cuda.synchronize()
        extraction_seconds = time.perf_counter() - started
        fit_started = time.perf_counter()
        cq, ck, cv = q[:calibration], k[:calibration], v[:calibration]
        greedy = greedy_masks(cq, ck, cv)
        centered_keys = ck - ck.mean(2, keepdim=True)
        scores = {
            "magnitude": (cq[:, :, None, :] * ck).square().mean((0, 2)),
            "centered": (cq[:, :, None, :] * centered_keys).square().mean((0, 2)),
            "random": torch.from_numpy(np.random.default_rng(42).random((HEADS, DIM))).to(device),
        }
        masks = {"greedy": greedy}
        for name, score in scores.items():
            masks[name] = {}
            for rank in RANKS:
                mask = torch.zeros_like(score)
                mask.scatter_(1, score.topk(rank, dim=1).indices, 1)
                masks[name][rank] = mask
        if device.type == "cuda":
            torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - fit_started
        full = attention(q, k, v)
        metrics = {}
        for name, ranks in masks.items():
            metrics[name] = {}
            for rank, mask in ranks.items():
                estimated = attention(q * mask, k, v)
                metrics[name][str(rank)] = {
                    "calibration": summary(full[:calibration], estimated[:calibration]),
                    "evaluation": summary(full[calibration:], estimated[calibration:]),
                }

        # Select by calibration loss alone, at a fixed half-width budget.
        chosen = min(masks, key=lambda name: metrics[name]["32"]["calibration"]["absolute_rmse"])
        mask = masks[chosen][32]
        eq, ek, ev = [tensor[calibration:].cpu().contiguous() for tensor in (q, k, v)]
        efull = full[calibration:].cpu()
        estimated = attention(q[calibration:] * mask, k[calibration:], v[calibration:]).cpu()
        mask = mask.cpu()
    # Plain binary64 scalars preserve each prepared FP32 activation exactly. The
    # replay intentionally measures TorchLean Float (binary64), not CUDA FP32.
    with (output / "attention.bin").open("wb") as handle:
        handle.write(b"HEATTN01")
        handle.write(struct.pack("<QQQ", evaluation * HEADS, context, DIM))
        for window in range(evaluation):
            for head in range(HEADS):
                handle.write(mask[head].numpy().astype("u1").tobytes())
                for tensor in (eq[window, head], ek[window, head], ev[window, head]):
                    handle.write(tensor.numpy().astype("<f8").tobytes())
    np.savez(output / "reference.npz", full=efull.numpy().reshape(-1, DIM),
             approximate=estimated.numpy().reshape(-1, DIM), q=eq.numpy(), k=ek.numpy(),
             v=ev.numpy(), mask=mask.numpy())
    result = {
        "checkpoint_sha256": digest(CHECKPOINT), "tokens_sha256": digest(TOKENS),
        "input_sha256": digest(output / "attention.bin"),
        "architecture": {"width": WIDTH, "heads": HEADS, "head_dim": DIM, "context": context},
        "layer": layer, "window_span": window_span, "position_anchor": position_anchor,
        "calibration_windows": calibration, "evaluation_windows": evaluation,
        "calibration_token_offsets": offsets[:calibration].tolist(),
        "evaluation_token_offsets": offsets[calibration:].tolist(),
        "layer_norm_epsilon": 1e-6, "projection_dtype": "float32", "attention_dtype": "float64",
        "preparation_device": str(device), "torch_version": torch.__version__,
        "extraction_seconds": extraction_seconds, "search_seconds": fit_seconds,
        "chosen_method": chosen, "chosen_rank": 32,
        "chosen_coordinates": [torch.where(row.bool())[0].tolist() for row in mask],
        "metrics": metrics,
    }
    (output / "search.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"chosen_method": chosen, "search_seconds": fit_seconds,
                      "evaluation": metrics[chosen]["32"]["evaluation"]}, indent=2), flush=True)


def compare(output: Path, replay: Path) -> None:
    observed = [json.loads(line) for line in replay.read_text().splitlines() if line]
    reference = np.load(output / "reference.npz")
    context = reference["k"].shape[-2]
    if [case["case"] for case in observed] != list(range(len(observed))):
        raise ValueError("replay cases are missing, repeated or reordered")
    if any(case["rank"] != 32 for case in observed):
        raise ValueError("replay did not use the exported half-width masks")
    # Lean's readable decimal display rounds its floats. Compare the binary64
    # words instead, so serialization cannot hide an implementation discrepancy.
    full = np.asarray([case["full"]["output_bits"] for case in observed], dtype=np.uint64).view(np.float64)
    approximate = np.asarray([case["sketch"]["output_bits"] for case in observed], dtype=np.uint64).view(np.float64)
    if full.shape != reference["full"].shape or approximate.shape != full.shape:
        raise ValueError("replay output shapes differ from the exported cases")
    if not np.isfinite(full).all() or not np.isfinite(approximate).all():
        raise ValueError("replay produced nonfinite outputs")
    full_error = float(np.max(np.abs(full - reference["full"])))
    approximate_error = float(np.max(np.abs(approximate - reference["approximate"])))
    # This diagnostic removes each head's evaluation mean from the denominator.
    # A large nearly constant output should not hide errors in its variation.
    grouped = full.reshape(-1, HEADS, DIM)
    centered_energy = np.square(grouped - grouped.mean(0, keepdims=True)).sum()
    centered_error = math.sqrt(float(np.square(approximate - full).sum() / centered_energy))
    # Absolute and relative checks protect near-zero outputs as well as large ones.
    np.testing.assert_allclose(full, reference["full"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(approximate, reference["approximate"], rtol=1e-10, atol=1e-10)
    intermediate_checks = {}
    for name, query in (("full", reference["q"]),
                        ("sketch", reference["q"] * reference["mask"])):
        logits = (np.einsum("bhd,bhtd->bht", query, reference["k"]) / 8).reshape(-1, context)
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        actual_logits = np.asarray([case[name]["logits_bits"] for case in observed],
                                   dtype=np.uint64).view(np.float64)
        actual_weights = np.asarray([case[name]["weights_bits"] for case in observed],
                                    dtype=np.uint64).view(np.float64)
        np.testing.assert_allclose(actual_logits, logits, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_weights, weights, rtol=1e-12, atol=1e-12)
        intermediate_checks[name] = {
            "max_logit_disagreement": float(np.max(np.abs(actual_logits - logits))),
            "max_probability_disagreement": float(np.max(np.abs(actual_weights - weights))),
        }
    report = {"cases": len(full), "output_scalars_per_run": int(full.size),
              "centered_relative_l2_error": centered_error,
              "full_max_absolute_disagreement": full_error,
              "approximate_max_absolute_disagreement": approximate_error,
              "full_seconds": sum(case["full_ns"] for case in observed) / 1e9,
              "packing_seconds": sum(case["packing_ns"] for case in observed) / 1e9,
              "sketch_seconds": sum(case["sketch_ns"] for case in observed) / 1e9,
              "intermediate_checks": intermediate_checks}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compare"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", type=int, default=64)
    parser.add_argument("--evaluation", type=int, default=256)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--layer", type=int, default=1, help="one-based attention block")
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--window-span", type=int, help="shared window span for nested prefixes")
    parser.add_argument("--position-anchor", type=int, help="shared final position plus one")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, args.calibration, args.evaluation, layer=args.layer,
                context=args.context, window_span=args.window_span,
                position_anchor=args.position_anchor)
    else:
        if args.replay is None:
            parser.error("compare requires --replay")
        compare(args.output, args.replay)


if __name__ == "__main__":
    main()
