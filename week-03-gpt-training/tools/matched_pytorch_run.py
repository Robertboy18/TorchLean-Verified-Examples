#!/usr/bin/env python3
"""Run a PyTorch reference with the exact Week 3 model, checkpoint, and batches.

This is an experimental control, not part of the Lean proof.  It reconstructs the
TorchLean GPT parameterization directly from its current checkpoint formats and uses the
same deterministic dialogue-record sampler, assistant-token weighting, AdamW
configuration, and learning-rate schedule as ``train_torchlean_gpt``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """Match ``Runtime.Autograd.TorchLean.Random.splitmix64`` exactly."""
    z = (value + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def sampled_record(seed: int, step: int, batch_index: int, count: int) -> int:
    key = splitmix64((seed + step) & MASK64)
    return splitmix64((key + batch_index) & MASK64) % count


class DialogueDataset:
    """Memory-mapped form of the three files consumed by the Lean runner."""

    def __init__(self, tokens: Path, mask: Path, records: Path) -> None:
        self.tokens = np.memmap(tokens, dtype="<u2", mode="r")
        self.mask = np.memmap(mask, dtype="u1", mode="r")
        raw_records = np.memmap(records, dtype="<u8", mode="r")
        if raw_records.size % 4:
            raise ValueError(f"{records} does not contain four-uint64 records")
        self.records = raw_records.reshape(-1, 4)
        if self.tokens.size != self.mask.size:
            raise ValueError("token and target-mask lengths differ")

    def batch(
        self,
        *,
        batch_size: int,
        context: int,
        seed: int,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = np.zeros((batch_size, context), dtype=np.int64)
        targets = np.zeros((batch_size, context), dtype=np.int64)
        enabled = np.zeros((batch_size, context), dtype=np.bool_)

        for batch_index in range(batch_size):
            record_index = sampled_record(seed, step, batch_index, len(self.records))
            offset, length, target_offset, target_length = map(
                int, self.records[record_index]
            )
            usable = min(context, length - 1)
            tokens[batch_index, :usable] = self.tokens[offset : offset + usable]
            targets[batch_index, :usable] = self.tokens[
                offset + 1 : offset + usable + 1
            ]
            target_indices = offset + np.arange(1, usable + 1)
            enabled[batch_index, :usable] = (
                (target_indices >= target_offset)
                & (target_indices < target_offset + target_length)
                & (self.mask[target_indices] != 0)
            )

        active = int(enabled.sum())
        weights = torch.from_numpy(enabled).to(device=device, dtype=dtype)
        if active:
            weights = weights / active
        return (
            torch.from_numpy(tokens).to(device),
            torch.from_numpy(targets).to(device),
            weights,
        )


class CheckpointReader:
    """Read binary32 streams or exact-bit binary64 JSON, checking the model layout."""

    def __init__(self, path: Path, device: torch.device) -> None:
        self.path = path
        self.device = device
        self.handle = path.open("rb")
        if self.handle.read(7) == b"TLSF32B":
            version = self._u64()
            if version != 1:
                raise ValueError(f"unsupported checkpoint version {version}")
            self.state = None
            self.remaining = self._u64()
        else:
            self.handle.seek(0)
            with self.handle:
                artifact = json.load(self.handle)
            if artifact.get("format") != "torchlean_state_bits_v1":
                raise ValueError(f"{path} is not a binary64 state checkpoint")
            self.state = artifact["state"]
            self.remaining = len(self.state)

    def _u64(self) -> int:
        raw = self.handle.read(8)
        if len(raw) != 8:
            raise ValueError("truncated checkpoint metadata")
        return struct.unpack("<Q", raw)[0]

    def tensor(self, expected: tuple[int, ...]) -> torch.Tensor:
        if self.remaining == 0:
            raise ValueError("checkpoint ended before the model layout")
        if self.state is not None:
            tensor = self.state[len(self.state) - self.remaining]
            shape = tuple(tensor["shape"])
            values = tensor["values"]
            if shape != expected or len(values) != math.prod(expected):
                raise ValueError(f"checkpoint shape {shape}/{len(values)} does not match {expected}")
            if any(type(bits) is not int or not 0 <= bits < 1 << 64 for bits in values):
                raise ValueError("checkpoint contains an invalid binary64 bit pattern")
            array = np.asarray(values, dtype="<u8").view("<f8").reshape(shape).copy()
            self.remaining -= 1
            return torch.from_numpy(array).to(self.device)
        rank = self._u64()
        shape = tuple(self._u64() for _ in range(rank))
        count = self._u64()
        if shape != expected or count != math.prod(expected):
            raise ValueError(
                f"checkpoint shape {shape}/{count} does not match {expected}"
            )
        raw = self.handle.read(4 * count)
        if len(raw) != 4 * count:
            raise ValueError("truncated checkpoint tensor")
        array = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
        self.remaining -= 1
        return torch.from_numpy(array).to(self.device)

    def finish(self) -> None:
        if self.remaining != 0:
            raise ValueError(f"checkpoint has {self.remaining} unused tensors")
        if self.state is None:
            if self.handle.read(1):
                raise ValueError("checkpoint has trailing bytes")
            self.handle.close()


class Block(nn.Module):
    def __init__(self, reader: CheckpointReader, width: int, hidden: int) -> None:
        super().__init__()
        self.norm1_weight = nn.Parameter(reader.tensor((width,)))
        self.norm1_bias = nn.Parameter(reader.tensor((width,)))
        # TorchLean stores attention projections for right multiplication: x @ W.
        self.query_weight = nn.Parameter(reader.tensor((width, width)))
        self.key_weight = nn.Parameter(reader.tensor((width, width)))
        self.value_weight = nn.Parameter(reader.tensor((width, width)))
        self.output_weight = nn.Parameter(reader.tensor((width, width)))
        self.output_bias = nn.Parameter(reader.tensor((width,)))
        self.norm2_weight = nn.Parameter(reader.tensor((width,)))
        self.norm2_bias = nn.Parameter(reader.tensor((width,)))
        self.ffn_input_weight = nn.Parameter(reader.tensor((hidden, width)))
        self.ffn_input_bias = nn.Parameter(reader.tensor((hidden,)))
        self.ffn_output_weight = nn.Parameter(reader.tensor((width, hidden)))
        self.ffn_output_bias = nn.Parameter(reader.tensor((width,)))

    def forward(self, x: torch.Tensor, heads: int, epsilon: float) -> torch.Tensor:
        batch, context, width = x.shape
        head_dim = width // heads
        normalized = F.layer_norm(
            x,
            (width,),
            self.norm1_weight,
            self.norm1_bias,
            eps=epsilon,
        )
        query = (normalized @ self.query_weight).view(
            batch, context, heads, head_dim
        ).transpose(1, 2)
        key = (normalized @ self.key_weight).view(
            batch, context, heads, head_dim
        ).transpose(1, 2)
        value = (normalized @ self.value_weight).view(
            batch, context, heads, head_dim
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, context, width)
        x = x + attended @ self.output_weight + self.output_bias

        normalized = F.layer_norm(
            x,
            (width,),
            self.norm2_weight,
            self.norm2_bias,
            eps=epsilon,
        )
        hidden = F.linear(
            normalized, self.ffn_input_weight, self.ffn_input_bias
        )
        hidden = F.gelu(hidden, approximate="tanh")
        return x + F.linear(
            hidden, self.ffn_output_weight, self.ffn_output_bias
        )


class MatchedGPT2(nn.Module):
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        *,
        context: int = 1024,
        vocab: int = 50257,
        width: int = 768,
        heads: int = 12,
        layers: int = 12,
    ) -> None:
        super().__init__()
        self.context = context
        self.vocab = vocab
        self.width = width
        self.heads = heads
        # Match TorchLeanGPT.normalizationEpsilon on the selected runtime.
        self.epsilon = 1e-6 if device.type == "cuda" else 1e-5
        reader = CheckpointReader(checkpoint, device)
        self.token_embedding = nn.Parameter(reader.tensor((vocab, width)))
        self.position_embedding = nn.Parameter(reader.tensor((context, width)))
        self.blocks = nn.ModuleList(
            [Block(reader, width, 4 * width) for _ in range(layers)]
        )
        self.final_norm_weight = nn.Parameter(reader.tensor((width,)))
        self.final_norm_bias = nn.Parameter(reader.tensor((width,)))
        reader.finish()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, context = tokens.shape
        positions = self.position_embedding[:context]
        x = F.embedding(tokens, self.token_embedding) + positions
        for block in self.blocks:
            x = block(x, self.heads, self.epsilon)
        x = F.layer_norm(
            x,
            (self.width,),
            self.final_norm_weight,
            self.final_norm_bias,
            eps=self.epsilon,
        )
        return x @ self.token_embedding.transpose(0, 1)


def weighted_loss(
    model: MatchedGPT2,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    logits = model(tokens)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    return torch.sum(losses * weights.reshape(-1))


def learning_rate(
    step: int, *, peak: float, minimum: float, warmup: int, total: int
) -> float:
    if total == 0 or step >= total:
        return minimum
    warmup = min(warmup, total)
    if step < warmup:
        return peak * (step + 1) / warmup
    decay = total - warmup
    if decay == 0:
        return minimum
    progress = (step - warmup) / decay
    cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
    return minimum + (peak - minimum) * cosine


@torch.no_grad()
def evaluate(
    model: MatchedGPT2,
    dataset: DialogueDataset,
    *,
    batches: int,
    batch_size: int,
    context: int,
    seed: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for batch_index in range(batches):
        batch = dataset.batch(
            batch_size=batch_size,
            context=context,
            seed=seed,
            step=batch_index,
            device=device,
            dtype=model.token_embedding.dtype,
        )
        losses.append(float(weighted_loss(model, *batch)))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--train-mask", type=Path, required=True)
    parser.add_argument("--train-records", type=Path, required=True)
    parser.add_argument("--val-bin", type=Path, required=True)
    parser.add_argument("--val-mask", type=Path, required=True)
    parser.add_argument("--val-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("the selected device requires CUDA")
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train_data = DialogueDataset(args.train_bin, args.train_mask, args.train_records)
    val_data = DialogueDataset(args.val_bin, args.val_mask, args.val_records)
    model = MatchedGPT2(
        args.checkpoint, device, context=args.context, vocab=args.vocab,
        width=args.width, heads=args.heads, layers=args.layers,
    ).to(dtype=getattr(torch, args.dtype))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    points: list[dict[str, float | int | str]] = []
    initial = evaluate(
        model,
        val_data,
        batches=args.eval_batches,
        batch_size=args.batch,
        context=args.context,
        seed=args.seed + 1_000_003,
        device=device,
    )
    points.append({"phase": "validation", "step": 0, "loss": initial})
    print(f"step=0 validation_loss={initial:.6f}", flush=True)

    for step in range(args.steps):
        lr = learning_rate(
            step,
            peak=args.lr,
            minimum=args.min_lr,
            warmup=args.warmup_steps,
            total=args.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        batch = train_data.batch(
            batch_size=args.batch,
            context=args.context,
            seed=args.seed,
            step=step,
            device=device,
            dtype=model.token_embedding.dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_loss(model, *batch)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        rate = args.batch * args.context / duration
        points.append(
            {
                "phase": "train",
                "step": step + 1,
                "loss": float(loss.detach()),
                "learning_rate": lr,
                "duration_ms": round(duration * 1000),
                "tokens_per_second": rate,
            }
        )
        print(
            f"step={step + 1} train_loss={float(loss.detach()):.6f} "
            f"lr={lr:.9g} tokens_per_second={rate:.2f}",
            flush=True,
        )
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val_loss = evaluate(
                model,
                val_data,
                batches=args.eval_batches,
                batch_size=args.batch,
                context=args.context,
                seed=args.seed + 1_000_003,
                device=device,
            )
            points.append(
                {"phase": "validation", "step": step + 1, "loss": val_loss}
            )
            print(
                f"step={step + 1} validation_loss={val_loss:.6f}", flush=True
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "torchlean.gpt-training.matched-pytorch.v1",
        "implementation": "PyTorch",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "dtype": args.dtype,
        "tf32": False,
        "checkpoint": str(args.checkpoint),
        "stored_parameters": parameter_count,
        "batch": args.batch,
        "context": args.context,
        "steps": args.steps,
        "seed": args.seed,
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
            "fused": device.type == "cuda",
        },
        "schedule": {
            "name": "warmup_cosine",
            "peak_learning_rate": args.lr,
            "minimum_learning_rate": args.min_lr,
            "warmup_steps": args.warmup_steps,
        },
        "evaluation": {
            "every_steps": args.eval_every,
            "batches": args.eval_batches,
        },
        "throughput_timing": "batch preparation, host-to-device transfer, forward, backward, and optimizer update",
        "points": points,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote metrics: {args.output}")


if __name__ == "__main__":
    main()
