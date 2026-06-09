#!/usr/bin/env python3
"""
Run the exact `neelsomani/verifiable-transformers` small-model pipeline.

This wrapper prepares Neel's small-model run that Lean later checks:

1. train the repo's actual GPT2LMHeadModel-based small verifiable Transformer;
2. use CUDA if available;
3. export SMT-compatible weights;
4. optionally run circuit extraction / verification.

Why the wrapper exists:
Neel's repo was written against a Transformers version whose GPT2Attention
forward signature used `layer_past`; the local environment currently has a newer
Transformers release that passes `past_key_values` / `cache_position`.  The
compatibility shim below preserves the repo's sparsemax attention semantics while
accepting the newer signature.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MethodType


def repo_root_from_args(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    default = Path(__file__).resolve().parents[2] / "verifiable-transformers"
    return Path(os.environ.get("VERIFIABLE_TRANSFORMERS_REPO", str(default))).resolve()


def patch_sparsemax_forward(vt_train):
    """Patch the upstream sparsemax forward for newer Transformers signatures."""

    def compat_gpt2_forward_with_sparsemax(
        self,
        hidden_states,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError("cross-attention requires q_attn")
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = query.view(*query.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(*key.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(*value.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)

        # The small verification/training path sets use_cache=False.  If a caller
        # passes a cache object, reject rather than silently using
        # a different KV-cache semantics from the SMT encoder.
        if past_key_values is not None or use_cache:
            raise ValueError("This exact SMT-friendly training wrapper expects use_cache=False")

        attn_output, attn_weights = vt_train.sparsemax_attention_forward(
            self, query, key, value, attention_mask, head_mask
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, (attn_weights if output_attentions else None)

    vt_train.gpt2_forward_with_sparsemax = compat_gpt2_forward_with_sparsemax


def train_small(args: argparse.Namespace, repo: Path) -> None:
    sys.path.insert(0, str(repo))
    from scripts.small import train as vt_train
    from scripts.small.config import get_default_config
    from transformers import TrainingArguments

    patch_sparsemax_forward(vt_train)

    import torch

    config = get_default_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(args.output_dir / "config.json")

    model = vt_train.create_small_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    # Fail-fast: verify sparsemax is actually used, as upstream does.
    vt_train._sparsemax_call_count = 0
    dummy = torch.randint(0, config.vocab_size, (1, config.max_seq_len), device=device)
    with torch.no_grad():
        model(dummy)
    if vt_train._sparsemax_call_count <= 0:
        raise RuntimeError("sparsemax patch was not exercised")

    train_dataset = vt_train.SmallVerifiableDataset(task_sampling=args.task_sampling)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.eval_every,
        save_steps=args.eval_every,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        bf16=False,
        fp16=False,
        max_grad_norm=1.0,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    callback = vt_train.TaskEvaluationCallback(
        eval_every_n_steps=args.eval_every,
        output_dir=str(args.output_dir),
    )
    trainer = vt_train.FinalTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vt_train.collate_fn,
        callbacks=[callback],
    )

    trainer.train()
    final_dir = args.output_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))

    final_metrics = {}
    for task_name in ["quote_close", "bracket_type"]:
        final_metrics[task_name] = vt_train.evaluate_task(model, task_name, device)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump({"final_metrics": final_metrics, "history": callback.metrics_history}, f, indent=2)


def export_weights(repo: Path, output_dir: Path) -> None:
    script = repo / "scripts" / "small" / "extract_weights.py"
    checkpoint = output_dir / "checkpoint-final"
    out = output_dir / "smt_weights.json"
    subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint), "--output", str(out)],
        cwd=str(repo),
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=None, help="Path to cloned verifiable-transformers repo")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/upstream-small-gpt"))
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-sampling", choices=["balanced", "proportional", "all"], default="balanced")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()

    repo = repo_root_from_args(args.repo)
    if not repo.exists():
        raise FileNotFoundError(f"Repo not found: {repo}")

    args.output_dir = args.output_dir.expanduser().resolve()
    train_small(args, repo)
    if not args.no_export:
        export_weights(repo, args.output_dir)

    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
