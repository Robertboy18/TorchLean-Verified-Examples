#!/usr/bin/env python3
"""
Train and export Neel Somani's small verifiable Transformer with his own code.

This runs `scripts/small/train.py` and then `scripts/small/extract_weights.py`
from `neelsomani/verifiable-transformers` at the pinned commit below, unmodified.
The training settings are the upstream defaults: seed 0, learning rate 0.003,
weight decay 0.01, batch size 64, at most 5000 steps, evaluation every 100 steps,
and a stop at the first evaluation where both tasks reach perfect candidate
accuracy. Upstream calls `set_seed` before it builds the model, so the
initialization and the data order both come from that seed.

The wrapper only adds what a reproducible run needs:

1. it refuses to run unless the upstream checkout is at the pinned commit and
   Transformers is the version pinned by upstream's `requirements.txt`
   (with that version upstream's sparsemax patch runs as written);
2. it turns on PyTorch's deterministic algorithms before training;
3. it deletes Trainer's intermediate step checkpoints and the duplicate
   `checkpoint-best` (after checking it holds the same tensors as
   `checkpoint-final`), then records versions, device, and file hashes in
   `provenance.json`.

Pick the GPU with CUDA_VISIBLE_DEVICES (upstream defaults to GPU 0 otherwise).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "8e3c54bad2494c17c91109fdb58cf9e2f1c98efc"
PINNED_TRANSFORMERS = "4.49.0"
WEEK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("artifacts/upstream-small-gpt")

# Upstream defaults from scripts/small/train.py at the pinned commit, spelled out
# so the recorded command is the command that ran.
UPSTREAM_TRAIN_ARGS = [
    "--learning_rate", "0.003",
    "--weight_decay", "0.01",
    "--batch_size", "64",
    "--max_steps", "5000",
    "--eval_every", "100",
    "--seed", "0",
    "--task_sampling", "balanced",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_repo(path: str | None) -> Path:
    raw = path or os.environ.get("VERIFIABLE_TRANSFORMERS_REPO")
    if not raw:
        raise SystemExit("pass --repo or set VERIFIABLE_TRANSFORMERS_REPO")
    repo = Path(raw).expanduser().resolve()
    if not (repo / "scripts" / "small" / "train.py").exists():
        raise SystemExit(f"not a verifiable-transformers checkout: {repo}")
    return repo


def check_pins(repo: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"upstream checkout is at {head}, expected {PINNED_COMMIT}")
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"upstream checkout has local changes:\n{dirty}")
    import transformers

    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise SystemExit(
            f"transformers {transformers.__version__} found, upstream pins {PINNED_TRANSFORMERS}"
        )


def make_deterministic() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def same_tensors(a: Path, b: Path) -> bool:
    from safetensors.torch import load_file

    ta, tb = load_file(str(a)), load_file(str(b))
    return ta.keys() == tb.keys() and all(ta[k].equal(tb[k]) for k in ta)


def tidy_outputs(out: Path) -> None:
    final = out / "checkpoint-final"
    best = out / "checkpoint-best"
    if best.exists():
        if not same_tensors(best / "model.safetensors", final / "model.safetensors"):
            raise SystemExit("checkpoint-best and checkpoint-final hold different tensors")
        shutil.rmtree(best)
    for step_dir in out.glob("checkpoint-[0-9]*"):
        shutil.rmtree(step_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--repo", default=None, help="verifiable-transformers checkout at the pinned commit")
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="output directory, relative to the Week 2 folder unless absolute",
    )
    args = p.parse_args()

    repo = resolve_repo(args.repo)
    check_pins(repo)
    make_deterministic()

    # Run from the Week 2 folder so the paths Trainer records are repo-relative.
    os.chdir(WEEK_DIR)
    out = args.output_dir
    (WEEK_DIR / out if not out.is_absolute() else out).mkdir(parents=True, exist_ok=True)

    train_script = repo / "scripts" / "small" / "train.py"
    train_argv = [str(train_script), "--output_dir", str(out), *UPSTREAM_TRAIN_ARGS]
    saved_argv = sys.argv
    sys.argv = train_argv
    try:
        runpy.run_path(str(train_script), run_name="__main__")
    finally:
        sys.argv = saved_argv

    out_abs = (WEEK_DIR / out).resolve() if not out.is_absolute() else out
    tidy_outputs(out_abs)

    export_script = repo / "scripts" / "small" / "extract_weights.py"
    subprocess.run(
        [sys.executable, str(export_script),
         "--checkpoint", str(out_abs / "checkpoint-final"),
         "--output", str(out_abs / "smt_weights.json")],
        cwd=str(repo), check=True,
    )

    import torch
    import transformers

    metrics = json.loads((out_abs / "metrics.json").read_text())
    history = metrics["history"]
    provenance = {
        "upstream_repo": "https://github.com/neelsomani/verifiable-transformers",
        "upstream_commit": PINNED_COMMIT,
        "train_command": ["python", "scripts/small/train.py", "--output_dir", str(out),
                          *UPSTREAM_TRAIN_ARGS],
        "export_command": ["python", "scripts/small/extract_weights.py",
                           "--checkpoint", f"{out}/checkpoint-final",
                           "--output", f"{out}/smt_weights.json"],
        "stopped_at_step": history[-1]["step"] if history else None,
        "all_tasks_perfect_at_stop": history[-1]["all_tasks_perfect"] if history else None,
        "deterministic_algorithms": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "sha256": {
            "checkpoint-final/model.safetensors":
                sha256_file(out_abs / "checkpoint-final" / "model.safetensors"),
            "smt_weights.json": sha256_file(out_abs / "smt_weights.json"),
        },
    }
    (out_abs / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote outputs to {out_abs}")


if __name__ == "__main__":
    main()
