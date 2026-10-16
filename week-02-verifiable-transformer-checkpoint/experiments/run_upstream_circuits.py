#!/usr/bin/env python3
"""
Extract and verify the quote/bracket circuits of our checkpoint with Neel's code.

For each task this runs, unmodified and at the pinned upstream commit:

1. `scripts/small/extract.py --threshold_sweep` (ACDC edge pruning with zero
   ablation, candidate KL, thresholds 0.001 to 0.05, keeping the smallest
   circuit that agrees with the full model on every prompt);
2. `scripts/small/verify.py --sanity_check` with the three SMT properties
   Neel ran for his own small-model results: projected functional
   equivalence, edge necessity, and continuous robustness at epsilon 0.01.

Both scripts run on the CPU so the ACDC decisions do not depend on the GPU.
Outputs go to `artifacts/upstream-small-gpt/circuits/<task>/`, together with a
`provenance.json` that records the commands, versions, and the SHA-256 of the
checkpoint and weight export the scripts read.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_upstream_model import (  # noqa: E402
    PINNED_COMMIT,
    WEEK_DIR,
    check_pins,
    resolve_repo,
    sha256_file,
)

RUN = Path("artifacts/upstream-small-gpt")
CHECKPOINT = RUN / "checkpoint-final"
WEIGHTS = RUN / "smt_weights.json"
TASKS = ["quote_close", "bracket_type"]
PROPERTIES = ["functional_equivalence", "edge_necessity", "continuous_robustness"]


def run_task(repo: Path, task: str, log_dir: Path) -> dict:
    out = RUN / "circuits" / task
    (WEEK_DIR / out).mkdir(parents=True, exist_ok=True)
    extract = [
        "scripts/small/extract.py",
        "--model_path", str(CHECKPOINT),
        "--task", task,
        "--threshold_sweep",
        "--output_dir", str(out),
    ]
    verify = [
        "scripts/small/verify.py",
        "--task", task,
        "--sanity_check",
        "--checkpoint", str(CHECKPOINT),
        "--weights_path", str(WEIGHTS),
        "--circuit_path", str(out / "circuit.json"),
        "--output_dir", str(out / "verification"),
        "--properties", *PROPERTIES,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    timings = {}
    for name, cmd in [("extract", extract), ("verify", verify)]:
        t0 = time.time()
        with (log_dir / f"{task}.{name}.log").open("w") as log:
            proc = subprocess.run(
                [sys.executable, str(repo / cmd[0]), *cmd[1:]],
                cwd=str(WEEK_DIR), env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        timings[name] = round(time.time() - t0, 1)
        # verify.py exits 1 after writing its results when the PyTorch sanity check
        # fails; the results file is still the answer for this checkpoint.
        if proc.returncode != 0 and not (name == "verify" and
                                         (WEEK_DIR / out / "verification" /
                                          "verification_results.json").exists()):
            raise RuntimeError(f"{task} {name} failed with exit code {proc.returncode}")
    return {
        "extract_command": ["python", *extract],
        "verify_command": ["python", *verify],
        "seconds": timings,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--repo", default=None, help="verifiable-transformers checkout at the pinned commit")
    p.add_argument("--log-dir", type=Path, default=WEEK_DIR / "build" / "circuit-logs",
                   help="where to write the raw extract/verify logs (not committed)")
    args = p.parse_args()

    repo = resolve_repo(args.repo)
    check_pins(repo)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=len(TASKS)) as pool:
        runs = dict(zip(TASKS, pool.map(lambda t: run_task(repo, t, log_dir), TASKS)))

    import torch
    import transformers
    import z3

    provenance = {
        "upstream_repo": "https://github.com/neelsomani/verifiable-transformers",
        "upstream_commit": PINNED_COMMIT,
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "z3": z3.get_version_string(),
        "sha256": {
            "checkpoint-final/model.safetensors":
                sha256_file(WEEK_DIR / CHECKPOINT / "model.safetensors"),
            "smt_weights.json": sha256_file(WEEK_DIR / WEIGHTS),
        },
        "tasks": runs,
    }
    path = WEEK_DIR / RUN / "circuits" / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
