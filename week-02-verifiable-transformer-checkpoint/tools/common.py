"""Shared helpers for certificate generators."""

from __future__ import annotations

import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Iterable

# Week 2 folder; every default path below is relative to it, wherever the script runs from.
WEEK_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = WEEK_DIR / "artifacts" / "upstream-small-gpt"
GENERATED_DIR = WEEK_DIR / "VerifiableTransformers" / "Generated"

# The upstream commit whose code trained, exported, and verified the checkpoint.
PINNED_COMMIT = "8e3c54bad2494c17c91109fdb58cf9e2f1c98efc"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for an input file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_display_path(path: Path) -> str:
    """Return a checkout-independent path for generated certificate metadata."""
    parts = path.parts
    try:
        artifact_index = parts.index("artifacts")
    except ValueError:
        return path.name
    return Path(*parts[artifact_index:]).as_posix()


def resolve_path(path: Path) -> Path:
    """Resolve a command-line path (relative paths are relative to the current directory)."""
    return path.expanduser().resolve()


def check_upstream_checkout(repo: Path) -> None:
    """Refuse to run upstream code from anything but the pinned commit."""
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"upstream checkout {repo} is at {head}, expected {PINNED_COMMIT}")


def load_json_exact(path: Path):
    """Load JSON keeping decimal literals exact, so `4.4` stays 44/10."""
    return json.loads(path.read_text(), parse_float=Decimal)


def lean_rat(value) -> str:
    """Exact Lean `Rat` literal for a JSON decimal such as `0.55` or `4.4`."""
    d = Decimal(str(value))
    sign, digits, exponent = d.as_tuple()
    numerator = int("".join(map(str, digits)) or "0")
    if sign:
        numerator = -numerator
    if numerator == 0:
        return "0"
    if exponent >= 0:
        return str(numerator * 10 ** exponent)
    return f"{numerator} / {10 ** (-exponent)}"


def lean_string(value: str) -> str:
    """Lean string literal (the inputs here are plain ASCII identifiers and paths)."""
    if not isinstance(value, str) or any(c in value for c in '"\\\n'):
        raise ValueError(f"unexpected string for a Lean literal: {value!r}")
    return f'"{value}"'


def lean_nat_list(xs: Iterable[int]) -> str:
    """Render natural-number data as a Lean list literal."""
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


def lean_nested_nat_list(xss: Iterable[Iterable[int]]) -> str:
    """Render nested natural-number data as a Lean list literal."""
    return "[" + ", ".join(lean_nat_list(xs) for xs in xss) + "]"
