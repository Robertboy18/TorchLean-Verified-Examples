"""Shared helpers for certificate generators."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for an input file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lean_nat_list(xs: Iterable[int]) -> str:
    """Render natural-number data as a Lean list literal."""
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


def lean_nested_nat_list(xss: Iterable[Iterable[int]]) -> str:
    """Render nested natural-number data as a Lean list literal."""
    return "[" + ", ".join(lean_nat_list(xs) for xs in xss) + "]"
