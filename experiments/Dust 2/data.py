"""Task 1: character-level language modelling on classical Latin (Cicero + Livy, ~11M characters).

Lower-cased, whitespace runs collapsed to one space, characters outside ALPHABET dropped. The whole corpus sits on the GPU
as uint8; a batch is B random windows of T + 1 characters. The last 5% is held out.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "corpora" / "latin_classical"
FILES = ["cicero.txt", "livy.txt"]
ALPHABET = " abcdefghijklmnopqrstuvwxyz.,;:?!'\"()-"
V = len(ALPHABET)
_CACHE = Path(__file__).resolve().parent / "runs" / "latin_chars.pt"


def load(dev="cuda"):
    if _CACHE.exists():
        ids = torch.load(_CACHE)
    else:
        text = "".join(open(ROOT / f, encoding="utf-8", errors="ignore").read() for f in FILES).lower()
        text = re.sub(r"\s+", " ", text)
        keep = {c: i for i, c in enumerate(ALPHABET)}
        ids = torch.tensor([keep[c] for c in text if c in keep], dtype=torch.uint8)
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ids, _CACHE)
    n = int(len(ids) * 0.95)
    return ids[:n].to(dev), ids[n:].to(dev)


def batch(ids, B, T, g):
    """(B, T) inputs and (B, T) next-character targets, int64."""
    start = torch.randint(0, len(ids) - T - 1, (B,), generator=g, device=ids.device)
    idx = start[:, None] + torch.arange(T + 1, device=ids.device)[None]
    w = ids[idx].long()
    return w[:, :-1], w[:, 1:]
