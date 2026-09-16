"""Planted latent macros and semantic fingerprints. No model is ever told these names; ground-truth programs are
always written fully expanded into primitives. The banned adjacencies are never adjacent in training and are required in
H-comp.
"""
from __future__ import annotations

import numpy as np

from .dsl import CANON_ARG, HOLE, N_OPS, ARITY, MAX_LEN, fill, parse_pattern, run
from .noise import derive_seed

TRAIN_MACROS = [parse_pattern(s) for s in [
    "sort rev", "fgt ? sort", "csum take ?", "rev rot ?", "swp rev",
    "diff uniq", "neg add ?", "mul ? fgt ?", "feven csum rev", "uniq sort take ?"]]
HELD_MACROS = [parse_pattern(s) for s in ["add ? swp csum", "drop ? neg sort"]]
ALL_MACROS = TRAIN_MACROS + HELD_MACROS               # P0..P9, Q0, Q1
N_TRAIN_MACROS = len(TRAIN_MACROS)
BANNED = [(0, 2), (1, 5), (2, 7), (3, 8), (4, 9), (5, 6), (6, 0), (7, 3)]
BANNED_SET = set(BANNED)


def fixed_inputs(n=64, seed=derive_seed("fingerprint-inputs")):
    """The 64 fixed inputs every fingerprint is taken on (lengths 4-8, uniform digits)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ln = int(rng.integers(4, MAX_LEN + 1))
        out.append(tuple(int(v) for v in rng.integers(0, 10, ln)))
    return out


FIXED_INPUTS = fixed_inputs()


def canonical_fill(pat):
    """Holes set to the canonical value of their op."""
    return tuple((op, CANON_ARG[op] if spec == HOLE else spec) for op, spec in pat)


def fingerprint(pat, extra=()):
    """Outputs on the 64 fixed inputs with holes at canonical values, plus any extra discriminator (templates add
    their free-arg count). Hashable."""
    prog = canonical_fill(pat)
    return (tuple(run(prog, x) for x in FIXED_INPUTS),) + tuple(extra)


def primitive_fingerprints():
    """Every single primitive (with canonical arg where arity 1), for 'semantically equal to a primitive' checks."""
    out = {}
    for op in range(N_OPS):
        pat = ((op, HOLE if ARITY[op] else None),)
        out[fingerprint(pat)] = pat
    return out


PRIM_FPS = primitive_fingerprints()
