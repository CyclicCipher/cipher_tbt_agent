import itertools
import math

import numpy as np

from env.dsl import ARITY, HOLE, N_OPS, fill, prog_to_str
from mdl.library import LOG2_10, Library, build_matches, c_unit, dp_cost, segment_units


def random_prog(rng, n):
    out = []
    for _ in range(n):
        op = int(rng.integers(N_OPS))
        out.append((op, int(rng.integers(10)) if ARITY[op] else None))
    return tuple(out)


def random_pattern(rng, n):
    out = []
    for _ in range(n):
        op = int(rng.integers(N_OPS))
        if ARITY[op]:
            out.append((op, HOLE if rng.random() < 0.5 else int(rng.integers(10))))
        else:
            out.append((op, None))
    return tuple(out)


def random_library(rng, k):
    lib = Library()
    for _ in range(k):
        lib.add(random_pattern(rng, int(rng.integers(2, 5))), 0)
    return lib


def expand_units(units, lib):
    out = []
    for u in units:
        if u[0] == "P":
            out.append((u[1], u[2]))
        else:
            out += list(lib.expand(u[1], list(u[2])))
    return tuple(out)


def test_expand_segment_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(5000):
        lib = random_library(rng, int(rng.integers(0, 6)))
        p = random_prog(rng, int(rng.integers(1, 12)))
        assert expand_units(segment_units(p, lib), lib) == p


def brute_force(prog, lib):
    """Every segmentation of a short program by exhaustive cut placement."""
    macros = lib.members()
    cu = c_unit(len(lib))
    n = len(prog)
    best = math.inf

    def rec(i, cost):
        nonlocal best
        if i == n:
            best = min(best, cost + cu)
            return
        rec(i + 1, cost + cu + (LOG2_10 if prog[i][1] is not None else 0.0))
        for m in macros:
            ln = m.length
            if i + ln <= n and all(
                    prog[i + k][0] == m.pattern[k][0] and (m.pattern[k][1] in (HOLE, prog[i + k][1]))
                    for k in range(ln)):
                rec(i + ln, cost + cu + m.holes * LOG2_10)
    rec(0, 0.0)
    return best


def test_dp_optimal_vs_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(300):
        lib = random_library(rng, int(rng.integers(0, 5)))
        p = random_prog(rng, int(rng.integers(1, 7)))
        dp = dp_cost(p, build_matches(prog_to_str(p), lib.members()), c_unit(len(lib)))
        assert abs(dp - brute_force(p, lib)) < 1e-9
