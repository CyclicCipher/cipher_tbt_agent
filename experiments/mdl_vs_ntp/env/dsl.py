"""The DSL: 14 total ops over digit lists (values 0-9, length <= 8), a tight pure-Python executor, and the shared
pattern representation used by planted macros, templates and the library.

A PROGRAM is a tuple of units `(op, arg)` with `arg is None` for arity-0 ops. A PATTERN is the same shape with
`arg in {None, 0..9, HOLE}`; `HOLE` marks an argument slot to be filled at use. Every hole is a distinct slot and they
are numbered by order of appearance, so a pattern needs no explicit hole ids.

The executor is the hot path of wake (256 tasks x 16 samples x 5 I/O per round) and of E3 (128 samples x task), so it is
written as closure-free branch tables over Python tuples: no numpy call overhead on 8-element lists, no allocation
beyond the output. ~1 us per op on this hardware.
"""
from __future__ import annotations

OPS = ["add", "mul", "neg", "rev", "sort", "rot", "take", "drop", "fgt", "feven", "uniq", "csum", "diff", "swp"]
OP_ID = {n: i for i, n in enumerate(OPS)}
N_OPS = len(OPS)
ARITY = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
HOLE = -1
MAX_LEN = 8

ADD, MUL, NEG, REV, SORT, ROT, TAKE, DROP, FGT, FEVEN, UNIQ, CSUM, DIFF, SWP = range(N_OPS)

# Generator argument ranges (what templates draw fixed args from and what free args are resampled from).
ARG_RANGE = {ADD: tuple(range(1, 10)), MUL: (2, 3, 4, 6, 7, 8, 9), ROT: tuple(range(1, 8)),
             TAKE: tuple(range(1, 8)), DROP: (1, 2, 3), FGT: tuple(range(0, 7))}
# Canonical hole values for fingerprints (add 3, mul 3, rot 2, take 3, drop 1, fgt 2).
CANON_ARG = {ADD: 3, MUL: 3, ROT: 2, TAKE: 3, DROP: 1, FGT: 2}

# Single-char op codes for the string encoding the segmentation kernel prefilters with regex. Letters never collide
# with digits or '_', which is what lets a pattern match only at unit boundaries.
OP_CH = "abcdefghijklmn"
CH_OP = {c: i for i, c in enumerate(OP_CH)}


def _add(x, k):
    return tuple((v + k) % 10 for v in x)


def _mul(x, k):
    return tuple((v * k) % 10 for v in x)


def _neg(x, _):
    return tuple((10 - v) % 10 for v in x)


def _rev(x, _):
    return x[::-1]


def _sort(x, _):
    return tuple(sorted(x))


def _rot(x, k):
    n = len(x)
    if n == 0:
        return x
    k %= n
    return x[k:] + x[:k]


def _take(x, k):
    return x[:k]


def _drop(x, k):
    return x[k:]


def _fgt(x, k):
    return tuple(v for v in x if v > k)


def _feven(x, _):
    return tuple(v for v in x if v % 2 == 0)


def _uniq(x, _):
    seen, out = set(), []
    for v in x:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


def _csum(x, _):
    out, s = [], 0
    for v in x:
        s = (s + v) % 10
        out.append(s)
    return tuple(out)


def _diff(x, _):
    return tuple((x[i + 1] - x[i]) % 10 for i in range(len(x) - 1))


def _swp(x, _):
    out = list(x)
    for i in range(0, len(x) - 1, 2):
        out[i], out[i + 1] = out[i + 1], out[i]
    return tuple(out)


FN = [_add, _mul, _neg, _rev, _sort, _rot, _take, _drop, _fgt, _feven, _uniq, _csum, _diff, _swp]


def run(prog, x):
    """Execute a fully-expanded program on one input tuple. Total for any digit argument 0-9."""
    for op, arg in prog:
        x = FN[op](x, arg)
    return x


def run_many(prog, xs):
    return [run(prog, x) for x in xs]


def prog_to_str(prog):
    """Two chars per unit: op letter + (digit | '_')."""
    return "".join(OP_CH[op] + ("_" if arg is None else str(arg)) for op, arg in prog)


def str_to_prog(s):
    return tuple((CH_OP[s[i]], None if s[i + 1] == "_" else int(s[i + 1])) for i in range(0, len(s), 2))


def pattern_to_regex(pat):
    """A pattern as a regex over the string encoding: const -> its digit, hole -> any digit, arity-0 -> '_'."""
    parts = []
    for op, spec in pat:
        parts.append(OP_CH[op] + ("_" if spec is None else (r"\d" if spec == HOLE else str(spec))))
    return "".join(parts)


def holes_of(pat):
    return sum(1 for _op, spec in pat if spec == HOLE)


def fill(pat, args):
    """Instantiate a pattern's holes, in order of appearance, from `args`."""
    it = iter(args)
    return tuple((op, next(it) if spec == HOLE else spec) for op, spec in pat)


def pattern_str(pat):
    """Human-readable, e.g. 'fgt ? sort'."""
    out = []
    for op, spec in pat:
        out.append(OPS[op])
        if spec == HOLE:
            out.append("?")
        elif spec is not None:
            out.append(str(spec))
    return " ".join(out)


def parse_pattern(text):
    """'fgt ? sort' -> pattern. Digits after an arity-1 op are consts, '?' a hole."""
    toks, out, i = text.split(), [], 0
    while i < len(toks):
        op = OP_ID[toks[i]]
        i += 1
        if ARITY[op]:
            spec = HOLE if toks[i] == "?" else int(toks[i])
            i += 1
        else:
            spec = None
        out.append((op, spec))
    return tuple(out)


REDUNDANT_SAME = {REV, NEG, SORT, UNIQ, FEVEN, ADD, MUL, ROT, FGT, TAKE, DROP}


def has_redundant_adjacency(ops):
    """`rev rev`, `neg neg`, `sort sort`, `uniq uniq`, `feven feven`, or a same-op pair of add/mul/rot/fgt/take/drop."""
    return any(a == b and a in REDUNDANT_SAME for a, b in zip(ops, ops[1:]))
