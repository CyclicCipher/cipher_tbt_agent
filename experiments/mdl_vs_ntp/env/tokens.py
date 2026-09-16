"""Vocabulary (73 tokens, fixed order) and the three fixed-width formats T / I / P with their loss masks.

Every list is written fixed-width: 8 slots right-padded with `_`, so all prompts of a format have one length and no
padding masks are needed. Formats are built as Python lists of ints and batched by the caller.
"""
from __future__ import annotations

import torch

from .dsl import ARITY, HOLE, MAX_LEN, N_OPS, OPS

SPECIAL = ["PAD", "BOS", "EOS", "<T>", "<I>", "<P>", "<ex>", "<in>", "<out>", "<q>", "<ans>", "<prog>", "</prog>",
           "<def>", "</def>", "?", "_"]
N_SLOTS = 32
VOCAB = SPECIAL + [str(d) for d in range(10)] + OPS + [f"M{i}" for i in range(N_SLOTS)]
V = len(VOCAB)                                     # 73
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS, T_, I_, P_, EX, IN, OUT, Q, ANS, PROG, EPROG, DEF, EDEF, QMARK, BLANK = range(17)
DIG0 = 17
OP0 = DIG0 + 10                                    # 27
M0 = OP0 + N_OPS                                   # 41
assert M0 + N_SLOTS == V == 73

LEN_T, LEN_I, LEN_P = 98, 108, 176
PROMPT_I = 79
PROG_BUDGET = 28                                   # program tokens including </prog>
SLOT_P = 30                                        # each P-format context slot
DIGITS = set(range(DIG0, DIG0 + 10))


def is_op(t):
    return OP0 <= t < M0


def is_macro(t):
    return t >= M0


def tok_arity(t, library):
    """Arity of an op or macro token; macros' arity is their hole count."""
    if is_op(t):
        return ARITY[t - OP0]
    return library.holes(t - M0)


# ── lists ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def enc_list(x):
    return [DIG0 + v for v in x] + [BLANK] * (MAX_LEN - len(x))


def dec_list(toks):
    out = []
    for t in toks:
        if t == BLANK:
            break
        if t not in DIGITS:
            return None
        out.append(t - DIG0)
    if any(t != BLANK for t in toks[len(out):]):
        return None
    return tuple(out)


# ── programs (library form) ───────────────────────────────────────────────────────────────────────────────────────────
def prog_tokens(units):
    """Library-form units [('P', op, arg) | ('M', slot, (args...))] -> tokens (no </prog>)."""
    out = []
    for u in units:
        if u[0] == "P":
            out.append(OP0 + u[1])
            if u[2] is not None:
                out.append(DIG0 + u[2])
        else:
            out.append(M0 + u[1])
            out += [DIG0 + a for a in u[2]]
    return out


def prim_units(prog):
    return [("P", op, arg) for op, arg in prog]


def parse_program(toks, library):
    """Tokens after <prog> (up to and excluding </prog>) -> fully expanded primitive program, or None if malformed.
    Macro tokens must be active slots; their holes are filled by the digits that follow."""
    prog, i = [], 0
    while i < len(toks):
        t = toks[i]
        i += 1
        if is_op(t):
            op = t - OP0
            if ARITY[op]:
                if i >= len(toks) or toks[i] not in DIGITS:
                    return None
                prog.append((op, toks[i] - DIG0))
                i += 1
            else:
                prog.append((op, None))
        elif is_macro(t):
            slot = t - M0
            if not library.active(slot):
                return None
            h = library.holes(slot)
            args = toks[i:i + h]
            if len(args) < h or any(a not in DIGITS for a in args):
                return None
            i += h
            prog += list(library.expand(slot, [a - DIG0 for a in args]))
        else:
            return None
    return tuple(prog)


def parse_def(toks, library):
    """Tokens between <def> and </def> -> a pattern (op, None | const | HOLE), fully expanded through active macros.
    None if malformed, fewer than 2 or more than 4 units, more than 2 holes, or an expansion longer than 8 ops."""
    pat, i, units, holes = [], 0, 0, 0
    while i < len(toks):
        t = toks[i]
        i += 1
        units += 1
        if is_op(t):
            op = t - OP0
            if ARITY[op]:
                if i >= len(toks):
                    return None
                a = toks[i]
                i += 1
                if a == QMARK:
                    holes += 1
                    pat.append((op, HOLE))
                elif a in DIGITS:
                    pat.append((op, a - DIG0))
                else:
                    return None
            else:
                pat.append((op, None))
        elif is_macro(t):
            slot = t - M0
            if not library.active(slot):
                return None
            h = library.holes(slot)
            specs = []
            for _ in range(h):
                if i >= len(toks):
                    return None
                a = toks[i]
                i += 1
                if a == QMARK:
                    holes += 1
                    specs.append(HOLE)
                elif a in DIGITS:
                    specs.append(a - DIG0)
                else:
                    return None
            pat += list(library.expand_pattern(slot, specs))
        else:
            return None
    if not (2 <= units <= 4) or holes > 2 or len(pat) > 8:
        return None
    return tuple(pat)


# ── formats ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _demos(inst):
    out = []
    for x, y in inst.demos:
        out += [EX, IN] + enc_list(x) + [OUT] + enc_list(y)
    return out


def encode_T(inst):
    """98 tokens. Loss on all 40 output slots plus EOS."""
    toks = [BOS, T_] + _demos(inst) + [Q, IN] + enc_list(inst.query[0]) + [ANS] + enc_list(inst.query[1]) + [EOS]
    assert len(toks) == LEN_T
    mask = [False] * LEN_T
    for k in range(4):
        base = 2 + k * 19 + 11
        for j in range(MAX_LEN):
            mask[base + j] = True
    for j in range(MAX_LEN):
        mask[2 + 76 + 11 + j] = True
    mask[LEN_T - 1] = True
    return toks, mask


def T_prompt_len():
    """Position of the first answer slot in T: everything before it is the E1 prompt."""
    return 2 + 76 + 11


def probe_positions():
    """Output slots of demos 2-4 plus the answer: 32 target positions in T."""
    pos = []
    for k in range(1, 4):
        base = 2 + k * 19 + 11
        pos += list(range(base, base + MAX_LEN))
    pos += list(range(2 + 76 + 11, 2 + 76 + 11 + MAX_LEN))
    return pos


def encode_I(inst, program_toks):
    """<= 108 tokens, PAD-padded. Loss on program tokens, </prog>, EOS. Prompt = 79 tokens."""
    body = program_toks + [EPROG]
    assert len(body) <= PROG_BUDGET, len(body)
    toks = [BOS, I_] + _demos(inst) + [PROG] + body + [EOS]
    assert len(toks) - len(body) - 1 == PROMPT_I
    mask = [False] * PROMPT_I + [True] * (len(body) + 1)
    pad = LEN_I - len(toks)
    return toks + [PAD] * pad, mask + [False] * pad


def encode_P(context_progs, def_toks=None):
    """<= 176 tokens. Five context slots of exactly 30 tokens each, then <def> DEF </def> EOS (prompt = 153)."""
    toks = [BOS, P_]
    for p in context_progs:
        slot = [PROG] + p + [EPROG]
        assert len(slot) <= SLOT_P, len(slot)
        toks += slot + [BLANK] * (SLOT_P - len(slot))
    toks.append(DEF)
    assert len(toks) == 153
    if def_toks is not None:
        toks += def_toks + [EDEF, EOS]
    pad = LEN_P - len(toks)
    return toks + [PAD] * pad


PROMPT_P = 153


def batch(seqs, masks=None, device="cpu"):
    ids = torch.tensor(seqs, dtype=torch.long, device=device)
    if masks is None:
        return ids
    return ids, torch.tensor(masks, dtype=torch.bool, device=device)
