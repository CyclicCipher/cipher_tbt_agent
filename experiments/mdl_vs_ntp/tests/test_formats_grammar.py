import numpy as np
import torch

from env.dsl import HOLE, run
from env.grammar import Grammar
from env.tokens import (ANS, DEF, EDEF, EOS, EPROG, LEN_I, LEN_P, LEN_T, PAD, PROG, PROG_BUDGET, PROMPT_I,
                        PROMPT_P, V, encode_I, encode_P, encode_T, parse_def, parse_program, probe_positions,
                        prog_tokens, prim_units)
from mdl.library import Library


def test_fixed_lengths_and_masks(env):
    inst = env.probe[env.by_split["train"][0]][0]
    t, m = encode_T(inst)
    assert len(t) == LEN_T and len(m) == LEN_T
    assert sum(m) == 41 and m[-1] and t[-1] == EOS                 # 40 output slots + EOS
    assert t[2 + 76] == 9 and t[2 + 76 + 10] == ANS
    pos = probe_positions()
    assert len(pos) == 32 and all(m[p] for p in pos)
    fam = env.fam(inst.fid)
    toks = prog_tokens(prim_units(fam.program(inst.args)))
    s, mm = encode_I(inst, toks)
    assert len(s) == LEN_I and len(mm) == LEN_I
    assert s[PROMPT_I - 1] == PROG and sum(mm) == len(toks) + 2     # program + </prog> + EOS
    assert all(not x for x in mm[:PROMPT_I])
    p = encode_P([toks] * 5, [])
    assert len(p) == LEN_P and p[PROMPT_P - 1] == DEF and all(p[2 + 30 * k] == PROG for k in range(5))


def test_grammar_10k_samples_parse_expand_execute():
    lib = Library()
    from env.dsl import parse_pattern
    for s in ("sort rev", "fgt ? sort", "mul ? fgt ?"):
        lib.add(parse_pattern(s), 0)
    g = Grammar(lib)
    rng = np.random.default_rng(0)
    x = (3, 1, 4, 1, 5, 9)
    n_macro_used = 0
    for i in range(10000):
        mode = "prog" if i % 2 == 0 else "def"
        st, toks = g.start(mode), []
        for _ in range(PROG_BUDGET + 1 if mode == "prog" else 20):
            allowed = sorted(g.allowed(st))
            t = int(allowed[rng.integers(len(allowed))])
            toks.append(t)
            st = g.advance(st, t)
            if st[1] == "pad":
                break
        assert toks[-1] == EOS, toks
        body = toks[:-2]
        if mode == "prog":
            assert toks[-2] == EPROG and len(toks) - 1 <= PROG_BUDGET
            prog = parse_program(body, lib)
            assert prog is not None, body
            n_macro_used += any(t >= 41 for t in body)
            run(prog, x)
        else:
            assert toks[-2] == EDEF
            pat = parse_def(body, lib)
            assert pat is not None, body
            assert 2 <= len(pat) and sum(1 for _o, s in pat if s == HOLE) <= 2
    assert n_macro_used > 0


def test_inactive_slots_masked_without_library():
    g = Grammar(None)
    assert not any(t >= 41 for t in g.allowed(g.start("prog")))
    m = g.masks([g.start("prog")] * 3, "cpu")
    assert m.shape == (3, V) and not m[:, 41:].any()
