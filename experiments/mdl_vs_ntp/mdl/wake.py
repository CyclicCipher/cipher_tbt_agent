"""Wake: the model proposes programs for fresh tasks, the executor verifies them, verified ones enter the buffer.

Each round: W tasks (family from the arm's distribution, fresh instance), N samples per task at T = 1.0 under the grammar
and the arm's active library, expanded and executed against all 4 demos plus the query. The buffer is keyed by
`(family_id, arg assignment)` and holds verified EXPANDED programs — 1 random one for `exit`, up to 4 distinct for the
rest — capped at 30k entries FIFO.

At batch time a stored entry is re-checked on FRESH inputs generated from its stored args; any program that disagrees
(a spurious solution) is dropped before a target is picked.
"""
from __future__ import annotations

import collections

import numpy as np
import torch

from env.dsl import run
from env.families import fresh_io, make_instance
from env.tokens import (DIG0, EPROG, PAD, PROG, PROG_BUDGET, PROMPT_I, encode_I, is_macro, parse_program,
                        prog_tokens, prim_units)
from mdl.library import primitive_length, segment_units

BUFFER_CAP = 30000
W_TASKS, N_SAMPLES = 256, 16


def prompt_I(inst):
    toks, _m = encode_I(inst, [])
    return toks[:PROMPT_I]


def verify(prog, inst):
    for x, y in inst.demos:
        if run(prog, x) != y:
            return False
    xq, yq = inst.query
    return run(prog, xq) == yq


class Buffer:
    def __init__(self, per_entry=4):
        self.entries = collections.OrderedDict()               # (fid, args) -> [programs]
        self.per_entry = per_entry

    def __len__(self):
        return len(self.entries)

    def add(self, fid, args, progs, rng):
        key = (fid, tuple(args))
        cur = self.entries.get(key)
        if cur is None:
            if len(self.entries) >= BUFFER_CAP:
                self.entries.popitem(last=False)
            cur = self.entries[key] = []
        if self.per_entry == 1:                                    # exit: ONE random verified program
            if not cur:
                cur.append(progs[int(rng.integers(len(progs)))])
            return
        for p in progs:
            if p not in cur and len(cur) < self.per_entry:
                cur.append(p)

    def sample_keys(self, rng, n):
        keys = list(self.entries)
        idx = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
        return [keys[i] for i in idx]

    def best_programs(self, fid, k=8):
        """Up to k distinct programs for a family across its entries, shortest first (for the proposer archive)."""
        seen = []
        for (f, _a), progs in self.entries.items():
            if f == fid:
                for p in progs:
                    if p not in seen:
                        seen.append(p)
        seen.sort(key=primitive_length)
        return seen[:k]

    def state(self):
        return dict(entries=list(self.entries.items()), per_entry=self.per_entry)

    def load_state(self, st):
        self.entries = collections.OrderedDict(st["entries"])
        self.per_entry = st["per_entry"]


def wake_round(adapter, env, library, grammar, sample_fids, rng, trng, buffer, rnd, dev, W=W_TASKS, N=N_SAMPLES):
    """Returns (stats dict, tokens for TE accounting)."""
    fids = sample_fids(W)
    insts = [make_instance(env.fam(f), int(rng.integers(1 << 30)), "wake") for f in fids]
    prompts = torch.tensor([prompt_I(i) for i in insts], dtype=torch.long, device=dev)
    prompts = prompts.repeat_interleave(N, dim=0)
    import time as _t
    t0 = _t.time()
    out = adapter.generate(prompts, PROG_BUDGET + 1, 1.0, grammar, "prog", rng=trng)     # program, </prog>, EOS
    gpu_seconds = _t.time() - t0
    gen = out[:, PROMPT_I:].tolist()
    te = int((out != PAD).sum())
    stats = collections.Counter()
    by_cat = collections.defaultdict(lambda: [0, 0])
    usage = collections.Counter()
    n_verified = 0
    for ti, inst in enumerate(insts):
        fam = env.fam(inst.fid)
        found = []
        for j in range(N):
            toks = gen[ti * N + j]
            body = toks[:toks.index(EPROG)] if EPROG in toks else None
            if body is None:
                stats["malformed"] += 1
                continue
            prog = parse_program(body, library)
            if prog is None:
                stats["malformed"] += 1
                continue
            if verify(prog, inst):
                found.append(prog)
                for t in body:
                    if is_macro(t):
                        usage[t] += 1
        by_cat[fam.category][1] += 1
        if found:
            by_cat[fam.category][0] += 1
            n_verified += 1
            buffer.add(inst.fid, inst.args, found, rng)
    stats["tasks"], stats["verified_tasks"] = W, n_verified
    stats["verified_rate"] = n_verified / W
    stats["by_category"] = {k: v[0] / v[1] for k, v in by_cat.items()}
    stats["macro_usage"] = {int(k): int(v) for k, v in usage.items()}
    stats["gpu_seconds"] = gpu_seconds
    return dict(stats), te


def buffer_batch(buffer, env, library, arm, n, rng):
    """`n` I-format sequences from the buffer: fresh I/O for the stored args, spurious programs dropped, the arm's
    target picked. Returns [(tokens, mask)]; may return fewer than n."""
    out = []
    for key in buffer.sample_keys(rng, n):
        fid, args = key
        fam = env.fam(fid)
        io = fresh_io(fam, args, rng, 5)
        progs = buffer.entries.get(key, [])
        keep = [p for p in progs if all(run(p, x) == y for x, y in io)]
        if not keep:
            buffer.entries.pop(key, None)
            continue
        buffer.entries[key] = keep
        if arm == "exit":
            target = prim_units(keep[0])
        elif arm == "mdl_nolib":
            target = prim_units(min(keep, key=primitive_length))
        else:
            forms = [segment_units(p, library) for p in keep]
            costs = [_form_cost(f, library) for f in forms]
            target = forms[int(np.argmin(costs))]
        toks = prog_tokens(target)
        if len(toks) + 1 > PROG_BUDGET:
            continue

        class _I:                                                # the encoder only needs .demos
            demos = io[:4]
        out.append(encode_I(_I, toks))
    return out


def _form_cost(units, library):
    from mdl.library import LOG2_10, c_unit
    cu = c_unit(len(library))
    n_args = sum((1 if u[2] is not None else 0) if u[0] == "P" else len(u[2]) for u in units)
    return (len(units) + 1) * cu + n_args * LOG2_10
