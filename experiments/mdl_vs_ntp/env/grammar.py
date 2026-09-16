"""Grammar-constrained decoding: a per-sequence state machine returning an allowed-token mask, applied to the logits
before temperature. Masks are cached per (library, state) and stacked per batch, so a 1024-sequence generation step costs
one dict lookup per sequence and one `torch.stack`.

Program states: `unit` allows ops, ACTIVE macro slots and </prog>; an arity-k token is followed by k `arg` states
(digits only); when the remaining budget only fits the pending args plus </prog>, completion is forced; after </prog>
comes EOS, then PAD. Def states are the same except args allow digits or `?` (masked once 2 holes are used), and </def>
is allowed only after >= 2 units and forced after 4. Inactive slots are always masked; for arms without a library every
slot is inactive. T-format answer decoding is masked to digits or `_`.
"""
from __future__ import annotations

import torch

from .dsl import ARITY, MAX_LEN
from .tokens import (BLANK, DIG0, EDEF, EOS, EPROG, M0, N_SLOTS, OP0, PAD, PROG_BUDGET, QMARK, V, is_macro, is_op)

MAX_DEF_UNITS, MIN_DEF_UNITS, MAX_DEF_HOLES = 4, 2, 2


class Grammar:
    """Built against one library snapshot; rebuild when the library changes."""

    def __init__(self, library=None):
        self.slots = [] if library is None else sorted(library.active_slots())
        self.slot_holes = {s: library.holes(s) for s in self.slots} if library is not None else {}
        self._cache = {}

    # states: (mode, phase, pend, units, holes, rem)
    @staticmethod
    def start(mode):
        if mode == "prog":
            return ("prog", "unit", 0, 0, 0, PROG_BUDGET)
        if mode == "def":
            return ("def", "unit", 0, 0, 0, 0)
        return ("ans", "slot", 0, 0, 0, MAX_LEN)

    def arity(self, t):
        return ARITY[t - OP0] if is_op(t) else self.slot_holes[t - M0]

    def allowed(self, state):
        """The set of allowed token ids in `state`."""
        mode, phase, pend, units, holes, rem = state
        if phase == "end":
            return {EOS}
        if phase == "pad":
            return {PAD}
        if mode == "ans":
            return set(range(DIG0, DIG0 + 10)) | {BLANK}
        if phase == "arg":
            out = set(range(DIG0, DIG0 + 10))
            if mode == "def" and holes < MAX_DEF_HOLES:
                out.add(QMARK)
            return out
        # unit
        out = set()
        if mode == "prog":
            for op in range(len(ARITY)):
                if 1 + ARITY[op] + 1 <= rem:
                    out.add(OP0 + op)
            for s in self.slots:
                if 1 + self.slot_holes[s] + 1 <= rem:
                    out.add(M0 + s)
            out.add(EPROG)
        else:
            if units < MAX_DEF_UNITS:
                out |= {OP0 + op for op in range(len(ARITY))} | {M0 + s for s in self.slots}
            if units >= MIN_DEF_UNITS:
                out.add(EDEF)
        return out

    def mask(self, state):
        m = self._cache.get(state)
        if m is None:
            m = torch.zeros(V, dtype=torch.bool)
            m[list(self.allowed(state))] = True
            self._cache[state] = m
        return m

    def masks(self, states, device):
        return torch.stack([self.mask(s) for s in states]).to(device, non_blocking=True)

    def advance(self, state, tok):
        mode, phase, pend, units, holes, rem = state
        if phase == "pad":
            return state
        if phase == "end":
            return (mode, "pad", 0, units, holes, rem)
        if mode == "ans":
            rem -= 1
            return (mode, "slot" if rem > 0 else "end", 0, units, holes, rem)
        if phase == "arg":
            if tok == QMARK:
                holes += 1
            pend -= 1
            return (mode, "arg" if pend > 0 else "unit", pend, units, holes, rem - 1)
        # unit
        if tok == EPROG or tok == EDEF:
            return (mode, "end", 0, units, holes, rem - 1)
        a = self.arity(tok)
        return (mode, "arg" if a > 0 else "unit", a, units + 1, holes, rem - 1)

    def replay(self, toks, mode):
        """Masks for each position of a generated token sequence (for `seq_logprob`): masks[i] constrained toks[i]."""
        st, out = self.start(mode), []
        for t in toks:
            out.append(self.mask(st))
            st = self.advance(st, t)
        return torch.stack(out)

    def finished(self, state):
        return state[1] == "pad"
