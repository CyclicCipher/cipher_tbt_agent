"""Description lengths: the macro library, the segmentation DP, and the per-round scoring kernel.

Unit costs: every emitted unit (op, macro, or terminator) costs `c_u = log2(14 + |Λ| + 1)`; every emitted argument
costs `c_a = log2(10)`. `L(y|Λ)` is the minimum over segmentations of units·c_u + args·c_a + c_u. A macro emits only its
holes as arguments; its consts are paid once, in `L_def`.

THE KERNEL, and why it is fast. A candidate must be scored against the whole archive (~540 families × ≤8 programs) in
well under 10 ms, so:
  * programs are encoded as strings (two chars per unit, op letter + digit/'_'), and every pattern becomes a regex, so
    the question "which programs could this macro touch at all" is one C-level `search` per program;
  * the library's own matches per program are cached for the round, so scoring a candidate re-runs the DP only on the
    programs its regex hits, with just the candidate's matches added;
  * the doc's "analytic c_u change for non-matching programs" is made EXACT rather than approximate: the round
    precomputes every program's optimal cost at c_u(|Λ|−1), c_u(|Λ|) and c_u(|Λ|+1), so a program the candidate cannot
    touch simply reads its cost at the larger unit cost (adding a macro raises c_u for everyone), and removing a member
    (`loo`) reads the smaller one. This costs two extra DP passes per round, not per candidate;
  * candidates are deduplicated by canonical pattern and their scores cached for the round.

Measured on this machine (CPU, full 294-family labelled archive = 2352 programs, a 32-macro library): Scorer build
57 ms per round, 0.47 ms per candidate, `loo` 2.4 ms and `support` 1.7 ms per member. The executor runs ~580k
program-executions/s, so a wake round's 20k verifications cost ~35 ms. Generation is what a round waits on.
"""
from __future__ import annotations

import math
import re

import numpy as np

from env.dsl import ARITY, HOLE, N_OPS, OP_CH, fill, holes_of, pattern_to_regex, prog_to_str
from env.macros import fingerprint

INF = float("inf")
LOG2_10, LOG2_11 = math.log2(10), math.log2(11)
MAX_MACROS = 32


def c_unit(n_lib):
    return math.log2(N_OPS + n_lib + 1)


def c_unit_def(n_lib):
    return math.log2(N_OPS + n_lib + 2)


def pattern_to_str(pat):
    """Pattern as a string with '?' for holes — the DEFINITION string that sub-macros are matched against."""
    return "".join(OP_CH[op] + ("_" if s is None else ("?" if s == HOLE else str(s))) for op, s in pat)


def pattern_to_defregex(pat):
    """A sub-macro as a regex over a definition string: hole matches a digit OR '?', const only its digit."""
    return "".join(OP_CH[op] + ("_" if s is None else (r"[\d?]" if s == HOLE else str(s))) for op, s in pat)


def canonical(pat):
    return tuple((op, s) for op, s in pat)


class Macro:
    __slots__ = ("pattern", "slot", "order", "round", "holes", "length", "regex", "search", "defregex", "ldef",
                 "unused_rounds", "fp")

    def __init__(self, pattern, slot, order, rnd):
        self.pattern, self.slot, self.order, self.round = canonical(pattern), slot, order, rnd
        self.holes, self.length = holes_of(pattern), len(pattern)
        rx = pattern_to_regex(pattern)
        self.regex = re.compile("(?=(" + rx + "))")           # overlapping starts
        self.search = re.compile(rx).search
        self.defregex = re.compile("(?=(" + pattern_to_defregex(pattern) + "))")
        self.ldef = 0.0
        self.unused_rounds = 0
        self.fp = fingerprint(self.pattern)             # survives checkpoint resume; duplicates are caught by it


class Library:
    """Active macros by slot. Slot numbers are what the model sees (tokens M0..M31)."""

    def __init__(self):
        self.macros = {}
        self.n_accepted = 0

    def __len__(self):
        return len(self.macros)

    def active(self, slot):
        return slot in self.macros

    def active_slots(self):
        return list(self.macros)

    def holes(self, slot):
        return self.macros[slot].holes

    def expand(self, slot, args):
        return fill(self.macros[slot].pattern, args)

    def expand_pattern(self, slot, specs):
        return fill(self.macros[slot].pattern, specs)

    def members(self):
        return sorted(self.macros.values(), key=lambda m: m.order)

    def free_slot(self):
        for s in range(MAX_MACROS):
            if s not in self.macros:
                return s
        return None

    def add(self, pattern, rnd):
        slot = self.free_slot()
        assert slot is not None
        self.macros[slot] = Macro(pattern, slot, self.n_accepted, rnd)
        self.n_accepted += 1
        return slot

    def remove(self, slot):
        del self.macros[slot]

    def patterns(self):
        return {m.pattern for m in self.macros.values()}

    def state(self):
        return [(m.slot, m.pattern, m.order, m.round, m.unused_rounds) for m in self.members()]

    def load_state(self, st):
        self.macros = {}
        for slot, pattern, order, rnd, unused in st:
            m = Macro(pattern, slot, order, rnd)
            m.unused_rounds = unused
            self.macros[slot] = m
        self.n_accepted = max([o for _s, _p, o, _r, _u in st], default=-1) + 1


# ── the DP ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def match_starts(prog_str, macro):
    """Unit indices where `macro` matches inside a program string (overlapping, C-level)."""
    return [m.start() >> 1 for m in macro.regex.finditer(prog_str)]


def build_matches(prog_str, macros):
    """{start: [(length, holes, slot), ...]} for one program over a list of macros."""
    out = {}
    for m in macros:
        for s in match_starts(prog_str, m):
            out.setdefault(s, []).append((m.length, m.holes, m.slot))
    return out


def dp_cost(units, matches, cu, ca=LOG2_10, track=None):
    """Optimal cost with terminator. If `track` is a slot/tag, also return whether the optimal path uses it."""
    n = len(units)
    best = [INF] * (n + 1)
    best[0] = 0.0
    used = [False] * (n + 1) if track is not None else None
    for i in range(n):
        d = best[i]
        if d == INF:
            continue
        c = d + cu + (ca if units[i][1] is not None else 0.0)
        if c < best[i + 1]:
            best[i + 1] = c
            if used is not None:
                used[i + 1] = used[i]
        ms = matches.get(i)
        if ms:
            for ln, h, tag in ms:
                c = d + cu + h * ca
                j = i + ln
                if c < best[j]:
                    best[j] = c
                    if used is not None:
                        used[j] = used[i] or tag == track
    if used is not None:
        return best[n] + cu, used[n]
    return best[n] + cu


def dp_segment(units, matches, cu, ca=LOG2_10):
    """Optimal segmentation as library-form units [('P', op, arg) | ('M', slot, args)]."""
    n = len(units)
    best = [INF] * (n + 1)
    best[0] = 0.0
    back = [None] * (n + 1)
    for i in range(n):
        d = best[i]
        if d == INF:
            continue
        c = d + cu + (ca if units[i][1] is not None else 0.0)
        if c < best[i + 1]:
            best[i + 1], back[i + 1] = c, (i, None, 0)
        ms = matches.get(i)
        if ms:
            for ln, h, tag in ms:
                c = d + cu + h * ca
                j = i + ln
                if c < best[j]:
                    best[j], back[j] = c, (i, tag, ln)
    out, j = [], n
    while j > 0:
        i, tag, ln = back[j]
        if tag is None:
            out.append(("P", units[i][0], units[i][1]))
        else:
            args = [units[k][1] for k in range(i, i + ln) if units[k][1] is not None]
            # only the HOLE positions are emitted; consts are baked into the macro
            out.append(("M", tag, tuple(args)))
        j = i
    out.reverse()
    return out


def segment_units(prog, library):
    """Library form of a program under the current library (used to rewrite targets at batch time)."""
    macros = library.members()
    if not macros:
        return [("P", op, a) for op, a in prog]
    s = prog_to_str(prog)
    matches = build_matches(s, macros)
    units = _fix_macro_args(dp_segment(prog, matches, c_unit(len(library))), library)
    return units


def _fix_macro_args(units, library):
    """`dp_segment` collects every arg in a matched span; keep only the ones at HOLE positions of the macro."""
    out = []
    for u in units:
        if u[0] == "M":
            pat = library.macros[u[1]].pattern
            all_args = list(u[2])
            it = iter(all_args)
            keep = []
            for _op, s in pat:
                if ARITY[_op]:
                    a = next(it)
                    if s == HOLE:
                        keep.append(a)
            out.append(("M", u[1], tuple(keep)))
        else:
            out.append(u)
    return out


def primitive_length(prog):
    return len(prog), sum(1 for _o, a in prog if a is not None)


# ── L_def ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def def_length(pattern, macros_before, n_lib):
    """Segment the pattern with the given macros at c_u' = log2(14+|Λ|+2); every argument slot (hole or const) costs
    log2(11); plus one terminator."""
    cu = c_unit_def(n_lib)
    s = pattern_to_str(pattern)
    matches = {}
    for m in macros_before:
        for mm in m.defregex.finditer(s):
            matches.setdefault(mm.start() >> 1, []).append((m.length, m.holes, m.slot))
    units = [(op, None if sp is None else 0) for op, sp in pattern]    # arg present -> costs LOG2_11
    return dp_cost(units, matches, cu, LOG2_11)


# ── the per-round scoring kernel ──────────────────────────────────────────────────────────────────────────────────────
class Scorer:
    """Built once per round from the archive and the library; scores candidates and members against it.

    `family_avg=True` is the J_S of the plan (family-averaged); `False` is the instance-summed variant `mdl_insample`
    uses on its own context families."""

    def __init__(self, archive, library, family_avg=True):
        self.library = library
        self.n = len(library)
        self.members = library.members()
        self.family_avg = family_avg
        self.fids, self.progs, self.strs = [], [], []
        self.fam_index = {}                                       # fid -> [program indices]
        for fid, progs in archive.items():
            for p in progs:
                self.fam_index.setdefault(fid, []).append(len(self.progs))
                self.fids.append(fid)
                self.progs.append(p)
                self.strs.append(prog_to_str(p))
        self.matches = [build_matches(s, self.members) for s in self.strs]
        self.weight = np.zeros(len(self.progs))
        for fid, idxs in self.fam_index.items():
            w = 1.0 / len(idxs) if family_avg else 1.0
            for i in idxs:
                self.weight[i] = w
        self.base = {}
        for k in (-1, 0, 1):
            cu = c_unit(max(0, self.n + k))
            self.base[k] = np.array([dp_cost(p, m, cu) for p, m in zip(self.progs, self.matches)])
        for m in self.members:                                    # members' L_def, recomputed every round
            before = [x for x in self.members if x.order < m.order]
            m.ldef = def_length(m.pattern, before, self.n)
        self.J = float((self.weight * self.base[0]).sum()) + sum(m.ldef for m in self.members)
        self._cache = {}

    def _member_matches_with(self, i, extra):
        if not extra:
            return self.matches[i]
        merged = {k: list(v) for k, v in self.matches[i].items()}
        for s, e in extra:
            merged.setdefault(s, []).append(e)
        return merged

    def score(self, pattern):
        """(ΔJ, support, L_def) of a candidate. Support = families whose best segmentation would use it."""
        key = canonical(pattern)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        cand = Macro(pattern, -1, 10 ** 9, -1)
        ldef = def_length(pattern, self.members, self.n)
        cu = c_unit(self.n + 1)
        new = self.base[1].copy()
        used_fams = set()
        for i, s in enumerate(self.strs):
            if cand.search(s) is None:
                continue
            extra = [(st, (cand.length, cand.holes, -1)) for st in match_starts(s, cand)]
            cost, used = dp_cost(self.progs[i], self._member_matches_with(i, extra), cu, track=-1)
            new[i] = cost
            if used:
                used_fams.add(self.fids[i])
        d = float((self.weight * (self.base[0] - new)).sum()) - ldef
        out = (d, len(used_fams), ldef)
        self._cache[key] = out
        return out

    def loo(self, slot):
        """J(Λ \\ {m}) − J(Λ): how much the library would LOSE by dropping a member (positive = worth keeping)."""
        m = self.library.macros[slot]
        cu = c_unit(self.n - 1)
        without = self.base[-1].copy()
        for i, mt in enumerate(self.matches):
            if any(tag == slot for ms in mt.values() for _l, _h, tag in ms):
                sub = {k: [e for e in v if e[2] != slot] for k, v in mt.items()}
                without[i] = dp_cost(self.progs[i], sub, cu)
        return float((self.weight * (without - self.base[0])).sum()) + m.ldef

    def support(self, slot):
        cu = c_unit(self.n)
        fams = set()
        for i, mt in enumerate(self.matches):
            if any(tag == slot for ms in mt.values() for _l, _h, tag in ms):
                _c, used = dp_cost(self.progs[i], mt, cu, track=slot)
                if used:
                    fams.add(self.fids[i])
        return len(fams)
