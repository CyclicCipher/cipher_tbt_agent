"""The task suite for the thinking experiments (planning doc §20; BRAINSTORM §0.1 item 3: diverse environments).

Six STRUCTURALLY DIFFERENT families, so that no conclusion rests on one kind of world. Each has a DEPTH knob — 8
levels, from "one step" to "far more steps than a no-thought model of our size should manage" — a single-token answer,
and an exact verifier (the generator computes the answer). Data is unlimited and fresh every draw (except Brainfuck,
drawn from large pre-built pools, see `Brainfuck`).

| family | structure                                   | depth knob h                     | answer classes |
|--------|---------------------------------------------|----------------------------------|----------------|
| ptr    | chain lookup: a random permutation of 16    | hops: pi^h(x)                    | 16             |
|        | nodes, one token per (node, successor) pair  |                                  |                |
| s5     | composition in a NON-SOLVABLE group: a word | word length                      | 5              |
|        | of h generators of S5 acting on 5 items     |                                  |                |
| bool   | tree evaluation: a prefix Boolean formula   | formula depth                    | 2 (balanced)   |
|        | (AND / OR / NOT) of depth exactly h         |                                  |                |
| ca     | parallel local dynamics: a random elementary | CA steps                        | 2 (balanced)   |
|        | cellular-automaton rule GIVEN IN CONTEXT,    |                                  |                |
|        | 16 circular cells, one cell after h steps    |                                  |                |
| bf     | a universal machine: a pure Brainfuck        | executed steps (bucketed)        | up to 256      |
|        | program (E46 semantics); the value of a      |                                  |                |
|        | queried cell when it halts                   |                                  |                |
| aff    | arithmetic in a SOLVABLE group: x0, then h   | number of operations             | 17             |
|        | operations +a or *a modulo 17                |                                  |                |

`s5` (non-solvable: its word problem is NC1-complete) and `aff` (solvable) are the pair the learning/complexity theory
separates: a fixed-depth transformer can in principle shortcut the second, not the first (planning doc §20).

One shared vocabulary (`V` tokens): PAD, separators, one TAG per family, operators, S5 generators, Brainfuck
instructions, and the numbers 0..255 (`num(v)`). A sequence is `[TAG, ...problem..., EQ]` and the answer is the next
token. Batches are RIGHT-padded; `make_batch` returns each problem's last position, where the answer is read.

Usage (smoke test: one example per family and level, pool statistics, timing):
    python experiments/neural_turing_architecture/thinking/tasks.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ziplearn"))

# ---------------------------------------------------------------------------------------------------------- vocabulary
PAD, SEP, Q, EQ = 0, 1, 2, 3
TAG = {"ptr": 4, "s5": 5, "bool": 6, "ca": 7, "bf": 8, "aff": 9}
AND, OR, NOT, PLUS, TIMES = 10, 11, 12, 13, 14
GEN0 = 15                                            # the four S5 generators: 15 .. 18
BF_TOK = {c: 19 + i for i, c in enumerate("><+-[]")}  # 19 .. 24
NUM = 32
PAIR0 = NUM + 256                                    # `ptr`: one token per (key, value) pair, 16 x 16 = 256 of them
CA_MARK = (25, 26)                                   # `ca` format "marked": the queried cell, holding 0 / 1
CA_RULE0 = PAIR0 + 256                               # `ca` format "marked_pairs": one token per (pattern, output bit)
V = CA_RULE0 + 16                                    # 560 tokens


def num(v) -> int:
    return NUM + int(v)


# ------------------------------------------------------------------------------------------------------------ families
class PointerChase:
    """A random permutation of N = 16 nodes, written as 16 PAIR TOKENS in random order — token `PAIR0 + 16a + pi(a)`
    holds a node and where it points; the query is (h, x); the answer is pi^h(x) as a number token. One wrong hop sends
    the walk to an effectively random node — no partial credit (planning §19.1).

    Why one token per pair (planning §20.1). Written as two tokens per pair — key then value, with or without separate
    key/value vocabularies, one query or eight per context, query last or not — ONE hop never left chance: 12,000 steps
    for our looped model, 10,000 for a plain 2-layer transformer (loss stuck at ln 16). Finding "the position after key
    x" needs two attention steps that are useful only TOGETHER (copy each key onto its value; match the query against
    it) — a leap of 2 from the outcome alone (§19.1). With the pair in one token a hop is one attention step (match the
    key part, read the value part): both models solve one hop in ~2,000 steps. The depth is untouched — h hops are still
    h dependent lookups."""
    name, levels, N = "ptr", [1, 2, 3, 4, 5, 6, 7, 8], 16

    def sample(self, rng, li):
        h = self.levels[li]
        perm, order = rng.permutation(self.N), rng.permutation(self.N)
        toks = [TAG["ptr"]] + [PAIR0 + self.N * int(a) + int(perm[a]) for a in order]
        x = int(rng.integers(self.N))
        y = x
        for _ in range(h):
            y = int(perm[y])
        return toks + [Q, num(h), num(x), EQ], num(y)


class PointerChaseTwoToken:
    """`ptr` as it was before §20.1: a KEY token (num(a)) and a VALUE token (num(16 + b)) per pair; answer num(16 + y).
    Not in the suite — kept as the leap probe's known negative (one hop never learned, planning §20.1)."""
    name, levels, N = "ptr2", [1, 2, 3, 4, 5, 6, 7, 8], 16

    def sample(self, rng, li):
        h = self.levels[li]
        perm, order = rng.permutation(self.N), rng.permutation(self.N)
        toks = [TAG["ptr"]]
        for a in order:
            toks += [num(a), num(self.N + perm[a])]
        x = int(rng.integers(self.N))
        y = x
        for _ in range(h):
            y = int(perm[y])
        return toks + [Q, num(x), num(h), EQ], num(self.N + y)


class S5Word:
    """Five items in positions 0..4; a word of h generators rearranges them; the query is a position, the answer the
    item there. Generators as position maps (new[i] = old[g[i]]): swap 0-1, rotate left, swap 1-2, rotate right —
    an adjacent transposition and a 5-cycle generate all of S5."""
    name, levels = "s5", [1, 2, 3, 4, 6, 8, 12, 16]
    GENS = [(1, 0, 2, 3, 4), (1, 2, 3, 4, 0), (0, 2, 1, 3, 4), (4, 0, 1, 2, 3)]

    def sample(self, rng, li):
        h = self.levels[li]
        arr = list(range(5))
        word = rng.integers(len(self.GENS), size=h)
        for w in word:
            g = self.GENS[int(w)]
            arr = [arr[g[i]] for i in range(5)]
        j = int(rng.integers(5))
        return [TAG["s5"]] + [GEN0 + int(w) for w in word] + [Q, num(j), EQ], num(arr[j])


class BoolFormula:
    """A prefix Boolean formula of depth exactly h: each AND/OR node has one child of depth h-1 and one of depth 0 or 1
    (order random); NOT with probability 0.15. Leaves are 0/1. Answers balanced 50/50 by rejection.
    The shallow sibling is NEUTRAL (1 under AND, 0 under OR — it lets the deep child decide) with probability 0.9. In
    P0's first run siblings were random, so half of them short-circuited their node and the answer was usually fixed
    within the top two levels: every arm scored 0.84-0.90 normalised at depth 8 without thoughts. With 0.9 neutral, the
    first short-circuit from the top is ~10 levels down on average, so depth h is real depth."""
    name, levels, NEUTRAL = "bool", [1, 2, 3, 4, 5, 6, 7, 8], 0.9

    def _f(self, rng, d):
        if d == 0:
            v = int(rng.integers(2))
            return [num(v)], v
        r = rng.random()
        if r < 0.15:
            t, v = self._f(rng, d - 1)
            return [NOT] + t, 1 - v
        op = AND if r < 0.575 else OR
        dt, dv = self._f(rng, d - 1)
        neutral = 1 if op == AND else 0                     # the sibling value that lets the deep child decide
        want = neutral if rng.random() < self.NEUTRAL else 1 - neutral
        while True:
            st, sv = self._f(rng, int(rng.integers(0, min(1, d - 1) + 1)))
            if sv == want:
                break
        t = [op] + (dt + st if rng.random() < 0.5 else st + dt)
        return t, (dv & sv) if op == AND else (dv | sv)

    def sample(self, rng, li):
        h, want = self.levels[li], int(rng.integers(2))
        while True:
            t, v = self._f(rng, h)
            if v == want:
                return [TAG["bool"]] + t + [EQ], num(v)


class CellularAutomaton:
    """A random elementary CA rule (its 8 output bits, neighbourhood 111 first) and a random row of 16 circular cells
    are given; the query is (cell c, steps h); the answer is that cell after h synchronous steps. The rule is IN THE
    CONTEXT, so it must be read, not memorised. Answers balanced 50/50 by rejection.

    `fmt` — how the problem is written (the leap probe compares them, planning §21):
      "index"        the rule as 8 bits in a fixed order, the row as 16 bits, the query as (c, h): finding cell c means
                     attending to a position named by a token's VALUE, and the rule bit for a neighbourhood is the bit at
                     the position that neighbourhood indexes — two computed-position lookups;
      "marked"       the queried cell is written in place with a marked token (CA_MARK), the query is just h: the cells
                     that matter are at fixed offsets from the mark;
      "marked_pairs" as "marked", and the rule as 8 tokens in random order, each holding a (neighbourhood, output bit)
                     pair — the rule lookup becomes a content match, as one token per pair did for `ptr`."""
    name, levels, W = "ca", [1, 2, 3, 4, 5, 6, 7, 8], 16

    def __init__(self, fmt="index"):
        self.fmt = fmt

    def sample(self, rng, li):
        h, want = self.levels[li], int(rng.integers(2))
        while True:
            rule, row0 = int(rng.integers(256)), rng.integers(0, 2, self.W)
            row = row0.copy()
            for _ in range(h):
                row = (rule >> (4 * np.roll(row, 1) + 2 * row + np.roll(row, -1))) & 1
            c = int(rng.integers(self.W))
            if int(row[c]) == want:
                break
        if self.fmt == "index":
            toks = [TAG["ca"]] + [num((rule >> k) & 1) for k in range(7, -1, -1)] + [num(b) for b in row0]
            return toks + [Q, num(c), num(h), EQ], num(want)
        cells = [CA_MARK[int(b)] if i == c else num(b) for i, b in enumerate(row0)]
        if self.fmt == "marked":
            ruletoks = [num((rule >> k) & 1) for k in range(7, -1, -1)]
        else:
            ruletoks = [CA_RULE0 + 2 * int(k) + ((rule >> int(k)) & 1) for k in rng.permutation(8)]
        return [TAG["ca"]] + ruletoks + cells + [Q, num(h), EQ], num(want)


class AffineMod:
    """x0, then h operations `+ a` or `* a` (a in 1..16) modulo 17; the answer is the result. The maps x -> x+a and
    x -> a*x generate the affine group of Z_17, which is SOLVABLE — the contrast with `s5`."""
    name, levels, P = "aff", [1, 2, 3, 4, 6, 8, 12, 16], 17

    def sample(self, rng, li):
        h = self.levels[li]
        x = int(rng.integers(self.P))
        toks = [TAG["aff"], num(x)]
        for _ in range(h):
            a = int(rng.integers(1, self.P))
            if rng.random() < 0.5:
                x, op = (x + a) % self.P, PLUS
            else:
                x, op = (x * a) % self.P, TIMES
            toks += [op, num(a)]
        return toks + [EQ], num(x)


class Brainfuck:
    """Pure Brainfuck (`> < + - [ ]`, no I/O) with E46's semantics — `run_reference` from `ziplearn/e46.py`: 16
    circular cells mod 256, unmatched brackets as no-ops, budget 256 steps; only programs that HALT are kept. The query
    is a cell c, the answer its value at the halt; c is drawn among the non-zero cells when there are any (a loop exits
    only on a zero cell, so "the cell under the pointer" was 0 most of the time — majority rate 0.88 in the first
    smoke test), and an all-zero tape is kept with probability 0.15. Levels are buckets of EXECUTED steps.

    Programs: E46's `sample_loopy` (random tokens, random brackets) halted too rarely at middle step counts (3M draws in
    6 minutes left two levels short), so `_gen` builds programs that tend to halt: a loop is `+`*r `[` body `]` where the
    body is a random program (nesting <= 3), followed by pointer moves that return to the loop's cell and one `-` — a
    counted loop, as the self-play generator learns to write. r is drawn up to a per-program cap from `LOOP_COUNTS`, and
    programs are at most 36 instructions. A body can still touch its own counter, so some loops run long or forever;
    the budget filters those. Measured (60k draws): the deepest level (65-256 steps) is ~1% of draws, about 4 minutes
    for 20,000. Each level is a pre-built pool, cached on disk (train and eval from different seeds; short programs can
    coincide across them — the shallow levels have few distinct programs and are sanity levels, not tests)."""
    name = "bf"
    BUCKETS = [(1, 2), (3, 4), (5, 6), (7, 10), (11, 16), (17, 32), (33, 64), (65, 256)]   # (1, 1) alone had ONE answer
    levels = [hi for _lo, hi in BUCKETS]              # a level is reported by its bucket's upper edge
    MAX_LEN, TAPE, BUDGET = 36, 16, 256
    LOOP_COUNTS = (2, 5, 8, 12)                       # per program, the largest `+` prefix of a loop

    def _gen(self, rng, depth=0, hi=None):
        hi = int(rng.choice(self.LOOP_COUNTS)) if hi is None else hi
        out = []
        for _ in range(int(rng.integers(1, 6))):
            if depth < 3 and rng.random() < 0.3:
                body = self._gen(rng, depth + 1, hi)
                net = body.count(">") - body.count("<")
                body += ("<" * net if net > 0 else ">" * -net) + "-"
                out.append("+" * int(rng.integers(0, hi + 1)) + "[" + body + "]")
            else:
                out.append(str(rng.choice(list("+-><"), p=[.35, .2, .225, .225])))
        return "".join(out)

    def __init__(self, seed=0, per_level=20000, max_draws=4_000_000, cache_dir=None):
        cache = Path(cache_dir) / f"bf_pools_seed{seed}_n{per_level}.pt" if cache_dir else None
        if cache is not None and cache.exists():
            d = torch.load(cache)
            self.pools, self.build_s, self.draws = d["pools"], 0.0, d["draws"]
            return
        from e46 import run_reference  # noqa: E402  (imports torch + h1_lid; cheap)
        rng = np.random.default_rng(seed)
        self.pools = [[] for _ in self.BUCKETS]
        t0, draws = time.time(), 0
        while min(len(p) for p in self.pools) < per_level and draws < max_draws:
            draws += 1
            prog = self._gen(rng)
            if len(prog) > self.MAX_LEN:
                continue
            _out, tape, _dp, steps, stop = run_reference(prog, [0], self.TAPE, 1 << 30, self.BUDGET, len(prog) + 1)
            if stop != "halt":
                continue
            b = next((i for i, (lo, hi) in enumerate(self.BUCKETS) if lo <= steps <= hi), None)
            if b is None or len(self.pools[b]) >= per_level:     # duplicates kept: a pool samples the generator's distribution
                continue
            nz = [i for i, x in enumerate(tape) if x]
            if not nz and rng.random() > 0.15:
                continue
            c = int(rng.choice(nz)) if nz else int(rng.integers(self.TAPE))
            self.pools[b].append(([TAG["bf"]] + [BF_TOK[ch] for ch in prog] + [Q, num(c), EQ], num(tape[c])))
        self.build_s, self.draws = time.time() - t0, draws
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(pools=self.pools, draws=draws), cache)

    def sample(self, rng, li):
        pool = self.pools[li]
        return pool[int(rng.integers(len(pool)))]


def make_families(bf_seed=0, bf_per_level=20000, cache_dir=None, ca_fmt="index"):
    return [PointerChase(), S5Word(), BoolFormula(), CellularAutomaton(ca_fmt),
            Brainfuck(bf_seed, bf_per_level, cache_dir=cache_dir), AffineMod()]


# ------------------------------------------------------------------------------------------------------------ batching
def collate(items, dev="cpu"):
    """RIGHT-pad a list of (tokens, answer) to one tensor; `last` is each problem's final position, where its answer is
    predicted. Under causal attention a problem never sees the padding after it, so its computation cannot depend on
    how long the other problems in the batch are. (P0 runs 1-2 LEFT-padded: a short problem sat behind ~20-35 PAD
    tokens in training and none in evaluation, and was scored on an input it had never seen — planning §20.3.)"""
    T = max(len(t) for t, _ in items)
    tok = torch.full((len(items), T), PAD, dtype=torch.long)
    for i, (t, _) in enumerate(items):
        tok[i, :len(t)] = torch.tensor(t)
    ans = torch.tensor([a for _, a in items], dtype=torch.long)
    last = torch.tensor([len(t) - 1 for t, _ in items], dtype=torch.long)
    return tok.to(dev), ans.to(dev), last.to(dev)


def make_batch(fams, rng, n_per_family, dev="cpu", level=None, max_level=-1, level_probs=None):
    """n_per_family examples from every family, levels uniform (or all at level index `level`, or uniform over the
    level indices <= `max_level` when that is >= 0, or drawn from `level_probs[f]` — one probability vector per
    family, e.g. a frontier curriculum). Returns tokens, answers, family index, level index."""
    items, fid, lid = [], [], []
    for f, fam in enumerate(fams):
        top = len(fam.levels) if max_level < 0 else max_level + 1
        for _ in range(n_per_family):
            if level is not None:
                li = level
            elif level_probs is not None:
                li = int(rng.choice(len(fam.levels), p=level_probs[f]))
            else:
                li = int(rng.integers(top))
            items.append(fam.sample(rng, li))
            fid.append(f)
            lid.append(li)
    tok, ans, last = collate(items, dev)
    return tok, ans, torch.tensor(fid, device=dev), torch.tensor(lid, device=dev), last


if __name__ == "__main__":
    t0 = time.time()
    fams = make_families(bf_per_level=2000)
    bf = fams[4]
    print(f"bf pools built in {bf.build_s:.1f}s from {bf.draws} draws: " + ", ".join(
        f"{lo}-{hi}:{len(p)}" for (lo, hi), p in zip(bf.BUCKETS, bf.pools)))
    rng = np.random.default_rng(1)
    inv = {v: k for k, v in TAG.items()}
    for fam in fams:
        lens, counts = [], {}
        for li in range(len(fam.levels)):
            for _ in range(200):
                t, a = fam.sample(rng, li)
                lens.append(len(t))
                counts.setdefault(li, {}).setdefault(a, 0)
                counts[li][a] += 1
        maj = [max(c.values()) / sum(c.values()) for c in counts.values()]
        t, a = fam.sample(rng, len(fam.levels) - 1)
        print(f"{fam.name:5s} levels {fam.levels}  len {min(lens)}-{max(lens)}  majority-class rate per level "
              + " ".join(f"{m:.2f}" for m in maj))
        print(f"      e.g. (deepest) {t} -> {a}")
    t1 = time.time()
    for _ in range(20):
        make_batch(fams, rng, 32)
    print(f"batch of 6 x 32: {(time.time() - t1) / 20 * 1000:.1f} ms;  total {time.time() - t0:.1f}s")
