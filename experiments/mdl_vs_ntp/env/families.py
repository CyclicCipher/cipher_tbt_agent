"""Families, splits, instances, and the cached environment.

A FAMILY is a template: a sequence of units, each a planted macro or a primitive, whose arguments are either FIXED in
the template or FREE (resampled per instance). Its EXPANDED pattern is the primitive sequence with `HOLE` at every free
argument. An INSTANCE is a family plus a free-argument assignment, 4 demo inputs and 1 query input.

Everything derives from `(ENV_SEED, split, family_id, instance_seed)`. Splits are generated once into `env_cache/` and
shared by every run.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dsl import ARG_RANGE, ARITY, HOLE, MAX_LEN, N_OPS, fill, has_redundant_adjacency, run
from .macros import ALL_MACROS, BANNED, BANNED_SET, HELD_MACROS, N_TRAIN_MACROS, fingerprint
from .noise import derive_seed, noisy, partial

ENV_SEED = 1234
SPLITS = ["train", "noisy", "partial", "hfresh", "hcomp", "hdepth", "hnovel"]
QUOTAS = {                                     # split -> {unit count: n families}
    "train": {1: 63, 2: 168, 3: 189},
    "hfresh": {1: 15, 2: 40, 3: 45},
    "hcomp": {2: 50, 3: 50},
    "hdepth": {4: 50, 5: 50},
    "hnovel": {2: 30, 3: 30},
}
N_NOISY, N_PARTIAL, N_LABELED = 90, 30, 294
LEAK_SPLITS = {"train", "hfresh", "hdepth"}      # must not contain a banned pair's or a Q macro's expanded pattern
MAX_ATTEMPTS = 20000
P_MACRO, P_FIXED = 0.6, 0.5


@dataclass
class Family:
    fid: int
    split: str
    kind: str                                   # structured | noisy | partial
    labeled: bool
    units: tuple = ()                           # ('M', k, (argspec,...)) | ('P', op, argspec); argspec None|('F',v)|('R',)
    pattern: tuple = ()                         # expanded (op, spec) with spec None | const | HOLE
    key: int = 0                                # noise key for noisy/partial
    n_units: int = 0

    @property
    def n_free(self):
        return sum(1 for _o, s in self.pattern if s == HOLE)

    @property
    def hole_ops(self):
        return [op for op, s in self.pattern if s == HOLE]

    def program(self, args):
        return fill(self.pattern, args)

    def output(self, x, args):
        if self.kind == "noisy":
            return noisy(self.key, x)
        if self.kind == "partial":
            return partial(self.key, x)
        return run(fill(self.pattern, args), x)

    @property
    def category(self):
        if self.kind != "structured":
            return self.kind
        return ("labeled" if self.labeled else "unlabeled") + f"-{self.n_units}u"


@dataclass
class Instance:
    fid: int
    args: tuple
    demos: list                                 # 4 x (x, y)
    query: tuple                                # (xq, yq)
    seed: int


@dataclass
class Env:
    families: list = field(default_factory=list)
    by_split: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)          # fid -> [Instance] x4  (train families)
    archive: dict = field(default_factory=dict)        # fid -> [(Instance, program)] x8 (labeled)
    eval_sets: dict = field(default_factory=dict)      # name -> [Instance]
    log: list = field(default_factory=list)

    def fam(self, fid):
        return self.families[fid]

    @property
    def train_fids(self):
        return self.by_split["train"] + self.by_split["noisy"] + self.by_split["partial"]


# ── template sampling ────────────────────────────────────────────────────────────────────────────────────────────────
def _argspec(rng, op):
    if not ARITY[op]:
        return None
    if rng.random() < P_FIXED:
        rr = ARG_RANGE[op]
        return ("F", int(rr[rng.integers(len(rr))]))
    return ("R",)


def _macro_unit(rng, k):
    pat = ALL_MACROS[k]
    return ("M", k, tuple(_argspec(rng, op) for op, s in pat if s == HOLE))


def _prim_unit(rng, op):
    return ("P", op, _argspec(rng, op))


def sample_unit(rng, macro_set):
    if macro_set and rng.random() < P_MACRO:
        return _macro_unit(rng, int(macro_set[rng.integers(len(macro_set))]))
    return _prim_unit(rng, int(rng.integers(N_OPS)))


def expand(units):
    """Template units -> expanded pattern (op, None | const | HOLE)."""
    out = []
    for u in units:
        if u[0] == "P":
            _t, op, spec = u
            out.append((op, None if spec is None else (spec[1] if spec[0] == "F" else HOLE)))
        else:
            _t, k, specs = u
            it = iter(specs)
            for op, s in ALL_MACROS[k]:
                if s == HOLE:
                    sp = next(it)
                    out.append((op, sp[1] if sp[0] == "F" else HOLE))
                else:
                    out.append((op, s))
    return tuple(out)


def ops_of(pattern):
    return tuple(op for op, _s in pattern)


def _contains(seq, sub):
    n, m = len(seq), len(sub)
    return any(seq[i:i + m] == sub for i in range(n - m + 1))


LEAK_PATTERNS = [ops_of(ALL_MACROS[a]) + ops_of(ALL_MACROS[b]) for a, b in BANNED] + [ops_of(q) for q in HELD_MACROS]


def leaks(pattern):
    ops = ops_of(pattern)
    return any(_contains(ops, p) for p in LEAK_PATTERNS)


def has_banned_adjacency(units):
    for a, b in zip(units, units[1:]):
        if a[0] == "M" and b[0] == "M" and (a[1], b[1]) in BANNED_SET:
            return True
    return False


def random_input(rng):
    ln = int(rng.integers(4, MAX_LEN + 1))
    return tuple(int(v) for v in rng.integers(0, 10, ln))


def random_args(rng, hole_ops):
    return tuple(int(ARG_RANGE[op][rng.integers(len(ARG_RANGE[op]))]) for op in hole_ops)


def template_valid(pattern, rng, registry, n=64):
    """64 random draws: >=75% non-empty output, >=50% output != input, >=10 distinct outputs; no redundant adjacency;
    fingerprint distinct from every accepted template in every split."""
    if has_redundant_adjacency(ops_of(pattern)):
        return False, None
    hole_ops = [op for op, s in pattern if s == HOLE]
    fp = fingerprint(pattern, (len(hole_ops),))
    if fp in registry:
        return False, None
    nonempty = changed = 0
    outs = set()
    for _ in range(n):
        x = random_input(rng)
        y = run(fill(pattern, random_args(rng, hole_ops)), x)
        nonempty += bool(y)
        changed += y != x
        outs.add(y)
    ok = nonempty >= 0.75 * n and changed >= 0.5 * n and len(outs) >= 10
    return ok, fp


def sample_template(rng, split, n_units):
    train_set = list(range(N_TRAIN_MACROS))
    if split == "hcomp":
        a, b = BANNED[rng.integers(len(BANNED))]
        pos = int(rng.integers(n_units - 1))
        units = [sample_unit(rng, train_set) for _ in range(n_units)]
        units[pos], units[pos + 1] = _macro_unit(rng, a), _macro_unit(rng, b)
        return tuple(units)
    if split == "hnovel":
        q = N_TRAIN_MACROS + int(rng.integers(len(HELD_MACROS)))
        pos = int(rng.integers(n_units))
        units = [sample_unit(rng, train_set) for _ in range(n_units)]
        units[pos] = _macro_unit(rng, q)
        return tuple(units)
    return tuple(sample_unit(rng, train_set) for _ in range(n_units))


def split_ok(split, units, pattern):
    if split in LEAK_SPLITS and (leaks(pattern) or has_banned_adjacency(units)):
        return False
    if split == "hcomp" and not has_banned_adjacency(units):
        return False
    if split == "hnovel" and sum(1 for u in units if u[0] == "M" and u[1] >= N_TRAIN_MACROS) != 1:
        return False
    return True


def generate_split(env, split, registry, rng):
    quotas = dict(QUOTAS[split])
    counts = sorted(quotas)
    fams = []
    for ci, n_units in enumerate(counts):
        need, attempts = quotas[n_units], 0
        while need > 0 and attempts < MAX_ATTEMPTS:
            attempts += 1
            units = sample_template(rng, split, n_units)
            pattern = expand(units)
            if not split_ok(split, units, pattern):
                continue
            ok, fp = template_valid(pattern, rng, registry)
            if not ok:
                continue
            registry.add(fp)
            fams.append(Family(fid=-1, split=split, kind="structured", labeled=False, units=units,
                               pattern=pattern, n_units=n_units))
            need -= 1
        if need > 0:                                                    # reallocate to the neighbouring count
            nb = counts[ci + 1] if ci + 1 < len(counts) else counts[ci - 1]
            quotas[nb] += need
            env.log.append(f"{split}: {need} families of {n_units} units reallocated to {nb} units")
    return fams


# ── instances ────────────────────────────────────────────────────────────────────────────────────────────────────────
def make_instance(fam, instance_seed, ns="inst"):
    rng = np.random.default_rng(derive_seed(ENV_SEED, fam.split, fam.fid, ns, instance_seed))
    hole_ops = fam.hole_ops
    while True:
        args = random_args(rng, hole_ops)
        for _ in range(20):
            xs = [random_input(rng) for _ in range(5)]
            ys = [fam.output(x, args) for x in xs]
            if sum(1 for y in ys if y) >= 4 and len(set(ys[:4])) >= 2:
                return Instance(fid=fam.fid, args=args, demos=list(zip(xs[:4], ys[:4])), query=(xs[4], ys[4]),
                                seed=instance_seed)


def fresh_io(fam, args, rng, n=5):
    """Fresh inputs for a stored arg assignment (the buffer's spurious-solution check). Same validity rule."""
    xs = ys = None
    for _ in range(20):
        xs = [random_input(rng) for _ in range(n)]
        ys = [fam.output(x, args) for x in xs]
        if sum(1 for y in ys if y) >= n - 1 and len(set(ys[:4])) >= 2:
            break
    return list(zip(xs, ys))


# ── the whole environment ────────────────────────────────────────────────────────────────────────────────────────────
def build_env():
    env = Env()
    registry = set()
    rng = np.random.default_rng(derive_seed(ENV_SEED, "splits"))
    fid = 0
    for split in ["train", "hfresh", "hcomp", "hdepth", "hnovel"]:
        fams = generate_split(env, split, registry, np.random.default_rng(derive_seed(ENV_SEED, "split", split)))
        if split == "train":                                            # labelled flag: balanced over unit counts
            order = rng.permutation(len(fams))
            for j in order[:N_LABELED]:
                fams[j].labeled = True
            for f in fams:
                f.fid = fid
                fid += 1
            env.by_split["train"] = [f.fid for f in fams]
            env.families += fams
            for split2, n in (("noisy", N_NOISY), ("partial", N_PARTIAL)):
                ids = []
                for _ in range(n):
                    env.families.append(Family(fid=fid, split=split2, kind=split2, labeled=False,
                                               key=derive_seed(ENV_SEED, split2, fid)))
                    ids.append(fid)
                    fid += 1
                env.by_split[split2] = ids
        else:
            for f in fams:
                f.fid = fid
                fid += 1
            env.by_split[split] = [f.fid for f in fams]
            env.families += fams

    # Fixed sets: probe (4 per train family), scoring archive (8 per labelled family, with ground truth), eval sets.
    for f in env.families:
        if f.split in ("train", "noisy", "partial"):
            env.probe[f.fid] = [make_instance(f, s, "probe") for s in range(4)]
        if f.split == "train" and f.labeled:
            env.archive[f.fid] = [(inst, f.program(inst.args)) for inst in
                                  (make_instance(f, s, "archive") for s in range(8))]
    e = env.eval_sets
    e["e1_id"] = [make_instance(env.fam(i), s, "e1") for i in env.by_split["train"] for s in range(2)]
    for split in ("hfresh", "hcomp", "hdepth", "hnovel"):
        e[f"e1_{split}"] = [make_instance(env.fam(i), s, "e1") for i in env.by_split[split] for s in range(8)]
        e[f"e3_{split}"] = [make_instance(env.fam(i), s, "e3") for i in env.by_split[split] for s in range(2)]
    e["e2_noisy"] = [make_instance(env.fam(i), s, "e2") for i in env.by_split["noisy"] for s in range(4)]
    e["e2_partial"] = [make_instance(env.fam(i), s, "e2") for i in env.by_split["partial"] for s in range(4)]
    lab = [i for i in env.by_split["train"] if env.fam(i).labeled]
    unl = [i for i in env.by_split["train"] if not env.fam(i).labeled]
    e["e3_id_labeled"] = [make_instance(env.fam(i), 0, "e3") for i in lab[:150]]
    e["e3_id_unlabeled"] = [make_instance(env.fam(i), 0, "e3") for i in unl[:126]]
    return env


def load_env(cache_dir=Path(__file__).resolve().parent.parent / "env_cache"):
    cache_dir = Path(cache_dir)
    path = cache_dir / "env.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    env = build_env()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(env, fh)
    with open(cache_dir / "generation.log", "w") as fh:
        fh.write("\n".join(env.log) + ("\n" if env.log else ""))
    return env
