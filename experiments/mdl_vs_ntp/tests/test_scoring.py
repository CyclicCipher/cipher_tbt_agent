import numpy as np

from env.dsl import ARITY, HOLE, N_OPS
from env.families import ops_of
from env.macros import TRAIN_MACROS
from mdl.library import Library, Scorer


def archive_of(env, n=60):
    fids = [f for f in env.by_split["train"] if env.fam(f).labeled][:n]
    return {f: [p for _i, p in env.archive[f]] for f in fids}


def test_planted_macros_score_positive(env):
    # DEVIATION from the plan's 60-family archive: under the plan's own cost model a new macro raises c_u for EVERY
    # program (log2(14+|L|+1) grows), an alphabet tax the macro must repay across the archive before its L_def.
    # Measured: 0/10 planted macros positive at 60 families, 4/10 at 147, 7/10 at 210, 10/10 at 294. The objective is
    # kept faithful and the test uses the full labelled archive, which is the scale the scoring set actually has.
    sc = Scorer(archive_of(env, 294), Library(), family_avg=True)
    pos = sum(1 for m in TRAIN_MACROS if sc.score(m)[0] > 0)
    assert pos >= 8, pos


def test_alphabet_tax_is_the_reason_small_archives_reject_macros(env):
    sc = Scorer(archive_of(env, 60), Library(), family_avg=True)
    assert all(sc.score(m)[1] >= 3 for m in TRAIN_MACROS)      # every planted macro is FOUND (support >= 3) ...
    tax = float((sc.weight * (sc.base[0] - sc.base[1])).sum())
    assert tax < 0                                              # ... it is the global c_u increase that outweighs it


def test_random_fragments_score_nonpositive(env):
    sc = Scorer(archive_of(env), Library(), family_avg=True)
    planted = [ops_of(m) for m in TRAIN_MACROS]
    rng = np.random.default_rng(0)
    frags, tried = [], 0
    while len(frags) < 200 and tried < 20000:
        tried += 1
        pat = []
        for _ in range(3):
            op = int(rng.integers(N_OPS))
            pat.append((op, (HOLE if rng.random() < 0.5 else int(rng.integers(10))) if ARITY[op] else None))
        pat = tuple(pat)
        ops = ops_of(pat)
        if any(ops[i:i + len(pm)] == pm for pm in planted for i in range(len(ops) - len(pm) + 1)):
            continue
        frags.append(pat)
    nonpos = sum(1 for f in frags if sc.score(f)[0] <= 0)
    assert nonpos >= 0.9 * len(frags), (nonpos, len(frags))


def test_leakage_in_sample_vs_disjoint(env):
    arch = archive_of(env, 120)
    strs = {f: [ops_of(p) for p in ps] for f, ps in arch.items()}
    chosen = None
    for f in arch:
        fam = env.fam(f)
        if fam.n_units < 2:
            continue
        pat = fam.pattern
        ops = ops_of(pat)
        clean = all(not any(o[i:i + len(ops)] == ops for i in range(len(o) - len(ops) + 1))
                    for g, oss in strs.items() if g != f for o in oss)
        if clean:
            chosen = (f, pat)
            break
    assert chosen is not None
    f, pat = chosen
    own = Scorer({f: arch[f]}, Library(), family_avg=False)
    others = Scorer({g: ps for g, ps in arch.items() if g != f}, Library(), family_avg=True)
    assert own.score(pat)[0] > 0
    assert others.score(pat)[0] <= 0


def test_scoring_speed(env):
    import time
    fids = [f for f in env.by_split["train"] if env.fam(f).labeled]
    sc = Scorer({f: [p for _i, p in env.archive[f]] for f in fids}, Library(), family_avg=True)
    t0 = time.time()
    for m in TRAIN_MACROS:
        sc.score(m)
    per = (time.time() - t0) / len(TRAIN_MACROS)
    assert per < 0.05, per                                   # target < 10 ms; generous bound for CI noise
