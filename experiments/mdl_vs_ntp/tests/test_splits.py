from env.families import (LEAK_SPLITS, N_LABELED, N_NOISY, N_PARTIAL, QUOTAS, build_env, has_banned_adjacency,
                          leaks, ops_of)
from env.macros import N_TRAIN_MACROS, fingerprint


def test_held_out_disjoint_from_train(env):
    train = {ops_of(env.fam(f).pattern) for f in env.by_split["train"]}
    train_fp = {fingerprint(env.fam(f).pattern, (env.fam(f).n_free,)) for f in env.by_split["train"]}
    for split in ("hfresh", "hcomp", "hdepth", "hnovel"):
        for f in env.by_split[split]:
            fam = env.fam(f)
            assert fam.pattern not in {env.fam(g).pattern for g in env.by_split["train"]}
            assert fingerprint(fam.pattern, (fam.n_free,)) not in train_fp


def test_banned_pair_and_q_rules(env):
    for split in LEAK_SPLITS:
        for f in env.by_split[split]:
            fam = env.fam(f)
            assert not leaks(fam.pattern), (split, fam.units)
            assert not has_banned_adjacency(fam.units)
    for f in env.by_split["hcomp"]:
        assert has_banned_adjacency(env.fam(f).units)
    for f in env.by_split["hnovel"]:
        units = env.fam(f).units
        assert sum(1 for u in units if u[0] == "M" and u[1] >= N_TRAIN_MACROS) == 1


def test_quotas_met_or_logged(env):
    for split, q in QUOTAS.items():
        fams = [env.fam(f) for f in env.by_split[split]]
        assert len(fams) == sum(q.values())
        got = {k: sum(1 for f in fams if f.n_units == k) for k in q}
        if got != q:
            assert any(split in line for line in env.log), (split, got, q)
    assert len(env.by_split["noisy"]) == N_NOISY and len(env.by_split["partial"]) == N_PARTIAL
    assert sum(env.fam(f).labeled for f in env.by_split["train"]) == N_LABELED


def test_regeneration_is_deterministic(env):
    again = build_env()
    assert [f.pattern for f in again.families] == [f.pattern for f in env.families]
    assert [f.key for f in again.families] == [f.key for f in env.families]
    a, b = env.eval_sets["e1_hcomp"][:20], again.eval_sets["e1_hcomp"][:20]
    assert [(i.args, i.demos, i.query) for i in a] == [(i.args, i.demos, i.query) for i in b]
