"""Check that ziplib.store reproduces the code it MOVES, on the evidence the regression numbers were measured on.

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/ziplib/_check_store.py          (CPU, ~1 min)

(1) Store.sleep vs LocalRule.sleep on the E24 LockPath evidence. The numbers 9,192 -> 660 bits and masks [3, 3, 3, 2]
    (runs/e24/play.json, level 0) are a property of the TRANSITIONS the E24 player saw; today's `arcgames.play`
    explores differently (E27's price of ignorance replaced E24's search), so the E24 evidence is obtained by running
    the E24 player itself (`git show cc443b6:experiments/ziplearn/arcgames.py`) on LockPath levels 0-1 with sleep,
    logging every (action, before, after) it observed and every sleep pass. Today's `play` is run the same way (as
    play_games.py --sleep 1 --budget 150 does, levels 0-1). On BOTH evidences a fresh LocalRule (radius 1 and 2, per
    action) and a fresh Store are fed the same pairs and slept at the same checkpoints; every sleep's (before, after,
    cells) and the final table, majority, mask, stats and cost must agree (cost to 1e-9, the rest exactly). Level 0's
    sums and radius-1 masks are printed against the E24 numbers.
(2) Store.as_tokens vs WrittenSim.memory[2:] on the E28 rules (today's play, best radius per action, `best_store` =
    `ActionModel.best`), with the codec of ziplib/_check_codec.py; as_bank re-sums to the tokens; as_rows' key + value
    rows re-sum to the tokens minus the class flag.
(3) Store.predict_nearest vs e35.NearestRule.predict_nearest on 50 corrupted frames (E35's corruption chain at t = 0.5,
    blocks fitted on the frames of today's play with E35's strict sleep), exactly; Store.nearest on the unknown rows.
Also: windows_seq / chain / sleep(price="prequential") vs ContextLM on a short text; record / refute / live vs
GoalModel on the play's goal events; candidates are exactly rescored and equal nearest's pick when k covers every sub-key; loo_kernel_width returns a grid value;
consolidate on the one-hot code moves every settled entry (no interference) and the sparse code runs; save/load.
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                                    # experiments/ziplearn
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent / "src"))
import arcgames as A                                                  # noqa: E402
import e28 as E28                                                     # noqa: E402
import e35 as E35                                                     # noqa: E402
import textlm as TL                                                   # noqa: E402
from tasks.games import LockPath                                      # noqa: E402
from tasks.harness import Environment                                 # noqa: E402
from ziplib.store import Store, Field, SparseProjection, best_store, windows_grid, windows_seq, HS   # noqa: E402
from ziplib import price as P                                         # noqa: E402
from _check_codec import e28_layout_and_codec                         # noqa: E402

V = A.V
OK = True


def check(name, good, detail=""):
    global OK
    OK &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'} {name}" + (f": {detail}" if detail else ""))


# ── evidence: a logged play ───────────────────────────────────────────────────────────────────────────────────────────
def logged_play(mod, **kw):
    """Run `mod.play` on LockPath with every Player.observe / Player.sleep / GoalModel.record / GoalModel.refute logged."""
    events = []
    P_obs, P_sleep = mod.Player.observe, mod.Player.sleep
    G_rec, G_ref = mod.GoalModel.record, mod.GoalModel.refute

    def observe(self, action, before, after, outcome):
        events.append(("observe", action.name, before.copy(), after.copy(), outcome))
        return P_obs(self, action, before, after, outcome)

    def sleep(self):
        events.append(("sleep",))
        return P_sleep(self)

    def record(self, action, before, after, outcome, recent=()):
        events.append(("record", action.name, before.copy(), after.copy(), outcome, set(recent)))
        return G_rec(self, action, before, after, outcome, recent)

    def refute(self, which, action, frame):
        events.append(("refute", which, action.name, frame.copy()))
        return G_ref(self, which, action, frame)

    mod.Player.observe, mod.Player.sleep, mod.GoalModel.record, mod.GoalModel.refute = observe, sleep, record, refute
    try:
        out = mod.play(Environment(LockPath()), budget_per_level=150, max_levels=2, sleep=True, **kw)
    finally:
        mod.Player.observe, mod.Player.sleep, mod.GoalModel.record, mod.GoalModel.refute = P_obs, P_sleep, G_rec, G_ref
    return events, out


def e24_module():
    """arcgames.py as it was when runs/e24/play.json was written (commit cc443b6), imported from a temp file."""
    src = subprocess.run(["git", "show", "cc443b6:experiments/ziplearn/arcgames.py"], capture_output=True, text=True,
                         cwd=str(ROOT), encoding="utf-8")
    if src.returncode != 0:
        return None
    d = Path(tempfile.mkdtemp(prefix="e24_"))
    p = d / "arcgames_e24.py"
    p.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("arcgames_e24", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arcgames_e24"] = mod
    spec.loader.exec_module(mod)
    return mod


def replay(events, radii=(1, 2)):
    """Fresh LocalRules and Stores fed the logged pairs; slept at the logged checkpoints. Returns
    (rules[action][r], stores[action][r], sleeps: [(action, r, (b, a, c)_rule, (b, a, c)_store)] per checkpoint)."""
    rules, stores, sleeps = {}, {}, []
    for ev in events:
        if ev[0] == "observe":
            _, a, before, after, outcome = ev
            if outcome == "win":
                continue                                              # Player.observe: a winning move's after-frame is not seen
            if a not in rules:
                rules[a] = {r: A.LocalRule(r) for r in radii}
                stores[a] = {r: Store(Field.grid(r), V) for r in radii}
            for r in radii:
                rules[a][r].observe(before, after)
                full, _m = windows_grid(before, stores[a][r].field)
                stores[a][r].observe(full, after.reshape(-1))
        elif ev[0] == "sleep":
            rep = []
            for a in rules:
                for r in radii:
                    rep.append((a, r, rules[a][r].sleep(), stores[a][r].sleep()))
            sleeps.append(rep)
    return rules, stores, sleeps


def same_rule(rule, store):
    """LocalRule vs Store: table, majority, mask, stats exactly; cost to 1e-9."""
    return (rule.table == store.table and rule.majority == store.majority and np.array_equal(rule.mask, store.mask)
            and rule.stats == store.stats and abs(rule.cost - store.cost) <= 1e-9 and rule.full == store.full)


def compare_evidence(label, events, want=None):
    rules, stores, sleeps = replay(events)
    actions = sorted(rules)
    good = all(same_rule(rules[a][r], stores[a][r]) for a in actions for r in (1, 2))
    check(f"[{label}] final LocalRule = Store for {len(actions)} actions x radii 1, 2 (table, majority, mask, stats, full; cost 1e-9)", good)
    good = all(abs(rb - sb) <= 1e-9 and abs(ra - sa) <= 1e-9 and rc == sc for rep in sleeps for (_a, _r, (rb, ra, rc), (sb, sa, sc)) in rep)
    check(f"[{label}] every sleep's (before, after, cells) agrees ({len(sleeps)} checkpoints x {len(sleeps[0])} rules)", good)
    rep0 = sleeps[0]
    before = sum(b for (_a, _r, _x, (b, _aa, _c)) in rep0)
    after = sum(aa for (_a, _r, _x, (_b, aa, _c)) in rep0)
    masks1 = [c for (a, r, _x, (_b, _aa, c)) in rep0 if r == 1]
    masks2 = [c for (a, r, _x, (_b, _aa, c)) in rep0 if r == 2]
    print(f"        level-0 sleep: {before:,.0f} -> {after:,.0f} bits over radii 1 + 2; radius-1 cells kept {masks1}, radius-2 {masks2}"
          f" (actions {actions})")
    if want is not None:
        check(f"[{label}] reproduces E24's 9,192 -> 660 bits and masks [3, 3, 3, 2]",
              round(before) == want[0] and round(after) == want[1] and masks1 == want[2] and masks2 == want[2],
              f"got {before:.0f} -> {after:.0f}, {masks1} / {masks2}")
    return rules, stores


def goal_events(events, actions):
    """GoalModel.record / refute replayed into one Store (field grid(1), mask all ones, tag = action) per outcome."""
    probe = Field.grid(1)
    goal = A.GoalModel()
    stores = {"win": Store(probe, V), "death": Store(probe, V)}
    for ev in events:
        if ev[0] == "record":
            _, a, before, after, outcome, recent = ev
            act = next(x for x in actions if x.name == a)
            goal.record(act, before, after, outcome, recent=recent)
            changed = set(zip(*np.nonzero(before != after))) | set(recent)
            if not changed:
                continue
            full, _m = windows_grid(before, probe)
            W = before.shape[1]
            idx = sorted({int(i) * W + int(j) for i, j in changed})
            stores[outcome].record(full[idx], tag=a)
        elif ev[0] == "refute":
            _, which, a, frame = ev
            act = next(x for x in actions if x.name == a)
            goal.refute(which, act, frame)
            full, _m = windows_grid(frame, probe)
            stores[which].refute(full, tag=a)
    return goal, stores


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("ziplib.store vs the moved code")
    # ── (1a) the E24 evidence: the E24 player's transitions ──
    mod = e24_module()
    if mod is None:
        check("[E24 evidence] git show cc443b6:experiments/ziplearn/arcgames.py", False, "git unavailable; skipped")
    else:
        events24, _out = logged_play(mod)
        compare_evidence("E24 evidence, cc443b6 player", events24, want=(9192, 660, [3, 3, 3, 2]))
    # ── (1b) today's evidence: arcgames.play as play_games.py --sleep 1 --budget 150 runs it (levels 0-1) ──
    events, (results, _stats, player) = logged_play(A, return_player=True)
    rules, stores = compare_evidence("today's evidence, arcgames.play", events)
    live = all(same_rule(player.models[a].rules[r], stores[a.name][r]) for a in player.actions if player.models[a].n_obs for r in (1, 2))
    check("[today] the replayed Store equals the LIVE player's rules after the play", live)
    # best_store = ActionModel.best
    acts = [a for a in player.actions if player.models[a].n_obs]
    good = all(best_store(stores[a.name].values()).r == player.models[a].best().r for a in acts)
    check("best_store picks ActionModel.best's radius per action", good, {a.name: best_store(stores[a.name].values()).r for a in acts})
    good = all(abs(stores[a.name][r].exception_rate() - (lambda m: (sum(s[1] for s in m.stats.values()) / max(1, sum(s[0] + s[1] for s in m.stats.values()))))(player.models[a].rules[r])) <= 1e-12
               for a in acts for r in (1, 2))
    check("exception_rate = ActionModel.exception_rate", good)
    # windows_grid = LocalRule._rows
    frame = rng.integers(0, V, size=(6, 8)).astype(np.int16)
    for r in (1, 2):
        rule = player.models[acts[0]].rules[r]
        f_old, m_old = rule._rows(frame)
        f_new, m_new = windows_grid(frame, Field.grid(r), rule.mask)
        check(f"windows_grid(r={r}) = LocalRule._rows (full and masked, dtype int16)", np.array_equal(f_old, f_new) and np.array_equal(m_old, m_new)
              and f_new.dtype == np.int16)
    # predict / predict_frame
    rule = player.models[acts[0]].best()
    st = best_store(stores[acts[0].name].values())
    p_old, u_old = rule.predict(frame)
    p_new, u_new = st.predict_frame(frame)
    full, _m = windows_grid(frame, st.field)
    hashed = st.predict(full)
    check("predict_frame = LocalRule.predict; predict(keys) = majority.get per row", np.array_equal(p_old, p_new) and u_old == u_new
          and sum(h is None for h in hashed) == u_new)
    # goal keys
    goal, gstores = goal_events(events, player.actions)
    good = True
    for which, gs in (("win", goal.win), ("death", goal.death)):
        for a, keys in gs.items():
            good &= gstores[which].outcome_keys.get(a.name, {}) == keys
            good &= gstores[which].live(a.name) == goal.live(which, a)
        good &= set(gstores[which].outcome_keys) == {a.name for a in gs}
    good &= gstores["win"].known() == goal.known()
    n_win = sum(len(k) for k in goal.win.values())
    check(f"record / refute / live / known = GoalModel on the play's goal events ({n_win} win keys)", good)
    # ── (2) as_tokens = WrittenSim.memory[2:] ──
    e28_rules = {a: player.models[a].best() for a in acts}
    sim = E28.WrittenSim(e28_rules, acts, 7, 9)
    layout, codec, _nbr = e28_layout_and_codec(sim)
    best = {a: best_store(stores[a.name].values()) for a in acts}
    tok = torch.cat([best[a].as_tokens(codec, action=ai) for ai, a in enumerate(acts)])
    err = float((tok.double() - sim.memory[2:].double()).abs().max()) if len(tok) else 0.0
    check(f"as_tokens x {len(tok)} = WrittenSim.memory[2:] (d = {sim.d})", tok.shape == sim.memory[2:].shape and err <= 1e-12, f"max |new - old| = {err:.3e}")
    keys_b, vals_b = zip(*[best[a].as_bank(codec, action=ai) for ai, a in enumerate(acts)])
    bank = torch.cat(keys_b) + torch.cat(vals_b)
    check("as_bank: keys + values = as_tokens; keys have no `out`, values only `out`",
          float((bank - tok).abs().max()) <= 1e-12 and float(torch.cat(keys_b)[:, layout["out"]].abs().max()) == 0.0
          and float(torch.cat(vals_b)[:, :layout["out"].start].abs().max()) == 0.0)
    rows = [r for ai, a in enumerate(acts) for r in best[a].as_rows(layout, action=ai)]
    resum = torch.stack([kr + vr for kr, _t, vr in rows])
    flag = torch.zeros(layout.d); flag[layout.flag("entry")] = 1.0
    good = float((resum + flag - tok).abs().max()) <= 1e-12 and all(abs(t - (float(kr.sum()) - 0.5)) <= 1e-12 for kr, t, _v in rows)
    check(f"as_rows x {len(rows)}: key_row + value_row + entry flag = token; threshold = k_in - 1/2", good)
    proj = SparseProjection(layout.d, layout.d, k=11, seed=0)
    srows = best[acts[0]].as_rows(layout, code="sparse", proj=proj, action=0)
    good = all(float(kr.sum()) >= 11 and set(kr.unique().tolist()) <= {0.0, 1.0} for kr, _t, _v in srows)
    check(f"as_rows(code='sparse') x {len(srows)}: binary key rows, >= k ones (the union of the key dims' codes)", good)
    # ── (3) nearest = NearestRule.predict_nearest on 50 corrupted frames ──
    frames = {}
    for ev in events:
        if ev[0] == "observe":
            frames[ev[2].tobytes() + bytes(ev[2].shape)] = ev[2]
    frames = list(frames.values())
    K = 8
    schedule = [k / K for k in range(K + 1)]
    chains = [E35.corrupt_chain(x, schedule, rng) for x in frames for _ in range(2)]
    pairs = [(c[4], c[3]) for c in chains]                            # block 4: t = 0.5 -> 0.375
    nr = E35.NearestRule(1)
    ns = Store(Field.grid(1), V)
    for before, after in pairs:
        nr.observe(before, after)
        ns.observe_frame(before, after)
    loose_r, loose_s = E35.NearestRule(1), Store(Field.grid(1), V)
    for before, after in pairs:
        loose_r.observe(before, after); loose_s.observe_frame(before, after)
    l_old, l_new = loose_r.sleep(strict=False), loose_s.sleep(strict=False)
    check(f"sleep(strict=False) on E35's block ({len(pairs)} pairs, {len(loose_r.full)} windows): NearestRule = Store",
          same_rule(loose_r, loose_s) and all(abs(x - y) <= 1e-9 for x, y in zip(l_old, l_new)), f"{l_old[0]:.0f} -> {l_old[1]:.0f} bits, {l_old[2]} cells")
    b_old = nr.sleep(strict=True)                                     # E35 strict: the nearest-neighbour denoiser (0.946 / 0.880 / 0.800)
    b_new = ns.sleep(strict=True)
    check(f"sleep(strict=True) on the same block: NearestRule = Store ({len(nr.table)} entries)",
          same_rule(nr, ns) and all(abs(x - y) <= 1e-9 for x, y in zip(b_old, b_new)), f"{b_old[0]:.0f} -> {b_old[1]:.0f} bits, {b_old[2]} cells")
    tests = [E35.corrupt_chain(frames[i % len(frames)], schedule, rng)[4] for i in range(50)]
    good, n_unknown = True, 0
    for x in tests:
        p_old, u_old = nr.predict_nearest(x)
        p_new, u_new = ns.predict_nearest(x)
        good &= np.array_equal(p_old, p_new) and u_old == u_new and p_new.dtype == p_old.dtype
        n_unknown += u_new
    check(f"predict_nearest = NearestRule.predict_nearest on 50 corrupted frames ({n_unknown} unknown cells)", good)
    x = tests[0]
    full, masked = windows_grid(x, ns.field, ns.mask)
    karr, kval = nr._keys_array()
    d = (masked[:, None, :] != karr[None, :, :]).sum(-1)
    want = kval[d.argmin(1)]
    got = ns.nearest(full)
    check("nearest(keys) = the Hamming argmin's majority for every window (ties by stored index)", np.array_equal(got, want))
    soft = ns.nearest(full, M=30.0)
    check("nearest(keys, M=30) agrees with the hard read where the nearest key is unique",
          bool(np.all((soft == want) | ((d == d.min(1, keepdims=True)).sum(1) > 1))))
    exact = ns._matches(masked[:20]).numpy()
    cands = ns.candidates(full[:20], k=8)
    good = all(all(exact[q, c[i]] >= exact[q, c[i + 1]] for i in range(len(c) - 1)) for q, c in enumerate(cands))
    check(f"candidates(k=8): each query's candidates are rescored exactly, best first ({sum(1 for c in cands if c)}/20 non-empty; "
          f"{sum(1 for q, c in enumerate(cands) if c and exact[q, c[0]] == exact[q].max())}/20 hold the global best)", good)
    n_sub = max(len(np.unique(karr[:, :karr.shape[1] // 2], axis=0)), len(np.unique(karr[:, karr.shape[1] // 2:], axis=0)))
    full_c = ns.candidates(full[:20], k=n_sub)
    check(f"candidates(k = every sub-key = {n_sub}): the first candidate is nearest's pick for all 20 queries",
          all(c and kval[c[0]] == want[q] and exact[q, c[0]] == exact[q].max() for q, c in enumerate(full_c)))
    h = ns.loo_kernel_width(sample=200, seed=0)
    check(f"loo_kernel_width returns a grid value (h = {h}; bits/obs {min(ns.loo_table.values()):.3f} at the pick)", h in HS)
    # ── text: windows_seq / chain / prequential sleep vs ContextLM ──
    Vt, R = 7, 4
    text = rng.integers(0, Vt, size=1500).astype(np.int16)
    for i in range(1, len(text)):                                     # a 2nd-order structure so the mask has something to keep
        if text[i - 1] == 3:
            text[i] = 5
    lm = TL.ContextLM(Vt, R)
    ctx_old, nxt_old = lm._contexts([text])
    ctx_old = np.where(ctx_old == Vt, -1, ctx_old)                    # ContextLM pads with V; the Store pads with -1
    full_seq, _m = windows_seq(text, Field.seq(R))
    check("windows_seq = ContextLM._contexts (column k-1 = k back; BORDER V -> -1)", np.array_equal(full_seq, ctx_old) and np.array_equal(nxt_old, text))
    lm.fit([text], sleep=True, limit=1000)
    ts = Store(Field.seq(R), Vt, keep_order=True)
    ts.observe_seq(text)
    b, a, c = ts.sleep(price="prequential", limit=1000)
    mask_pos = [int(j) + 1 for j in np.flatnonzero(ts.mask)]
    check(f"sleep(price='prequential') = ContextLM.fit's mask {lm.mask} and bits {lm.bits_before:.1f} -> {lm.bits_after:.1f}",
          mask_pos == lm.mask and abs(b - lm.bits_before) <= 1e-6 and abs(a - lm.bits_after) <= 1e-6, f"store {mask_pos}, {b:.1f} -> {a:.1f}")
    chain_pos = [tuple(int(j) + 1 for j in np.flatnonzero(m)) for m in ts.chain()]
    check("chain() = backoff_chain(mask) (drop the oldest kept position down to the unigram)", chain_pos == P.backoff_chain(lm.mask))
    for m in lm.chain:
        ck, ct, pk, pc = lm.tables[m]
        merged = ts.merged(np.array([(j + 1) in m for j in range(R)]))
        good = len(merged) == len(ck) and sum(sum(cnt.values()) for cnt in merged.values()) == int(ct.sum()) and sum(len(cnt) for cnt in merged.values()) == len(pk)
        if not good:
            break
    check("merged(mask) per chain level has ContextLM's context / pair counts", good)
    # ── consolidate ──
    cs = best[acts[0]]
    moved = cs.consolidate(n0=2, eps0=0.5)
    settled = [i for i, k in enumerate(cs.table) if sum(cs.stats[k]) >= 2 and cs.stats[k][1] / max(1, sum(cs.stats[k])) <= 0.5]
    check(f"consolidate(onehot): every settled entry moves (no interference on a partitioned one-hot code): {len(moved)} of {len(cs)}",
          moved == settled)
    check("as_tokens(which='tokens') + as_rows(which='rows') partition the entries",
          len(cs.as_tokens(codec, action=0)) + len(cs.as_rows(layout, which="rows", action=0)) == len(cs)
          and len(cs.as_tokens(codec, which="all", action=0)) == len(cs))
    cs.forms = {}
    moved_s = cs.consolidate(n0=2, eps0=0.5, code="sparse", k=3)
    check(f"consolidate(sparse, k=3) runs and prices interference: {len(moved_s)} of {len(settled)} settled entries moved", isinstance(moved_s, list))
    cs.forms = {}
    # ── save / load ──
    with tempfile.TemporaryDirectory() as d_:
        p = Path(d_) / "store.npz"
        for s_ in (cs, ts, gstores["win"]):
            s_.save_npz(p)
            back = Store.load_npz(p)
            good = (back.table == s_.table and back.full == s_.full and back.majority == s_.majority and back.stats == s_.stats
                    and np.array_equal(back.mask, s_.mask) and abs(back.cost - s_.cost) <= 1e-12 and back.outcome_keys == s_.outcome_keys
                    and back.field.offsets == s_.field.offsets and back.field.names == s_.field.names
                    and ((s_.log is None) == (back.log is None)) and (s_.log is None or all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(s_.log, back.log))))
            if not good:
                break
        check("save_npz / load_npz round-trips the evidence, table, mask, cost, outcome keys, log (3 stores)", good)
    print("PASS: ziplib.store reproduces the moved code" if OK else "FAIL: see above")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
