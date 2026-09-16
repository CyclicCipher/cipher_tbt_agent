"""The proposer and the library update (`mdl`, `curio`, `mdl_insample`).

Each round the archive (labelled families' ground truth, unlabelled families' best buffer programs) is split 50/50 into
a PROPOSAL set P_r and a SCORING set S_r. The model reads 5 programs from 5 distinct P_r families and emits a DEF; the
reward is `clip(ΔJ(m) / L_def(m), −1, 5)` with ΔJ measured on S_r — families the proposal never saw — so a macro is paid
for what it does to OTHER families' solutions, never its own. `mdl_insample` is the leakage control: the same pipeline
scored on the proposal's own context families, instance-summed.

Acceptance is greedy, at most 2 per round, re-scoring after each; pruning drops a member whose leave-one-out value is
non-positive or that no wake-verified program has used for 3 consecutive rounds (the bounded-learner check). A symbolic
miner over S_r runs as a diagnostic only, giving the proposer's regret against the best contiguous window.
"""
from __future__ import annotations

import collections
import itertools

import numpy as np
import torch

from env.dsl import ARITY, HOLE, pattern_str
from env.macros import PRIM_FPS, fingerprint
from env.tokens import DEF, EDEF, LEN_P, OP0, PAD, PROMPT_P, encode_P, parse_def, prog_tokens
from mdl.library import MAX_MACROS, Scorer, canonical, segment_units

N_RL_STEPS, N_CONTEXTS, N_CTX_PROGS = 8, 64, 5
MAX_ACCEPT, MIN_SUPPORT, UNUSED_LIMIT = 2, 3, 3


def build_archive(env, buffer):
    arch = {}
    for fid in env.by_split["train"]:
        fam = env.fam(fid)
        if fam.labeled:
            arch[fid] = [p for _i, p in env.archive[fid]]
        else:
            ps = buffer.best_programs(fid, 8)
            if ps:
                arch[fid] = ps
    return arch


def candidate_reward(pattern, scorer, library, member_fps):
    """(reward, ΔJ, support, L_def, fingerprint). Invalid handled by the caller (−1)."""
    fp = fingerprint(pattern)
    if fp in PRIM_FPS or fp in member_fps:
        return 0.0, 0.0, 0, 0.0, fp
    dj, sup, ldef = scorer.score(pattern)
    return float(np.clip(dj / max(ldef, 1e-6), -1.0, 5.0)), dj, sup, ldef, fp


def propose_round(adapter, opt, env, library, buffer, grammar, arm, rnd, rng, trng, dev, beta, used_slots):
    """Runs REINFORCE, acceptance, pruning and diagnostics. Returns (round record, TE consumed)."""
    arch = build_archive(env, buffer)
    fids = list(arch)
    rng.shuffle(fids)
    half = len(fids) // 2
    P_r, S_r = fids[:half], fids[half:]
    insample = arm == "mdl_insample"
    min_support = 1 if insample else MIN_SUPPORT
    scorer_S = Scorer({f: arch[f] for f in S_r}, library, family_avg=True)
    member_fps = {m.fp for m in library.members()}
    te = 0
    cands = {}                                                   # pattern -> (dj, support, ldef)
    rl_log = collections.Counter()
    rewards_all = []
    for _step in range(N_RL_STEPS):
        ctx_fams = [rng.choice(P_r, size=N_CTX_PROGS, replace=False) for _ in range(N_CONTEXTS)]
        prompts = []
        for fs in ctx_fams:
            progs = [arch[f][int(rng.integers(len(arch[f])))] for f in fs]
            prompts.append(encode_P([prog_tokens(segment_units(p, library)) for p in progs])[:PROMPT_P])
        prompts = torch.tensor(prompts, dtype=torch.long, device=dev)
        out = adapter.generate(prompts, LEN_P - PROMPT_P, 1.0, grammar, "def", rng=trng)
        te += int((out != PAD).sum())
        gen = out[:, PROMPT_P:].tolist()
        rewards = []
        for i, toks in enumerate(gen):
            body = toks[:toks.index(EDEF)] if EDEF in toks else None
            pat = parse_def(body, library) if body is not None else None
            if pat is None:
                rewards.append(-1.0)
                rl_log["invalid"] += 1
                continue
            pat = canonical(pat)
            if pat in library.patterns():
                rewards.append(0.0)
                rl_log["duplicate"] += 1
                continue
            sc = scorer_S
            if insample:                                         # score on the proposal's OWN context families
                sc = Scorer({f: arch[f] for f in ctx_fams[i]}, library, family_avg=False)
            r, dj, sup, ldef, fp = candidate_reward(pat, sc, library, member_fps)
            rewards.append(r)
            rl_log["valid"] += 1
            if fp not in PRIM_FPS and fp not in member_fps:
                prev = cands.get(pat)
                if prev is None or dj > prev[0]:
                    cands[pat] = (dj, sup, ldef)
        r = torch.tensor(rewards, device=dev)
        rewards_all += rewards
        n = len(rewards)
        loo_mean = (r.sum() - r) / max(n - 1, 1)                  # RLOO
        adv = (r - loo_mean)
        adv = adv / (adv.std() + 1e-8)
        gen_mask = torch.zeros_like(out, dtype=torch.bool)
        gen_mask[:, PROMPT_P:] = out[:, PROMPT_P:] != PAD
        adapter.train()
        lp = adapter.seq_logprob(out, gen_mask, grammar, "def", PROMPT_P)
        loss = beta * (-(adv.detach() * lp)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        te += 3 * int((out != PAD).sum())

    # ── accept ───────────────────────────────────────────────────────────────────────────────────────────────────────
    events = []
    accepted = 0
    for _ in range(MAX_ACCEPT):
        if not cands:
            break
        scorer_S = Scorer({f: arch[f] for f in S_r}, library, family_avg=True)
        ranked = sorted(cands.items(), key=lambda kv: -kv[1][0])[:20]
        best = None
        for pat, _old in ranked:
            if pat in library.patterns():
                continue
            if insample:
                dj, sup, ldef = _insample_best(pat, arch, P_r, library)
            else:
                dj, sup, ldef = scorer_S.score(pat)
            cands[pat] = (dj, sup, ldef)
            if dj > 0 and sup >= min_support and (best is None or dj > best[1]):
                best = (pat, dj, sup, ldef)
        if best is None:
            break
        pat, dj, sup, ldef = best
        if len(library) >= MAX_MACROS:
            worst = min(library.members(), key=lambda m: scorer_S.loo(m.slot))
            if dj <= scorer_S.loo(worst.slot):
                break
            events.append(dict(event="replace", slot=worst.slot, pattern=pattern_str(worst.pattern)))
            library.remove(worst.slot)
            adapter.reset_slot(worst.slot)
        slot = library.add(pat, rnd)
        library.macros[slot].fp = fingerprint(pat)
        adapter.init_slot(slot, [OP0 + op for op, _s in pat])
        events.append(dict(event="accept", slot=slot, pattern=pattern_str(pat), dJ=dj, support=sup, L_def=ldef))
        cands.pop(pat, None)
        accepted += 1

    # ── prune ────────────────────────────────────────────────────────────────────────────────────────────────────────
    scorer_S = Scorer({f: arch[f] for f in S_r}, library, family_avg=True)
    scorer_P = Scorer({f: arch[f] for f in P_r}, library, family_avg=False) if insample else scorer_S
    lib_rows = []
    for m in list(library.members()):
        loo = scorer_P.loo(m.slot)
        sup = scorer_S.support(m.slot)
        if m.round < rnd:                                          # usage pruning starts after the acceptance round
            m.unused_rounds = 0 if m.slot in used_slots else m.unused_rounds + 1
        reason = None
        if loo <= 0:
            reason = "loo"
        elif m.unused_rounds >= UNUSED_LIMIT:
            reason = "unused"
        lib_rows.append(dict(slot=m.slot, pattern=pattern_str(m.pattern), loo=loo, support=sup,
                             unused_rounds=m.unused_rounds, round=m.round))
        if reason is not None:
            events.append(dict(event="prune", slot=m.slot, pattern=pattern_str(m.pattern), reason=reason, loo=loo))
            library.remove(m.slot)
            adapter.reset_slot(m.slot)

    # ── diagnostics: the symbolic miner ──────────────────────────────────────────────────────────────────────────────
    scorer_S = Scorer({f: arch[f] for f in S_r}, library, family_avg=True)
    miner_best, miner_pat = mine(arch, S_r, scorer_S)
    best_prop = max((v[0] for v in cands.values()), default=float("nan"))
    rec = dict(round=rnd, arm=arm, n_archive=len(arch), n_P=len(P_r), n_S=len(S_r),
               proposals=dict(rl_log), mean_reward=float(np.mean(rewards_all)) if rewards_all else float("nan"),
               validity_rate=rl_log["valid"] / max(1, sum(rl_log.values())),
               best_proposed_dJ=best_prop, miner_best=miner_best, miner_pattern=miner_pat,
               proposer_regret=(miner_best - best_prop) if best_prop == best_prop else float("nan"),
               J_S=scorer_S.J, library=lib_rows, events=events, accepted=accepted, library_size=len(library))
    return rec, te


def _insample_best(pat, arch, P_r, library):
    """For `mdl_insample`, a candidate's acceptance score is the best instance-summed ΔJ over context-sized subsets;
    approximated by scoring against all of P_r instance-summed (a superset, so never smaller)."""
    sc = Scorer({f: arch[f] for f in P_r}, library, family_avg=False)
    return sc.score(pat)


def mine(arch, S_r, scorer, top=200):
    """Every contiguous 2-4-op window in S_r programs, with up to 2 args turned into holes; the top-200 by family
    support are scored. Returns (best ΔJ, its pattern)."""
    support = collections.defaultdict(set)
    for f in S_r:
        for p in arch[f]:
            n = len(p)
            for ln in (2, 3, 4):
                for s in range(n - ln + 1):
                    w = p[s:s + ln]
                    argpos = [i for i, (_o, a) in enumerate(w) if a is not None]
                    for k in range(0, min(2, len(argpos)) + 1):
                        for hs in itertools.combinations(argpos, k):
                            pat = tuple((op, HOLE if i in hs else a) for i, (op, a) in enumerate(w))
                            support[pat].add(f)
    ranked = sorted(support.items(), key=lambda kv: -len(kv[1]))[:top]
    best, best_pat = float("-inf"), None
    for pat, _fs in ranked:
        dj, _sup, _ld = scorer.score(pat)
        if dj > best:
            best, best_pat = dj, pattern_str(pat)
    return best, best_pat
