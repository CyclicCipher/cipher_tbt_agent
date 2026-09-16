"""Evaluation, identical for every arm, on final checkpoints (E1 also on the 25/50/75% ones for learning curves).

    python experiments/mdl_vs_ntp/evaluate.py --run runs/mdl_s0            # writes runs/mdl_s0/eval.json
    python experiments/mdl_vs_ntp/evaluate.py --run runs/mdl_s0 --curve    # E1 at every checkpoint too

Eval instances and per-task sampling seeds are fixed and shared across arms (they come from env_cache). Per-task
outcomes are stored, not just means, so `analyze.py` can bootstrap over tasks within seeds.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from env.dsl import run                                                                   # noqa: E402
from env.families import load_env, ops_of                                                 # noqa: E402
from env.grammar import Grammar                                                            # noqa: E402
from env.macros import ALL_MACROS, TRAIN_MACROS                                            # noqa: E402
from env.tokens import (DIGITS, EPROG, PAD, PROG_BUDGET, PROMPT_I, enc_list, encode_T, is_macro,  # noqa: E402
                        parse_program, probe_positions)
from mdl.library import HOLE, Library, Scorer                                              # noqa: E402
from mdl.wake import prompt_I, verify                                                     # noqa: E402
from model_adapter import ModelAdapter                                                     # noqa: E402

PROBE_POS = probe_positions()
ANS_PROMPT = 2 + 76 + 11                                       # up to and including <ans>
WAVES = [4, 4, 8, 16, 32, 64]
KS = [1, 4, 16, 64, 128]


def load_run(run_dir, tag="final"):
    run_dir = Path(run_dir)
    import yaml
    with open(run_dir / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = ModelAdapter(cfg.get("size", "4M"), amp=cfg.get("amp", True)).to(dev)
    st = torch.load(run_dir / f"ckpt_{tag}.pt", map_location=dev, weights_only=False)
    adapter.load_state_dict(st["model"])
    adapter.eval()
    library = Library()
    library.load_state(st["library"])
    has_lib = cfg["arm"] in ("mdl", "curio", "mdl_insample")
    return adapter, library, Grammar(library if has_lib else None), cfg, dev


# ── E1: transduction exact match ─────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def e1(adapter, env, grammar, dev, sets):
    out = {}
    for name in sets:
        insts = env.eval_sets[name]
        prompts = torch.tensor([encode_T(i)[0][:ANS_PROMPT] for i in insts], dtype=torch.long, device=dev)
        gen = adapter.generate(prompts, 8, 0.0, grammar, "ans")[:, ANS_PROMPT:].tolist()
        hits = [int(g == enc_list(i.query[1])) for g, i in zip(gen, insts)]
        by_units = collections.defaultdict(list)
        for h, i in zip(hits, insts):
            by_units[env.fam(i.fid).n_units].append(h)
        out[name] = dict(em=float(np.mean(hits)), per_task=hits, fids=[i.fid for i in insts],
                         by_units={k: float(np.mean(v)) for k, v in by_units.items()})
    return out


# ── E2: NLL ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def e2(adapter, env, dev):
    out = {}
    for name, digits_only in (("e1_id", False), ("e1_hfresh", False), ("e1_hcomp", False), ("e1_hdepth", False),
                              ("e1_hnovel", False), ("e2_noisy", True), ("e2_partial", False)):
        insts = env.eval_sets[name]
        ids = torch.tensor([encode_T(i)[0] for i in insts], dtype=torch.long)
        vals = []
        for s in range(0, len(ids), 256):
            chunk = ids[s:s + 256].to(dev)
            mean_bits, bits = adapter.probe_bits(chunk, PROBE_POS)
            if digits_only:
                tgt = chunk[:, PROBE_POS]
                dm = torch.tensor([[int(t) in DIGITS for t in row] for row in tgt.tolist()], device=dev)
                vals += ((bits * dm).sum(1) / dm.sum(1).clamp(min=1)).tolist()
            else:
                vals += mean_bits.tolist()
        out[name] = dict(bits=float(np.mean(vals)), per_task=vals)
    return out


# ── E3: induction search ─────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def e3(adapter, env, library, grammar, dev, sets, seed=0):
    trng = torch.Generator(device=dev).manual_seed(seed)
    out = {}
    for name in sets:
        insts = env.eval_sets[name]
        n = len(insts)
        first_consistent = [None] * n                          # (sample index, program, used macro)
        first_correct = [None] * n
        n_seen = [0] * n
        active = list(range(n))
        prompts_all = torch.tensor([prompt_I(i) for i in insts], dtype=torch.long, device=dev)
        for wave in WAVES:
            if not active:
                break
            prompts = prompts_all[active].repeat_interleave(wave, dim=0)
            gen = adapter.generate(prompts, PROG_BUDGET + 1, 1.0, grammar, "prog", rng=trng)[:, PROMPT_I:].tolist()
            still = []
            for ai, ti in enumerate(active):
                inst = insts[ti]
                stop = False
                for j in range(wave):
                    toks = gen[ai * wave + j]
                    k = n_seen[ti] + j + 1
                    body = toks[:toks.index(EPROG)] if EPROG in toks else None
                    prog = parse_program(body, library) if body is not None else None
                    if prog is None:
                        continue
                    if all(run(prog, x) == y for x, y in inst.demos):
                        used = any(is_macro(t) for t in body)
                        if first_consistent[ti] is None:
                            first_consistent[ti] = (k, prog, used)
                        if run(prog, inst.query[0]) == inst.query[1] and first_correct[ti] is None:
                            first_correct[ti] = (k, prog, used)
                            stop = True
                            break
                n_seen[ti] += wave
                if not stop:
                    still.append(ti)
            active = still
        rec = dict(verified_at={}, pass_at={}, per_task_verified={}, per_task_pass={})
        for k in KS:
            # verified@k: the FIRST demo-consistent sample within k is query-correct
            v = [int(first_consistent[i] is not None and first_consistent[i][0] <= k and first_correct[i] is not None
                     and first_correct[i][0] == first_consistent[i][0]) for i in range(n)]
            p = [int(first_correct[i] is not None and first_correct[i][0] <= k) for i in range(n)]
            rec["verified_at"][k] = float(np.mean(v))
            rec["pass_at"][k] = float(np.mean(p))
            rec["per_task_verified"][k] = v
            rec["per_task_pass"][k] = p
        sts = [fc[0] if fc is not None else None for fc in first_correct]
        solved = [s for s in sts if s is not None]
        rec["samples_to_first_correct_median"] = float(np.median(solved)) if solved else float("nan")
        rec["censored_frac"] = 1 - len(solved) / n
        rec["prim_length_first_correct"] = float(np.mean([len(fc[1]) for fc in first_correct if fc])) if solved else float("nan")
        rec["macro_usage_rate"] = float(np.mean([fc[2] for fc in first_correct if fc])) if solved else 0.0
        rec["fids"] = [i.fid for i in insts]
        out[name] = rec
    return out


# ── E4: library ──────────────────────────────────────────────────────────────────────────────────────────────────────
def e4(env, library, run_dir):
    members = library.members()
    planted = [m for m in TRAIN_MACROS]
    exact = partial = 0
    rec_planted = []
    for pm in planted:
        pops = ops_of(pm)
        status = "missed"
        for m in members:
            if m.pattern == pm:
                status = "exact"
                break
            if ops_of(m.pattern) == pops and all(
                    (a == b) or (a == HOLE) or (b == HOLE) for (_o, a), (_p, b) in zip(m.pattern, pm)):
                status = "partial"
        exact += status == "exact"
        partial += status == "partial"
        rec_planted.append(status)
    planted_ops = [ops_of(p) for p in ALL_MACROS]
    aligned = 0
    for m in members:
        ops = ops_of(m.pattern)
        sub = any(any(po[i:i + len(ops)] == ops for i in range(len(po) - len(ops) + 1)) for po in planted_ops)
        cat = any(po + qo == ops for po in planted_ops for qo in planted_ops)
        aligned += sub or cat
    arch = {f: [p for _i, p in env.archive[f]] for f in env.archive}
    fam_specific = 0
    if members:
        sc = Scorer(arch, library, family_avg=True)
        fam_specific = sum(1 for m in members if sc.support(m.slot) <= 2)
    traj = []
    rp = Path(run_dir) / "rounds.jsonl"
    if rp.exists():
        for line in open(rp):
            r = json.loads(line)
            traj.append(dict(round=r["round"], size=r["library_size"], J_S=r["J_S"], reward=r["mean_reward"],
                             regret=r["proposer_regret"], validity=r["validity_rate"]))
    return dict(recovered_exact=exact, recovered_partial=partial, planted=rec_planted, n_macros=len(members),
                precision_aligned=(aligned / len(members)) if members else float("nan"),
                family_specific_rate=(fam_specific / len(members)) if members else float("nan"),
                library=[dict(slot=m.slot, pattern=[(o, s) for o, s in m.pattern]) for m in members], trajectory=traj)


# ── E5: curriculum ───────────────────────────────────────────────────────────────────────────────────────────────────
def e5(run_dir):
    rounds = []
    pp = Path(run_dir) / "probe.jsonl"
    if pp.exists():
        for line in open(pp):
            r = json.loads(line)
            rounds.append(dict(round=r["round"], mass=r.get("mass", {}), noisy_probe=r.get("noisy_probe")))
    comp = collections.Counter()
    tp = Path(run_dir) / "train.jsonl"
    if tp.exists():
        for line in open(tp):
            for k, v in json.loads(line).get("batch", {}).items():
                comp[k] += v
    tot = sum(comp.values())
    return dict(rounds=rounds, realized_share={k: v / tot for k, v in comp.items()} if tot else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--curve", action="store_true", help="E1 at the 25/50/75% checkpoints too")
    ap.add_argument("--e3_cap", type=int, default=0, help="1 instance per H family if search eval is too slow")
    args = ap.parse_args()
    env = load_env()
    adapter, library, grammar, cfg, dev = load_run(args.run)
    t0 = time.time()
    res = dict(arm=cfg["arm"], seed=cfg["seed"], params=cfg.get("params"))
    e1_sets = ["e1_id", "e1_hfresh", "e1_hcomp", "e1_hdepth", "e1_hnovel"]
    res["e1"] = e1(adapter, env, grammar, dev, e1_sets)
    res["e2"] = e2(adapter, env, dev)
    e3_sets = ["e3_hfresh", "e3_hcomp", "e3_hdepth", "e3_hnovel", "e3_id_labeled", "e3_id_unlabeled"]
    if args.e3_cap:
        for s in e3_sets[:4]:
            env.eval_sets[s] = env.eval_sets[s][::2]
    res["e3"] = e3(adapter, env, library, grammar, dev, e3_sets)
    res["e4"] = e4(env, library, args.run)
    res["e5"] = e5(args.run)
    if args.curve:
        res["e1_curve"] = {}
        for tag in ("25", "50", "75"):
            if (Path(args.run) / f"ckpt_{tag}.pt").exists():
                a2, _l, g2, _c, _d = load_run(args.run, tag)
                res["e1_curve"][tag] = {k: v["em"] for k, v in e1(a2, env, g2, dev, ["e1_id"]).items()}
    res["eval_seconds"] = time.time() - t0
    with open(Path(args.run) / "eval.json", "w") as fh:
        json.dump(res, fh)
    print(f"{cfg['arm']} s{cfg['seed']}: ID EM {res['e1']['e1_id']['em']:.3f} | "
          f"H-comp v@16 {res['e3']['e3_hcomp']['verified_at'][16]:.3f} | "
          f"noisy-digit NLL {res['e2']['e2_noisy']['bits']:.2f} | {res['eval_seconds']:.0f}s")


if __name__ == "__main__":
    main()
