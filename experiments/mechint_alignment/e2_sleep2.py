"""E2 arm K4: label-free sleep on a trained world model. Rumination chooses WHAT to rehearse by a test that can lose.
(EXPERIMENT_E2.md)

1. Open refutations: among 100k random stream rows, the 10 % the model (J2) is most surprised by, each with its
   surprise kind (e2_world.signature). No event labels are used anywhere in training.
2. Each kind's open rows are split in half: a REHEARSE half and a PROBE half.
3. Sleep: gentle updates (lr 1e-4, batch 32) = 8 rows from the rehearse halves (kinds chosen by GENERALIZATION
   PROGRESS) + 12 from a pool of past refutations (even across kinds) + 12 stream rows (the base rate of no change).
4. Every 25 updates, each kind's probe half is scored. Its surprise drop since the last probe is the kind's progress.
   - Learning a general rule transfers to the unrehearsed probes.
   - Rationalizing (memorizing the rehearsed cases) and noise do not.
   A rehearsed row retires once its surprise has halved.
--uniform: kinds chosen uniformly (the control).
"""
import argparse
import json
import os

import numpy as np
import torch

from e2_fewshot2 import batch, surprise_and_kind
from e2_world import eval_set, evaluate, load, losses, robot_mask
from village import SPEC
from worldmodel import WorldModel

RULES = ("move", "turn", "mine_work", "mined", "deliver", "theft", "interact_full", "push", "robot_burn",
         "trapped_hurt", "button_off", "frozen_tick")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/e2/J2_focus_questions")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--ema", type=float, default=0.0, help="evaluate a slow EMA copy of the weights (0 = off)")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="runs/data")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    focus = cfg.get("focus", 2)
    train, test = load(os.path.join(a.data, "train.npz")), load(os.path.join(a.data, "test.npz"))
    ev, by_event = eval_set(test)
    model = WorldModel(SPEC, cfg["d"], cfg["layers"], cfg["heads"], pos=cfg["pos"], persist=cfg["persist"],
                       radius=cfg.get("radius", 1)).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))

    cand = rng.choice(len(train["act"]), 100000, replace=False)
    s, k = surprise_and_kind(model, train, cand, dev, focus)
    hot = s > np.quantile(s, 0.9)
    open_rows, open_kind = cand[hot], k[hot]
    mid = (s > np.median(s)) & ~hot
    pool_rows, pool_kind = cand[mid], k[mid]
    kinds = [kd for kd in np.unique(open_kind) if (open_kind == kd).sum() >= 4]
    reh, probe = {}, {}
    for kd in kinds:
        rows = rng.permutation(open_rows[open_kind == kd])
        reh[kd], probe[kd] = list(rows[0::2]), rows[1::2][:32]
    pool_by_kind = {kd: pool_rows[pool_kind == kd] for kd in np.unique(pool_kind)}
    pk = list(pool_by_kind)
    s_probe = {kd: float(surprise_and_kind(model, train, probe[kd], dev, focus)[0].mean()) for kd in kinds}
    s0 = dict(zip(open_rows.tolist(), s[hot].tolist()))
    prog = {kd: 0.0 for kd in kinds}
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    slow = None
    if a.ema:
        import copy
        slow = copy.deepcopy(model)
    log = open(a.out, "w")
    print(f"{len(open_rows)} open refutations in {len(kinds)} kinds (>= 4 rows); pool {len(pool_rows)}", flush=True)
    for u in range(a.updates + 1):
        if u % 50 == 0:
            r = evaluate(slow if slow is not None else model, ev, by_event, dev)
            top = sorted(prog, key=lambda kd: -prog[kd])[:6]
            row = dict(update=u, acc={e: r[e]["acc"] for e in RULES}, changed_acc=r["_overall"]["changed_acc"],
                       unchanged_acc=r["_overall"]["unchanged_acc"], bits=r["_overall"]["bits_per_transition"],
                       open=sum(len(v) for v in reh.values()), top_kinds=[(int(kd), round(prog[kd], 3)) for kd in top])
            log.write(json.dumps(row) + "\n")
            log.flush()
            if u % 200 == 0:
                print(f"update {u:4d}: " + "  ".join(f"{e} {row['acc'][e]:.2f}" for e in RULES) +
                      f"  unchanged {row['unchanged_acc']:.3f}  top kinds {row['top_kinds'][:4]}", flush=True)
        if u == a.updates:
            break
        live = [kd for kd in kinds if reh[kd]]
        if a.uniform or not live:
            pick = rng.choice(live, 8) if live else []
        else:
            w = np.array([max(prog[kd], 0.0) + 0.02 for kd in live])
            pick = rng.choice(live, 8, p=w / w.sum())
        rrows = [reh[kd][rng.integers(len(reh[kd]))] for kd in pick]
        prow = [pool_by_kind[pk[i]][rng.integers(len(pool_by_kind[pk[i]]))] for i in rng.integers(0, len(pk), 12)]
        srow = rng.integers(0, len(train["act"]), 12).tolist()
        rows = np.array(rrows + prow + srow, int)
        o, g, act, nx, gx = batch(train, rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = model(o, g, act)
        loss, tot = losses(cl, gl, nx, gx, per_row=True, mask=robot_mask(o, focus))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if slow is not None:
            with torch.no_grad():
                for ps, pf in zip(slow.parameters(), model.parameters()):
                    ps.mul_(a.ema).add_(pf.detach(), alpha=1 - a.ema)
        st = tot[:len(rrows)].cpu().numpy()
        for kd, r_, sv in zip(pick, rrows, st):                       # retire when explained
            if sv < 0.5 * s0[int(r_)] and r_ in reh[kd]:
                reh[kd].remove(r_)
        if u + 1 == a.updates:                                       # keep the final model (the slow copy if any)
            torch.save((slow if slow is not None else model).state_dict(), a.out.replace(".jsonl", ".pt"))
        if (u + 1) % a.probe_every == 0:                              # the test that can lose: unrehearsed probes
            for kd in kinds:
                cur = float(surprise_and_kind(model, train, probe[kd], dev, focus)[0].mean())
                prog[kd] = 0.7 * prog[kd] + 0.3 * (s_probe[kd] - cur)
                s_probe[kd] = cur


if __name__ == "__main__":
    main()
