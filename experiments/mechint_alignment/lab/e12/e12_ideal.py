"""E12a: the ideal observer of robot motion against E8c's learned world models, on the same worlds, instances and
chunks (EXPERIMENT_E12.md).

    python -m lab.e12.e12_ideal
"""
import json
import os
import time

import numpy as np

from agents.ideal_observer import PERMS, PUSH, MotionObserver, robot_state
from lab.e8.e8_hier import world_index
from worlds.village import FIRE, sample_physics
from worlds.village_events import EVENTS

DATA, OUT, LENGTH, N_WORLDS, CHUNK = "runs/data_family_long", "runs/e12", 512, 64, 8
ARMS = {"L_Ac": True, "L_A": True, "L_A0": False}          # E8c arm -> evaluated with in-world adaptation?
MOVE, TURN = EVENTS.index("move"), EVENTS.index("turn")


def true_hypothesis(seed, phys_row):
    """The world's (perm row, fire code index, push index), regenerated from its seed; checked against the stored
    physics. Tries the training split first, then the held-out one."""
    for split in ("train", "heldout"):
        ph = sample_physics(np.random.default_rng([int(seed), 3]), split)
        stored = (ph.mine_work, ph.carry, ph.fire_damage, PUSH.index(ph.push), ph.freeze, int(ph.reward_per_gold * 2))
        if tuple(int(x) for x in phys_row) == stored:
            perm = int(np.nonzero((PERMS == np.array(ph.action_perm, np.int8)).all(1))[0][0])
            return perm, ph.code_map[FIRE] - 2, PUSH.index(ph.push)
    raise ValueError(f"could not regenerate the physics of world {seed}")


def run_split(name):
    with np.load(os.path.join(DATA, f"{name}.npz")) as f:      # into memory once: an npz re-reads on every access
        z = {k: f[k] for k in ("obs", "gob", "act", "nxt", "gnx", "events", "episode", "tstep", "phys")}
    worlds = world_index(z, LENGTH)[:N_WORLDS]
    n_chunks = LENGTH // CHUNK
    hit = {"move": np.zeros(n_chunks), "turn": np.zeros(n_chunks)}
    tot = {"move": np.zeros(n_chunks), "turn": np.zeros(n_chunks)}
    bits, entropy, violations = np.zeros(n_chunks), np.zeros(n_chunks + 1), 0
    for rows in worlds:
        obs = MotionObserver()
        truth = true_hypothesis(z["episode"][rows[0]], z["phys"][rows[0]])
        entropy[0] += obs.entropy()
        for t, r in enumerate(rows):
            pred, prob = obs.step(z["obs"][r], z["gob"][r], int(z["act"][r]), z["nxt"][r], z["gnx"][r])
            k = t // CHUNK
            bits[k] += -np.log2(max(prob, 1e-12))
            seen = robot_state(z["nxt"][r], z["gnx"][r])
            ev = int(z["events"][r])
            if (ev >> MOVE) & 1:
                tot["move"][k] += 1
                hit["move"][k] += pred[0] == seen[0]
            if (ev >> TURN) & 1:
                tot["turn"][k] += 1
                hit["turn"][k] += pred[1] == seen[1]
            violations += not obs.mask[truth]
            if (t + 1) % CHUNK == 0:
                entropy[k + 1] += obs.entropy()
    acc = {e: (hit[e] / np.maximum(tot[e], 1)).tolist() for e in hit}
    return dict(acc=acc, count={e: tot[e].tolist() for e in tot}, motion_bits=(bits / (len(worlds) * CHUNK)).tolist(),
                entropy=(entropy / len(worlds)).tolist(), violations=int(violations), worlds=len(worlds))


def model_curves(arm, adapt, split):
    last = None
    for line in open(f"runs/e8/{arm}/log.jsonl"):
        r = json.loads(line)
        if r.get("type") == "eval" and r.get("tag") == "end" and r["split"] == split and r["adapt"] == adapt:
            last = r                                           # the last end evaluation = E8c's
    return last["curves"]


def first_chunk(acc, level):
    hits = [k for k, a in enumerate(acc) if a >= level]
    return hits[0] if hits else None


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    out = {}
    for name, split in (("test", "test"), ("heldout", "heldout")):
        ideal = run_split(name)
        rec = dict(ideal=ideal)
        for arm, adapt in ARMS.items():
            c = model_curves(arm, adapt, split)
            rec[arm] = dict(acc={e: c["acc"][e] for e in ("move", "turn")}, count={e: c["count"][e] for e in ("move", "turn")})
        for who in ["ideal"] + list(ARMS):
            a = rec[who]["acc"]
            rec[who]["steps_to"] = {f"{e}_{lv}": (None if first_chunk(a[e], lv) is None else (first_chunk(a[e], lv) + 1) * CHUNK)
                                    for e in ("move", "turn") for lv in (0.9, 0.95, 0.99)}
        out[split] = rec
        print(f"[{split}] {time.time() - t0:.0f}s  violations {ideal['violations']}  entropy start {ideal['entropy'][0]:.1f} "
              f"-> chunk 4 {ideal['entropy'][4]:.1f} -> end {ideal['entropy'][-1]:.1f} bits", flush=True)
        for who in ["ideal"] + list(ARMS):
            a = rec[who]["acc"]
            print(f"   {who:6s} move {[round(x, 2) for x in a['move'][:8]]} ... {round(np.mean(a['move'][-4:]), 3)}   "
                  f"turn {[round(x, 2) for x in a['turn'][:8]]} ... {round(np.mean(a['turn'][-4:]), 3)}   steps_to {rec[who]['steps_to']}")
    json.dump(out, open(os.path.join(OUT, "ideal_vs_model.json"), "w"))


if __name__ == "__main__":
    main()
