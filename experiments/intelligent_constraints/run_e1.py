"""Experiments 1 and 1b driver: the capacity pilot, then the grid. Existing result files are reused, never re-run.

    python run_e1.py            # Experiment 1  (learned token codes)  -> runs/e1/
    python run_e1.py --frozen   # Experiment 1b (frozen random codes)  -> runs/e1b/

A configuration is (mlp kind, MLP width, tied embeddings, seed); arms map onto configurations, so a configuration two
arms share (arm C at m = arm A at 2m) is trained once.
"""
import json
import sys
from pathlib import Path

from train import run, save

FROZEN = "--frozen" in sys.argv
OUT = Path(__file__).parent / "runs" / ("e1b" if FROZEN else "e1")
ARMS = {"A": lambda m: ("plain", m, True),         # plain MLP (baseline)
        "B": lambda m: ("dagger", m, True),        # dagger MLP (the hypothesis)
        "C": lambda m: ("plain", 2 * m, True),     # compute control: same MLP FLOPs as B, 2x parameters
        "D": lambda m: ("dagger", m, False)}       # dagger MLP, untied embeddings (code-match test)
PILOT = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def get(mlp, w, tied, seed):
    p = OUT / f"{mlp}_w{w}_{'tied' if tied else 'untied'}_s{seed}.json"
    if p.exists():
        return json.load(open(p))
    r = run(mlp, w, tied, seed, frozen=FROZEN)
    save(r, p)
    print(f"ran {p.name}: train {r['final']['train_acc']}  one-way clean {r['final']['rev_oneway_clean']['hits']}"
          f"  {r['secs']}s", flush=True)
    return r


def main():
    tight = None
    for m in PILOT:                                # smallest width at which the plain MLP memorises the training set
        if get("plain", m, True, 0)["final"]["train_acc"] >= 0.99:
            tight = m
            break
    if tight is None:
        print("no width memorised the training set: stop")
        return
    caps = (("tight", tight), ("ample", 8 * tight))
    print(f"\ntight m = {tight}, ample m = {8 * tight}\n")
    hdr = (f"{'cap':6s}{'m':>5s} arm s  params  train reach99 | fwd   revP  | 1-way: hits  mass   leak "
           f"| transpose off: 1-way revP | secs")
    print(hdr + "\n" + "-" * len(hdr))
    for cap, m in caps:
        for arm, f in ARMS.items():
            for seed in (0, 1):
                r = get(*f(m), seed)
                fin, off = r["final"], r.get("final_rev_off")
                c = fin["rev_oneway_clean"]
                o = (f"{off['rev_oneway_clean']['hits']:.3f}  {off['rev_paired']['hits']:.3f}" if off
                     else "  -      -  ")
                print(f"{cap:6s}{m:5d}  {arm}  {seed} {r['params']:7d}  {fin['train_acc']:.3f} {str(r['reach99']):>6s} | "
                      f"{fin['fwd']['hits']:.3f} {fin['rev_paired']['hits']:.3f} | {c['hits']:.3f}  {c['mass']:.3f}  "
                      f"{c['leak']:.3f} |       {o}      | {r['secs']}")
    print(f"\nchance hits@1 on one-way questions: {r['chance']}")


if __name__ == "__main__":
    main()
