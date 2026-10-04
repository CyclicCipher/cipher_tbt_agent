"""Follow-up to G0/G1 (planning doc §13): two measurements the first run got wrong or lacked.

  signal       mean ‖Δc‖ of a transition, relative to a full move (a demonstration's): how much a random thought, or a
               thought the search simulated, actually changes the state code. (G1's "share that changed the state" used a
               threshold so low that a tiny leak of mass counted, so it read 1.00 everywhere.)
  predictability  a precondition probe that needs no knowledge of the right thoughts, so it transfers to a real model:
               fit the ridge inverse model on half of a world's OWN search transitions, and on the other half measure the
               cosine between W·Δc and the thought that produced Δc. G0's invariance turned out not to predict whether
               W works (V2sr: invariance 0.05, W progress 0.88-0.96); this is the candidate replacement.
    python experiments/neural_turing_architecture/search_bench/probe_gcml.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from mock_task import TaskConfig, _unit  # noqa: E402
from structured_task import StructuredWalk, VARIANTS  # noqa: E402
from searchers import ALGOS  # noqa: E402
from gcml import transitions_random, transitions_demo, transitions_search, InverseModel  # noqa: E402

torch.set_num_threads(4)
rows = []
for v, V in VARIANTS.items():
    for d in (64, 256):
        for world in range(3):
            cfg = TaskConfig(d=d, beta=16.0, gate=0.3, value_noise=0.1)
            make = lambda ps: StructuredWalk(cfg, world, ps, graph=V["graph"], keys=V["keys"])  # noqa: E731
            gen = torch.Generator().manual_seed(world * 31 + d)
            t0 = make(world * 1000 + 999)
            code = V["code"]
            mag = lambda tr: float((t0.code(tr[2], code) - t0.code(tr[0], code)).norm(dim=1).mean())  # noqa: E731
            demo = transitions_demo(t0, 2048, gen)
            rand = transitions_random(t0, 2048, gen)
            srch = transitions_search(make, [world * 1000 + j for j in range(8)], ALGOS["mcts_hybrid"], 512, gen)
            full = mag(demo)
            n = srch[0].shape[0]
            perm = torch.randperm(n, generator=gen)
            a, b = perm[: n // 2], perm[n // 2:]
            W = InverseModel("ridge", code).fit(t0, *(x[a] for x in srch))
            dc = t0.code(srch[2][b], code) - t0.code(srch[0][b], code)
            keep = dc.norm(dim=1) > 0.05 * full                       # only transitions that actually moved
            pred = W.propose(t0, srch[0][b][keep], dc[keep])
            cos = float((pred * srch[1][b][keep]).sum(1).mean()) if keep.any() else float("nan")
            rows.append(dict(variant=v, d=d, world=world, random_signal=mag(rand) / full, search_signal=mag(srch) / full,
                             predictability=cos, n_moved=int(keep.sum()), n=int(len(b))))
            print(f"{v:8s} d={d:4d} w={world}  signal random {mag(rand) / full:.3f}  search {mag(srch) / full:.3f}  "
                  f"predictability {cos:.2f}  ({int(keep.sum())}/{len(b)} moved)", flush=True)
json.dump(rows, open(HERE / "runs" / "gcml_probe.json", "w"), indent=0)
