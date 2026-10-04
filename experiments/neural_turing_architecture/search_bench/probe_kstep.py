"""A transferable precondition probe for GCML's inverse model (planning doc §13.4).

GCML's use is to map a MULTI-step difference (goal − state) to a good FIRST thought. The probe measures exactly that on
the model's own experience, with no knowledge of the right thoughts or of any goal: run value-gradient chains (the
`grad_greedy` rule: one normalised ∇_z V step per thought) on training problems; fit the ridge inverse model on their
ONE-step pairs; on held-out chains, measure the cosine between W·(c_{t+k} − c_t) and the first thought z_t, for
k = 1, 2, 3. A real looped model can produce such chains from its own value head. Compared, per variant, with G2's
direct measure of usefulness (one-step progress of W·(goal − state) toward real goals).
    python experiments/neural_turing_architecture/search_bench/probe_kstep.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from mock_task import TaskConfig  # noqa: E402
from structured_task import StructuredWalk, VARIANTS  # noqa: E402
from gcml import value_gradient_chains as chains, kstep_probe  # noqa: E402

torch.set_num_threads(4)


rows = []
for v, V in VARIANTS.items():
    for d in (64, 256):
        for world in range(4):
            cfg = TaskConfig(d=d, beta=16.0, gate=0.3, value_noise=0.1)
            make = lambda ps: StructuredWalk(cfg, world, ps, graph=V["graph"], keys=V["keys"])  # noqa: E731
            gen = torch.Generator().manual_seed(world * 31 + d + 5)
            t0 = make(world * 1000 + 999)
            code = V["code"]
            tr = chains(make, [world * 1000 + j for j in range(24)], gen)
            te = chains(make, [world * 1000 + 600 + j for j in range(24)], gen)
            res = dict(variant=v, d=d, world=world)
            res.update({f"k{k}": x for k, x in kstep_probe(t0, code, tr, te).items()})
            rows.append(res)
agg = {}
for r in rows:
    agg.setdefault((r["variant"], r["d"]), []).append(r)
print("| variant d | k=1 | k=2 | k=3 |\n|---|---|---|---|")
for (v, d), rs in agg.items():
    print(f"| {v} d={d} | " + " | ".join(f"{sum(r[f'k{k}'] for r in rs) / len(rs):.2f}" for k in (1, 2, 3)) + " |")
json.dump(rows, open(HERE / "runs" / "gcml_probe_kstep.json", "w"), indent=0)
