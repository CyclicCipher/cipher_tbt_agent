"""Re-score P0 runs 1 and 2 with the padding they were TRAINED under (planning §20.3).

The bug: training batches mix all six families, LEFT-padded to the longest problem in the batch (~37-41 tokens), so a
short problem (an `aff` or `s5` one is 5-20 tokens) was always seen behind ~20-35 PAD tokens. `p0_ceiling.evaluate`
batched one family at a time and padded only to that family's own length, so the same short problem was scored with
NO padding — an input the model never saw. Attention reads the PAD tokens, so the two differ: the run-2 `loop4` model
scores `aff` h=1 at 0.08 unpadded and 1.00 padded to 40.

This script re-scores every saved checkpoint with every problem left-padded to 40 tokens (the training regime; the
longest problem is 40 tokens) and prints the same table as `p0_ceiling.py`. Runs from §20.4 on RIGHT-pad instead, so
the answer position never sees padding and the question does not arise.

Usage: python reeval_leftpad.py runs/p0/r1_loop4.pt runs/p0/r2_loop4.pt ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "transformers"))
import h1_lid as H1  # noqa: E402
import tasks  # noqa: E402
from p0_ceiling import FAILED, SOLVED, ceilings  # noqa: E402

PAD_TO = 40


@torch.no_grad()
def score(model, fams, n=512, seed=12345, dev="cuda"):
    rng = np.random.default_rng(seed)
    out = {}
    for fam in fams:
        rows = []
        for li in range(len(fam.levels)):
            items = [fam.sample(rng, li) for _ in range(n)]
            T = max(PAD_TO, max(len(t) for t, _ in items))
            tok = torch.full((n, T), tasks.PAD, dtype=torch.long)
            for i, (t, _) in enumerate(items):
                tok[i, T - len(t):] = torch.tensor(t)
            ans = torch.tensor([a for _, a in items])
            pred = torch.cat([model(tok[i:i + 256].to(dev))[:, -1].argmax(-1).cpu() for i in range(0, n, 256)])
            acc = float((pred == ans).float().mean())
            chance = float(np.bincount(ans.numpy()).max() / n)
            norm = round((acc - chance) / (1 - chance), 4) if chance < 0.999 else None
            rows.append(dict(h=fam.levels[li], acc=round(acc, 4), chance=round(chance, 4), norm=norm))
        out[fam.name] = rows
    return out


def main():
    results = {}
    for path in sys.argv[1:]:
        ck = torch.load(path)
        a = ck["args"]
        # run 1's `ptr` used two tokens per pair; its PointerChase is no longer in tasks.py, so run 1's `ptr` is skipped
        run1 = Path(path).name.startswith("r1_")
        n_vocab = ck["state"]["emb.weight"].shape[0]
        model = H1.LoopedModel(d_model=a["d"], n_head=a["heads"], max_len=a["max_len"], pos="rope", n_vocab=n_vocab,
                               loops=a["loops"], tied=True, n_prelude=1, n_coda=1).cuda().eval()
        model.load_state_dict(ck["state"])
        fams = tasks.make_families(bf_seed=1, bf_per_level=a["bf_eval"], cache_dir=a["cache"])
        if run1:          # run 1 trained `bool` with RANDOM siblings; score it on that distribution
            tasks.BoolFormula.NEUTRAL = 0.5
            fams = [f for f in fams if f.name != "ptr"]
        res = score(model, fams)
        tasks.BoolFormula.NEUTRAL = 0.9
        results[Path(path).name] = res
        print(f"\n{Path(path).name} — left-padded to {PAD_TO} (solved >= {SOLVED}, failed <= {FAILED})")
        for name, rows in res.items():
            hs, hf = ceilings(rows)
            cells = "  ".join(f"{r['h']}:{r['norm'] if r['norm'] is None else format(r['norm'], '+.2f')}" for r in rows)
            print(f"  {name:5s} h_solved {hs!s:>4}  h_fail {hf!s:>4}  | {cells}")
    json.dump(results, open(HERE / "runs" / "p0" / "reeval_leftpad.json", "w"), indent=1)


if __name__ == "__main__":
    main()
