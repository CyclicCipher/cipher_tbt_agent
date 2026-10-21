# E10: the preference graph, first tests

The architecture is in `PREFERENCE_GRAPH.md`; the code is `prefgraph.py`.

## E10a, pre-registered (2026-10-21): appreciation + preservation

**Question.** The appreciator of E9c overwrote 27% of the nature it had learned from. Does a preservation node, whose
content is written by appreciation (a content edge), stop that? Does creation continue, moved elsewhere? And does it
matter whether preservation enters as a fixed exchange rate or as a veto?

**Nodes** (`prefgraph.py`):
- **Appreciation:** E9's appreciator, the compression progress of its own 5 x 5 view. τ = 16, summed bits; that is
  25× E9's mean-per-cell reward, so alone it chooses the same actions.
- **Preservation:** each learned context k has a value v_k, the progress felt on it (per view, the mean over its
  instances; summed over views). A step's loss is Σ_k max(v_k, 0) · max(0, log2(1 + n_k before) − log2(1 + n_k after)),
  where n_k counts the paintable cells with context k:
  - destroying the last instances of an appreciated structure costs the most;
  - destroying one of many copies costs little;
  - making copies earns nothing (the loss is one-sided).
- **Not low impact:** it protects only what was appreciated, so painting on blank is cheap (many copies of blank
  exist).
- **Not harm avoidance:** there are no other agents in Canvas.

**Arms** (everything else as E9c: the same 8 layouts 100–107, the cross-entropy planner 3 x 48 x 8 with warm start,
450 steps):

| arm | combination |
|---|---|
| **baseline** | E9c's appreciator. **Reused, not re-run**: identical configuration |
| **pres_sum** | appreciation − preservation loss: a fixed exchange rate of 1 bit for 1 bit |
| **pres_veto** | appreciation, with preservation as a veto: any step's loss above 0.5 bit dominates every trade |

    python e9_aesthetic.py --out runs/e10a --arms pres_sum,pres_veto --seed0 100 --seeds 8 --t-max 450 --n-seq 48 --horizon 8 --iters 3 --warm-start

**Predictions:**

| # | prediction |
|---|---|
| E1 | **Preservation protects.** Nature repainted: pres_veto ≤ 5%; pres_sum ≤ 13.5% (half the baseline's 27%) |
| E2 | **Creation continues.** Both arms paint ≥ 60% as many cells as the baseline (≥ 18 of its 30) |
| E3 | **Still no copies.** The neighbourhood distinct ratio is 1.00 in ≥ 7 of 8 layouts, in both arms |
| E4 | **The price in appreciation is modest.** pres_veto's total compression progress ≥ 70% of the baseline's |
| E5 | **Creation moves to blank.** In both arms, the share of painted cells that were originally blank exceeds the baseline's |

**Known design choices that shape the result:**
- During the first τ steps the lag model is empty, so everything seen at the start is felt as progress and gains
  value. Blank near the start position is therefore valued too; it is cheap to paint only because blank has many
  copies.
- The 0.5-bit veto tolerance allows small losses every step.
- The exchange rate of 1 is a choice, not a measurement.
