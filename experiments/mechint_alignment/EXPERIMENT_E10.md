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

## E10a results (2026-10-21; 16 episodes, 339 s; `runs/e10a.txt`, `runs/e10a/`): an apparatus defect

| arm | painted cells | nature painted over | neighbourhood distinct ratio (per layout; – = < 5 painted) | total progress (bits) | share painted on blank | WAIT | screen in view |
|---|---|---|---|---|---|---|---|
| baseline (E9c) | 30.0 | 27.3% | 1.00 × 8 | 499.8 | 0.46 | 0.33 | 0.06 |
| pres_sum | 16.8 | **4.0%** | 1.00 × 8 | 335.4 | 0.85 | 0.21 | 0.12 |
| pres_veto | 6.5 | **0.0%** | 1.00, 1.00, –, –, 0.60, 1.00, –, 1.00 | 95.8 | 1.00 | 0.14 | **0.38** |

Verdicts as run: E1 confirmed, E2 refuted, E3 confirmed for pres_sum and failed for pres_veto (3 layouts untestable),
E4 refuted (19%), E5 confirmed.

**The defect.** The veto arm watched the noisy screen 38% of the time, against the plain appreciator's 6%. The
preservation loss excluded the screen's own cells but not cells whose *context* includes a screen cell (the row
below the screen, the column to its right). Their contexts change every step whatever the agent does.
- Measured with the agent only walking and waiting, never painting: an exogenous loss of 3.0 bits per step (layout
  100) and 4.9 (layout 103), above the veto's 0.5-bit tolerance on 54% and 60% of steps.
- In planning, every imagined future therefore drew random vetoes and penalties unrelated to its actions. The veto
  arm behaved erratically; the sum arm's comparisons were noisy.
- **E10a's verdicts are confounded and not counted.**
- **Fix** (`prefgraph.world_contexts`): a cell is left out of the preservation count if it or any of its four causal
  neighbours is on the screen. A test now checks that walking without painting costs exactly 0.

## E10b, pre-registered (2026-10-21): E10a with the fixed apparatus

Everything as E10a (arms pres_sum and pres_veto, the same 8 layouts, the same planner, the E9c appreciator as the
reused baseline), with the screen excluded from preservation. **Predictions E1–E5 unchanged.**

    python e9_aesthetic.py --out runs/e10b --arms pres_sum,pres_veto --seed0 100 --seeds 8 --t-max 450 --n-seq 48 --horizon 8 --iters 3 --warm-start
