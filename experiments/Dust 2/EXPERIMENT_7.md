# Experiment 7 — Block AttnRes and DiffusionBlocks (pre-registered 2026-10-10, stage by stage)

**Where it comes from.** The user (2026-10-10): "move to DiffusionBlocks-like approaches and integrate Block AttnRes into
both the control model and the one we're testing the new algorithm on. These are complementary."

**What this experiment's design takes from the earlier ones:**
- **Depth does not raise Dust's noise** (Experiment 1, M0). DiffusionBlocks' headline benefit, a credit path one block
  deep, therefore targets a cost Dust does not have here.
- **Local scoring transfers.** Local scoring (auxiliary heads, reruns truncated at the next head) was our clearest
  training win, and DiffusionBlocks is a principled version of it: each block gets an exact, independent target.
- **The cost is higher.** For autoregressive text, every block processes the context plus a noisy copy of each target:
  2× the tokens.

## Stage 1 (7a) — Block AttnRes alone (`attnres.py`)

**The model.** TinyGPT with one AttnRes block per transformer block:
- before each sublayer and the head, a softmax mix over the embedding, the completed blocks' outputs, and the partial
  sum;
- zero-initialised pseudo-queries;
- RMSNorm'd keys and raw values.

**How Dust trains it.** Dust perturbs each mix's scores as a site (K_mix = 8 draws; at most 5 sources).

| arm | settings | compare (reused, not re-run) |
|---|---|---|
| **bp_ar** (the control) | backprop, Adam lr 2e-2 | backprop without AttnRes: 2.1586 |
| **dust_ar_O4** | Dust O4, K = 56 | O4 without AttnRes: 2.3329 |

**Apparatus check first.** At a backprop-trained AttnRes checkpoint, the per-site error cosines must follow √(K/(K+D))
as in Experiment 1. If not, the sites or reruns are wrong.

**Predictions.**
- **A1:** AttnRes changes backprop by less than 0.02 at this depth (4 blocks). The paper's gains are about a 1.25×
  compute equivalent at 16–48 layers. Prior work in this repo (ZipLearn E30) found AttnRes's value mostly at depth.
- **A2:** dust_ar_O4 is within ±0.02 of O4 without AttnRes. The mix sites add ~9 cheap rerun sites.

## Stage 2 (7b) — DiffusionBlocks with Block AttnRes across blocks

Pre-registered once stage 1 has run; designed in `DIFFUSIONBLOCKS.md`.

### 7a — amendment, before it ran

At the pre-registered lr 2e-2, bp_ar ended at 2.364. The apparatus checkpoint (same model, lr 3e-3, fp32, seed 0) reached
2.337. So the control gets its own learning-rate check: lr 3e-3 and 1e-2, two seeds, 300 steps. These are seconds-long
backprop runs; the lr-2e-2 runs are kept, not re-run.

**Rule violation, recorded.** In the lr check, the plain model's seed-0 runs at lr 3e-3 and 1e-2 repeated earlier smoke
runs exactly (identical results: 2.3437, 2.1776), against the user's rule. Since then, `train.py` refuses to run when its
result file already exists.

### 7a — second amendment, before it ran

**The finding.** AttnRes cannot use high learning rates:

| lr | plain | AttnRes |
|---|---|---|
| 3e-3 | 2.3509 | 2.3404 |
| 1e-2 | 2.1885 | 2.3300 (unstable seeds) |
| 2e-2 | 2.1586 | 2.3641 |

**Hypothesis.** The pseudo-queries' scores jump at high Adam rates and lock the softmax onto one source; the paper calls
zero initialisation crucial against this kind of volatility.

**The test.** lr 2e-2, with the pseudo-queries at 0.1× and 0.01× that rate (`--mix_lr`), two seeds, backprop.

## 7a — results (`runs/e7/`)

| arm | validation per seed | mean | compare |
|---|---|---|---|
| bp_ar, lr 2e-2 (pre-registered) | 2.3741 / 2.3541 | 2.3641 | plain backprop 2.1586: **+0.206** |
| bp_ar, lr 1e-2 | 2.2965 / 2.3635 | 2.3300 | plain lr 1e-2 2.1885 |
| bp_ar, lr 3e-3 | 2.3366 / 2.3442 | 2.3404 | plain lr 3e-3 2.3509: −0.011 |
| bp_ar, lr 2e-2, pseudo-queries × 0.1 | 2.2753 / 2.3595 | 2.3174 | |
| bp_ar, lr 2e-2, pseudo-queries × 0.01 | 2.1572 / 2.2324 | 2.1948 | plain 2.1586: seed 0 level (−0.001), seed 1 +0.074 |
| dust_ar_O4, K = 56 | 2.3444 / 2.3785 | 2.3615 | O4 without AttnRes 2.3329: **+0.029**; **151–156 s per run (over the limit)** |

**Apparatus check passed.** At the AttnRes checkpoint, the per-site error cosines follow √(K/(K+D)): MLP output 0.561
against 0.577 at K = 32, and 0.885 against 0.894 at K = 256. The mix weights reach cos 0.5–0.9 with 8 draws. The mixes
add 12% to Dust's cost.

### Against the predictions

- **A1 — at the pre-registered lr, REFUTED.** AttnRes makes backprop much worse.
  - **At lr 3e-3:** it is level (−0.011).
  - **At lr 2e-2 with pseudo-queries slowed 100×:** level on one seed, worse on the other.
  - **Overall:** at this depth (4 blocks) AttnRes gives no gain and is fragile at the high learning rates that train the
    plain model best.
- **A2 — REFUTED.** Dust with AttnRes is 0.029 worse than without, and slower: its rerun path is not compiled.

### What it says

**Block AttnRes needs depth to pay.** The paper reports about a 1.25× compute equivalent at 16–48 layers. Our model has
4. As integrated here, it adds fragility (the pseudo-query learning rate) and cost (mix sites; an uncompiled rerun).

**DiffusionBlocks is also built for deep models.** Its blocks of about 3 layers save memory in proportion to the number
of blocks.

**The next step:** a fair test of the user's plan needs a deeper model, in BOTH the backprop control and the Dust model.
That costs more per run than our 4-layer runs. The choice of regime is the user's.
