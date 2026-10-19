# Experiment H3b — longer sleep, and self-checked eviction

**Pre-registered 2026-10-18, before any H3b result run.** The user: "Do so [H3b, then sleep inside the stream] … Keep
doing steps without pause until I come back."

## Why

H3 found two limits.

1. **The sleep budget.** With 2,000 steps, the self-generated-inference arm (I) brought the weights only to about 0.6
   in each direction. Each fact had about 44 exposures, and the forward and reverse questions shared the consolidation
   batch.
2. **An unsafe eviction rule.** "p ≥ 0.1 in both directions" evicted facts whose weights-only *top* answer was wrong,
   and cost the combined system 0.13.

## Changes (`h3.py` parameters, `run_h3b.py`)

- **Self-checked eviction:** evict an episode only if the weights alone (empty memory) give the stored answer as their
  **top** answer, forward ([a, r] → b) **and** reverse ([b, r°] → a).
  - The store holds the answer, so no label is needed.
  - By construction, nothing is evicted that the weights answer wrongly.
- **Longer sleep:** 5,000 steps instead of 2,000 (about 110 exposures per consolidation example at 1:1).
- **Two consolidation : rehearsal mixes,** both batch 256:

| arm | steps | consolidation + rehearsal per batch | eviction |
|---|---|---|---|
| **I5k** | 5,000 | 128 + 128 (1:1, as H3) | self-checked |
| **I5k31** | 5,000 | 192 + 64 (3:1) | self-checked |

Everything else is as in H3: E1 readers of seeds 4 and 5, world 0, inferences from the normal read path at p ≥ 0.5, lr
5e-4 cosine, the combined switch (normal read on a full match ≥ 60 bits, otherwise an empty memory), and held-out world
50. For comparison, each run also reports what the p ≥ 0.1 rule would have evicted.

## Predictions

- **B1 (more sleep, better weights):** I5k reaches weights-only forward ≥ 0.85 and reverse ≥ 0.75, in both seeds.
- **B2:** the 3:1 mix reaches weights-only accuracy ≥ the 1:1 mix (averaged over seeds, in each direction).
- **B3 (reading survives):** world-0 memory reads and held-out world 50 keep reverse ≥ their pre-sleep value − 0.05, in
  both arms. Less rehearsal (3:1) is the risk.
- **B4 (self-checked eviction is loss-free):** the combined system after eviction is within **0.02** of the combined
  system right after sleep, forward and reverse, in every run.
- **B5 (compression):** self-checked eviction removes ≥ 50% of the episodes for I5k.

## Cost

4 runs of about 90–110 s each. Each must stay under 2 minutes; the last H3 runs took 42–48 s for 2,000 steps.

---

## Results (2026-10-18; `runs/h3/I5k*_s*.json`, log `runs/h3b_log.txt`)

All runs took 93–100 s.

| arm | seed | **weights only fwd / rev** | memory rev, before → after | held-out rev, before → after | evicted (self-checked) / p ≥ 0.1 would / rows | combined fwd / rev: after sleep → **after eviction** (read share) |
|---|---|---|---|---|---|---|
| I5k (1:1) | 4 | **0.998 / 0.968** | 0.975 → 0.979 | 0.976 → 0.981 | **1,933** / 2,486 / 2,994 | 0.956 / 0.975 → **0.955 / 0.970** (0.34) |
| I5k (1:1) | 5 | **0.997 / 0.964** | 0.972 → 0.972 | 0.981 → 0.974 | **1,921** / 2,495 / 2,994 | 0.959 / 0.972 → **0.947 / 0.963** (0.34) |
| I5k31 (3:1) | 4 | **0.998 / 0.968** | 0.975 → 0.978 | 0.976 → 0.975 | **1,934** / 2,508 / 2,994 | 0.963 / 0.976 → **0.953 / 0.970** (0.34) |
| I5k31 (3:1) | 5 | **0.998 / 0.966** | 0.972 → 0.972 | 0.981 → 0.973 | **1,924** / 2,508 / 2,994 | 0.962 / 0.972 → **0.950 / 0.964** (0.34) |

**Against the predictions:**
- **B1 (more sleep, better weights): holds, strongly.** The weights alone reach forward 0.997–0.998 and **reverse
  0.964–0.968**, about the store's own read accuracy (0.97). Every reverse fact in the weights came from the reader's
  own inferences.
- **B2 (3:1 ≥ 1:1): holds, as a tie.** 0.998 / 0.967 against 0.998 / 0.966. At 5,000 steps both mixes saturate.
- **B3 (reading survives): holds in every run.** Memory reads change by −0.000 to +0.004; held-out world 50 by
  −0.008 to +0.005. Even 64 rehearsal examples per batch suffice.
- **B4 (self-checked eviction is loss-free within 0.02): holds in every run.** The largest drop is 0.012 (forward). The
  small losses come from questions with several answers: evicting some of a group's episodes changes what the reader
  sees for the rest.
- **B5 (≥ 50% evicted): holds,** 64–65%.
  - Why not more: the self-check needs the weights' *top* answer to be the stored one, so for a question with several
    answers only one of its episodes can pass.
  - The p ≥ 0.1 rule would have evicted 83–84%, unsafely (H3: −0.13).

**What H3b establishes:**
1. **Complementary learning systems, working end to end:**
   - one write per fact into the store;
   - reading in both directions at once;
   - sleep (replay of the episodes plus the reader's own reverse inferences, with reading rehearsed) moves **both
     directions** into the weights at about 0.97;
   - **64% of the store is then freed at ≤ 0.012 cost.**
   - Compression tier 3 (`PLAN.md` §2b) works.
2. **No reverse data is needed anywhere.** The reverse knowledge in the weights is the model's own inference.
3. **The remaining store is the part the weights cannot yet hold:** mostly set-valued facts, which the self-check keeps
   conservatively.

**Next:** H3c, sleep inside the continual stream, so consolidation keeps the long tail that value eviction loses.
