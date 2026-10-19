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

## Results

*(to be filled in after the runs)*
