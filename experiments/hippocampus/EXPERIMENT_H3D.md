# Experiment H3d — sleep with pseudo-rehearsal: the weights replay what they already know

**Pre-registered 2026-10-18, before any H3d result run.** It continues the user's "keep doing steps without pause".

## Why

In H3c, sleeping inside the stream suffered **catastrophic interference.**
- Facts consolidated at the first sleep and then evicted by the self-check were never replayed again.
- Later sleeps overwrote them (retention 0.18–0.24).
- The long tail improved only +0.04.

Complementary-learning-systems theory's answer is **interleaving**: the slow learner must keep seeing old material
while it learns new. The old material no longer exists in the store, and keeping it there would defeat the
compression. So the weights must **replay themselves**: pseudo-rehearsal (Robins 1995), or generative replay.

## The change (`h3.what_weights_know`, `h3c.py` arm `sleep_pr`)

- **Before each sleep:** ask the weights, with an **empty memory**, every question in the question space:
  - all 1,000 entities × 6 relations, forward;
  - all 1,000 entities × 6 converse tokens, reverse;
  - 12,000 questions in all.
  
  Keep the confident answers (p ≥ 0.5) as targets: what the weights already know.
- **During sleep:** each batch is **128 consolidation + 64 pseudo-rehearsal + 64 reading rehearsal**. At the first
  sleep, when nothing is yet known, it falls back to H3c's 192 + 64.
- **Generality caveat:** enumerating every question is a convenience of this small world. A general system would
  *sample* cues (for example recent entities, or from its own generative model). The test is whether self-replay
  prevents the forgetting, not how its cues are chosen.

Everything else is identical to H3c's `sleep` arm: the stream, 1,000-step sleeps after phases 6/12/18/24 at lr 5e-4
cosine, self-checked eviction, the combined switch, E1 readers 4/5, stream seeds 0/1. H3c's results are the reference
(`value`, `sleep`, `uncapped`), and are not re-run.

## Smoke test (a throwaway reader, 100-step sleeps; the confidence filter forced to 0 in-process to exercise the path)

The path runs, with all 12,000 pseudo examples in that forced setting. A run takes about 17 s at 100-step sleeps, so
about 90 s at 1,000.

## Predictions (each for every reader × stream-seed pair)

- **D1 (forgetting stopped):** the first sleep's evicted facts are still answered by the weights at the end at
  **≥ 0.8** (mean of forward and reverse). H3c: 0.18–0.24.
- **D2 (the long tail kept):** final per-fact accuracy on all facts (combined, mean of forward and reverse) is
  ≥ H3c's `value` + 0.15, that is ≥ 0.50.
- **D3:** the weights alone on all facts ≥ 0.6 (mean of forward and reverse). H3c sleep: 0.40.
- **D4 (reading survives):** held-out reverse after ≥ before − 0.05.
- **D5 (no cost on frequent facts):** online mean ≥ H3c's `value` − 0.02.

## Cost

4 runs of about 90 s each.

---

## Results (2026-10-18; `runs/h3c/sleep_pr_*.json`, log `runs/h3d_log.txt`)

All runs took 82–86 s.

| reader | stream seed | **final, all facts, combined: fwd / rev / mean** | weights alone fwd / rev | online mean | rows at end | evicted per sleep | pseudo examples per sleep | first-sleep retention fwd / rev | held-out rev |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 0.583 / 0.620 / **0.601** | 0.637 / 0.619 | 0.824 | 583 | 324, 823, 856, 917 | 2, 870, 4,371, 5,756 | 0.46 / 0.49 (324 facts) | 0.979 → 0.973 |
| 4 | 1 | 0.580 / 0.623 / **0.602** | 0.622 / 0.619 | 0.797 | 551 | 412, 830, 913, 949 | 2, 1,355, 5,391, 6,119 | 0.50 / 0.47 (412) | 0.977 → 0.975 |
| 5 | 0 | 0.603 / 0.596 / **0.599** | 0.647 / 0.597 | 0.828 | 555 | 726, 803, 896, 945 | 42, 3,017, 5,826, 6,211 | 0.52 / 0.49 (726) | 0.985 → 0.973 |
| 5 | 1 | 0.609 / 0.617 / **0.613** | 0.664 / 0.602 | 0.832 | 549 | 722, 815, 958, 951 | 42, 3,011, 5,902, 6,522 | 0.53 / 0.48 (722) | 0.980 → 0.973 |

H3c's references: value 0.346–0.357, sleep 0.392–0.397, uncapped 0.854–0.864 on the same measure; first-sleep retention
under plain sleep was 0.18–0.24.

**A deviation from the pre-registration (implementation).**
- The design said the first sleep falls back to 192 consolidation + 64 reading "when nothing is yet known".
- The code fell back only when the pseudo set was *empty*. At the first sleep the weights had **2–42 spurious confident
  answers**, so the pseudo path ran: 128 consolidation + 64 draws from those few examples.
- That weakened the first sleep. It evicted 324–726 facts, against 659–777 under H3c's plain sleep, and the retention
  measure is over that smaller set.
- The fix (a minimum pseudo-set size) is part of H3e.

**Against the predictions:**
- **D1 (retention ≥ 0.8): REFUTED.** 0.46–0.53, up from 0.18–0.24. Forgetting is halved, not stopped.
- **D2 (long tail ≥ 0.50): holds in all four.** 0.599–0.613, against 0.35 without sleep and 0.39 with plain sleep:
  **+0.25.**
- **D3 (weights alone ≥ 0.6): holds in all four.** 0.62–0.63, against 0.40.
- **D4 (reading survives): holds.** −0.012 to −0.002.
- **D5 (online ≥ value − 0.02): holds in all four.** −0.016 to +0.025.

**What H3d establishes:**
1. **Self-replay works against catastrophic interference,** partially. Replaying the weights' own confident answers
   (no stored data) lifts the long tail from 0.39 to 0.60, and the weights alone from 0.40 to 0.63, at no cost to
   frequent facts or to reading.
2. **The store ends at about 550 rows against the uncapped 5,950 (11× smaller),** holding 0.60 per fact against the
   full store's 0.86. The remaining gap is forgetting: about half of what was consolidated early is lost by the end.
3. **A likely cause is a dose problem.** By the last sleep the pseudo set is about 6,000 answers, at 64 per batch, so
   about 11 replays each per sleep, against about 43 for new facts.

**Next (H3e):** the first-sleep fallback fixed (pseudo set < 100 → no pseudo path), and a larger pseudo share
(96 consolidation + 96 pseudo + 64 reading).
