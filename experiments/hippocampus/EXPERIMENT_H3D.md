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

## Results

*(to be filled in after the runs)*
