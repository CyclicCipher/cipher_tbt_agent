# Experiment H3c — sleep inside the continual stream

**Pre-registered 2026-10-18, before any H3c result run.** The user: "Do so [self-checked eviction + longer sleep, then
sleep inside the stream] … Keep doing steps without pause until I come back."

## Question

H1c showed that without crowding, value eviction keeps the often-asked facts at a quarter of the size but loses the long
tail (per fact 0.64 against 0.78 for the full store). H3b showed that sleep moves stored facts into the weights in both
directions, and that self-checked eviction then frees about 64% of the store at almost no cost.

**Does sleeping during a stream let a small store keep the long tail,** because facts leave the store by consolidation
rather than by being forgotten?

## Design (`h3c.py`, `run_h3c.py`)

- **The stream:** the union of worlds **100–101 (5,957 facts)** in a fixed random order, **24 phases**, **1,000
  Zipf(1) queries per phase** (half forward, half reverse), store **capped at 1,500 rows** (a quarter, as in H2), value
  eviction as in H2 (prior age = one phase).
  - This world is smaller than H2's 11,930, so four sleeps fit under the 2-minute rule.
- **Asking:** every question goes through the **combined switch**. A full match in the store (≥ 60 bits) gets the
  normal read; otherwise the memory is empty and the weights answer.
- **Readers:** E1 seeds 4 and 5, from `runs/h1c/`. **Stream seeds:** 0 and 1.

| arm | rule |
|---|---|
| **value** | capped, value eviction, no sleep |
| **sleep** | capped, value eviction, and after phases **6, 12, 18 and 24** a sleep: the store's episodes, plus the reader's own reverse inferences (normal read, p ≥ 0.5), **1,000 steps at 3:1** with rehearsal of reading (lr 5e-4 cosine), then **self-checked eviction** (H3b) |
| **uncapped** | no cap, no sleep (reference) |

**Final measures, after phase 24:**
- **Per-fact accuracy on every stream fact**, forward and reverse, through the combined system. This is the long-tail
  measure: every fact counts once, asked or not.
- **The weights alone** on every fact.
- **Online accuracy** (popularity-weighted, scored before credit).
- **Retention:** the weights' accuracy, at the end, on the facts evicted at the *first* sleep. Later sleeps may overwrite
  them, since evicted facts are never replayed again.
- **Held-out world 50**, read with its own store, before and after the stream.

## Smoke test (a throwaway 100-step reader, 100-step sleeps, not saved)

- The mechanics run: 4.8 s for the value arm, 14.8 s for the sleep arm. With 1,000-step sleeps, about 70 s.
- That reader is too weak to consolidate anything (0 evicted per sleep), so no numbers carry over.

## Predictions

- **C1 (sleep keeps the long tail):** final per-fact accuracy on all facts (mean of forward and reverse), sleep ≥ value
  + 0.15, for every reader × stream seed pair.
- **C2:** sleep ≥ uncapped − 0.10 on the same measure, for every pair.
- **C3 (no cost on frequent facts):** online mean, sleep ≥ value − 0.02, for every pair.
- **C4 (reading survives four sleeps):** held-out reverse after the stream ≥ before − 0.05 (sleep arm).
- **C5 (consolidated facts are retained):** at the end, the weights alone answer the first sleep's evicted facts at
  ≥ 0.8 (mean of forward and reverse).
- **Reported:** rows at the end (≤ 1,500 for the capped arms), episodes evicted at each sleep, and the weights alone on
  all facts.

## Cost

12 runs: 4 sleep runs of about 70–90 s, 8 others of 5–20 s. About 7 minutes.

---

## Results

*(to be filled in after the runs)*
