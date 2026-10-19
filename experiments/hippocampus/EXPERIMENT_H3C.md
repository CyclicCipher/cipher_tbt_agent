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

## Results (2026-10-18; `runs/h3c/`, log `runs/h3c_log.txt`)

Sleep runs took 84–88 s, the others 6–8 s.

| arm | reader | stream seed | **final, all facts, combined: fwd / rev / mean** | weights alone fwd / rev | online mean | rows at end | evicted per sleep | first-sleep retention (fwd / rev) | held-out rev before → after |
|---|---|---|---|---|---|---|---|---|---|
| value | 4 | 0 | 0.335 / 0.358 / **0.346** | 0.003 / 0.001 | 0.808 | 1,500 | – | – | 0.979 → 0.981 |
| **sleep** | 4 | 0 | 0.396 / 0.397 / **0.397** | 0.435 / 0.374 | 0.800 | 480 | 777, 1,030, 1,031, 1,020 | **0.24 / 0.18** (777 facts) | 0.979 → 0.978 |
| uncapped | 4 | 0 | 0.859 / 0.869 / **0.864** | 0.003 / 0.001 | 0.872 | 5,950 | – | – | 0.979 → 0.977 |
| value | 4 | 1 | 0.334 / 0.362 / 0.348 | 0.003 / 0.001 | 0.813 | 1,500 | – | – | 0.977 → 0.980 |
| **sleep** | 4 | 1 | 0.379 / 0.405 / **0.392** | 0.430 / 0.376 | 0.777 | 485 | 754, 1,040, 1,035, 1,015 | **0.23 / 0.18** (754) | 0.977 → 0.970 |
| uncapped | 4 | 1 | 0.856 / 0.865 / 0.861 | 0.003 / 0.001 | 0.875 | 5,957 | – | – | 0.977 → 0.976 |
| value | 5 | 0 | 0.345 / 0.369 / 0.357 | 0.002 / 0.001 | 0.803 | 1,500 | – | – | 0.985 → 0.976 |
| **sleep** | 5 | 0 | 0.393 / 0.397 / **0.395** | 0.432 / 0.373 | 0.757 | 478 | 659, 1,053, 1,030, 1,022 | **0.22 / 0.20** (659) | 0.985 → 0.981 |
| uncapped | 5 | 0 | 0.863 / 0.859 / 0.861 | 0.002 / 0.001 | 0.874 | 5,950 | – | – | 0.985 → 0.983 |
| value | 5 | 1 | 0.334 / 0.361 / 0.348 | 0.002 / 0.001 | 0.818 | 1,500 | – | – | 0.980 → 0.983 |
| **sleep** | 5 | 1 | 0.381 / 0.403 / **0.392** | 0.430 / 0.384 | 0.795 | 458 | 704, 1,035, 1,071, 1,042 | **0.20 / 0.19** (704) | 0.980 → 0.977 |
| uncapped | 5 | 1 | 0.853 / 0.856 / 0.854 | 0.002 / 0.001 | 0.875 | 5,957 | – | – | 0.980 → 0.983 |

**Against the predictions:**
- **C1 (sleep ≥ value + 0.15 on all facts): REFUTED.** The gain is only +0.04 to +0.05 (0.39–0.40 against 0.35).
- **C2 (sleep ≥ uncapped − 0.10): REFUTED.** 0.39 against 0.86.
- **C3 (online, sleep ≥ value − 0.02): fails in 3 of 4 pairs** (−0.008, −0.036, −0.046, −0.023).
- **C4 (reading survives four sleeps): holds.** −0.007 to −0.001.
- **C5 (first-sleep facts retained ≥ 0.8): REFUTED.** Retention is **0.18–0.24**.

**What happened: catastrophic interference in the weights.**
- Each sleep consolidates the store's current episodes, and self-checked eviction then removes the ones the weights
  answer. Those facts are **never replayed again**.
- The next sleeps train the weights on newer facts, and overwrite them: the first sleep's evicted facts fall from 1.00
  (the eviction self-check) to about 0.2 in the weights by the end.
- The weights hold about 40% of all facts at the end; the 480-row store holds a little more.
- Online accuracy dips, because a forgotten fact that has left the store is answered by weights that no longer know it.
- This is the failure complementary-learning-systems theory exists to explain: the slow learner must be trained on
  **interleaved** old and new material, or new learning overwrites old. Rehearsal here covered only the *reading skill*,
  which is why reading survived and the facts did not.

**The lesson:** self-checked eviction is loss-free *at the moment of eviction* (H3b), but it removes the facts'
replay source, so later learning erases them. **Sleep must also replay what the weights already know.**

**Next (H3d):** pseudo-rehearsal (Robins 1995; generative replay). Before each sleep, record the weights' own
confident empty-memory answers over the question space, and replay them during sleep alongside the new consolidation
examples and the reading rehearsal. The weights rehearse themselves, so no evicted fact needs to be kept.
