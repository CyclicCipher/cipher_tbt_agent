# Experiment H2 — a continual stream: interference by crowding, and eviction by value

**Pre-registered 2026-10-18, before any H2 result run.** The user: "Update the plan documents with learnings and
results. Then do H2."

## Question

H1/H1b wrote one fixed world into an empty store. H2 asks what happens over a **stream** of facts and questions, and
answers two things.

1. **Interference.** The store has no weight interference: rows are separate. But as it grows, an entity accumulates
   more episodes than the R = 12 read slots. Its correct episode then competes for a slot (**crowding**). How much does
   that cost on old facts?
2. **Forgetting by value.** With the store capped, which rows should go? ZipLearn's rule is "forgetting chosen by value:
   the cheapest to lose goes first". Here value is a row's observed rate of usefulness, compared against
   first-in-first-out, random, and an oracle.

## Readers (step 1)

- **Two new B2 readers** (H1b's carried-forward configuration: compile, fused AdamW, 4,500 steps, cosine decay,
  familiarity gate), seeds **2 and 3**.
- They are new seeds, so H1b's runs are not repeated. Their world-0 results are recorded as a **replication of H1b**.
- Weights are saved to `runs/h2/reader_s{2,3}.pt` (git-ignored), for H2 and later H3.

## The stream (step 2, `h2.py`)

- **World:** the union of worlds 100–103, giving **11,930 facts** over the same 1,000 entities and 6 relations. Mean
  entity degree is 23.9; an entity is the object of 11.9 facts on average (max 26).
- **Arrival:** a fixed random order, in 24 phases of about 497 facts.
- **Each phase:**
  1. write the phase's facts through the gate: familiarity, then surprise > τ = 2 bits;
  2. evict down to the capacity (capped runs);
  3. answer **2,000 queries**, half forward and half reverse.
- **Queries:** sampled from all facts written so far, evicted ones included (the world asks about anything it has
  shown). Each fact has a fixed **Zipf(1) popularity**, assigned by a random rank independent of arrival time.
- **Scoring:** each query is scored *before* any credit is given. That is **online accuracy**, as in ZipLearn's
  prequential measure.
- **Evidence:** a read row earns +1 when it supplied the answer the reader gave: (a, r, answer) for a forward question,
  (answer, r, b) for a reverse one. No ground truth is used.
- **Probe:** after every phase, forward and reverse questions on the **first phase's 497 facts** (uniform, not
  popularity-weighted). This measures retention of old facts.
- **Stream randomness** is seed 0 for every run, so policies and readers face the same stream.

| policy | capacity | rule |
|---|---|---|
| uncapped | none | keep everything |
| fifo | 3,000 rows | evict the oldest written |
| random | 3,000 | evict at random |
| **value** | 3,000 | evict the lowest **(evidence + 1) / (age + 2,000)**, with age in queries since writing. This is a usefulness rate with a prior of one use per phase, so a new row is not evicted before it has had a chance to be used |
| oracle | 3,000 | evict the lowest *true* popularity (not observable; an upper bound) |

That makes 5 policies × 2 readers = 10 stream runs of about 7 s each, plus 2 reader runs of about 90 s.

## Smoke tests before writing this (not saved)

- A 200-step reader (too weak for reverse questions) ran the stream under `uncapped` and `value`, in about 7 s each.
- The mechanics work: 11,930 facts written, and the capped store holds at 3,000.
- **Reader-independent fact seen there: retrieval recall.** In the uncapped store, reverse recall on the first phase's
  facts falls from 1.00 (497 rows) to **0.42** (11,930 rows). An entity's ~24 episodes all overlap a reverse cue by the
  same 40 bits, and only 12 can be read. The prediction below is therefore stated on *accuracy*, not recall.
- With the weak reader, forward online accuracy fell from 0.98 to 0.61 across the uncapped stream, and was 0.82 overall
  with `value` against 0.76 uncapped. Reverse was about 0 throughout (the reader was too weak).

## Predictions

- **R (replication of H1b):** both new readers reach reverse ≥ 0.6 and forward ≥ 0.93 on world 0, with 0 duplicates.
- **S1 (interference by crowding, uncapped):** the probe's reverse accuracy at the end of the stream is ≤ 0.7 × its
  value at phase 6 (about 3,000 rows), in both seeds.
- **S2 (eviction by value):** online accuracy, the mean of forward and reverse over the whole stream:
  - value ≥ fifo + 0.10 and ≥ random + 0.10, in both seeds;
  - oracle ≥ value.
- **S3 (forgetting by value beats keeping everything here):** value-capped online accuracy ≥ uncapped, in both seeds.
  Under skewed queries, crowding should cost the full store more than forgetting costs the value-capped one.
- **S4 (writes):** ≥ 99% of the stream's facts are written on first presentation, since none is familiar.
- **Reported, no prediction:**
  - the probe under each capped policy (fifo should lose the first phase entirely once the store turns over);
  - rows and bytes at the end (14 bytes per row);
  - the forward/reverse split.

## Cost

About 3.5 minutes of reader training plus about 1.5 minutes of streams. Every run is ≤ 2 minutes.

---

## Results (2026-10-18; `runs/h2/`, log `runs/h2_log.txt`)

Reader runs took 88–93 s and stream runs 8–15 s, all within the 2-minute rule.

**Readers on world 0 (the H1b replication):**

| seed | fwd | REV one-way clean | reverse curve (every 500 steps) | first presentation / second |
|---|---|---|---|---|
| 2 | 0.956 | **0.475** | 0.21 … 0.47, 0.48, 0.48: **never left the plateau** | 2,997 / 0 |
| 3 | 0.958 | **0.963** | 0.25 … 0.53, 0.58 → **0.88 → 0.97** | 2,997 / 2 |

Seed 3's "2 on the second presentation" are *not* duplicates. The familiarity gate recognised all 2,997 stored facts.
The 2 were facts the gate had skipped as unsurprising the first time, and wrote the second time.

**Streams** (online accuracy over all 48,000 queries; the probe is the first phase's 497 facts):

| policy | seed | online fwd / rev / **mean** | probe rev: phase 6 → 24 (recall at 24) | probe fwd at 24 | rows at end | bytes |
|---|---|---|---|---|---|---|
| uncapped | 2 | 0.730 / 0.438 / **0.584** | 0.523 → 0.211 (0.43) | 0.702 | 11,929 | 167 KB |
| fifo | 2 | 0.572 / 0.437 / 0.504 | 0.523 → 0.189 (0.00) | 0.342 | 3,000 | 42 KB |
| random | 2 | 0.578 / 0.430 / 0.504 | 0.523 → 0.193 (0.06) | 0.368 | 3,000 | 42 KB |
| **value** | 2 | 0.731 / 0.549 / **0.640** | 0.523 → 0.282 (0.28) | 0.463 | 3,000 | 42 KB |
| oracle | 2 | 0.753 / 0.535 / 0.644 | 0.523 → 0.272 (0.24) | 0.422 | 3,000 | 42 KB |
| uncapped | 3 | 0.799 / 0.664 / **0.732** | 0.915 → 0.284 (0.41) | 0.771 | 11,928 | 167 KB |
| fifo | 3 | 0.607 / 0.687 / 0.647 | 0.915 → 0.378 (0.00) | 0.356 | 3,000 | 42 KB |
| random | 3 | 0.623 / 0.707 / 0.665 | 0.915 → 0.398 (0.06) | 0.380 | 3,000 | 42 KB |
| **value** | 3 | 0.771 / 0.877 / **0.824** | 0.915 → 0.511 (0.26) | 0.461 | 3,000 | 42 KB |
| oracle | 3 | 0.794 / 0.907 / 0.851 | 0.915 → 0.491 (0.24) | 0.424 | 3,000 | 42 KB |

**Crowding in the uncapped store, phase by phase** (rows: probe fwd / probe rev / reverse recall):

| phase | rows | seed 2 | seed 3 |
|---|---|---|---|
| 1 | 497 | 0.86 / 0.87 / 1.00 | 0.85 / 1.00 / 1.00 |
| 7 | 3,478 | 0.81 / 0.46 / 0.99 | 0.88 / 0.87 / 0.99 |
| 13 | 6,461 | 0.79 / 0.31 / 0.81 | 0.85 / 0.48 / 0.81 |
| 22 | 10,934 | 0.71 / 0.23 / 0.48 | 0.80 / 0.35 / 0.46 |

**Against the predictions:**
- **R (replication): half met.** Seed 3 passes (0.963). **Seed 2 fails** (0.475 < 0.6): it stayed on the plateau to the
  end. Across the four B2 readers so far (H1b s0/s1, H2 s2/s3), three escaped the plateau and one did not. Forward ≥ 0.93
  and no true duplicates hold for both.
- **S1 (crowding): holds in both seeds.** The probe's reverse accuracy at the end is 0.40× (seed 2) and 0.31× (seed 3)
  its phase-6 value. Reverse recall falls to 0.43 / 0.41. Forward decays too, but less (0.86 → 0.70 and 0.85 → 0.77).
  **The store has no weight interference, but it does have interference by crowding.**
- **S2 (eviction by value): holds in both seeds.** Value beats fifo by 0.136 / 0.177 and random by 0.136 / 0.159. It
  comes within **0.004 / 0.027** of the oracle, which knows the true popularity.
- **S3 (forgetting by value beats keeping everything): holds in both seeds.** Value 0.640 vs 0.584 and 0.824 vs 0.732.
  **The capped store is a quarter of the size (42 against 167 KB) and better.** The gain is on reverse questions (0.549
  vs 0.438, 0.877 vs 0.664): a less crowded store reads better. Forward is about equal.
- **S4 (writes): holds.** 11,924–11,929 of 11,930 facts were written on first presentation.

**Caveats:**
1. **Zipf(1) over 11,930 facts puts about 10% of all queries on the single most popular fact.** Online accuracy is
   therefore dominated by a few facts. This explains its phase-to-phase swings (for example, uncapped forward drops to
   0.60 at phase 4 while the uniform probe stays at 0.85). Every policy faces the same stream, so the *differences*
   between policies are paired and meaningful, but their *levels* depend on which facts happen to be popular.
   - **Future streams:** several stream seeds, and a popularity-weighted per-fact score.
2. **The probe can be answered without its own episode.** In the union world, a question's correct set includes answers
   from other facts that share its key. That is why fifo, with recall 0, still scores 0.19–0.38 on the probe.

**What H2 establishes:**
1. **Crowding is the store's form of interference.** It needs a fix before the store can grow far: a relation-aware
   query, a two-stage read, a larger R, or keeping each entity's live episodes few.
2. **ZipLearn's "forgetting chosen by value" transfers.** A usefulness rate with a prior, computed from the reader's
   own answers with no labels, is almost as good as an oracle.
3. **Forgetting by value is better than remembering everything** while crowding exists, and it is 4× smaller.
   Compression and accuracy point the same way.
4. **The reader's plateau escape is unreliable** (3 of 4 seeds). That is now the main fragility of the read path.
