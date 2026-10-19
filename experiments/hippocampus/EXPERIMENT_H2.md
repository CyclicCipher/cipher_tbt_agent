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

## Results

*(to be filled in after the runs)*
