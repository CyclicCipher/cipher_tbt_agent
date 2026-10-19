# Experiment H1c — a reliable reader: train it with the retrieval it will use

**Pre-registered 2026-10-18, before any H1c result run.** The user's task 2: make the reader reliable.

## Why

1. **B2 readers escaped the reverse plateau in only 3 of 4 seeds** (H1b s0 0.93, s1 0.83; H2 s2 0.475, s3 0.963).
2. **H2b showed the reader co-adapts to its retrieval.** Readers trained under the old index left entity matching to
   retrieval, and collapsed to about 1/12 when the learned query expansion changed what was read.

So the reader must be trained *with* the expansion. Under the expansion:
- a reverse question's answer episode overlaps its query by 80 bits, and everything else by ≤ 40;
- the answer episode is therefore ranked best, and placed last, next to the question.

The reader may then learn simply to copy from that position: the matching is done by retrieval. Whether that division of
labour is what makes reading reliable is tested by shuffling the read order.

## Arms (`run_h1c.py`)

Both arms use H1b's B2 recipe (compile, fused AdamW, 4,500 steps, cosine decay, familiarity gate), plus the learned
expansion (`expand.py`: converse token → its relation) in **every** store: training worlds, test, and leak control.

| arm | read order | seeds |
|---|---|---|
| **E1** | ranked (best last, as before) | 4, 5, 6, 7 |
| **E2** | **shuffled** per question, so rank position carries nothing and the reader must match by content | 4, 5 |

**New diagnostic: reverse error classes.** Each reverse question's answer is classed as:
- correct;
- right relation, wrong entity (the subject of a read episode with relation r, but not object b);
- right entity, wrong relation (the subject of a read episode with object b, but another relation);
- another entity in context;
- not in context.

**Streams.** The E1 readers of **seeds 4 and 5** (fixed in advance, not chosen by result) re-run the H2 stream with
expansion: policies uncapped and value × stream seeds 0, 1, 2. Readers are saved to `runs/h1c/reader_*.pt` for H3.

## Smoke test (a throwaway seed, 300 steps, not saved)

- **E1:** forward 0.89, reverse **0.88** (B2 was at about 0.2 at 300 steps). Every error was "right relation, wrong
  entity". 19 s.
- **E2:** forward 0.02, reverse 0.02; 78% of answers were not in context at all. 12 s.

## Predictions

- **P1 (reliability):** E1 reaches reverse ≥ 0.8 on world 0 in **4 of 4** seeds, with a mean ≥ 0.9, and forward ≥ 0.95.
- **P2 (the reliability comes from retrieval doing the matching):** E2's mean reverse < E1's mean − 0.1.
- **P3:** E1's reverse errors are mostly "right relation, wrong entity".
- **P4 (crowding fixed end to end):** in every uncapped stream with an E1 reader, the probe's reverse accuracy at
  phase 24 is ≥ 0.8 × its phase-6 value. Without expansion (H2), it was 0.31–0.40×.
- **P5 (keeping everything wins once crowding is gone):** uncapped ≥ value, averaged over stream seeds, for both
  readers, both popularity-weighted (online) and per fact.

## Cost

6 readers × about 90 s, plus 12 streams × about 8 s. About 11 minutes.

---

## Results (2026-10-18; `runs/h1c/`, log `runs/h1c_log.txt`)

**One crashed attempt first.** The driver's first attempt trained E1 seed 4 fully, then crashed when saving the weights,
because `runs/h1c/` did not exist. No result was recorded, so it is not a re-run. The driver now creates the folder.
Reader runs then took 91–95 s, and streams 8–15 s.

**Readers on world 0:**

| arm | seed | fwd | **REV one-way clean** | reverse errors | reverse curve (every 500 steps) |
|---|---|---|---|---|---|
| E1 | 4 | 0.958 | **0.973** | 2.7% "right relation, wrong entity", nothing else | 0.86 0.95 0.96 … 0.98 |
| E1 | 5 | 0.956 | **0.968** | 3.2% right relation, wrong entity | 0.88 0.96 … 0.97 |
| E1 | 6 | 0.950 | **0.973** | 2.7% right relation, wrong entity | 0.89 0.97 … 0.98 |
| E1 | 7 | 0.950 | **0.979** | 2.1% right relation, wrong entity | 0.88 0.96 … 0.98 |
| E2 (shuffled) | 4 | 0.064 | 0.199 | 65% right relation, wrong entity; 15% not in context | 0.05 0.16 … 0.19 |
| E2 (shuffled) | 5 | 0.054 | 0.209 | 62% right relation, wrong entity; 16% not in context | 0.07 0.18 … 0.21 |

Recall is 1.00 for every arm.

**Streams with E1 readers and expansion** (online = popularity-weighted, per-fact = each fact weighted equally):

| policy | reader | online mean, stream seeds 0 / 1 / 2 (avg) | per-fact mean (avg) | probe rev, phase 6 → 24 (recall) |
|---|---|---|---|---|
| uncapped | 4 | 0.802 / 0.814 / 0.785 (**0.800**) | **0.785** | 0.85→0.73, 0.84→0.75, 0.85→0.72 (1.00) |
| value | 4 | 0.801 / 0.809 / 0.803 (**0.804**) | 0.638 | 0.85→0.47, 0.84→0.51, 0.85→0.47 (0.26–0.31) |
| uncapped | 5 | 0.814 / 0.814 / 0.781 (**0.803**) | **0.780** | 0.88→0.75, 0.84→0.72, 0.86→0.71 (1.00) |
| value | 5 | 0.817 / 0.816 / 0.796 (**0.810**) | 0.637 | 0.88→0.46, 0.84→0.52, 0.86→0.47 (0.24–0.32) |

**Against the predictions:**
- **P1 (reliability): holds.** 4 of 4 E1 seeds reach ≥ 0.968 reverse (mean 0.973), with forward 0.950–0.958.
  **No plateau at all:** 0.86–0.89 already at step 500. The plateau of H1–H2 came from retrieval that left the reader the
  hard part of the matching.
- **P2 (the reliability comes from retrieval doing the matching): holds, strongly.** Shuffled read order gives
  0.20–0.21 reverse and even breaks *forward* (0.05–0.06). Without a rank cue, this reader does not learn to match by
  content in 4,500 steps. Its errors are "right relation, wrong entity": it matches the relation and fails on the
  entity.
- **P3: holds.** All of E1's reverse errors are "right relation, wrong entity".
- **P4 (crowding fixed end to end): holds in all 6 streams.** The probe keeps 0.83–0.89× of its phase-6 reverse
  accuracy, against 0.31–0.40× in H2. Reverse recall is 1.00 throughout. The remaining decay (about 0.85 → 0.72) is the
  reader, not retrieval.
- **P5 (keeping everything wins once crowding is gone): split.**
  - **Per fact: holds clearly,** uncapped 0.78 vs value 0.64.
  - **Popularity-weighted: refuted narrowly,** value 0.804 / 0.810 vs uncapped 0.800 / 0.803, a tie within 0.01.
  - **So with crowding fixed, a value-capped store at a quarter of the size loses nothing on what is asked often, and
    loses the long tail.**

**What H1c establishes:**
1. **A reliable read path is retrieval that ranks by full match, plus a reader that copies from the top.** Four of four
   seeds, with no plateau, from the learned expansion alone.
2. **The plateau was a division-of-labour problem:** the reader had to learn matching the retrieval did not do.
3. **Content matching without rank cues is hard for this reader** (E2). Harder reading tasks (several facts, chains)
   will need it, so it is noted as a future risk.
4. **Crowding is solved at both retrieval and accuracy level.** The H2 verdict "forgetting by value beats keeping
   everything" was a crowding artefact. Without crowding, value eviction is a pure compression trade: a quarter of the
   size, the same popularity-weighted accuracy, and the long tail lost. **H3 (consolidation) is the way to keep the long
   tail without the rows.**

**E1 readers (seeds 4–7) are the read path from here.** Their weights are in `runs/h1c/` (git-ignored).
