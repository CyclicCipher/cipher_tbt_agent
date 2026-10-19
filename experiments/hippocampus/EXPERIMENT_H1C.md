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

## Results

*(to be filled in after the runs)*
