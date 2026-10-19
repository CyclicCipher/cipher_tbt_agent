# Experiment H2b — learned query expansion against crowding

**Pre-registered 2026-10-18, before any H2b result run.** The user: "Do 1 [fix crowding], then 2 [make the reader
reliable], then H3."

## Why

In H2, a reverse question (b, r°) shared only b's code with its answer episode (a, r, b). It therefore tied, at 40 bits,
with all of b's other episodes (about 24 in the full stream), and only 12 could be read. Reverse recall fell from 1.00
to 0.42, and old facts' reverse accuracy fell 2.5–3×.

## The fix: learned query expansion (`expand.py`)

**How it is learned.**
- Over the reader's training worlds (1–20, labelled, as the reader's own training is), take every question.
- Count the tokens of its answer episode that are *not* in the question.
- A question token whose count is dominated by one partner (share ≥ 0.5, over ≥ 20 questions) gets that partner's code
  added to its query (union).
- Episodes' own index codes do not change.

**What it learned** (computed before this pre-registration; deterministic counting, no training):
- every converse token learned its relation, **1006→1000, 1007→1001, …, 1011→1005**;
- **no other token** qualified: the best entity share is 0.19.

Nothing pairs relations by hand. The pairing is co-occurrence statistics in the index layer, a Hebbian association
between query tokens and the tokens of episodes that answered them.

**The effect:** a reverse question now overlaps its answer episode by b + r = 80 bits, and b's other episodes by only
40.

## A second question it answers: what is the plateau?

H2's seed-2 reader stalled at 0.475 reverse on world 0. If the plateau is *relation-blind reading* (find an episode with
b as the object, ignore the relation), then retrieval that matches the relation should lift that reader without
retraining.

## Smoke test (a throwaway 200-step reader, not saved)

The mechanics work: about 7.5 s per stream, about 2 s per world-0 evaluation.

**Reader-independent facts seen there** (stream seed 1, uncapped, with expansion, at phase 24, 11,930 rows):
- **reverse recall is 1.00** (H2 without expansion: 0.42);
- **forward recall is also 1.00.** So H2's forward decay (0.86 → 0.70) was never a retrieval failure: it is the reader,
  confused by crowded contexts.

The predictions below are therefore on *accuracy*.

## Runs (`run_h2b.py`)

- **World 0 with expansion:** the two H2 readers (seeds 2, 3), loaded from `runs/h2/reader_s*.pt`, with no retraining.
  Gated write (familiarity, then surprise), then the H1 evaluation sets.
- **Streams:** policies uncapped and value × readers 2, 3 × **stream seeds 0, 1, 2** × expansion off/on.
  - Stream seed 0 without expansion is H2's result, reused. That leaves 20 new stream runs of about 8 s.
  - Extra stream seeds answer H2's caveat (Zipf(1) puts about 10% of queries on one fact).
  - New in the logs: a **per-fact** score (every queried fact weighted equally) and **forward recall**.

## Predictions

- **W1 (the plateau is relation-blind reading):** reader 2's world-0 reverse rises from 0.475 to **≥ 0.7** with
  expansion. Reader 3 stays ≥ 0.9. If reader 2 does not rise, the plateau is something else.
- **X1 (crowding fixed):** in the uncapped stream, the probe's reverse accuracy at phase 24 is ≥ 0.8 × its phase-6
  value, for both readers and every stream seed. Without expansion, H2 found 0.31–0.40×.
- **X2 (stream accuracy):** uncapped with expansion ≥ uncapped without + 0.05, in online mean (forward and reverse),
  averaged over stream seeds, for both readers.
- **X3 (was forgetting's advantage only crowding?):** with expansion, uncapped ≥ value-capped (mean over stream seeds),
  for both readers. H2's "forgetting beats keeping everything" should reverse once crowding is gone, because the full
  store then loses nothing.
- **X4 (forward):** expansion changes forward online accuracy by ≤ 0.02 (mean over stream seeds). The uncapped forward
  decay persists; it is the reader's.
- **X5 (H2 replicated on new stream seeds):** without expansion, value ≥ uncapped on stream seeds 1 and 2 as well, for
  both readers.

## Cost

20 streams × about 8 s, plus 2 world-0 evaluations. About 3 minutes.

---

## Results

*(to be filled in after the runs)*
