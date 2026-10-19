# Experiment H6b — a retrieval-confidence gate on real text

**Pre-registered 2026-10-18, before any H6b result run** (a distribution analysis on the data, disclosed). The user:
"Go ahead."

## Why

In H6 the store gave one-shot learning of a re-read passage (second reading 0.29–0.53 bits/char cheaper) and a gain
under shift (Middle High German −0.43 bits/char). But it **cost +0.06 bits/char on non-repeating Latin**, because a
retrieved passage was shown for every window, relevant or not.

The fix is the H3 combined switch carried to text: **show the memory only on a confident retrieval**; otherwise show
padding, as the no-memory model sees.

## Changes (`h6.py` `select_pair`, arms in `PAIR`)

- **Pair scoring.** A candidate is scored by the words of a *pair* of chunks (j, j+1), 64 characters like the cue, as
  the **share of the cue's idf mass they contain (0 to 1)**. Words split across the two chunks are not double-counted.
  The passage delivered is the same as before: j to j+3 (the match and its continuation).
  - **Why normalise:** raw idf scores do not transfer between texts. Every Middle High German word is unseen in the
    Latin training text and gets the maximum idf, so any shared word would pass a raw threshold.
  - **Why pairs:** a single 32-character chunk can contain at most about half of a 64-character cue, so a share of 1.0
    (a full match) is only reachable with pairs.
- **The gate:** if the best share is below τ, no memory is shown.

| arm | scoring | gate τ |
|---|---|---|
| **pair** | pair share | none (isolates the scoring change) |
| **pair_g3** | pair share | 0.3 |
| **pair_g5** | pair share | 0.5 |

Everything else is identical to H6: the PoPE reader, 3,000 steps, batch 48, 25% re-read training windows (the gate is
applied in training too, so the reader learns that memory shown means memory relevant), streams, and the second
reading. H6's runs (none, recent, retrieve) are reused, not repeated. Seeds 0, 1.

## The data, analysed before registering (deterministic, no training)

Best-share distribution (quantiles 50 / 75 / 90 / 95 / 99%), and the fraction of windows at or above each gate:

| windows | quantiles | ≥ 0.3 | ≥ 0.5 |
|---|---|---|---|
| ordinary training windows | 0.15 / 0.19 / 0.25 / 0.30 / 0.46 | about 5% | 0.8% |
| re-read training windows | 0.78 / 0.89 / 1.0 / 1.0 / 1.0 | about 100% | 98.5% |
| held-out Latin stream | 0.15 / 0.20 / 0.23 / 0.28 / 0.43 | about 4% | 0.7% |
| Middle High German stream | 0.33 / 0.38 / 0.43 / 0.46 / 0.52 | about 60% | 1.8% |
| second reading (2nd pass) | 1.0 throughout | 100% | 99.7% |

On the second reading the retrieved passage covers the first pass's copy of the window 99.7% of the time.

**The expected trade-off:**
- τ = 0.5 keeps only near-copies. It should remove the Latin cost, but also most of the Middle High German gain, which
  came from partial matches.
- τ = 0.3 shows memory on most Middle High German windows and about 4% of Latin ones.

## Predictions (means over the two seeds; "none" is H6's no-memory arm: heldout 2.132, mhg 5.418)

- **G1 (the cost removed):** on held-out Latin, pair_g3 and pair_g5 are each ≤ none + 0.01 bits/char. H6's ungated
  retrieve was +0.056.
- **G2 (one-reading learning kept):** pair_g3's and pair_g5's second-reading ratios are ≤ the ungated pair arm's
  + 0.05.
- **G3 (the shift gain):** on Middle High German, pair_g3 keeps ≥ 50% of the ungated pair arm's gain over none.
  pair_g5 keeps less than pair_g3.
- **Reported:** the ungated pair arm against H6's retrieve (the scoring change alone), caesar, and the fraction of
  windows shown memory per stream.

## Cost

6 runs of about 95 s each.

---

## Results

*(to be filled in after the runs)*
