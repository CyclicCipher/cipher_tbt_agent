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

## Results (2026-10-18; `runs/h6/pair*_s*.json`, log `runs/h6b_log.txt`)

All runs took 93–98 s.

| arm | seed | **heldout** | caesar | **mhg** | **second reading: first → second (ratio)** | memory shown: heldout / caesar / mhg | final training loss |
|---|---|---|---|---|---|---|---|
| pair | 0 | 2.191 | 2.116 | 4.931 | 2.199 → 1.481 (0.673) | 99.7% / 99.7% / 99.7% | 1.897 |
| pair | 1 | 2.215 | 2.150 | 5.074 | 2.234 → 1.159 (0.519) | 99.7% / 99.7% / 99.7% | 1.941 |
| pair_g3 | 0 | 2.174 | 2.093 | 5.154 | 2.181 → **0.121 (0.056)** | 2.8% / 3.5% / 63.7% | 1.632 |
| pair_g3 | 1 | 2.202 | 2.114 | 5.003 | 2.215 → **0.163 (0.074)** | 2.8% / 3.5% / 63.7% | 1.789 |
| pair_g5 | 0 | 2.168 | 2.076 | 5.136 | 2.176 → **0.148 (0.068)** | 0.6% / 0.1% / 1.8% | 1.618 |
| pair_g5 | 1 | 2.194 | 2.112 | 5.132 | 2.211 → **0.105 (0.048)** | 0.6% / 0.1% / 1.8% | 1.805 |

**Means over seeds, with H6's arms:**

| arm | heldout | caesar | mhg | second-reading ratio |
|---|---|---|---|---|
| none | 2.132 | 2.050 | 5.418 | 1.000 |
| recent | 2.125 | 2.058 | 5.173 | 0.995 |
| retrieve (H6) | 2.188 | 2.114 | 4.992 | 0.814 |
| pair | 2.203 | 2.133 | 5.003 | 0.596 |
| **pair_g3** | 2.188 | 2.104 | 5.078 | **0.065** |
| **pair_g5** | 2.181 | 2.094 | 5.134 | **0.058** |

**Against the predictions:**
- **G1 (the gate removes the Latin cost, ≤ none + 0.01): REFUTED.** pair_g3 is +0.056 and pair_g5 +0.049, though
  pair_g5 shows memory on only 0.6% of held-out windows.
- **G2 (one-reading learning kept): holds, far beyond the prediction.** The gated ratios are **0.065 and 0.058**,
  against the ungated pair arm's 0.596. A second reading costs 0.11–0.16 bits/char instead of about 2.2.
- **G3 (the shift gain): holds.** pair_g3 keeps 82% of the ungated pair arm's Middle High German gain over none (0.340
  of 0.415); pair_g5 keeps less (0.284, 68%).
- **Reported:** pair scoring without a gate is similar to H6's retrieve on prose (+0.015 heldout), but much better on the
  second reading (0.596 vs 0.814), because the pair match aligns the continuation with the window.

**What H6b establishes, and what it corrects in H6:**
1. **With a confidence gate, the store learns a passage in one reading almost perfectly:** about 16× cheaper on the
   second pass. The reader learns that memory shown means memory to copy.
2. **The cost on prose is in the *weights*, not in what memory shows.**
   - pair_g5's input equals the no-memory model's on 99.4% of held-out windows, yet it is 0.05 bits/char worse.
   - The difference comes from training: 25% of training windows are re-reads, where copying is trivial, so about a
     quarter of the training signal does not teach ordinary Latin.
   - **H6's reading** ("a passage is shown even when irrelevant, so the reader sometimes copies wrongly") **was wrong**,
     or at most a small part; corrected in `EXPERIMENT_H6.md`.
3. **The same weights effect explains most of the Middle High German gain.**
   - pair_g5 shows memory on only 1.8% of Middle High German windows, yet gains 0.28 bits/char.
   - Copy-rich training appears to make the model better at copying within its own 128-character context (induction),
     which the repetitive verse rewards. The longer-context control (recent, −0.25) agrees that short-range repetition
     is what Middle High German offers.
   - The retrieved memory itself adds 0.06 (g3 against g5) to 0.13 (pair against g5) on top.
4. **Next (H6c, a hypothesis to test):** make the copy training **additive**: re-read windows *in addition to* a full
   batch of ordinary text. If the cost is lost training signal, it should disappear, and the store's benefits remain.
