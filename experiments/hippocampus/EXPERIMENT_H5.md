# Experiment H5 — a neural substrate: a Willshaw CA3 network against the table

**Pre-registered 2026-10-18, before any H5 result run** (one mechanics check at 30 episodes, disclosed below). It is
`PLAN.md`'s H5, run as part of the user's "keep doing steps without pause".

## Question

The table is the *ideal* hippocampus: perfect separation, unlimited capacity, recall 1.00 at any size with the learned
expansion (H2b/H1c). How close is a real autoassociative network, the classic model of CA3 (Marr 1971; Willshaw 1969),
built on the same sparse codes? And how does its capacity depend on the codes' sparsity?

## Design (`h5.py`)

- **Codes:** every token has a fixed random binary code, D = 2,048 bits with **k active**, for **k ∈ {10, 20, 40}**.
  40 is the table's current code. An episode (a, r, b) is the union of its three codes.
- **Store:** clipped Hebbian learning, **W[i, j] = 1 if bits i and j were ever co-active in a stored episode**.
  - This is one-shot and symmetric, storing the joint pattern.
  - It costs a fixed D² bits = 512 KB, whatever the number of episodes.
- **Recall:** the cue is the stored tokens the question points at:
  - forward: a and r;
  - reverse: b and r, where the learned expansion maps r° → r, since r° never occurs in a stored episode.

  Dendritic sums are s = W · cue. The output is every bit whose sum reaches the cue's size (the Willshaw threshold),
  plus the cue.
- **Decode:** entities ranked by overlap of their code with the recalled pattern / k, excluding the cue's entity. Two
  scores:
  - **top-1:** the decoded answer is correct;
  - **@12:** the answer is among the 12 best, the analogue of the table's 12-episode read set.

  The union code carries no roles, so a cue {x, r} also completes episodes where x has the *other* role. Top-1 pays for
  that; @12 mostly does not.
- **Episodes:** the union of worlds 100–103 (11,930 facts) in a fixed random order. They are measured after
  250, 500, 1,000, 2,000, 4,000, 8,000 and 11,930 are stored, on 1,000 probe facts drawn from those stored, forward and
  reverse.

## Mechanics check (disclosed)

k = 40, 30 episodes stored: top-1 and @12 are 1.00 in both directions, matrix density 0.082, 4.6 s.

## Predictions (from Willshaw capacity theory)

A spurious bit fires when all cue bits connect to it. With matrix density p₁ ≈ 1 − exp(−M m² / D²) (M stored, m ≈ 3k
active per episode) and a 2k-bit cue, there are about D · p₁^(2k) spurious bits. Decoding fails as these approach
saturation.

- **W1 (k = 40):** @12 stays ≥ 0.95 up to 1,000 episodes and falls below 0.5 by 4,000 (saturation at about
  1,000–2,000).
- **W2 (k = 20):** @12 ≥ 0.95 up to 2,000, and below 0.5 by 11,930.
- **W3 (k = 10):** @12 ≥ 0.9 at all sizes up to 11,930.
- **W4 (direction-free):** forward and reverse recall are within 0.05 of each other at every point, since the network
  stores the joint pattern.
- **W5 (roles):** top-1 is below @12 wherever entities occur in both roles of a relation. Reported, not bounded.
- **Cost comparison** (reported): the network is 512 KB at any size; the table is 14 bytes per episode (167 KB at
  11,930).

## Cost

3 runs, seconds each.

---

## Results (2026-10-18; `runs/h5/willshaw_k*.json`, log `runs/h5_log.txt`)

Each run took seconds.

| k | episodes stored | matrix density | top-1 fwd / rev | **@12 fwd / rev** |
|---|---|---|---|---|
| 40 | 250 | 0.40 | 0.968 / 0.976 | 1.000 / 1.000 |
| 40 | 1,000 | 0.72 | 0.522 / 0.510 | 1.000 / 1.000 |
| 40 | 2,000 | 0.86 | 0.305 / 0.304 | 1.000 / 1.000 |
| 40 | 4,000 | 0.96 | 0.182 / 0.216 | 0.994 / 0.994 |
| 40 | 8,000 | 0.994 | 0.099 / 0.083 | 0.630 / 0.653 |
| 40 | 11,930 | 0.999 | 0.007 / 0.008 | **0.078 / 0.080** |
| 20 | 250 | 0.14 | 0.972 / 0.972 | 1.000 / 1.000 |
| 20 | 1,000 | 0.31 | 0.661 / 0.645 | 1.000 / 1.000 |
| 20 | 4,000 | 0.59 | 0.179 / 0.232 | 0.994 / 0.994 |
| 20 | 8,000 | 0.78 | 0.151 / 0.136 | 0.910 / 0.929 |
| 20 | 11,930 | 0.87 | 0.109 / 0.091 | **0.836 / 0.830** |
| 10 | 250 | 0.04 | 0.964 / 0.972 | 1.000 / 1.000 |
| 10 | 1,000 | 0.10 | 0.743 / 0.704 | 1.000 / 1.000 |
| 10 | 4,000 | 0.23 | 0.209 / 0.224 | 0.995 / 0.995 |
| 10 | 8,000 | 0.35 | 0.157 / 0.139 | 0.911 / 0.929 |
| 10 | 11,930 | 0.44 | 0.109 / 0.091 | **0.836 / 0.830** |

(The sizes 500 and 2,000 for k = 10/20 follow the same pattern; see the log.)

**Against the predictions:**
- **W1 (k = 40):** @12 ≥ 0.95 up to 1,000 ✓. "Below 0.5 by 4,000" **refuted**: 0.994 at 4,000, 0.63 at 8,000, and
  below 0.5 only at 11,930 (0.08). The network saturates later than my estimate.
- **W2 (k = 20):** ≥ 0.95 up to 2,000 ✓. "Below 0.5 by 11,930" **refuted**: 0.83.
- **W3 (k = 10, ≥ 0.9 at all sizes): REFUTED narrowly.** 0.836 at 11,930.
- **W4 (direction-free):** @12 forward and reverse are within 0.024 everywhere ✓. Top-1 is within 0.05 except one
  point (k = 20, 500 episodes: 0.926 vs 0.874, a gap of 0.052).
- **W5:** top-1 falls far below @12 as the store fills: 0.97 → 0.11 at 11,930, for every k.

**What H5 establishes:**
1. **Capacity is not the limit with sparse codes.** At k ≤ 20 the fixed 512 KB network holds all 11,930 episodes
   without saturating. **k = 10 and k = 20 give identical results** at 8,000 and 11,930, which shows that their shared
   ceiling (0.83 @12) is not spurious bits. Only k = 40, the table's current code, saturates (density 0.999).
2. **What the network loses against the table is episode structure, not capacity.**
   - An episode is stored as an unordered union of its tokens' codes. A cue {x, r} completes *every* episode containing
     x and r, with x in either role, and returns their union blended together.
   - Decoding then sees many fully overlapping entities, which tie: top-1 falls to 0.11, and @12 to 0.83 once more than
     about 12 entities share the cue.
   - The table keeps episodes separate, and the reader resolves roles by reading their tokens in order.
3. **The cost comparison favours the table at this scale:** 14 bytes per episode (167 KB at 11,930) against a fixed
   512 KB.

**Next (H5b):** role-bound codes. Each token's code is permuted by a fixed random permutation **per slot within the
episode** (subject, relation, object): generic sequence-position binding, like a positional encoding, not a schema.
Which slot a question's tokens fill, and which slot it asks for, is learned from the same co-occurrence counts as the
expansion.
