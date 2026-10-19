# Experiment H5b — the Willshaw CA3 network with slot-bound codes

**Pre-registered 2026-10-18, before any H5b result run** (one mechanics check at 30 episodes, disclosed). It continues
the user's "keep doing steps without pause".

## Why

H5 showed that a Willshaw network on sparse codes (k ≤ 20) has the capacity for all 11,930 episodes, but loses
**episode structure**. An episode stored as an unordered union of its tokens' codes is completed by any cue sharing its
tokens, whatever their roles. Every matching episode then comes back blended, so the top decoded answer falls to 0.11 at
11,930.

## The change (`h5.py --slots`)

- **Slot binding.** Each token's code is permuted by a fixed random permutation **per slot within the episode**
  (subject, relation, object), and an episode is the union of its tokens' **slot-bound** codes.
  - This is generic sequence-position binding, the way a positional encoding binds position. It is not a domain schema.
  - Every 3-token episode uses the same three permutations.
- **Which slot a question fills, and which it asks for, is learned,** from the training worlds' counts, like the
  expansion (`learn_slots`).
  - **Learned before this pre-registration** (deterministic counting): every relation token puts its entity in slot 0
    and asks for slot 2. Every converse token puts its entity in slot 2 and asks for slot 0. That is correct for all 12.
- **Cue:** the question's entity at its learned slot, plus the relation at slot 1 (the converse maps to its relation by
  the expansion).
- **Decode:** entities' codes at the learned answer slot.

Everything else is as in H5: D = 2,048, k ∈ {10, 20, 40}, clipped Hebbian, the Willshaw threshold, the same episodes and
sizes, 1,000 probes per size and direction, top-1 and @12.

## Mechanics check (disclosed)

k = 40, 30 episodes stored: top-1 and @12 are 1.00 in both directions, density 0.083, 4.6 s.

## Predictions

- **S1 (role confusion removed):** for k = 10 and k = 20, top-1 is ≥ 0.9 at every size up to 11,930. H5: 0.11 at
  11,930.
- **S2 (saturation unchanged):** for k = 40, top-1 is ≥ 0.9 up to 2,000 and is below the k = 20 value at 11,930.
  Binding does not change pattern size, so the matrix saturates as in H5.
- **S3 (direction-free):** forward and reverse top-1 are within 0.05 at every point.
- **S4:** for k ≤ 20, @12 is ≥ 0.95 at every size.

## Cost

3 runs, seconds each.

---

## Results (2026-10-18; `runs/h5/willshaw_slots_k*.json`, log `runs/h5b_log.txt`)

| k | 250 | 500 | 1,000 | 2,000 | 4,000 | 8,000 | 11,930 |
|---|---|---|---|---|---|---|---|
| **top-1 fwd / rev** | | | | | | | |
| 10 | 0.996 / 0.996 | 0.980 / 0.980 | 0.925 / 0.906 | 0.649 / 0.649 | 0.372 / 0.364 | 0.289 / 0.270 | **0.245 / 0.227** |
| 20 | 0.996 / 0.996 | 0.958 / 0.974 | 0.829 / 0.825 | 0.540 / 0.532 | 0.384 / 0.363 | 0.288 / 0.284 | **0.243 / 0.226** |
| 40 | 0.996 / 0.996 | 0.956 / 0.950 | 0.709 / 0.721 | 0.522 / 0.534 | 0.369 / 0.361 | 0.023 / 0.030 | 0.006 / 0.004 |
| **@12 fwd / rev** | | | | | | | |
| 10 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.996 / 0.998 | **0.989 / 0.989** |
| 20 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.995 / 0.998 | **0.989 / 0.989** |
| 40 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.270 / 0.271 | 0.040 / 0.044 |

The matrix densities match H5's (k = 10 reaches 0.47 at 11,930; k = 40 reaches 0.9997).

**Against the predictions:**
- **S1 (top-1 ≥ 0.9 at every size, k ≤ 20): REFUTED.** It holds to 1,000 (k = 10) or 500 (k = 20), then falls to
  0.24 at 11,930. That is better than H5's 0.11, far from 0.9.
- **S2 (k = 40: top-1 ≥ 0.9 up to 2,000): REFUTED** (0.71 at 1,000, 0.52 at 2,000). "Below k = 20 at 11,930" holds
  (0.006 vs 0.243).
- **S3 (direction-free within 0.05): holds** everywhere (largest gap 0.019).
- **S4 (@12 ≥ 0.95 for k ≤ 20): holds,** ≥ 0.989 at every size, against H5's 0.83. **Slot binding removes the role
  confusion.**

**Why top-1 still falls: a pairwise matrix cannot store a conjunction.**
- Clipped Hebbian learning records only *pairs* of co-active bits.
- Suppose b′'s object-slot bits were paired with a's subject-slot bits by some episode (a, r₂, b′), and with r's bits by
  another episode (a′, r, b′). Then b′ receives the full Willshaw sum for the cue {a, r}, although (a, r, b′) was never
  stored.
- The number of such "pairwise-consistent" impostors grows with the store and ties with the true answers. That is why
  k = 10 and k = 20, both far from saturation, fall identically: the failure is the network's order (pairwise), not its
  capacity.
- **The table does not have this problem:** each row is a unit for one episode, a conjunction of its three tokens.

**What H5/H5b establish about a neural CA3:**
1. **Capacity is not the obstacle.** Sparse codes hold about 12k episodes in a fixed 512 KB.
2. **Roles need binding.** Per-slot permutations work, learned slot maps included.
3. **Conjunctions need units of a higher order than pairs.** These are per-episode or conjunctive hidden units: the
   table's rows, the hidden memory units of a dense associative memory (Krotov & Hopfield 2016; equivalently softmax
   attention over stored patterns, Ramsauer et al. 2020), or, in the brain, the dentate gyrus's sparse conjunctive
   cells feeding CA3. A pairwise Hebbian matrix alone is the wrong substrate for relational episodes.
4. **A neural replacement for the table should therefore be a layer of conjunctive units.** Allocated per episode, that
   is the table with soft reads; random and sparse, that is a dentate gyrus. The second is the open, non-trivial
   version.
