# Experiment H5c — conjunctive (dentate-gyrus-like) units for the neural store

**Pre-registered 2026-10-18, before any H5c result run** (one mechanics check at 30 episodes, disclosed). It continues
the user's "keep doing steps without pause".

## Why

H5b showed the limit of a pairwise Hebbian CA3. With slot binding, the right answers are among the best 12 (0.99), but
the top answer falls to 0.24 at 11,930 episodes. A clipped Hebbian matrix stores only pairs of bits, so an impostor
paired with the entity by one episode and with the relation by another gets full support. **A conjunction needs a unit
of higher order.**

## Design (`h5c.py`)

- **The conjunctive layer.** Every pair of active bits from two different slots of an episode, (bit i of slot x, bit j
  of slot y), activates one **pair unit**, chosen by a fixed hash into **N = 65,536 units**. This is a random, fixed
  expansion, like the dentate gyrus's sparse conjunctive cells.
- **Write (one-shot).** For each slot pair, (0,1)→2, (1,2)→0, (0,2)→1, a clipped Hebbian map from pair units to the
  third slot's bits: W_xy[unit] |= third-slot code.
- **Recall.** The cue's two slots come from the learned slot map (H5b). Its k² pair units each vote for third-slot
  bits; the output is the bits that *all* of them vote for (the Willshaw threshold).
- **Decode:** entities by overlap / k; top-1 and @12.
- **Memory:** 3 × 65,536 × 2,048 bits = **50 MB**, against 512 KB for the pairwise network and 167 KB for the table at
  11,930 episodes.
- **Same as H5/H5b:** codes (D = 2,048), episodes (worlds 100–103, the same order), sizes, and probes.
- **k ∈ {10, 20}.** H5/H5b showed that k = 40 saturates.

## Mechanics check (disclosed)

k = 10, 30 episodes: top-1 and @12 are 1.00 in both directions, layer density 0.0002, 4.9 s.

## Predictions

- **K1 (conjunctions stored):** for k = 10, top-1 is ≥ 0.9 at every size up to 11,930, in both directions. H5b: 0.24
  at 11,930.
- **K2 (k = 20):** top-1 ≥ 0.9 up to 4,000. At 11,930 it may degrade from crosstalk (each episode writes 400 pair units
  per slot pair; the layer density approaches about 0.5), so predicted ≥ 0.7.
- **K3 (direction-free):** forward and reverse top-1 within 0.05 at every point.
- **Reported:** layer density; memory per episode against the table.

## Cost

2 runs, seconds to a minute each.

---

## Results (2026-10-18; `runs/h5/conjunctive_k*.json`, log `runs/h5c_log.txt`)

| k | 250 | 1,000 | 4,000 | 8,000 | **11,930** | layer density at 11,930 |
|---|---|---|---|---|---|---|
| 10 | top-1 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 / 1.000** | 0.064 |
| 20 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 / 1.000** | 0.327 |

Forward / reverse; @12 is 1.000 everywhere.

**Checking the ceiling** (a near-perfect score is first a bug report):
- The correct-answer sets come only from facts stored so far, and the cue's own entity is excluded from decoding.
- A spurious output bit needs the votes of *all* k² cue units. At layer density ≤ 0.33, with 100–400 votes, that
  probability is negligible.
- What remains is exactly the conjunction: only an episode containing both cue tokens, in the right slots, wrote those
  units.
- The hash overflows int64 arithmetic for large bit indices and wraps around. That gives a different but fixed hash,
  so it changes nothing.

**Against the predictions:**
- **K1 (k = 10, top-1 ≥ 0.9 at every size): holds,** at 1.000 throughout. H5b: 0.24 at 11,930.
- **K2 (k = 20, ≥ 0.9 to 4,000, ≥ 0.7 at 11,930): holds,** at 1.000 throughout. The crosstalk I expected did not
  appear: density reached only 0.33.
- **K3 (direction-free): holds,** with identical forward and reverse.

**What H5c establishes:**
1. **Conjunctive units are what relational episodes need,** and with them a neural store matches the table: top-1
   1.000 in both directions at 11,930 episodes, written in one shot, with no gradient.
2. **The price is memory.** The random layer costs **50 MB** (3 × 65,536 × 2,048 bits), about 34 kbit per episode at
   11,930, against the table's 112 bits per episode (14 bytes, 167 KB in all) and the pairwise network's 512 KB.
   - A *random* conjunctive expansion must be large to keep collisions rare.
   - An *allocated* one (a unit per episode: the table's rows, or a dense associative memory's hidden units) is about
     300× cheaper.
   - The brain's dentate gyrus has about 5–10× more cells than its input layer, which looks like this trade made with
     biological parts.
3. **Neural substrate (H5–H5c), summed up:**
   - a **pairwise** CA3 has the capacity but cannot hold conjunctions;
   - adding **slot binding** fixes roles;
   - adding **conjunctive units** fixes conjunctions.
   - The table is the cheapest implementation of the result, and the conjunctive layer is the neural one.
