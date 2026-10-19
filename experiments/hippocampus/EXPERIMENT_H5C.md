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

## Results

*(to be filled in after the runs)*
