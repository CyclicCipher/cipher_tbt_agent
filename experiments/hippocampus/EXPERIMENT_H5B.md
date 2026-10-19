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

## Results

*(to be filled in after the runs)*
