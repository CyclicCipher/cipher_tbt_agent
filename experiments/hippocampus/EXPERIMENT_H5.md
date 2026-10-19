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

## Results

*(to be filled in after the runs)*
