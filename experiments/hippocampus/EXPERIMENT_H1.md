# Experiment H1 — store plus read on synthetic facts

**Pre-registered 2026-10-18, before any result run.** The user approved H1 ("Proceed"), on condition that the
hippocampus has a solution for compressing knowledge efficiently. See `PLAN.md` §2b. H1 measures two of its mechanisms:
the write gate and bytes per fact.

## Smoke tests before writing this (not saved; disclosed so nothing is hidden)

**Arm B:**
- **At 300 steps:** forward hits@1 0.91 (mass 0.15); reverse (one-way clean) 0.21.
- **At 1,000 steps:** forward 0.98 (mass 0.91); reverse 0.43.
- **Gate:** on a second presentation it re-wrote 2,724 of 2,999 facts at 300 steps, and 641 at 1,000 steps.
- **A design flaw these exposed:** duplicate rows crowd an entity's correct episode out of the 12 read slots, lowering
  reverse recall. The main evaluation was therefore moved to the store right after the *first* presentation, and
  duplicates are counted separately, by relation kind.

**Arm A:** e = 8 after 500 phase-1 steps gave forward 0.89 and reverse 0.002.

**Timing:** about 25 ms per B step, which set the budgets below (B/C 3,500 steps; A phase 1 3,000 steps) to stay under
2 minutes.

**Bug fixed after the first smoke run:** recall and the oracle had used fact indices as row ids; the gated store writes
in random order. They now look rows up by episode tokens.

## Question

A reader can be trained once, on other worlds, to use a store. Then a new world's facts are written once, forward only,
through a surprise gate. Does the store answer **reverse** questions about those facts, which the weights alone cannot
learn even with many exposures?

## Data

- **Worlds:** `../intelligent_constraints/data.py`, `make_facts(seed=w)`. A world is 1,000 entities, 6 relations with
  converse tokens, and about 3,000 random facts. The entities are shared across worlds, and the facts are re-drawn.
- **Reader training:** worlds 1–20. Each world's facts are written whole (ungated) into its own store. Training questions
  are every fact in both directions: forward [a, r] → b and reverse [b, r°] → a. That gives about 120k examples.
  - Because the facts conflict across worlds, the weights cannot hold them. The model has to learn to *read*.
- **Test world:** world 0, the same world as Experiment 1's.
- **Evaluation sets, identical for every arm** (world 0's split into paired and one-way groups):
  - `fwd_oneway`: forward questions about one-way facts;
  - **`rev_oneway_clean`, PRIMARY:** reverse questions about one-way facts, as in Experiment 1;
  - `rev_paired`.
- **Chance:** hits@1 0.0012.

## The store (`store.py`) and the read (`h1.py`)

- **Index (arm B):**
  - every token id gets a fixed random sparse binary code (D = 2048 bits, 40 active);
  - an episode's index is the union of its tokens' codes;
  - a question's code is the union of its two tokens' codes;
  - overlap is the number of shared bits.
- **Read:** the best R = 12 episodes with overlap ≥ θ = 20 bits (random ties broken at random), inserted as
  `[MEM] ep₁₂ [SEP] … ep₁ [SEP]`, with the best episode last, right before the question.
- **Write gate:** a fact is written only if its surprise, −log₂ p(b | memory read for [a, r]), exceeds τ = 2 bits.
- **The test store:** starts empty. World 0's facts are presented once, in random order, in chunks of 256 (a chunk does
  not see its own writes). The facts are then presented **again**, to count the duplicates the gate lets through.
- **Arm C (control):** each token gets a fixed random N(0, 1) 128-d vector, an episode's key is the sum (also
  order-free), and the top 12 are taken by cosine. Same read and gate otherwise.
- **Reader:** `../intelligent_constraints/model.py` `Net`: 4 layers, d = 128, 4 heads, MLP width 512, learned tied
  embeddings, context 51.
- **Training:** AdamW, lr 1e-3, weight decay 0.01, 200-step warm-up, batch 256, 3,500 steps, bf16.
- **Logging:** world-0 curves every 500 steps, on a store holding every fact (ungated). These are for the curve only.

## Arms (`run_h1.py`)

| arm | what | runs |
|---|---|---|
| A, e ∈ {1, 8, 64} | weights only, same 4-layer model, no memory. Phase 1: world 0's paired facts in both directions, 3,000 steps. Phase 2: e epochs, each presenting every one-way fact once (forward), mixed 1:1 with paired rehearsal | 6 (2 seeds) |
| B | memory, sparse union index | 2 seeds |
| C | memory, dense summed index | 2 seeds |

Arm A gets every advantage the weights can have: the paired facts teach it the converse tokens *on world 0 itself*. B and
C never see world 0 except through one write per fact.

## Diagnostics (memory arms)

- **recall:** the fraction of questions whose correct episode is among those read.
- **oracle:** the correct episode forced into the read set, which separates the read skill from retrieval.
- **leak control:** world 1's facts in the store instead of world 0's.
- **gate:** facts written on the first presentation, and duplicates written on the second, by relation kind.
- **store size:** minimal bytes per row, and the size of the overlap cache.

## Predictions

- **Apparatus:**
  - the leak control's hits@1 is ≤ 0.02 on every set;
  - recall is ≥ 0.98 forward and ≥ 0.9 reverse for B (smoke tests: about 1.00 and 0.94–1.00).
- **P1 (weights need many exposures, and stay cursed):**
  - A's forward hits@1 is ≤ 0.3 at e = 1 and ≥ 0.95 at e = 64;
  - A's reverse one-way clean hits@1 is ≤ 0.02 at every e.
- **P2 (one write suffices in both directions):** B's forward hits@1 ≥ 0.95 and reverse one-way clean hits@1 ≥ 0.5, in
  both seeds.
  - **Strongly supported** if reverse ≥ 0.8.
  - **Refuted** if reverse ≤ 0.1.
- **P3 (pattern separation):** C's reverse is ≤ B's, averaged over seeds. No prior on the size of the gap at this
  scale.
- **P4 (read against retrieval):** B with the oracle exceeds B by at most the recall shortfall, plus 0.05.
- **P5 (compression by the gate):**
  - every new fact is written on the first presentation (≥ 99%);
  - on the second presentation, B writes ≤ 10% duplicates, mostly in the one-to-many relations, where even a perfect
    reader spreads probability over 1–3 answers;
  - a stored fact costs ≤ 16 bytes as a minimal row (tokens plus metadata); the overlap cache is reported separately.

## Cost

10 runs, each ≤ 2 minutes (about 100 s for B/C, about 60–90 s for A). About 15 minutes in total.

---

## Results

*(to be filled in after the runs)*
