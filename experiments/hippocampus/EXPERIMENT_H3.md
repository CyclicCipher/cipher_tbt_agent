# Experiment H3 — consolidation: sleep moves the store into the weights, then the store shrinks

**Pre-registered 2026-10-18, before any H3 result run.** The user's order: "Do 1, then 2, then H3."

## Question

The store learns in one shot and reads in both directions (H1–H1c). Can "sleep" then move what it holds into the
weights, so that the weights answer without memory and the rows can be evicted? And does the reversal curse come back in
the weights, unless sleep also replays the model's own **reverse inferences**? This is compression mechanism 3 of
`PLAN.md` §2b (consolidate, then evict), and the complementary-learning-systems claim.

## Design (`h3.py`, `run_h3.py`)

- **Readers:** E1, seeds **4 and 5** (H1c; world-0 reverse 0.973 / 0.968). They are fixed by the rule "the first two
  E1 seeds", which both meet the ≥ 0.8 bar. Weights come from `runs/h1c/`.
- **The store:** world 0 written once through the gate (familiarity, then surprise), with the learned expansion.
- **Sleep:** 2,000 fine-tuning steps, batch 128 consolidation + 128 rehearsal, AdamW, lr 5e-4 peak, 200-step warm-up,
  cosine to 10%, compiled.
  - **Consolidation examples** are questions with an **empty memory** (every slot padded):
    - **F (forward only):** each stored episode (a, r, b) as [a, r] → b.
    - **I (forward + inferences):** F's examples, plus for each episode the reverse question [b, r°] answered *by the
      reader itself through its normal read path* (retrieval from the store). If that answer's probability is
      ≥ 0.5, it becomes the empty-memory target for [b, r°]. No reverse label is ever used.
  - **Rehearsal:** the reader's own training task (worlds 1–20, read with the expansion), to keep reading alive.
- **Measures:** before and after sleep, world 0's one-way sets:
  - **weights only** (empty memory);
  - **memory** (the normal read);
  - **combined**, which reads normally if the store holds a full match (best overlap ≥ 60 bits: entity and relation),
    and otherwise shows an *empty* memory so the weights answer;
  - and a **held-out world 50** read with its own store (interference).
- **Eviction after sleep:** an episode counts as consolidated if the weights alone give its answer p ≥ 0.1 in **both**
  directions. Consolidated episodes are removed, and the combined system is measured again.

## Checks before this pre-registration (eval only, no training; disclosed)

1. **Smoke tests** with throwaway 100–200-step readers and 100–300-step sleeps. The mechanics work, at about 18 s per
   300 sleep steps, so about 50 s per run.
2. **A design flaw found and fixed.** The first design generated inferences by putting each episode *alone* in memory
   (11 of the 12 slots padded). On the real E1 readers that gives only **0.5–8%** correct reverse answers: they were
   always trained on full 12-episode contexts. That is reader–context co-adaptation again. Hence:
   - inferences now come from the **normal read path**;
   - the combined system never shows a half-empty context: either the normal read, or an entirely empty memory (the
     context sleep trains).
3. **The normal-read inferences on the real readers:** 97.9% correct. 94.3% / 95.0% pass p ≥ 0.5, and those are 98.4% /
   98.3% correct.

## Predictions

- **C0:** before sleep, weights-only accuracy on world 0 is ≤ 0.02 in both directions (the weights have never seen it).
- **C1 (forward consolidates):** after sleep, weights-only forward ≥ 0.8, for F and I.
- **C2 (the reversal curse returns in the weights):** F's weights-only reverse ≤ 0.05 after sleep.
- **C3 (self-generated inferences carry reverse into the weights):** I's weights-only reverse ≥ 0.6 after sleep.
- **C4 (reading survives):** after sleep, world 0 with memory, and held-out world 50 with its store, each keep reverse
  ≥ their pre-sleep value − 0.05.
- **C5 (consolidate, then evict):**
  - I consolidates ≥ 50% of the episodes in both directions; F ≤ 10% (its reverse is not consolidated).
  - For I, the combined system after eviction is within 0.05 of the combined system before eviction, forward and
    reverse.

## Cost

4 runs of about 50–70 s each.

---

## Results (2026-10-18; `runs/h3/`, log `runs/h3_log.txt`)

All runs took 42–48 s.

| arm | seed | inferences kept (correct) | **weights only: fwd / rev**, before → after sleep | memory rev, before → after | held-out rev, before → after |
|---|---|---|---|---|---|
| F | 4 | – | 0.001 / 0.002 → **0.985 / 0.000** | 0.975 → 0.978 | 0.976 → 0.979 |
| F | 5 | – | 0.001 / 0.000 → **0.987 / 0.000** | 0.972 → 0.974 | 0.981 → 0.987 |
| I | 4 | 2,841 (98.4%) | 0.001 / 0.002 → **0.613 / 0.588** | 0.975 → 0.974 | 0.976 → 0.977 |
| I | 5 | 2,834 (98.3%) | 0.001 / 0.000 → **0.609 / 0.651** | 0.972 → 0.977 | 0.981 → 0.980 |

| arm | seed | consolidated (p ≥ 0.1): fwd / rev / **both**, of 2,994 | rows after eviction | combined fwd / rev: before sleep → after sleep → **after eviction** (share of questions read from the store) |
|---|---|---|---|---|
| F | 4 | 2,969 / 2 / **2** | 2,992 | 0.950 / 0.979 → 0.951 / 0.976 → 0.952 / 0.973 (1.00) |
| F | 5 | 2,949 / 2 / **2** | 2,992 | 0.959 / 0.978 → 0.960 / 0.976 → 0.960 / 0.972 (1.00) |
| I | 4 | 1,936 / 1,826 / **1,222** | 1,772 | 0.950 / 0.979 → 0.964 / 0.975 → **0.831 / 0.838** (0.65) |
| I | 5 | 1,951 / 1,935 / **1,266** | 1,728 | 0.959 / 0.978 → 0.955 / 0.975 → **0.825 / 0.858** (0.65) |

**Against the predictions:**
- **C0: holds.** Before sleep the weights know nothing of world 0 (≤ 0.002).
- **C1 (forward consolidates): holds for F (0.985 / 0.987), fails for I (0.613 / 0.609).** I spends half its
  consolidation batch on reverse examples, so each forward fact gets about 44 exposures against F's 85, and the two kinds
  of question compete.
- **C2 (the reversal curse returns in the weights): holds, starkly.** F's weights answer forward 0.985 and reverse
  **0.000**, although the store they were consolidated from reads both ways at 0.97.
- **C3 (self-generated inferences carry reverse): holds for seed 5 (0.651), fails narrowly for seed 4 (0.588 < 0.6).**
  Reverse in the weights goes from 0.000 to 0.59–0.65 with **no reverse label ever used**: only the reader's own
  answers, 98% correct.
- **C4 (reading survives): holds in every run.** World-0 memory reads and held-out world 50 stay within ±0.01 (one
  rises by 0.006). Rehearsal fully protects the reading skill.
- **C5 (consolidate, then evict): fails for I.**
  - Only 41–42% of episodes passed "p ≥ 0.1 in both directions" (prediction ≥ 50%).
  - Evicting them dropped the combined system by **0.12–0.14**, forward and reverse.
  - The criterion is too lax: p ≥ 0.1 admits facts whose weights-only *top* answer is wrong. About 35% of questions are
    then answered by weights that are right only about 60% of the time.
  - F holds its half: it consolidated only 0.07% (≤ 10%), because its reverse never consolidates.

**What H3 establishes:**
1. **The reversal curse is a property of how weights are trained,** not of the knowledge. Consolidating a store that
   reads both ways by one-way replay gives one-way weights (forward 0.985, reverse 0.000).
2. **Self-generated inferences fix it in principle,** as complementary-learning-systems theory says. Replaying the
   reader's own reverse answers puts reverse knowledge into the weights (0.000 → 0.59–0.65) with no reverse data. This is
   the brain's "duplicate as a cache generated from a symmetric master copy".
3. **Sleep does not damage reading,** as long as reading is rehearsed (C4).
4. **Two things limit the result:**
   - **Sleep budget:** I's weights reach only about 0.6 in each direction in 2,000 steps.
   - **The eviction criterion:** p ≥ 0.1 is unsafe.

**Next (not pre-registered):**
- **A strict eviction criterion:** evict only if the weights' *top* answer equals the stored episode's answer in both
  directions. The store holds the answer, so this self-check needs no label, and makes eviction loss-free by
  construction.
- **More sleep,** or a larger consolidation share, to raise the weights' accuracy and so the evictable fraction.
- **Then sleep inside the H2 stream:** consolidation keeping the long tail that value eviction loses.
