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

## Results

*(to be filled in after the runs)*
