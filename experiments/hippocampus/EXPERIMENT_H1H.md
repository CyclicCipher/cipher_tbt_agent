# Experiment H1h — non-causal attention for forward content matching

**Pre-registered 2026-10-18, before any H1h result run** (one timing run on a throwaway seed, disclosed). It follows
H1g within the user's "Go ahead and proceed".

## Why

Across H1f and H1g, PoPE readers learned **reverse** content matching (4 of 4 seeds with enough updates) but **never
forward** (10 readers, ≤ 0.16; 18–34% of forward answers not even in context).

The hypothesis is **causal masking.**
- An episode reads subject → relation → object, so only the object's position sees the whole episode.
- **Reverse** matches the question's entity against the object's *own* token and reads the subject copied there.
- **Forward** must match against a *copied* subject, superposed on the object's own code in the same embedding space.
- With **non-causal** attention, the subject's position also sees its object, so forward becomes the mirror image of
  reverse.
- This is safe here: the reader predicts one answer after the input, and the answer never appears in the input.

## Change

`Net(causal=False)` in the shared `../intelligent_constraints/model.py` sets every attention layer to see the whole
input (the PoPE attention's own `causal` flag, and `is_causal` in the standard attention). The default is unchanged.

## Arms (`run_h1h.py`)

PoPE, learned expansion, **shuffled** read order, from scratch, B2 recipe (compile, fused AdamW, cosine, familiarity
gate), **batch 128 × 5,000 steps**, evaluated every 1,250 steps, seeds **6–9** (paired with H1g):

| arm | attention |
|---|---|
| **PC** | causal (as H1f/H1g) |
| **PN** | **non-causal** |

**Budget, measured this time** (H1g's 8,000-step runs broke the 2-minute rule): a throwaway-seed run of 1,000 steps
(non-causal, batch 128) took 33.6 s end to end. With H1g's 125–156 s for 8,000 steps, that is about 13–17 ms per step
plus about 20 s of fixed cost, so 5,000 steps is ≤ about 110 s.

**Evaluation:** both read orders on world 0, forward and reverse error classes, predictions saved.

## Predictions

A breakthrough means ≥ 0.8 with the shuffled order.

- **N1 (non-causal attention enables forward):** PN breaks through in **forward** in ≥ 2 of 4 seeds. Causal readers:
  0 of 10 so far.
- **N2:** PN breaks through in reverse in at least as many seeds as PC.
- **N3 (replication):** PC's forward is ≤ 0.3 in every seed.
- **Rule:** every run ≤ 2 minutes. Any overrun is reported.

## Cost

8 runs of about 100–110 s, plus evaluations.

---

## Results (2026-10-18; `runs/h1h/`, log `runs/h1h_log.txt`)

Every run took 68–94 s, within the 2-minute rule; the measured budget worked.

| reader | **shuffled: REV / FWD** | ranked: REV / FWD | forward errors (shuffled) | reverse curve | forward curve |
|---|---|---|---|---|---|
| PC s6 | 0.178 / 0.064 | 0.102 / 0.054 | 61% right relation, wrong entity; 31% not in context | 0.10 → 0.18 | 0.05 → 0.06 |
| PC s7 | 0.463 / 0.070 | 0.492 / 0.090 | 59%; 33% | 0.10 0.16 0.28 0.46 (rising) | 0.03 → 0.06 |
| PC s8 | **0.973** / 0.061 | 0.973 / 0.058 | 53%; 40% | 0.14 0.27 **0.95** 0.97 | 0.03 → 0.06 |
| PC s9 | 0.322 / 0.064 | 0.325 / 0.067 | 59%; 33% | 0.06 → 0.31 (rising) | 0.02 → 0.06 |
| PN s6 | 0.210 / 0.060 | 0.122 / 0.047 | 56%; 37% | 0.11 → 0.19 | 0.04 → 0.07 |
| PN s7 | 0.088 / 0.057 | 0.090 / 0.108 | 48%; 37% | 0.07 → 0.11 | 0.02 → 0.05 |
| PN s8 | 0.158 / 0.061 | 0.143 / 0.062 | 52%; 40% | 0.08 → 0.17 | 0.02 → 0.06 |
| PN s9 | 0.131 / 0.059 | 0.090 / 0.066 | 60%; 33% | 0.07 → 0.12 | 0.04 → 0.07 |

**Against the predictions:**
- **N1 (non-causal enables forward in ≥ 2 of 4): REFUTED.** Forward is 0.06 in every PN seed.
- **N2 (PN's reverse breakthroughs ≥ PC's): REFUTED.** PN 0 of 4 (≤ 0.21); PC 1 of 4.
- **N3 (PC forward ≤ 0.3): holds.** 0.06–0.07.

**What H1h establishes:**
1. **Causal masking is not what blocks forward content matching.** Removing it did not help forward at all.
2. **Non-causal attention slows reverse learning** within this budget. A plausible cause, untested: every position now
   also sees the *next* episode's tokens, so "the token two positions away" is ambiguous in both directions.
3. **The causal PoPE arm matches the update-count picture from H1g:**

   | budget (batch × steps) | reverse breakthroughs |
   |---|---|
   | 128 × 5,000 | 1 of 4 (two more still rising) |
   | 256 × 4,000 | 2 of 4 |
   | 128 × 8,000 | 4 of 4 |

**State of the content-matching question (H1d–H1h, about 20 shuffled-order readers):**
- Reverse content matching is learnable with PoPE, given about 8,000 updates.
- Forward content matching has never been learned by any reader: absolute positions, RoPE, PoPE, causal or not, more
  updates or not.
- Its errors always include 30–40% answers *not in the read set at all*, so the reader does not locate the right
  episode.
- The remaining explanations are untested:
  - it needs many more updates than the 2-minute rule allows;
  - matching on a *copied* feature (the subject, carried to the object's position) is structurally harder than matching
    on a token's own identity.

**Practical position:** the ranked read path (E1: retrieval ranks the full match first, the reader copies) answers
forward and reverse at about 0.96–0.97 for single-fact questions. Content matching only matters once retrieval cannot
rank: several relevant episodes, or chains.
