# Experiment 1b — the dagger memory with frozen token codes (C1, corrected)

**Pre-registered 2026-10-18, before any 1b run.**

## Why 1b

Experiment 1 did not test C1. A plain MLP with 8 hidden units memorised all the facts, so the facts were stored in the
learned embedding table and attention, not in the MLP the dagger constraint was applied to (`EXPERIMENT_1.md`).

1b fixes every token's code at a random N(0, 1) vector, and the codes are not trained. Facts can no longer be written
into the embedding table.

Attention alone (about 131k parameters) is mostly linear in the frozen code of a, with a two-position softmax. A linear
map on random 128-d codes can fit only about 128 arbitrary input→output pairs, against 2,999 facts. So the facts must go
into the MLPs.

The tied output side reads the same frozen codes, so "one code per entity" (C3) holds by construction in arms A–C. The
answer logits are s · LN(x)·code / √d, with a learned scalar s.

## What changes from Experiment 1

- **`frozen=True`** in `model.py`.
- **Arm D:** the input codes stay frozen, and the output codes are learned (untied, initialised at the same scale). So D
  still tests whether the cue code and the answer code must be the same vector.
- **Pilot:** extended to m = 4096. If no width memorises, 1b stops and reports.

Everything else is unchanged and reused as pre-registered in `EXPERIMENT_1.md`: data, evaluation sets (primary =
`rev_oneway_clean`, chance 0.0012), training, arms, the capacity rule (tight = smallest m with train_acc ≥ 0.99,
ample = 8× tight), seeds {0, 1}, and measures.

**Driver:** `python run_e1.py --frozen` writes to `runs/e1b/`.

## Apparatus check (must pass before anything is read)

- **The facts must now need the MLP:** the plain MLP at m = 8 must NOT memorise (train_acc < 0.99).
  - If it does, the facts are still outside the MLP. The grid result is then reported as uninformative about C1, as in
    Experiment 1.
- **Memorisation must be possible:** some m ≤ 4096 reaches train_acc ≥ 0.99 within 6,000 steps.

## Predictions (same thresholds as Experiment 1)

- **P1 (baseline cursed).** A's one-way clean hits@1 ≤ 0.02 at both capacities and both seeds. With frozen random codes
  there is no co-adapted embedding table to give the small "free" reversal seen in Experiment 1.
- **P2 (the hypothesis).** At one capacity or more, in both seeds, B's one-way clean hits@1 is ≥ 0.10 and at least 0.05
  above C's.
  - **Supported** if met.
  - **Strongly supported** if B is ≥ 0.5.
  - **Refuted** if B ≤ 0.02 at both capacities.
- **P3 (mechanism; applies if P2 holds).** Switching the transpose read off at test time drops B's one-way clean hits@1
  to ≤ 0.02.
  - **Diagnostic:** B's paired reverse hits@1 with the transpose off. A drop means the paired reverse facts are read
    through the transpose rather than stored separately.
- **P4 (capacity).** B's one-way clean hits@1, averaged over seeds, is higher at tight than at ample.
- **P5 (code match).** D is below B, averaged over seeds.
- **P6 (cost).** B reaches train_acc ≥ 0.99 at both capacities. reach99 is reported against A.

## Cost

- **Pilot:** up to 10 runs.
- **Grid:** up to 16 configurations, minus reuse; each run ≤ 2 min.
- **Total:** about 15–25 minutes.

---

## Results (2026-10-18; `runs/e1b/`, log `runs/e1b_log.txt`)

**The apparatus check FAILED.** The plain MLP at m = 8 memorised again (train_acc 1.000, reached at step 1250), even
with frozen codes. So tight = 8 and ample = 64, and the facts are again mostly outside the MLP.

My argument that attention could not store them was wrong:
- A linear map on 128-d random codes can fit only about 128 pairs *exactly*.
- But hits@1 among 1,000 random codes needs only about 0.5 correlation with the right code (the largest of 999 random
  overlaps is about 3σ ≈ 0.28).
- Attention's relation-gated linear maps reach that easily.

As pre-registered, the grid is therefore uninformative about C1 as a clean test. The numbers are still worth recording.

One-way clean hits@1 (chance 0.0012), and paired reverse hits@1 with the transpose read switched off at test time:

| capacity | arm | one-way (s0, s1) | transpose off: paired reverse (s0, s1) | leak (s0, s1) |
|---|---|---|---|---|
| tight (8) | A plain | 0.000, 0.002 | – | 0.07, 0.09 |
| | B dagger | 0.002, 0.000 | 0.845, 0.835 | 0.07, 0.07 |
| | C plain 2m | 0.002, 0.001 | – | 0.06, 0.07 |
| | D dagger untied | 0.003, 0.001 | 0.933, 0.986 | 0.13, 0.10 |
| ample (64) | A plain | 0.002, 0.000 | – | 0.07, 0.08 |
| | B dagger | 0.000, 0.001 | **0.707, 0.465** | 0.07, 0.07 |
| | C plain 2m | 0.000, 0.000 | – | 0.06, 0.08 |
| | D dagger untied | 0.001, 0.002 | 0.899, 0.858 | 0.12, 0.10 |

All arms memorised (train_acc ≥ 0.998, reach99 750–1500). Every run took 33–55 s.

**Against the predictions:**
- **P1 (baseline cursed): holds.** A is at 0.000–0.002, which is chance. With frozen codes, the small "free" reversal
  that learned tied embeddings gave in Experiment 1 is gone.
- **P2: refuted under these conditions.** B ≤ 0.002 at both capacities.
- **P3:** not applicable.
- **P4:** not applicable (everything is at zero).
- **P5:** not meaningful, since B and D are both at chance.
- **P6: holds.** Every arm memorises.
- **Leak fell from about 0.55 (Experiment 1) to about 0.07.** E1's habit of naming b's other partners lived in the
  learned embedding table.

**The informative finding: the transpose path is used, as extra storage, without transfer.**
- Switching the transpose read off drops B's *paired* reverse accuracy to 0.84 at tight, and to 0.71 / 0.47 at ample.
- So up to half of the paired reverse facts are answered through the transpose.
- Yet one-way reverse accuracy stays at chance.
- The model stores the reverse direction of each paired fact in the transpose path *as a separate association*, not as a
  re-read of the forward one.

That can happen because sharing K and V does not force sharing a fact:
- a unit's row of K is both the forward *key* (a cue in the MLP's input space) and the reverse *value* (an answer
  written toward the output codes);
- the two coincide only if an entity's code at the MLP input equals its code at the output.

Attention did not learn to carry codes unchanged, so the two roles stayed in different spaces. This is C3 (one code per
entity) at the MLP interface, not only at the embedding table. Tying the embeddings is not enough when attention sits in
between.

**Next (not pre-registered yet, awaiting the user):**
- **Weight decay** (0.1, 1.0): under an L2 penalty, the one-unit-both-ways solution is cheaper than separate storage.
- **A per-run apparatus check:** zero the MLP at test time and measure the drop in forward accuracy.
- **A fixed-transport variant:** replace attention with a fixed sum of the two token codes. That separates "can the
  dagger memory transfer" from "can attention learn to carry codes unchanged".
