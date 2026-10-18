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

## Results

*(to be filled in after the runs)*
