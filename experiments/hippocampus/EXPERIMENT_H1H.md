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

## Results

*(to be filled in after the runs)*
