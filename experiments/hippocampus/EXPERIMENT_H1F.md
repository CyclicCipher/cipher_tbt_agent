# Experiment H1f — PoPE attention for content matching (the user's suggestion)

**Pre-registered 2026-10-18, before any H1f result run.** The user, on seeing H1e: "Have you considered using PoPE
instead of RoPE? It should be in the repo."

## Why PoPE fits this failure

- **The failure.** In H1d/H1e the reader, without a rank cue, finds episodes with the asked relation but not the one
  whose *object* is the question's entity ("right relation, wrong entity", 62–91% of answers). RoPE did not help
  (0.19–0.22).
- **The task decomposes into content plus position:** match the question's entity b to the object slot of a read
  episode (*content*), then read the token two positions earlier (*position*).
- **PoPE's argument** (arXiv:2509.10534): RoPE **entangles** the two. Its score Σ μ_q μ_k cos((s−t)θ + φ_k − φ_q)
  contains content-dependent phases, so a head cannot match on what and offset by where independently. PoPE puts content
  in the magnitude (softplus of q and k) and position alone in the phase, giving Σ μ_q μ_k cos((s−t)θ + δ) with no
  interaction term.
- **Its reported gain is on an indexing diagnostic of exactly this kind:** 11% → 95%.

## What the repo already knows

- **`../transformers/NOTES.md`:** PoPE 7/17 vs RoPE 9/17 on a composition task, one seed, far below the paper's scale.
  "Not evidence against the paper."
- **ZipLearn E0:** PoPE's *position-free channels* (an addition of mine) destroyed training and were withdrawn. Plain
  PoPE trained normally.

## Change

`Net(pope=True)`, in `../intelligent_constraints/model.py`:
- every block's attention is **the repo's PoPE attention, reused unchanged** from `../transformers/h1_lid.py`
  (`Attn(d, n_head, "pope")`: softplus magnitudes, position-only phases over all hd channels, learnable δ initialised in
  U(−2π, 0) and clipped there, SDPA scaled by 1/√hd);
- no absolute position embedding;
- existing configurations are unchanged.

## Arms (`run_h1f.py`)

The B2 recipe with the expansion, from scratch, seeds 4 and 5, but **4,000 steps**:

| arm | attention | read order in training |
|---|---|---|
| **P_sh** | PoPE | shuffled |
| **P_rk** | PoPE | ranked |

- **Why 4,000 steps:** PoPE runs at about 18 ms per step (a microbenchmark), and the 2-minute rule includes
  compilation. This is a small handicap against PoPE relative to H1e's 4,500.
- **Evaluation:** both read orders on world 0, with error classes, as in H1d/H1e.

## Predictions

- **P1 (disentangled content and position enable content matching):** P_sh's shuffled-order reverse is **≥ 0.6** in
  both seeds. **Refuted if ≤ 0.3**, the level of absolute positions (0.20) and RoPE (0.19–0.22).
- **P2 (PoPE keeps the copy route):** P_rk's ranked-order reverse is ≥ 0.9.
- **P3 (reported):** P_rk evaluated shuffled, against E1 (0.10–0.13) and the RoPE reader (0.09–0.12).

## Cost

4 runs of about 100–115 s each.

---

## Results

*(to be filled in after the runs)*
