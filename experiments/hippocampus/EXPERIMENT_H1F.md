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

## Results (2026-10-18; `runs/h1f/`, log `runs/h1f_log.txt`)

Training runs took 100–107 s, within the 2-minute rule.

| reader | REV one-way clean: ranked / **shuffled** | fwd: ranked / shuffled | shuffled-order reverse errors | reverse curve (training order, every 500 steps) |
|---|---|---|---|---|
| **P_sh s4** | **0.977 / 0.971** | 0.140 / 0.094 | 2.4% right relation, wrong entity; 0.5% right entity, wrong relation | 0.12 0.17 0.46 **0.91** 0.95 0.97 0.97 0.97 |
| P_sh s5 | 0.147 / 0.155 | 0.070 / 0.078 | 71% right relation, wrong entity; 12% not in context | 0.06 0.08 0.10 0.12 0.14 0.14 0.15 0.16 |
| P_rk s4 | **0.974** / 0.096 | 0.950 / 0.093 | 90% right relation, wrong entity | 0.87 … 0.97 |
| P_rk s5 | **0.967** / 0.126 | 0.965 / 0.127 | 87% right relation, wrong entity | 0.88 … 0.95 |

**Against the predictions:**
- **P1 (content matching ≥ 0.6 in both seeds): met in seed 4, not seed 5.** Not met as pre-registered.
  - **Seed 4 learned content matching:** 0.971 reverse with the read order shuffled, and 0.977 ranked. It is
    *order-independent*, the first such reader in this line, with a sharp transition at about steps 1,500–2,000
    (0.46 → 0.91).
  - Seed 5 stayed at 0.16, like every earlier shuffled reader.
- **P2 (PoPE keeps the copy route ≥ 0.9): holds,** 0.967–0.974 ranked.
- **P3 (reported):** a PoPE reader trained on ranked order is as rank-dependent as E1 and RoPE (0.10–0.13 shuffled).

**Unexplained, recorded:** P_sh s4 learned *reverse* content matching but not *forward* (0.09–0.14 forward, in either
order). The two look symmetric (match one entity and the relation, copy the other entity), and the training mix is
50/50. Only the reverse curve was logged per step, so it is unknown whether forward was about to follow. The next run
should log both directions and save predictions.

**What H1f establishes:**
- Across 10 shuffled-order readers in H1c–H1f (absolute positions ×6, counting warm-start and mixed; RoPE ×2; PoPE ×2),
  **the only one that ever learned content matching is a PoPE reader:** 0.97, order-independent. Every other reader
  stayed at or below 0.23.
- That supports the user's suggestion and PoPE's argument: disentangling content (magnitude) from position (phase) is
  what lets a head match on *what* and offset by *where*.
- It is **one seed of two**, with a late, seed-dependent transition like H1b's plateau escape, and with forward not
  learned. Not yet reliable.

**Next:** more seeds, and longer training where the 2-minute rule allows (or fewer steps per second spent elsewhere);
log forward per step; save predictions.
