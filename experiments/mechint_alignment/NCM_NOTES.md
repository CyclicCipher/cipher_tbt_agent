# Reading notes: Next-Concept Modeling (NCM)

**Source:** Kumbong, Khare, Mlauzi, Woen, Ding, Buchanan, Arora, Mirhoseini, Ré (Stanford / Together AI), "Next-Concept
Modeling: Fast, Parallel Coarse-to-Fine Text Generation", arXiv 2610.14162 (9 Oct 2026). Code:
github.com/HazyResearch/ncm (MIT; evaluation code and checkpoints on Hugging Face; the README documents no training
procedure). Read 2026-10-21 at the user's suggestion: "Because speed is a concern here, it is worth experimenting with
an NCM model to see if it gets us faster training and inference."

## What it is

Two stages:
1. **A multiscale residual VQ-VAE tokenizer.**
   - An encoder maps L tokens to L latent vectors.
   - Level k downsamples the remaining *residual* to ℓ_k vectors, with ℓ_1 < ℓ_2 < … < ℓ_K = L, and quantizes each
     to its nearest entry in level k's codebook. The quantized correction is mapped back to length L and added to
     the reconstruction.
   - **Nested dropout** (training keeps only the first k levels, for a random k) forces the early, coarse levels to
     carry the most important information.
   - A decoder maps the final reconstruction to tokens. Exact reconstruction is 98.8–100% (100% on Sudoku).
2. **A generator transformer.**
   - It predicts all ℓ_k codes of level k *in parallel* (independently, given the cumulative reconstruction of the
     earlier levels), so generation takes K forward passes, coarse to fine, instead of L.
   - Training predicts all levels in one pass. Some conditioning codes are randomly replaced, so generation is
     robust to its own earlier errors.

"Concept" means only "the information carried by the coarse levels". It is learned compression, not a semantic unit.
It follows next-scale prediction for images (VAR), adapted to text.

## The numbers (≈100M-parameter generators; 25–29M on Sudoku)

- **Speed against diffusion LMs:** 10× (OpenWebText), 21× (LM1B), 31× (XSum) faster wall-clock on one H100, at
  matched or better quality. From 4–16× fewer function evaluations, and cheaper early steps because coarse levels
  are short.
- **Training:** 4–5× fewer training tokens than diffusion baselines on LM1B and OpenWebText, tokenizer included
  (13.4B against 66B token positions on LM1B).
- **Against autoregressive models:** no speed comparison. Quality against the AR baselines of the same size:
  - WMT14 De–En BLEU 22.6, against AR 25.2 (*worse*; 25.2 with MBR decoding over many samples);
  - XSum ROUGE-L 28.0 against AR 24.4 (better);
  - Sudoku solved boards: hard 86.1% against AR 0.6–1.0%, and against the best diffusion model 81.3% with 12 vs 16
    passes. The coarse-to-fine refinement of the whole board helps where every output constrains every other.
- **Stated limitations:**
  - a fixed depth and per-level lengths (short outputs still traverse the whole hierarchy);
  - a separately trained tokenizer (not end to end);
  - scale studied only up to 725M parameters;
  - occasional artifacts at coarse-level transitions.

**A correction to the framing:** the paper does not claim diffusion is 10× faster than autoregression, and does not
compare NCM's speed with AR. Its speedups are against diffusion LMs, which process the whole sequence at every step;
an AR model with a KV cache is cheap per token.

## Is it relevant to our speed problems?

| where our time goes | does NCM help? |
|---|---|
| **world-model inference (E2–E8):** one step = one forward pass predicting every cell in parallel | no: already 1 pass per step; NCM's gain is fewer passes per *output* |
| **world-model training:** data-bound (transitions), small models | unclear: its training gain is against diffusion training, not against our one-step objective |
| **imagination in planning (E9–E10):** 3 × 48 sequences × 8 steps per decision, each step a simulation | **plausibly yes, later:** with a learned world model, planning means H-step rollouts, H passes each. NCM-style generation of a whole H-step future (or a whole plan) coarse to fine in K ≪ H passes is the analogue of its Sudoku result, where global constraints are satisfied by refinement rather than left to right |
| **language lines** (NTA, hippocampus) | yes, if generation speed matters there; it is a text model |

**A link to the beauty work:** a multiscale residual code *is* a hierarchy of compressions (coarse = the global
structure, fine = the detail). The proposed multi-scale appreciator (BEAUTY_RESULTS.md, next experiments) could
measure compression progress per level of such a code.

## A proposed first experiment (not designed in detail; needs the user's go-ahead)

**Trajectory generation for planning:** train a small NCM-style model (our own implementation, one file each for the
tokenizer and the generator; no downloaded code is run) on Canvas or Village rollouts. Generate whole H-step futures
in K levels, and compare against H one-step world-model passes:
- wall-clock time;
- exactness of the predicted states, which matters as in Sudoku: one wrong cell misleads a planner;
- the plans chosen.

Runs under 10 minutes.
