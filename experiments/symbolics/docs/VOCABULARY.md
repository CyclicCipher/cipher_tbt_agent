# Experiment B3: a wider vocabulary (composition, combined inputs, relations)

**Status:** pre-registered 2026-10-26, before any of it was built.

**Why (the user, 2026-10-26):** extend the vocabulary, taking problems 1–3 of the list the user approved:
composition by inversion, atoms over a fitted combination of inputs, and relational templates with detection of
interacting pairs. These are problems 3, 2 and 5 of `OPEN_PROBLEMS.md`; the theory of the first is in
`RESIDUAL_TRICK.md`.

**This is the polynomial line.** Every construction below has a fixed, bounded candidate set: depth 2,
O(d²) directions, two pair sets. The design is informed by B and B2's held-out results.

## Base: B2, unchanged

Selection by BIC with the noise level known, σ = max(roughness, 0.002) × std of the training y. The search (beam 4,
refine 8, at most 8 blocks plus 4 after a decomposition), the guide (G1–G3), library learning (Stitch-style
description length), the curriculum (48 tasks, 3 cycles), the held-out tasks (H1–H6, n = 30 and 100, 3 seeds), A0, the
GP and the textbook laws (reused from B) are all as before.

## The three extensions (finder version 3)

### 1. Composition by inversion, depth 2 (problem 3)

The formula becomes y = f(T(x)), with T(x) the usual searched sum (with optional denominator) and f an outer function
from a fixed family. The inner sum is searched on the transformed target t = f⁻¹(y). The family has 11 members:

| f⁻¹ (the target t) | f (the prediction) | b |
|---|---|---|
| y | t | — |
| log(y − b) | e^t + b | b ∈ {0, min y − 0.05 · range y} |
| (y − b)^q, q ∈ {−1, −½, ½, 2} | t^(1/q) + b | b ∈ {0, min y − 0.05 · range y} |

Searching all 11 costs 11 inner searches, so the finder **scores every member by how additive its target is** and
searches only the best: the first-order sensitivity fit's R² (a degree-4 polynomial in each input, summed). That is
the coordinate-change route in `RESIDUAL_TRICK.md` §5. It searches the top 3, plus y and log y (B2's pair): at most 5
inner searches. All are compared by BIC on the y scale.

### 2. Atoms over a fitted combination of inputs (problem 2)

- **New factor kinds:** (a·u + b)^p and log(a·u + b). Here u is the inputs scaled to [0, 1] over the box, a is a
  direction (normalized, d parameters), and b is set so the base stays positive on the domain with any singularity
  at least 10% of the base's range outside it: b = −min(a·u) + (0.1 + softplus(β)) · range(a·u). p keeps its sign and
  |p| ≥ 0.25.
- **Starts:** directions e_i + e_j and e_i − e_j for every pair, and the all-ones direction; p ∈ {−2, −1, −½, ½, 1, 2};
  β ∈ {−2, 0}. Only for 2 or more inputs.
- **Margins** stay as they were; these atoms generalize them.

### 3. Relational templates and detection of interacting pairs (problem 5)

- **Detection, in the guide:** a second-order sensitivity fit, in the transform the guide chooses:
  t = b₀ + Σᵢ P₄(zᵢ) + Σ_{i<j} (c₁ zᵢzⱼ + c₂ zᵢ²zⱼ + c₃ zᵢzⱼ²), by least squares with a 10⁻⁶ ridge.
  - A pair's share is the variance of its fitted interaction part over the variance of t.
  - The pairs detected are those whose share is ≥ ¼ of the largest share and ≥ 10⁻⁴.
  - Input names are never used.
- **The relational group block:** xᵢ^p·xⱼ^q summed over a set of pairs, with p and q shared and one coefficient per
  pair. It is offered over the detected pairs (guide arms) and over all pairs (every arm, for 2 or more inputs).
- **Library learning:** group folding extends to two-input terms. Two or more terms with the same two-input structure
  over different pairs, parameters within 0.1, fold into a relational template. At fit time such a template is bound
  to the detected pairs and to all pairs.

**Cost (polynomial):** at most 5 inner searches; O(d²) affine directions; 2 pair bindings per relational template.

## Arms

- **Held-out: A1–A4** as before (finder v3 alone; + guide; + library; + all), after a fresh curriculum with v3 (3
  cycles).
- **Ablations, held-out only, n = 30 and 100:** A2 without inversion (targets y and log y only); A2 without combined-input
  atoms; A2 without relational groups.

## Predictions (2026-10-26, before anything was built)

| # | prediction |
|---|---|
| R1 | **Detection finds the steps:** on every stepped task with ≥ 3 segments (curriculum and held-out, pooled), the detected pairs include ≥ 90% of the adjacent pairs, and ≥ 80% of the detected pairs are adjacent |
| R2 | **The 6-segment beam:** A4 on H1 at n = 100 reaches in range ≤ 0.05 and far ≤ 0.15 (B2: 0.069 / 0.30; the textbook law with neighbour terms: 0.043 in range) |
| R3 | **Combined inputs give the plate a law:** a candidate law (≤ 6 terms, in ≤ 0.02, far ≤ 0.1) on H3 from A2 or A4 in ≥ 1 of 3 seeds at n = 100 |
| R4 | **Each extension contributes:** each ablation's in-range NRMSE, geometric mean over the 6 tasks at n = 100, is ≥ 10% higher than full A2's |
| R5 | **Safety is still unsolved:** the largest finder-v3 far NRMSE at n = 100 still exceeds 10 (growth outside the box isn't addressed) |
| R6 | **The yardstick (reference only):** A4 beats the GP in range on ≥ 4 of 6 and in far extrapolation on ≥ 4 of 6 at n = 100 (B2: 3 and 3) |
| R7 | **Discovery:** at least one candidate law on any held-out task from A1–A4 at n = 100 (B2: none) |
| R8 | **Cost:** a held-out fit takes ≤ 180 s at n = 100; the curriculum ≤ 60 CPU-minutes |
| R9 | **Composition gets used:** the outer function of the best formula is neither y nor log y in ≥ ⅓ of A4's held-out fits at n = 100 |

    python -m regression.claude.curriculum --variant v3 wake 1   (then sleep 1, wake 2, sleep 2, wake 3, sleep 3)
    python -m regression.claude.curriculum --variant v3 evaluate
    python -m regression.claude.curriculum --variant v3 report

## Amendments (during the build, before any curriculum or held-out run)

- **2026-10-26, how outer functions are scored.** As specified (first-order sensitivity fit R²), the score failed two
  build checks:
  - In one dimension every transform scores about 0.999, so it can't discriminate: for 1/(1 + 4x²) the right
    transform, 1/y, ranked 5th and wasn't searched.
  - It penalizes products: for √(1 + x²y) the right transform, y², whose target is the single term 1 + x²y, ranked 10th
    of 11.

  Amended to what the inner search can use: a 3-term greedy screen of the target (grid starts, no refinement),
  measured by the squared error on the y scale. After the change, √(1 + x²y) is found exactly as y² = 1 + 0.9992·x²y.
  A limitation seen on the way: the family's outer powers are fixed, so (2 + ab²)^−1.5, whose inverse power −2/3 is
  missing, gets only the nearest member, y^−½.
- **Build checks of the other two extensions** (recorded, not amendments):
  - Combined-input atoms recover (1.2 − a − b)^−1.5 and 3 log(1 + 2a + b) exactly, with the right directions.
  - Pair detection on the 3-segment curriculum beams finds only adjacent pairs, but sometimes only one of the two (the
    ¼-of-the-largest rule).
  - On a 4-input test function with neighbour interactions, detection failed (all shares ≈ 10⁻⁴). The all-pairs
    relational group still put coefficients ≈ 4.5 on the adjacent pairs and ≈ 0 on the others.
