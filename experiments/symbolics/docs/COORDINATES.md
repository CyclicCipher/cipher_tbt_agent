# Computed coordinates: can the right embedding be derived from the data instead of handed over?

**Status:** pre-registered 2026-10-26, before the code was written.

**Why (the user, 2026-10-26):** every easy result in this folder had its geometry handed over (`ANALYSIS.md`), e.g.
addition mod 105 groks in 12 epochs with the residue embedding against 88 without (`CONTROLS.md`). The claim to
test: the coordinates in which an operation becomes addition are the eigenvectors of the operators that carry it out
(`RESIDUAL_TRICK.md`), so they can be **computed** from the operation table, without being told that 105 = 3·5·7.

## Method (`neural/claude/coordinates_mod.py`)

The data are pairs (a, b) → (a + b) mod 105. Only the training half of the table is seen, the same split as the
grokking controls (torch seed 42).

1. **Hypothesis: the operation is a commutative group.** Complete the table from the seen half using only
   commutativity (a ⊕ b = b ⊕ a) and associativity: if a ⊕ b = c, b ⊕ d = f and c ⊕ d = e, then a ⊕ f = e. Iterate
   to a fixed point. A proposal that conflicts with a known entry is a contradiction, and contradictions reject the
   hypothesis. Polynomial: O(N³) per pass.
2. **Coordinates:** from the completed table, the shift operators Pₐ (x ↦ a ⊕ x) as permutation matrices. Take
   the eigenvectors of one random combination S = Σₐ wₐPₐ; they are common to all the commuting Pₐ. Normalize each
   at the identity element (found from the table), to give a function χ on the elements.
3. **A minimal generating set:** each χ's order (the smallest k with χᵏ = 1, from its values). One character per
   prime-power order of maximal order, plus one of the largest order, gives the coordinates: (cos, sin) of each
   one's angle.
4. **The learner:** the same small transformer as the grokking runs (`grokking_adelic.py`: AttnRes, RoPE, AdamW
   2e-3, weight decay 0.01, dropout 0.1, 120 epochs). The only change is the embedding: a linear map of the computed
   coordinates; '=' is zero. CE on the answer; 3 seeds.

**Control:** the same completion on a random operation table (each (a, b) given a random answer), with the same
seen half.

## Predictions (2026-10-26)

| # | prediction |
|---|---|
| K1 | completion fills the whole table from the seen half with no contradiction, and its answers on the held-out pairs are 100% correct: the hypothesis "commutative group" plus associativity solves the task without any learning |
| K2 | on the random table, completion produces contradictions: the hypothesis is rejected |
| K3 | the eigenvectors are exactly the 105 characters e^(2πikx/105), up to relabelling: after normalization, each matches one with maximum deviation < 10⁻⁶ |
| K4 | the generating set has orders 3, 5 and 7 (plus one of order 105), found without being told 105 = 3·5·7 |
| K5 | the transformer with computed coordinates reaches 99% held-out accuracy within ≤ 20 epochs, mean over 3 seeds (residue embedding given: 12; plain: 88) |

**Cost:** under 2 minutes in total.

    python -m neural.claude.coordinates_mod      (from experiments/symbolics)

## Results (2026-10-26; `runs/coordinates_mod.txt`, `runs/coordinates_mod.json`, `runs/coordinates_mod_tables.*`; about 2 minutes in total)

| step | result |
|---|---|
| completion under "commutative group" | the whole table (11,025 entries) from the seen 5,512, in 2 passes and 0.05 s, with 0 contradictions; **100% correct on all held-out pairs** |
| the same on a random table | 1,630,914 contradictions: rejected |
| eigenvectors of the shift operators | exactly the 105 characters: largest deviation 2.4·10⁻¹³, homomorphism error 4.3·10⁻¹³ |
| character orders | 1, 3, 5, 7, 15, 21, 35, 105; generating set of orders **3, 5, 7** plus 105, computed from the table |
| transformer, computed coordinates as (cos, sin) circles | **no seed reaches 99% in 120 epochs** (final 96.4%, 96.3%, 96.7%) |
| transformer, computed residue classes as lookup tables (exploratory, after K5 failed) | 99% at epochs **15, 14, 12** (mean 13.7). Given residue embedding: 9, 19, 9 (mean 12); plain: 88 |

**Verdicts: K1–K4 confirmed; K5 refuted** as specified, with circle-encoded coordinates.

**Reading:**
- **The coordinates can be computed, not handed over.** From half the table, hypothesis-then-diagonalization
  recovers exactly the residue and Fourier structure of Z/105, including its factorization, in a fraction of a
  second, with no knowledge of the task's geometry.
- **The hypothesis did the real work, before any learning.** "This is a commutative group", tested against the data
  (no contradiction, while a random table yields 1.6 million), plus associativity, completes the whole task
  symbolically.
- **What the learner can use is a separate question from what is true.** The same computed information speeds a
  network up as residue lookup tables (mean 13.7 epochs, equal to the given embedding), but not as points on circles
  (96% after 120 epochs). Adding angles needs multiplicative interactions this small network learns slowly. Handing
  over coordinates needs an encoding matched to the learner.
- **What this does not show:** generality beyond groups. The method needs a hypothesized algebraic structure that the
  data can confirm or refute. Here it was given one hypothesis; choosing among many is the open part.
