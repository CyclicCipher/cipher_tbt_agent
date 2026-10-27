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

# Part 2: competing hypotheses (pre-registered 2026-10-26, before the code was written)

**Why (the user):** run the competing-hypotheses test. Part 1 gave the pipeline one hypothesis. Here several compete,
on tasks the pipeline wasn't designed for, and the data must pick.

**Tasks:** binary operations on 60 elements, each relabelled by a random permutation, so no numeric shortcut exists:

| task | operation | true structure |
|---|---|---|
| T1 | addition mod 60 | cyclic abelian group |
| T2 | Z/6 × Z/10, componentwise | abelian, not cyclic |
| T3 | A₅, the even permutations of 5 objects, composed | non-commutative group |
| T4 | max over a chain of 60 | semilattice (a total order) |
| T5 | subtraction mod 60 | a quasigroup, isotopic to a group; not associative, not commutative, and outside the hypothesis space as a group |
| T6 | a random table | none |

**Hypotheses:** each is a set of axioms that both completes the table and can be contradicted.
- **Completion rules:**
  - commutativity: a ⊕ b = b ⊕ a;
  - associativity: from a ⊕ b = c, b ⊕ d = f and c ⊕ d = e, infer a ⊕ f = e;
  - idempotence: a ⊕ a = a;
  - Latin: every row and column is a permutation. A duplicate is a contradiction, and a row or column missing exactly
    one value gets it filled.
- **The hypotheses:** none; commutative {comm}; commutative semigroup {comm, assoc}; semilattice {comm, assoc, idem};
  quasigroup {Latin}; group {assoc, Latin}; abelian group {comm, assoc, Latin}.

**Score** (a predictive likelihood, Occam built in):
- The observed entries are split 80/20 three times. Each hypothesis completes the table from the 80% and predicts the
  20%.
- Per predicted entry: log(1 − ε) if determined and right; log ε if determined and wrong; log(1/60) if undetermined.
  Plus log ε per contradiction met while completing. ε = 0.01.
- Sum over the three splits. The highest score wins; ties (within 10⁻⁶) go to the hypothesis with more axioms, the
  strongest consistent statement.

**Coordinates for the winner,** from the completion of all observed entries:
- **Abelian group:** characters, and their orders. Cyclic if one has order 60.
- **Group:** the size of its centre (the elements that commute with all others).
- **Semilattice:** each element's rank, #{b : a ⊕ b = a}, against the true order.

Observed fractions: 50% and 25% of the table.

| # | prediction |
|---|---|
| C1 | at 50%, the winner is right on T1 (abelian group), T2 (abelian group), T3 (group), T4 (semilattice) and T6 (none) |
| C2 | on T5 the winner is quasigroup: the strongest hypothesis in the space consistent with the data |
| C3 | T1 comes out cyclic (a character of order 60); T2 non-cyclic (largest order 30); T3 has a centre of size 1; T4's ranks equal the true order exactly |
| C4 | at 25%, the winner is still right on at least 5 of the 6 tasks |

**Cost:** seconds, CPU only.

    python -m structure.claude.structure_hypotheses      (from experiments/symbolics)

## Part 2 results (2026-10-26; `runs/structure_hypotheses.txt`, `.json`; 2.4 s in total)

| task | winner at 50% | winner at 25% | unseen entries filled, and correct | coordinates |
|---|---|---|---|---|
| T1 Z/60 | abelian group | abelian group | all, 100% (both fractions) | a character of order 60: **cyclic** |
| T2 Z/6 × Z/10 | abelian group | abelian group | all, 100% | largest order 30: **not cyclic** |
| T3 A₅ | group | group | all, 100% | centre of size **1** |
| T4 chain max | semilattice | semilattice | 97% (50%) and 88% (25%) of them, 100% correct | ranks not computable: the table is incomplete (below) |
| T5 subtraction | quasigroup | quasigroup | none | — |
| T6 random | none | none | none | — |

**Verdicts:**
- **C1 confirmed:** all 5 right at 50%.
- **C2 confirmed, but only by the tie-break.** Quasigroup and none score the same on T5: Latin never fills anything,
  because no row or column is ever one value short. The evidence for Latin is that it is never contradicted on T5,
  while on T6 it meets about 1,480 duplicates. The score gives that no credit.
- **C3: 3 of 4.** T1 is cyclic, T2 non-cyclic with largest order 30, and T3 has a centre of size 1. T4 fails as
  specified: completion leaves 54 entries (27 pairs) undetermined at 50%, so the ranks can't be counted.
- **C4 confirmed:** 6 of 6 right at 25%.

**Exploratory (after C3 failed on T4):** the order the completion does determine (a ≤ b iff a ⊕ b = b), against the
true chain:

| observed | pairs ordered correctly | wrongly | undetermined |
|---|---|---|---|
| 50% | 1,743 | 0 | 27 |
| 25% | 1,611 | 0 | 159 |

The undetermined pairs are close neighbours in the chain. Ordering a and b by inference needs an element c between
them with both a ⊕ c and c ⊕ b known, so nearly adjacent pairs are only settled by direct observation. "The order is
total" isn't in the hypothesis space; with it, the rest would follow.

**Reading:**
- **The data picked the structure in every case, from a quarter of the table, in seconds.** On T1–T3 the winner
  completed the whole table exactly. The coordinates then told cyclic from non-cyclic, and found A₅'s trivial centre.
  The relabelling meant no numeric shortcut was available.
- **The tie-break did real work in two places.**
  - On T1 and T2, commutative semigroup, group and abelian group tie: each completes the table perfectly.
  - On T5, quasigroup ties with none.

  Held-out prediction credits a hypothesis for what it fills, never for a constraint that was never violated.
- **The fix, not yet run:** a Bayesian marginal likelihood, coded entry by entry (prequential). Under Latin, an entry
  can only be one of the values not yet in its row and column, so each observed entry costs fewer bits than under
  none. That would credit an unviolated constraint directly, and should separate quasigroup from none on T5 without a
  tie-break.
