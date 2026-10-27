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

# Parts 3–5 (pre-registered 2026-10-26, before the code was written)

**Why (the user):** run the prequential marginal likelihood scoring, and work through the options part 2 left open:
1. score hypotheses by the full Bayesian evidence (part 3);
2. add "the order is total" (part 4);
3. use computed coordinates as the addresses of a Spotlight-style memory (part 5).

## Part 3: prequential marginal likelihood

**The problem it fixes:** part 2's score credits a hypothesis only for entries it fills. An axiom that restricts
without filling (Latin, on T5), and a hypothesis that is simply stronger (abelian group against group, on T1 and
T2), were decided by the tie-break.

**Method:** the same tasks, relabellings and observed entries as part 2 (same seeds).
- The observed entries are coded one at a time, in a random order. Each entry's probability under hypothesis H
  comes from the completion of the entries before it:

      P(v) = (1 − ε)·q_H(v) + ε/60,   ε = 0.01

  - **q_H when the completion determines the entry:** a point mass on the completed value.
  - **Otherwise,** uniform over the allowed values:
    - Latin: the values not yet in the entry's row or column;
    - total (part 4): {a, b};
    - both together: the intersection;
    - otherwise: all 60.
- **Score:** the log marginal likelihood, Σ log P(v). The prior over hypotheses is uniform. The highest score wins,
  with no tie-break. The margin is the winner's score minus the runner-up's.
- **Speed:** completions are recomputed at checkpoints, after 20 entries and then each time the count has grown by
  15%. The entries between two checkpoints are coded from the completion at the earlier one.
- 3 random orders; the mean is reported.

| # | prediction |
|---|---|
| P1 | the winner is right on all 6 tasks at both fractions, with a positive margin in each of the 3 orders |
| P2 | on T5, quasigroup beats none by ≥ 500 nats at 50% observed and ≥ 100 at 25%. (Estimate: with a fraction ρ of the table known, an entry has about 60(1 − ρ)² allowed values, so Latin saves about 1,100 and 250 nats) |
| P3 | on T1 and T2, abelian group beats both group and commutative semigroup by ≥ 5 nats: no tie |
| P4 | on T4, semilattice beats commutative semigroup (idempotence predicts the diagonal); on T6, none beats every other hypothesis by ≥ 100 nats |

## Part 4: totality

**A new hypothesis, chain** = {comm, assoc, idem, total}. "Total" says every a ⊕ b is a or b:
- a known entry outside {a, b} is a contradiction;
- an undetermined entry has two allowed values.

It is scored as in part 3, and added to the hypothesis space for all six tasks. Part 3's verdicts use the original 7
hypotheses only.

**Rank read-out** from the completion of all observed entries:
- "a is below b" iff a ⊕ b = b.
- Each element scores s(a) = #{b known to be below a} − #{b known to be above a}.
- The ranks are the order by s.

**The information limit:** two elements adjacent in the chain are ordered only if one of their two entries was
observed. No axiom recovers this. So exact ranks were never fully reachable, and part 2's C3 asked for more than the
data hold.

| # | prediction |
|---|---|
| Q1 | on T4, chain wins, beating semilattice by ≥ 500 nats at both fractions |
| Q2 | on T1–T3, T5 and T6, chain loses to the part-3 winner by ≥ 100 nats |
| Q3 | ranks: Spearman correlation with the true order ≥ 0.999, and largest rank error ≤ 2, at 50%; largest error ≤ 4 at 25% |

## Part 5: computed coordinates as memory addresses (Spotlight-style, in the Tolman–Eichenbaum machine's setting)

**The setting (the Tolman–Eichenbaum machine's):** an agent walks a graph using actions whose meaning never changes,
and sees a label at each node. Labels are reshuffled every episode; the structure stays.

**Environments:**
- **E1:** a 6 × 10 torus with actions N, S, E, W. This is the group Z/6 × Z/10, and Spotlight's case: a 2-D lattice.
- **E2:** a cycle of 60 nodes with actions ±1 and ±7 (Z/60).
- **E3:** the Cayley graph of A₅ (one node per group element) with actions a, b and b⁻¹, where a = (1 2)(3 4) and
  b = (1 3 5). This group is not commutative.

**Episode 1:** walk until every (node, action) edge has been seen. From the transition table:
1. **The commutator test:** does every pair of actions commute at every node?
2. **If yes, compute the characters:** the common eigenvectors of the actions' shift operators, normalized at the
   start node.
   - Each character has an order, and a step per action (an integer modulo the order).
   - Characters are added greedily, largest order first, until every node has a distinct address (the tuple of
     phase indices).
3. **If no, the fallback is the regular representation:** a node's address is its episode-1 node, and path
   integration uses the episode-1 transition table.

**Transferred to episode 2:** only the per-action steps (in the fallback, the table).

**Episode 2:** new labels and a new 300-step random walk. Two predictors of the next label:
- **Coordinate memory:** path-integrate the address from the start, and write address → label at each visit. These
  are the Spotlight cells: one label per cell, with no neighbourhood read needed, because the addresses are exact.
- **Edge memory (no structure):** (current label, action) → next label, for edges traversed in this episode.

| # | prediction |
|---|---|
| S1 | the commutator test: E1's and E2's actions commute; E3's don't |
| S2 | E1 needs 2 characters (Z/6 × Z/10 isn't cyclic); E2 needs 1, of order 60. On E3 only the trivial character exists (A₅ equals its own commutator subgroup), so characters give all 60 nodes one address, and the fallback is used |
| S3 | on steps that reach a node already visited in episode 2, coordinate memory is 100% correct in all three environments. Its overall accuracy over the 300 steps beats edge memory's by ≥ 0.15 in each |
| S4 | the transferred structure: E1 8 integers (2 characters × 4 actions), E2 4, E3's fallback 180 (60 nodes × 3 actions) |

3 episode pairs per environment.

**What part 5 does not test:**
- repeated labels (the Tolman–Eichenbaum machine's harder case, which needs inference over location);
- learned addresses;
- transfer to a different size of environment.

**Cost:** about 2 minutes in total, CPU only.

    python -m structure.claude.structure_prequential      (parts 3 and 4, from experiments/symbolics)
    python -m structure.claude.structure_transfer         (part 5)

## Part 3 results (2026-10-26; `runs/structure_prequential.txt`, `.json`; 84.5 s for parts 3 and 4 together)

Log marginal likelihood in nats, the mean over 3 orders. The margin is the winner's score minus the runner-up's;
the smallest margin over the orders is in brackets.

| task | winner at 50% | margin | winner at 25% | margin |
|---|---|---|---|---|
| T1 Z/60 | abelian group | 48 (45) over commutative semigroup; 425 over group | abelian group | 55 (51) |
| T2 Z/6 × Z/10 | abelian group | 51 (47) over commutative semigroup; 459 over group | abelian group | 54 (46) |
| T3 A₅ | group | 5,479 (5,476) over quasigroup | group | 2,802 (2,760) |
| T4 chain max | semilattice | 123 (123) over commutative semigroup | semilattice | 49 (49) |
| T5 subtraction | quasigroup | 987 (984) over none | quasigroup | 200 (200) |
| T6 random | none | 1,750 (1,730) over commutative | none | 421 (415) |

**Verdicts: P1–P4 all confirmed.**
- **P1:** right on all 6 tasks at both fractions, with a positive margin in every order.
- **P2:** quasigroup beats none by 987 and 200 nats (estimates were 1,100 and 250).
- **P3:** abelian group beats commutative semigroup by 48–55 nats and group by 425 or more.
- **P4:** semilattice beats commutative semigroup by 123 and 49; none beats the rest by 1,750 and 421.

**No tie-break was needed anywhere.** The two cases part 2 decided by tie-break are now decided by evidence:
- **T5:** e^987 in favour of the Latin rule. It never fills an entry, but every entry it restricts is cheaper.
- **T1 and T2:** commutativity is worth about 50 nats over commutative semigroup, and Latin about 425 over group. The
  strongest consistent hypothesis wins, because it is the one that predicts most. That is Occam's razor running the
  other way from parameter counting: an axiom costs nothing, and it pays whenever it is true.

**Exploratory (arithmetic on the results above): the code against the information-theoretic minimum.**
- **Abelian groups:** a labelled abelian group of order 60 is one of 60!/16 + 60!/48 tables (Z/60, plus Z/2 × Z/30,
  whose automorphism groups have orders 16 and 48). Pinning one down therefore takes at least 186.1 nats. The
  prequential code spends 352–393 nats on T1 and T2: about twice the minimum.
- **A₅:** tables isomorphic to A₅ number 60!/120, a minimum of 183.8 nats. The code spends 810–862: about 4.5 times.
- **The structure is settled within the first quarter of the table.** On T1 the 50% code costs only 10 nats more than
  the 25% code, which is what 900 more correctly predicted entries cost at ε = 0.01.
- **Reading:** the completion rules are a sound but incomplete inference engine. The excess over the minimum
  measures what they miss. They miss more without commutativity.

## Part 4 results

| | 50% | 25% |
|---|---|---|
| chain against semilattice on T4 | chain wins by 1,292 nats | chain wins by 1,190 |
| chain on the other five tasks | last or second to last on every one; it loses to the winner by ≥ 1,700 nats | the same pattern, by ≥ 1,500 |
| ranks: Spearman correlation | 0.9992 | 0.9935 |
| largest rank error | 2 | **6** |
| elements misplaced | 22 | 43 |

**Verdicts:**
- **Q1 confirmed.**
- **Q2 confirmed.**
- **Q3 confirmed at 50% and refuted at 25%:** the largest error was 6 against a predicted maximum of 4.

**Reading:**
- **Totality is the axiom that identifies the chain.** It is worth about 1,200–1,300 nats: every undetermined entry
  has 2 allowed values instead of 60.
- **The rank errors sit where the information limit says they must.** At 25%, each pair of chain neighbours has both
  of its entries unobserved with probability 0.56. Runs of mutually unordered neighbours then grow long enough to
  misplace an element by 6.

## Part 5 results (`runs/structure_transfer.txt`, `.json`; 0.3 s)

| | E1 torus 6 × 10 | E2 cycle 60 | E3 A₅ |
|---|---|---|---|
| episode-1 steps to see every edge | 1,076–1,830 | 1,209–1,943 | 1,233–1,633 |
| non-commuting (node, action pair) cases | 0 | 0 | 120 (a with b, and a with b⁻¹, at every node) |
| characters found | 60 | 60 | 1 (the trivial one) |
| characters chosen (orders) | 2 (30, 30) | 1 (60) | none; 1 distinct address, fallback used |
| transferred integers | 8 | 4 | 180 |
| coordinate memory on revisits | 100% (all seeds) | 100% | 100% (fallback) |
| overall accuracy over 300 steps, coordinate vs edge memory | 0.80 vs 0.43–0.50 | 0.81–0.82 vs 0.45–0.49 | 0.81–0.83 vs 0.55–0.60 |
| edge memory on revisits by an untraversed edge | 0 of 91–111 | 0 of 98–106 | 0 of 69–78 |

**Verdicts: S1–S4 all confirmed.**
- **S1:** the commutator test splits the environments as predicted.
- **S2:** 2 characters for the torus, 1 of order 60 for the cycle, and only the trivial character for A₅. The
  torus's two characters have orders 30 and 30, not 6 and 10: Z/6 × Z/10 ≅ Z/2 × Z/30, and any pair whose phases
  separate all 60 nodes serves.
- **S3:** the gap between coordinate and edge memory is 0.23–0.37.
- **S4:** 8, 4 and 180 integers.

**Reading:**
- **Coordinates move the learning out of the episode.** With the coordinates from episode 1, the second episode
  needs one visit per node, and every edge is predicted on first use: the Tolman–Eichenbaum machine's result. Edge
  memory needs one traversal per edge.
- **The commutator test is the switch.** Commuting actions get compressed coordinates: 4–8 integers, a lattice of
  cells.
- **A₅ shows the abelian method's edge.** A₅ equals its own commutator subgroup, so its only character is trivial,
  and every node gets the same address. The fallback, the episode-1 graph itself, still transfers, but at 180
  integers.
- **The compressed alternative is not built:** A₅'s 3-dimensional representation as the rotations of an icosahedron.
  Addresses would be 3 × 3 rotation matrices and each action a fixed rotation, 27 numbers in all. That is the
  non-commutative frontier: an address that is a matrix, not a phase.
- **What is Spotlight-like and what isn't:**
  - Like Spotlight: the split between where (the address) and what (the label stored there), and, on the torus, a
    lattice of cells.
  - Not like Spotlight: the addresses are computed and path-integrated, not learned by projecting a key. They are
    exact, so no 3 × 3 read is needed, and a cell holds one label, not a DeltaNet state.

# Parts 6 and 7 (pre-registered 2026-10-27, before the code was written)

**Why (the user):** do next steps 1 and 2:
1. irreducible representations from data, so A₅ gets matrix coordinates instead of the fallback;
2. a stronger completion engine, to close part of the gap to the information-theoretic minimum.

## Part 6: irreducible representations from data: one method for commuting and non-commuting actions

**Method** (`structure/claude/structure_irreps.py`). The episode-1 transition table from part 5 is the input, with
the same seeds; part 5's episode-2 walks are reproduced exactly.
1. **Words:** a breadth-first search from the start node gives each node a word, the actions that reach it. The start
   node plays the identity.
2. **The group's multiplication:** x·y is the node reached by following y's word from x. It is computed from the
   transition table alone.
3. **Left translations:** (L_h f)(y) = f(h·y). They commute with the actions, which act on the right.
4. **The invariant subspaces:** Z = Σ_h z_h (L_h + L_hᵀ), with random z, is real symmetric and commutes with every
   action. Its eigenspaces (eigenvalues grouped within 10⁻⁶) are invariant subspaces of the regular
   representation, generically one copy of one irreducible representation each.
5. **The matrices:** on each subspace U, every action s gets a matrix A_s = Uᵀ R_s U, where R_s is the action's
   shift operator. A node's matrix is the product of its word's matrices. Its character is the trace. Irreducible
   over the reals means character norm 1 (real type) or 2 (complex type: a rotation block that splits into two
   complex characters).
6. **Addresses:** irreducible representations, one per character class, are added greedily: smallest dimension
   first, then most distinct addresses. Each is added as a block until every node's address is distinct.
7. **Episode 2:** the address starts at the identity. Each action multiplies it on the right by that action's
   matrices. Memory cells are keyed by the address rounded to 10⁻⁴.

**This replaces part 5's switch.** Part 5 chose between characters and the fallback by the commutator test. Here,
"abelian" is read off the decomposition: every real irreducible has dimension 1 with norm 1, or dimension 2 with
norm 2. Equivalently, every complex irreducible is 1-dimensional.

| # | prediction |
|---|---|
| R1 | A₅: 16 invariant subspaces, of dimensions 1, 3 (×3), 3 (×3), 4 (×4) and 5 (×5), forming 5 distinct irreducibles (by character), each of norm 1. Invariance residual < 10⁻⁹. Torus: 4 of dimension 1 and 28 of dimension 2 (norm 2). Cycle: 2 of dimension 1 and 29 of dimension 2 |
| R2 | the smallest injective address: A₅ one 3-dimensional irreducible, whose matrices are rotations (determinant +1): a by 180°, b by 120°, and a·b (order 5) by 72° in one 3-dimensional irreducible and 144° in the other. These are the two icosahedral representations. Torus: two 2-dimensional blocks. Cycle: one 2-dimensional block of order 60 |
| R3 | episode 2, with one method on all three environments: coordinate memory 100% correct on revisits; transferred numbers A₅ 27 (3 actions × 3 × 3), torus 32 (4 actions × two 2 × 2 blocks), cycle 16 |
| R4 | the read-out "abelian" agrees with part 5's commutator test on all three environments |

**Cost:** seconds.

## Part 7: a stronger completion engine

**A diagnostic of part 3's engine, run before this pre-registration** (scratch script; T1 and T3 at 25% observed,
order seed 100):
- **Batching:** re-completing at every 2% instead of every 15% lowers the code from 388 to 357 nats (T1) and from
  852 to 764 (T3).
- **Slow crystallization:** almost all the rest is entries coded while still undetermined.
  - On T1 the completion knows 221 cells after 60 entries and 1,958 after 80.
  - On T3 it knows 226 after 150 entries and 3,154 after 200.
  - The minimum, 186 nats, is the equivalent of about 45 entries at full price.

**New rules.** All are sound consequences of the axioms, and apply to any hypothesis that contains those axioms:
- **Associativity in all four forms.** The four facts a ⊕ b = c, b ⊕ d = f, c ⊕ d = e and a ⊕ f = e are linked:
  - any three of them give the fourth;
  - the old rule used only one of the four patterns (the first three give the fourth);
  - the reverse pattern needs associativity alone;
  - the two patterns that recover a ⊕ b or b ⊕ d need cancellation, which the Latin axiom supplies.
- **Latin propagation, as in Sudoku** (only under Latin):
  - a naked single: a cell with one allowed value gets it;
  - a hidden single: a value that fits in only one cell of a row or column goes there;
  - a cell or value with no option left is a contradiction.

**Arms:**
- **Arm A (does anything break?):** the full engine, all 8 hypotheses, all tasks, both fractions, part 3's
  checkpoints, order seed 100.
- **Arm B (the gap):** the true hypothesis on T1, T2 and T3, both fractions, checkpoints every 2%, incremental
  completion (exact for a hypothesis that is never contradicted).
  - The full engine runs with orders 100–102.
  - The ablation runs with order 100 only: the base rules (part 3's), base plus associativity, and base plus
    Sudoku.

| # | prediction |
|---|---|
| V1 | soundness: in arm B, the full engine never meets a contradiction under the true hypothesis, and every entry it determines is correct |
| V2 | arm A: the winner is unchanged on all 12 task × fraction cells (part 3's verdicts), and chain still wins T4 (part 4) |
| V3 | abelian group's code on T1 and T2 (full engine, every 2%, mean of 3 orders) is ≤ 330 nats at both fractions (part 3: 352–393) |
| V4 | group's code on T3 is ≤ 650 nats at both fractions (part 3: 810–862) |
| V5 | the limit: no run gets below 250 nats on T1 or T2, or below 400 on T3. Local propagation doesn't track which labels are already placed in the structure, an inference that needs global counting, so it stays well above the minimum (186 and 184) |

**Cost:** a build check measures the engine's speed first. If the full run projects past 8 minutes, the ablation is
cut to T1 and T3 (to be recorded here).

    python -m structure.claude.structure_irreps        (part 6, from experiments/symbolics)
    python -m structure.claude.structure_engine        (part 7)

## Part 6 results (2026-10-27; `runs/structure_irreps.txt`, `.json`; 0.5 s)

| | E1 torus 6 × 10 | E2 cycle 60 | E3 A₅ |
|---|---|---|---|
| invariant subspaces | 32: 4 of dim 1, 28 of dim 2 | 31: 2 of dim 1, 29 of dim 2 | 16: 1 + 3·3 + 3·3 + 4·4 + 5·5 |
| distinct irreducibles (dim, character norm) | 32: (1, 1) ×4, (2, 2) ×28 | 31: (1, 1) ×2, (2, 2) ×29 | 5: (1, 1), (3, 1) ×2, (4, 1), (5, 1) |
| invariance residual | ≤ 2·10⁻¹³ | ≤ 4·10⁻¹³ | ≤ 2·10⁻¹³ |
| chosen blocks (dims) | 1, 1, 2 | 1, 2 | 3 |
| transferred numbers | 24 | 20 | **27** (part 5's fallback: 180) |
| "abelian" read-out = commutator test | yes = yes | yes = yes | no = no |
| coordinate memory on revisits | 100% | 100% | 100% |

**A₅'s two 3-dimensional irreducibles,** all seeds:
- Every matrix has determinant +1, so every one is a rotation.
- a rotates by 180.0° and b by 120.0°.
- a·b rotates by 72.0° in one irreducible and 144.0° in the other.

These are the rotation group of the icosahedron, in its two forms, which differ by √5 → −√5.

**Verdicts:**
- **R1 confirmed.**
- **R2 confirmed for A₅, refuted for the torus and the cycle.** The rule takes the smallest dimension first, so it
  picks 1-dimensional sign characters (values ±1) before any rotation block. The addresses are still distinct for
  every node, but the blocks are not the predicted ones.
- **R3 confirmed for accuracy, partly refuted for size.** Coordinate memory is 100% on revisits everywhere. Transferred
  sizes: A₅ 27 as predicted; torus 24 (predicted 32); cycle 20 (predicted 16).
- **R4 confirmed.**

Episode-2 accuracies equal part 5's exactly, because the walks are the same by design.

**Exploratory (after R2):** dropping any block whose removal keeps the addresses distinct gives:
- torus: one sign character plus one rotation of order 30, which is Z/2 × Z/30 (20 numbers);
- cycle: a single rotation of order 60 (16 numbers).

**Reading:**
- **One method now covers both cases.** The regular representation's decomposition, computed from the transition
  table alone, gives:
  - commuting actions: rotation blocks, which are part 5's characters written as 2 × 2 matrices;
  - A₅: a 3 × 3 rotation address.
- **The commutator switch becomes a read-out:** "abelian" means no irreducible is larger than a rotation block.
- **For A₅, the structure that transfers shrinks from 180 integers to 27 numbers.** Path integration becomes
  multiplying 3 × 3 rotation matrices, as on a sphere.
- **For commuting actions, matrices are a redundant encoding:** a 2 × 2 rotation spends 4 numbers on 1 angle.

## Part 7 results (`runs/structure_engine.txt`, `.json`; 91 s)

**Arm A** (full engine, part 3's checkpoints, order 100): **the winner is unchanged on all 12 cells,** for parts 3
and 4 alike. The true hypothesis's code falls on T1–T3:

| | T1, 50% | T2, 50% | T3, 50% | T1, 25% | T2, 25% | T3, 25% |
|---|---|---|---|---|---|---|
| part 3 engine | 365 | 329 | 860 | 388 | 370 | 852 |
| full engine | 271 | 257 | 367 | 266 | 249 | 359 |

**Arm B** (true hypothesis, checkpoints every 2%), code in nats:

| | T1, 50% | T2, 50% | T3, 50% | T1, 25% | T2, 25% | T3, 25% |
|---|---|---|---|---|---|---|
| full engine, mean of 3 orders | 252.7 | 254.1 | 381.8 | 256.3 | 246.8 | 338.7 |
| range over orders | 247–259 | 250–256 | 369–406 | 248–264 | 242–255 | 323–348 |
| entries until the table is fully known | 63–110 | 67–110 | 89–104 | 71–85 | 71–116 | 79–87 |
| minimum (information-theoretic) | 186.1 | 186.1 | 183.8 | 186.1 | 186.1 | 183.8 |

**Ablation** (order 100), code, and the entries until the table is fully known:

| | T1, 50% | T2, 50% | T3, 50% | T1, 25% | T2, 25% | T3, 25% |
|---|---|---|---|---|---|---|
| base | 356.1 (168) | 311.4 (172) | 795.1 (249) | 356.9 (296) | 347.9 (176) | 764.4 (260) |
| base + associativity | 258.7 (69) | 250.3 (104) | 370.5 (137) | 248.1 (85) | 242.2 (71) | 323.3 (79) |
| base + Sudoku | 356.1 (168) | 306.7 (172) | 795.1 (249) | 356.9 (296) | 347.9 (176) | 764.4 (260) |
| full | 258.7 (63) | 250.3 (77) | 370.5 (95) | 248.1 (85) | 242.2 (71) | 323.3 (79) |

**Verdicts:**
- **V1 confirmed:** 0 contradictions and 0 wrongly determined entries in all 18 full-engine runs.
- **V2 confirmed.**
- **V3 confirmed:** abelian group codes 247–264.
- **V4 confirmed:** A₅ codes 323–406.
- **V5 refuted:** the engine got closer than predicted. T1 and T2 went below 250 in 4 runs, and T3 below 400 in 5
  of 6.

**Reading:**
- **The gap to the minimum closed from about 2× to about 1.35× for abelian groups, and from about 4.5× to about 2×
  for A₅.**
- **Nearly all the gain is associativity's other three patterns,** above all the two cancellation patterns. These
  let one known product be solved for an unknown factor. The table now becomes fully known after 63–116 entries,
  against 168–296 before.
- **Sudoku propagation changes the code almost nowhere.** A naked single is an entry with one allowed value, and the
  Latin coding already prices it at 0 nats. Sudoku makes the table fully known sooner, but on entries that were
  never asked about.
- **What remains (a rough decomposition, by arithmetic).** On T1 at 50%, the 60 entries coded while undetermined
  cost (253 − about 17 for ε on determined entries) / 60 ≈ 3.9 nats each.
  - **The ideal averages 3.14 nats per label,** ln(60!)/60. It knows the answer is one of the labels not yet placed
    in the structure, and their number falls from 60 to 1.
  - **Our coding is uniform over the values its row and column don't exclude.**
  - **That difference,** about 0.8 nats × 60 ≈ 45 nats, plus the ε overhead, accounts for the rest.
  - **Tracking which labels are placed** means tracking the labelling itself: coordinates, as in parts 5–6, not
    local rules.
- **A₅ still needs more undetermined entries (78–97) before its table is known.** Without commutativity there are
  fewer patterns to propagate through.

# Parts 8 and 9: repeated labels (pre-registered 2026-10-27, before the code was written)

**Why (the user):** run 3b, then 3a.

**The setting.** Parts 5–6 identified each node by its label. Here many nodes share a colour, so a colour no longer
says where the agent is. Actions still do what they say, except where part 9 adds slips.

## Part 8 (3b): learning the structure from repeated colours

**Environments:**
- **E1:** the 6 × 10 torus, actions N, S, E, W (true structure inside the hypothesis family).
- **E2:** the 60-cycle with actions ±1, ±7 (true structure outside the family: the control).

**The walk:** every node gets a random colour from C colours, C ∈ {2, 3, 6, 60}. At C = 60 every colour is unique.
A random walk of 3,000 steps is shared across the colour counts within each seed. 3 seeds.

**Hypotheses** (675 tori, plus 2 structure-free models):
- **Tori:** Z/h × Z/w with 1 ≤ h, w ≤ 15. Each pairs the 4 actions into two inverse pairs (3 pairings): the first
  pair moves ±1 along the h-axis, the second ±1 along the w-axis. The pairings, orientations and periods are all
  unknown. Each hypothesis appears once (no relabelled duplicates).
- **Null:** every step lands on a new cell.
- **Edge memory:** the next colour is the one last seen after the same (colour, action).

**Score** (prequential, as in part 3, over the colour sequence):
- **Tori:** path integration gives each step a cell. A cell visited before predicts its last colour:
  P = (1 − ε)·[match] + ε/C. A new cell predicts 1/C. ε = 0.01.
- **Null:** 1/C at every step.
- **Edge memory:** predicts like a cell, keyed by (colour, action).
- **Prior:** uniform over all 677 models.
- **T\*** is the step after which the true hypothesis (pairing N/S with E/W, periods 6 × 10) leads every other model
  for the rest of the walk.

**The expected bottleneck.** 12 × 10 is the only multiple of 6 × 10 within range. It never contradicts the data, and
loses only at a revisit whose vertical displacement is an odd multiple of 6, where it predicts a new cell. Smaller
tori, and any wrong pairing, merge distinct nodes, and are refuted when the merged colours differ. So T\* should be
set mostly by the walk, not by the number of colours.

| # | prediction |
|---|---|
| B1 | on E1, the true hypothesis wins at step 3,000 for every colour count and seed |
| B2 | on E1, T\* ≤ 500 for every colour count and seed |
| B3 | on E1, within each seed, T\* at 2 colours exceeds T\* at 60 colours by at most 200 steps |
| B4 | on E2, no torus wins: the winner at step 3,000 is the null or edge memory, for every colour count and seed |

## Part 9 (3a): slips, with the structure known

**The setting:**
- The 6 × 10 torus. The structure is given, as transferred from an earlier episode: the actions' displacements and
  the periods.
- C ∈ {3, 6, 60} colours.
- With probability p ∈ {0, 0.05, 0.2} an action does nothing (a slip), and the agent isn't told.
- A 600-step walk; the memory starts empty; 3 seeds.

**Predictors of the next colour,** each made before the colour is seen:
- **Oracle:** the true position, plus a cell memory (position → last colour).
- **Dead reckoning:** the position from the intended actions, plus a cell memory.
- **Particle filter:** up to 128 slip histories, each with its own position and its own map. At every step each
  history branches into "moved" and "slipped" (weights 1 − p and p) and is reweighted by how well it predicted the
  observed colour. The 128 heaviest are kept.
- **Soft filter** (the version proposed in my last message): one belief over the 60 cells.
  - Each step shifts the belief by the action, blurred by the slip chance.
  - The belief is reweighted by the colour.
  - The colour is written into one shared map of colour counts, weighted by the belief.
- **Edge memory:** (colour, action) → next colour.

**Measures:**
- next-colour accuracy (the most probable colour; ties split evenly);
- log loss per step;
- for the filters, how often the most probable position is the true one.

| # | prediction |
|---|---|
| A1 | sanity: at p = 0, dead reckoning and the particle filter equal the oracle exactly |
| A2 | dead reckoning collapses under slips: at p = 0.05 its accuracy is ≥ 0.3 below the oracle's for every C |
| A3 | the particle filter stays close: within 0.05 of the oracle's accuracy at p = 0.05 for every C, and within 0.1 at p = 0.2 for C ≥ 6 |
| A4 | the soft filter is ≥ 0.1 below the particle filter at C = 3, p = 0.2; at C = 60 it is within 0.05 of the oracle at both slip rates (a unique colour re-localizes in one look) |
| A5 | edge memory is ≥ 0.2 below the particle filter at C ≤ 6, at every p |

**Cost:** seconds for both parts, CPU only.

    python -m structure.claude.structure_aliased      (part 8, from experiments/symbolics)
    python -m structure.claude.structure_slip         (part 9)

## Part 8 results (2026-10-27; `runs/structure_aliased.txt`, `.json`; 4.3 s)

**E1 (6 × 10 torus):**

| colours | winner at step 3,000 (all seeds) | margin over the runner-up (12 × 10) | T\* by seed |
|---|---|---|---|
| 2 | true: N/S with E/W, 6 × 10 | 41.3 | 89, 117, 105 |
| 3 | true | 65.5 | 86, 90, 103 |
| 6 | true | 107.0 | 86, 90, 103 |
| 60 | true | 245.1 | 86, 90, 103 |

The last rival before T\* was a wrong second period every time (6 × 8, 6 × 9 or 6 × 11), never 12 × 10.

**E2 (60-cycle, ±1 ±7; outside the family):**

| colours | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 2 | null | torus 12 × 12, right pairing | null |
| 3 | torus 15 × 15, right pairing | torus 15 × 15, right pairing | torus 15 × 15, right pairing |
| 6 | torus 15 × 15, right pairing | torus 15 × 15, right pairing | torus 15 × 15, right pairing |
| 60 | edge memory | edge memory | edge memory |

**Verdicts:**
- **B1 confirmed.**
- **B2 confirmed:** T\* = 86–117.
- **B3 confirmed:** 2 colours cost 2–27 more steps than 60 colours.
- **B4 refuted:** a torus wins E2 in 7 of 12 runs.

**Reading:**
- **The structure is found from repeated colours almost as fast as from unique ones.** With 3 or more colours, T\*
  is the same as with unique labels in every seed. What sets T\* is the walk: the first horizontal excursion long
  enough to tell period 10 from 9 or 11. My expected bottleneck (12 × 10) was wrong. That rival falls behind as soon
  as the walk wraps vertically.
- **A consistent structure that is too big pays a bounded price.** At step 3,000 the margin over 12 × 10 is exactly
  60 × [ln C + ln(1 − ε + ε/C)]: the cost of a new cell, once for each of its 60 extra cells. Once the walk has
  visited all 120 cells, the gap stops growing. The data never refute 12 × 10; Occam's factor just keeps it e⁻⁴¹ to
  e⁻²⁴⁵ behind.
- **Why B4 failed: the family contains a good partial model of E2.** A large torus with the right pairing
  approximates the plane Z², which is the unwrapped version of E2's true group (Z/60 = Z²/Λ).
  - A return to the same point of the plane is always a true revisit. Wrong merges happen only at wrap-arounds
    (displacements that are multiples of 15), which a random walk rarely makes.
  - So the winning tori carry real structure: which actions are inverse pairs, and that they commute. Every E2 torus
    win had the right pairing.
  - With 2 colours, the wrap-around errors cost more than the gains, and the null wins. With unique colours,
    memorizing the 240 transitions (edge memory) beats any wrong geometry.
- **The fix, not run:** the family should hold every lattice Λ ⊂ Z² (Hermite normal form, with a shear), not just
  rectangular ones. E2's lattice is spanned by (1, 17) and (0, 60): one step of +1 plus 17 steps of +7 returns to the
  start.

## Part 9 results (`runs/structure_slip.txt`, `.json`; 3.1 s)

Next-colour accuracy, the mean of 3 seeds. "Located" is how often the filter's most probable position was the true
one.

| colours, slip | oracle (pre-registered) | dead reckoning | particle filter (128) | soft filter | edge memory |
|---|---|---|---|---|---|
| 3, 0 | 0.934 | 0.934 | 0.934 | 0.934 | 0.412 |
| 3, 0.05 | 0.935 | 0.633 | 0.776 (located 0.36) | 0.812 (0.32) | 0.422 |
| 3, 0.2 | 0.935 | 0.524 | 0.590 (0.15) | 0.524 (0.02) | 0.396 |
| 6, 0 | 0.918 | 0.918 | 0.918 | 0.918 | 0.321 |
| 6, 0.05 | 0.919 | 0.538 | 0.878 (0.92) | 0.746 (0.57) | 0.302 |
| 6, 0.2 | 0.919 | 0.402 | 0.650 (0.55) | 0.478 (0.06) | 0.299 |
| 60, 0 | 0.903 | 0.903 | 0.903 | 0.903 | 0.645 |
| 60, 0.05 | 0.903 | 0.492 | 0.864 (0.96) | 0.862 (0.96) | 0.594 |
| 60, 0.2 | 0.904 | 0.286 | 0.735 (0.81) | 0.727 (0.81) | 0.453 |

**Verdicts:**
- **A1 confirmed:** at p = 0, dead reckoning and the particle filter equal the oracle exactly.
- **A2 confirmed:** dead reckoning falls 0.30 (borderline: 0.3015), 0.38 and 0.41 below the oracle at p = 0.05.
- **A3 refuted.** It holds at p = 0.05 for 6 and 60 colours (gaps 0.040, 0.039), and fails at 3 colours (0.159) and
  at p = 0.2 (0.27, 0.17).
- **A4 refuted.** At 3 colours, p = 0.2, the soft filter is 0.066 below the particle filter, not ≥ 0.1. At 60
  colours it is within 0.05 of the oracle at p = 0.05 (0.041), but not at p = 0.2 (0.177).
- **A5 narrowly refuted:** 5 of 6 cells hold; at 3 colours, p = 0.2, the gap is 0.194.

**Exploratory 1 (after the results): the pre-registered oracle was a bound no predictor can reach.** It predicts from
the position after the step, so it knows whether this step slipped. The fair bound knows every past slip but must
hedge on the current one: (1 − p)·(the target cell's colour) + p·(the current cell's colour). Adding it consumes no
random draws, so every other number is unchanged.

| colours, slip | fair bound | particle filter, 128 | particle filter, 1,024 | particle filter, 8,192 |
|---|---|---|---|---|
| 3, 0.05 | 0.909 | 0.776 | 0.892 | 0.892 |
| 3, 0.2 | 0.829 | 0.590 | 0.649 | 0.637 |
| 6, 0.05 | 0.887 | 0.878 | 0.878 | 0.878 |
| 6, 0.2 | 0.778 | 0.650 | 0.743 | 0.743 |
| 60, 0.05 | 0.864 | 0.864 | 0.864 | 0.864 |
| 60, 0.2 | 0.735 | 0.735 | 0.735 | 0.735 |

**Exploratory 2:** the number of particles (128, 1,024, 8,192), in the last three columns
(`runs/structure_slip_k.txt`; 153 s).

**Reading:**
- **Dead reckoning collapses at the first slip.** Its memory is written in a drifting frame.
- **With unique colours a slip is visible at once:** the agent sees the colour it just saw. The particle filter then
  sits exactly on the fair bound, so its only loss is the slip itself, which nothing can predict.
- **With repeated colours, a slip in unvisited territory can't be seen when it happens.** Both branches predict "a
  new cell". The slip shows only at a later revisit, and by then the histories have multiplied.
  - With 128 histories the true one gets pruned: at 3 colours and 5% slips, accuracy rises from 0.776 to 0.892 with
    1,024. Beyond that, more histories change nothing.
  - At 3 colours and 20% slips, a gap of about 0.19 to the fair bound remains even at 8,192. This run doesn't
    separate how much of that is the data's limit (past slips the colours can't reveal) and how much is the beam's.
- **The soft filter works only with unique colours.** With repeated colours and frequent slips it loses its position
  almost entirely (located 2–6% of steps): blurred writes into one shared map reinforce their own errors.
  Spotlight-style blurred writing, without a separate map per hypothesis, isn't enough. This is SLAM's
  data-association problem. The established answer is a separate map per hypothesis, which is what the particle
  filter is (FastSLAM), plus loop closure.

# Part 10: the thesis in miniature (pre-registered 2026-10-27, before the code was written)

**Why (the user):** run option 3, after the design note `arc_agent/claude/DESIGN.md`. This tests its predictions F1
(transfer follows shared geometry) and F2 (content generalizes exactly when it is a function of a quotient of the
geometry).

**Environments:** both inside part 8's family, actions N, S, E, W:
- **E1:** the 6 × 10 torus, Z/6 × Z/10, with 20 subgroups.
- **E4:** the 4 × 15 torus, Z/4 × Z/15 ≅ Z/60, which is cyclic, with 12 subgroups.

**Episode 1, learning the geometry:**
- A random colouring with 6 colours and a 300-step walk. Part 8's 677 models compete; the one with the most
  evidence is taken.
- From the winner's transition table:
  1. part 6's decomposition gives the addresses (the computed coordinates);
  2. the group's multiplication gives its subgroups (the content hypotheses).

**Transfer episodes:** 6 per run, each with a new start, a new 200-step walk and a new colouring:
- **3 random:** 6 colours.
- **3 striped:** the colour is constant on the cosets of a random subgroup H of the true group, with between 2 and
  20 cosets. Each coset gets a random colour out of 6.

**Predictors of the next colour:**
1. **Carried geometry, cell memory:** path-integrate the computed address; address → last colour.
2. **Carried geometry, content model:** a Bayesian mixture over the learned group's subgroups. Hypothesis H predicts
   a cell's colour from the last colour seen anywhere in its coset. The trivial subgroup is cell memory; the whole
   group is "one colour everywhere". Uniform prior.
3. **Relearning:** no carried geometry. A Bayesian average over part 8's 677 models, started fresh in each episode.
4. **Edge memory:** (colour, action) → next colour.

**Measures:**
- accuracy (the most probable colour; ties split);
- first-visit accuracy (steps onto a cell not yet visited in this episode);
- log loss.

3 seeds per environment.

| # | prediction |
|---|---|
| D1 | episode 1 recovers the true geometry in all 6 runs. Part 6's decomposition of the learned table gives distinct addresses for all 60 cells, and the learned groups have 20 (E1) and 12 (E4) subgroups |
| D2 | random colourings: carried geometry with cell memory beats relearning by ≥ 0.03 accuracy over the 200 steps |
| D3 | random colourings: the content model is within 0.01 of cell memory |
| D4 | striped colourings: the content model's first-visit accuracy is ≥ 0.6, while cell memory's and relearning's are ≤ 0.25 (chance: 1/6) |
| D5 | striped colourings: the content model's overall accuracy is ≥ cell memory's + 0.05 |
| D6 | edge memory is ≥ 0.2 below cell memory on both kinds of colouring |

**Cost:** seconds, CPU only.

    python -m structure.claude.structure_miniature      (from experiments/symbolics)

## Part 10 results (2026-10-27; `runs/structure_miniature.txt`, `.json`; 2.4 s)

**Episode 1:** all 6 runs learned the true torus (6 × 10 or 4 × 15, pairing N/S with E/W). The learned groups have
20 and 12 subgroups, the true counts. Part 6's decomposition gave distinct addresses for all 60 cells: blocks [1, 2]
for 6 × 10 (Z/2 × Z/30) and [2] for 4 × 15 (a single rotation of order 60).

**Transfer episodes,** the mean of 18 episodes per kind (2 environments × 3 seeds × 3). The striped episodes had 2 to
20 cosets.

| colouring | predictor | accuracy | first-visit accuracy | log loss |
|---|---|---|---|---|
| random | carried geometry, cell memory | 0.802 | 0.167 | 0.43 |
| random | carried geometry, content model | 0.802 | 0.165 | 0.45 |
| random | relearning | 0.799 | 0.162 | 0.46 |
| random | edge memory | 0.344 | 0.166 | 3.78 |
| striped | carried geometry, cell memory | 0.789 | 0.167 | 0.46 |
| striped | carried geometry, content model | **0.969** | **0.872** | 0.09 |
| striped | relearning | 0.966 | 0.860 | 0.11 |
| striped | edge memory | 0.704 | 0.593 | 1.65 |

**Exploratory (after D2 failed):** accuracy by window of steps.

| colouring | predictor | steps 1–25 | 26–50 | 51–100 | 101–200 |
|---|---|---|---|---|---|
| random | cell memory | 0.535 | 0.637 | 0.839 | 0.891 |
| random | relearning | 0.539 | 0.617 | 0.840 | 0.889 |
| striped | content model | 0.829 | 0.955 | 0.985 | 0.999 |
| striped | relearning | 0.804 | 0.957 | 0.983 | 0.999 |

**Verdicts:**
- **D1 confirmed.**
- **D2 refuted:** carried geometry beats relearning by 0.003, not ≥ 0.03.
- **D3 confirmed:** content model 0.802 against cell memory 0.802.
- **D4 partly refuted:** the content model's first-visit accuracy is 0.872 (≥ 0.6) and cell memory's 0.167 (≤ 0.25),
  but relearning's is 0.860.
- **D5 confirmed:** +0.18.
- **D6 partly refuted:** edge memory is 0.46 below cell memory on random colourings, but only 0.085 below on striped
  ones.

**Reading:**
- **F2 holds.** Content that is a function of one of the geometry's quotients is predicted at cells never visited
  (0.87; chance 0.17), and arbitrary content is not (0.165).
- **But a learner with no carried geometry does as well, because its geometry hypotheses are quotient hypotheses.**
  Any finite-index subgroup of the plane contains a rectangle aZ × bZ. So a striped colouring is always a function of
  (x mod a, y mod b): it is a colouring of a smaller torus, which is in the relearner's family. Within one episode,
  colours alone can't tell "the world is a 3 × 1 torus" from "the world is 6 × 10, with stripes of period 3". The
  geometry of the world and the regularity of its content are the same kind of object, a quotient.
- **What separates structure from content is what persists across episodes,** and what actions do. A relearner that
  takes the 3 × 1 torus as the world would predict colours perfectly but plan badly: it would expect three steps
  north to bring it home. Structure is the Tolman–Eichenbaum machine's operational notion: the part shared across
  environments.
- **F1, as measured here, is not supported.** Next-colour prediction rewards only revisits. On a torus, early
  revisits are short loops (N then S) on which every hypothesis with the right pairing agrees. So relearning costs
  about 0.02 accuracy in steps 26–50 and nothing after.
- **F1 needs a test where geometry matters before any revisit:** a shortcut. Which actions reach a remembered place
  fastest? Tolman's own test. Carried geometry knows from the first step that 6 steps north return home; a relearner
  knows only after it has wrapped once. Edge memory does better on stripes (0.59 at first visits) because stripes
  make (colour, action) → next colour partly predictable.

# Parts 11 and 12 (pre-registered 2026-10-27, before the code was written)

**Why (the user):** run the shortcut test, then the moving-piece test. The geometry-discovery techniques are meant for
the columns of the symbolic agent, so that task manifolds are learned from data.

## Part 11: the shortcut test (Tolman's), where geometry should pay before any revisit

**Environments:** the 6 × 10 and 4 × 15 tori, with 6 colours, and with unique colours (60). 3 seeds.

**Episode 1:** a 300-step walk; the most probable torus of part 8's family becomes the carried geometry, as in part
10.

**Trials,** 20 per seed and colour regime. Each trial has a new colouring and a new start, then:
1. a 40-step random walk (exploration);
2. an excursion: k steps of action 0, then m steps of action 2, with k uniform in 1..h−1 and m in 1..w−1;
3. the query: go back to the start.

**Agents:**
- **Carried geometry:** its address by path integration in the learned torus, and that torus's shortest way home.
- **Relearning (a Bayesian planner):**
  - The 675 tori's evidence on this trial's colours gives a posterior.
  - Each torus proposes its own shortest way home.
  - The agent takes the proposal with the highest posterior probability of ending at the start, the shortest on
    ties.
- **Graph memory** (unique colours only): breadth-first search over the transitions (colour, action) → colour seen
  in this trial.
- **True geometry:** a sanity check.

**Measures:**
- **Success:** the agent ends at the start.
- **Length ratio:** the path's length over the true shortest length.
- **Shortcut trials:** those where the shortest way home wraps around the torus, which is shorter than undoing the
  walk's displacement in the plane.

| # | prediction |
|---|---|
| S1 | carried geometry succeeds in every trial, with the shortest length, at both colour counts |
| S2 | relearning succeeds in ≥ 95% of trials. On shortcut trials it takes the shortest way in ≤ 20%, and its mean length ratio there is ≥ 1.2 |
| S3 | graph memory's success rate is ≥ 0.3 below relearning's |
| S4 | design check: ≥ 30% of trials are shortcut trials |

## Part 12: the moving piece: objects as frames that move together (DESIGN.md, F3)

**The scene:**
- A 12 × 12 board that wraps at the edges, with background 0.
- 8 landmark cells in colours 1–3, plus one distractor landmark in a colour the piece also uses.
- The piece: a random 5-cell polyomino, each cell coloured 4, 5 or 6, chosen so that no rotation maps it onto itself.

**Actions:**
- 0–3 translate the piece by one cell in the world's directions.
- 4 rotates it 90° about its own pivot.
- The piece is drawn over landmarks, and they reappear when it leaves.
- Random action sequences of 30 steps.

**The learner has no object primitive. Its geometry is the innate one:** the board's translations and 90° rotations,
576 group elements.
- **Discovery:** it explains the first frame change as one set of cells moving by one group element, with everything
  else staying, and picks the element that leaves the fewest cells wrong. The set that moves is the object, an
  output rather than an assumption.
- **After discovery:**
  - the object's content is kept in its own frame;
  - its pose is the group element read off each new frame by matching that content;
  - a static memory holds the rest of the board.
- **Each action is either a world-frame element** (the new pose is g·pose) **or a body-frame element** (the new pose
  is pose·h). The frame is whichever stays constant across the action's transitions; on a tie, world-frame.

**Baselines:**
- **No change:** predict the same frame again.
- **Colour as the object** (the symbolic agent's assumption): each colour's cells move by a translation learned per
  action and colour.

**Transfer:** after a 30-step trial, a new board and a new piece for 10 steps.
- **Carried learner:** keeps the world-frame rules, and the angle of body-frame rules. The rotation's centre belongs
  to the piece, so it must be observed once.
- **Fresh learner:** starts from nothing.

20 trials per seed, 3 seeds.

| # | prediction |
|---|---|
| P1 | after the first transition, the discovered moving set equals the piece's cells exactly in ≥ 95% of trials |
| P2 | by the end of a trial, the translations are classed world-frame and the rotation body-frame in ≥ 95% of trials |
| P3 | exact next-frame prediction from transition 2 on: the learner ≥ 0.75, with its only errors at first sightings of an action plus ≤ 2 per trial from the world/body tie; colour as the object ≤ 0.2; no change 0 |
| P4 | after discovery, the read-off pose places the piece's cells correctly in every frame |
| P5 | transfer, frame errors in the first 10 transitions: carried learner ≤ 2.5 per trial (mean); fresh learner ≥ 3.5 |

**Cost:** seconds for each part, CPU only.

    python -m structure.claude.structure_shortcut      (part 11, from experiments/symbolics)
    python -m structure.claude.structure_piece         (part 12)

## Part 11 results (2026-10-27; `runs/structure_shortcut.txt`, `.json`; 7.8 s)

60 trials per row: 3 seeds × 20.

| environment, colours | shortcut trials | carried geometry: success, length ratio | relearning: success, shortcut taken, length ratio on shortcut trials | graph memory: success |
|---|---|---|---|---|
| 6 × 10, 6 | 0.78 | 1.00, 1.00 | 0.83, 0.23, 2.13 | — |
| 6 × 10, 60 | 0.75 | 1.00, 1.00 | 0.95, 0.36, 1.92 | 0.15 |
| 4 × 15, 6 | 0.85 | 1.00, 1.00 | 0.78, 0.22, 1.89 | — |
| 4 × 15, 60 | 0.78 | 1.00, 1.00 | 0.73, 0.40, 1.45 | 0.13 |

True geometry matched carried geometry in every trial: carried geometry was learned correctly in all runs.

**Verdicts:**
- **S1 confirmed:** carried geometry got home in every trial, always by the shortest way, wrap-arounds included.
- **S2 refuted.** Relearning's success was 0.73–0.95 (only one row reached 0.95), and it took the shortcut in 22–40%
  of shortcut trials (predicted ≤ 20%). Its length ratio on shortcut trials, 1.45–2.13, did meet ≥ 1.2.
- **S3 confirmed:** graph memory succeeded in 13–15% of trials.
- **S4 confirmed.**

**Why S2 failed: a flaw in my candidate set.** Each torus proposes its own shortest way home, and periods stop at 15.
Once the walk has strayed more than about 7 cells along an axis, no torus proposes simply undoing the displacement,
the route that works under every hypothesis. The planner is then forced to gamble on a wrap.

**Exploratory (after S2 failed):** adding "undo the displacement in the plane" as a candidate, one per action pairing.

| environment, colours | relearning + plane route: success | shortcut taken | length ratio on shortcut trials |
|---|---|---|---|
| 6 × 10, 6 | 1.00 | 0.02 | 3.83 |
| 6 × 10, 60 | 1.00 | 0.07 | 3.61 |
| 4 × 15, 6 | 1.00 | 0.04 | 4.88 |
| 4 × 15, 60 | 1.00 | 0.21 | 3.16 |

**Reading:**
- **This is where geometry pays: before any experience of the route.** Carried geometry knows from the first step
  that the torus wraps, so it cuts through territory it has never crossed. A learner without it either gambles
  (fails 5–27% of the time) or plays safe by undoing its walk: always home, but taking the shortcut in only 2–21% of
  shortcut trials, with routes 3.2–4.9 times the shortest. That is Tolman's distinction between a cognitive map and
  remembered routes.
- **Graph memory (remembered transitions) almost never gets home.** It only knows the directed steps it has taken.

## Part 12 results (`runs/structure_piece.txt`, `.json`; 37 s)

60 trials (3 seeds × 20) of 30 actions; transfer over 10 actions.

| measure | pre-registered learner | exploratory: revisable segmentation |
|---|---|---|
| moving set exact after the first transition (P1) | 0.783 | — |
| final object model exact (pose × body = the piece's cells at the end) | 0.783 | **1.000** |
| translations world-frame and rotation body-frame (P2) | 0.733 | 0.750 |
| exact next frame from transition 2: learner (P3) | 0.676 | 0.818 |
| exact next frame: colour as the object | 0.003 | 0.003 |
| exact next frame: no change | 0.000 | 0.000 |
| errors per trial: first sightings / world-body tie / other | 3.98 / 4.93 (max 25) / 0.47 | 3.98 / 1.02 (max 2) / 0.28 |
| pose places the piece's cells correctly (P4) | 0.783 of frames | 0.989 of frames |
| transfer, frame errors in 10 steps: carried / fresh (P5) | 3.98 / 6.38 | 2.12 / 5.18 |
| revisions per trial | — | 0.32 |

**Verdicts (pre-registered learner): P1–P5 refuted**, with two parts holding:
- in P3, colour as the object ≤ 0.2 and no change = 0;
- in P5, the fresh learner ≥ 3.5.

**Diagnosis:** segmentation failed in 13 of 60 trials, and every failure spoiled the whole trial.
- **9 failures:** the first action was a rotation. All 9 such trials failed: the pivot is a fixed point of the
  rotation, so nothing in its cell changes, and one frame pair can't show it belongs to the piece.
- **4 failures:** the aperture problem. Inside a run of same-coloured cells along the direction of motion, nothing
  changes either.

Both are limits of what a single frame pair can reveal. The flaw was that my learner never revised the set it first
discovered.
- **On the 47 trials whose segmentation was exact,** the learner did as predicted: 0.829 exact frames, and at most 1
  tie error per trial.
- **The remaining class errors are "both":** an action seen only at one orientation, where the data really can't tell
  world-frame from body-frame.

**Exploratory (after P1 failed):** revise on refutation. When the read-off pose doesn't reproduce the observed
frame, re-explain that transition from scratch. The old body-frame observations are carried into the new body frame
by conjugation (h → P h P⁻¹).

**Reading:**
- **Objects come out as frames that move together, with no object primitive, but only if the segmentation can be
  revised.** With revision, every trial ends with the exact object after 0.32 revisions on average. This is "mint on
  refutation", and the design's generalize-then-correct thesis: a first guess, corrected when a prediction fails.
- **Colour as the object, the symbolic agent's assumption, predicts 0.3% of frames.** Rotation can't be expressed as
  a per-colour translation, and the distractor landmark moves with its colour.
- **The learner discovers which coordinate system each action lives in.** Translations act on the left (world frame:
  pose' = g·pose). The rotation acts on the right (body frame: pose' = pose·h, about the piece's own pivot).
- **Transfer:** the action-to-group-element map carries to a new piece and board, and cuts errors from 5.2 to 2.1
  per 10 steps. What transfers is the geometry of the actions, not the object.
- **The dominant remaining cost is first sightings,** about 4 per trial: each action's effect has to be seen once.
