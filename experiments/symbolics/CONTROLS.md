# Controls for the two transformer experiments

**Why:** both experiments report near-perfect held-out accuracy almost at once. Before reading that as a property of
the adelic idea, check which component carries it. Each experiment gives the network some structure of the task as
input. Each control removes one piece and leaves everything else as written: model, optimizer, epochs, data and
split. Script: `controls_transformers.py`. It imports the original files and does not change them.

**Where the originals are:** Gemini's three adelic scripts (`test_adele_resnet.py`, `test_transformer_grokking.py`,
`test_transformer_discrete_logic.py`) are frozen in `gemini_original/`, and the controls import them from there.
The files of the same name in this folder are now the real versions (`REAL_ADELIC.md`).

## Pre-registration (2026-10-26, before any control was run)

### Grokking: addition mod 105, half of all pairs for training (`test_transformer_grokking.py`)

The adelic embedding gives each number as its residues mod 3, 5 and 7, plus its angle on the mod-105 circle. With
residues, (a + b) mod 3 depends only on a mod 3 and b mod 3. That is a 3 × 3 table, and every cell of it appears
thousands of times in training. The same holds for mod 5 (5 × 5) and mod 7 (7 × 7). Reading out the answer then maps
the 105 residue triples to the 105 answers, and all 105 triples appear in training. So no unseen combination is left
to generalize to. The angle gives addition as a rotation, which is the Fourier solution that grokking transformers
otherwise have to discover.

Separately, the Tate term in the loss uses argmax, so it has no gradient and does not affect training.

| arm | embedding | loss |
|---|---|---|
| as_written | adelic | answer + residue heads + Tate term |
| adelic_ce | adelic | answer only |
| plain_ce | learned token embedding | answer only (the standard grokking setup) |
| plain_heads | learned token embedding | answer + residue heads + Tate term |

3 seeds each (model init and batch order; the split is the original's), 120 epochs, held-out accuracy every epoch.

| # | prediction (mean over seeds) |
|---|---|
| G1 | as_written reaches ≥ 99% held-out accuracy within 10 epochs (replicates the report) |
| G2 | adelic_ce reaches ≥ 99% within 20 epochs: the embedding carries the result, not the loss |
| G3 | plain_ce stays < 50% at epoch 120 |
| G4 | plain_heads: no prediction (exploratory). It tests the "circular/adelic loss" claim, i.e. whether the residue targets help without the embedding |

### Discrete logic: Boolean expression trees (`test_transformer_discrete_logic.py`)

The network is given the parse tree: each token's path from the root as an embedding, and an attention bias by
tree distance. It is also given every subtree's value as a training target, weighted 2^-depth. With both, each node
only needs one local step: apply its operator to its children's values. A 5-layer network covers depth-5 trees.

| arm | tree structure given | supervised |
|---|---|---|
| as_written | yes | every node |
| no_structure | no | every node |
| root_only | yes | the answer only |
| plain_root_only | no | the answer only (the plain baseline) |

1 seed first (more if the arms are close), 120 epochs; the majority-class rate is printed as the floor.

| # | prediction |
|---|---|
| L1 | as_written reaches ≥ 95% held-out accuracy (replicates the report) |
| L2 | plain_root_only is < 85% at epoch 120 |
| L3 | the given structure matters more than the node targets: root_only > no_structure at epoch 120 |

### Adelic distance in greedy subset sum (`test_adele_resnet.py`)

The four test cases were built by hand so that the solution's elements share prime factors with the target (50 + 50
= 100, 30 + 30 + 30 = 90) and the decoys do not. Planted instances: 10 numbers in 1..100, and the target is the sum
of a random 2 to 5 of them. 2,000 instances per family.

| family | instances |
|---|---|
| random | all numbers uniform |
| structured | the solution elements are multiples of a factor f ∈ {2, 3, 5, 7, 10} of the target; the decoys are not |

| # | prediction |
|---|---|
| S1 | random: the adelic greedy's success rate is within 5 points of the Euclidean greedy's (no general advantage) |
| S2 | structured: the adelic greedy beats the Euclidean greedy by ≥ 20 points (it helps where the test was built that way) |

**Commands** (from this folder, the repo venv):

    ../../venv/Scripts/python.exe controls_transformers.py grok --seeds 0 1 2
    ../../venv/Scripts/python.exe controls_transformers.py logic --seeds 0
    ../../venv/Scripts/python.exe controls_adelic_search.py

## The math: is it the real adelic / p-adic thing? (2026-10-26; `check_adelic_math.py`, `runs/check_adelic_math.txt`)

The user suspected the "adelic" constructions were cheap proxies (they came from a Gemini session). Checked against
the definitions they are named after, three of the four constructions are proxies. The embedding is the real
thing. My first version of this table said it was not, and that was wrong.

| script | named as | what it is | check |
|---|---|---|---|
| grokking | the adelic embedding A_105 = R/105Z × F_3 × F_5 × F_7 | **genuine.** Z/105 = (1/105)Z/Z sits inside Q/Z = ⊕_p Q_p/Z_p. The p-adic parts of x/105 are nonzero only at p = 3, 5, 7, and there they are x mod p up to multiplication by a unit, which is irrelevant to an embedding table (this is the CRT). The circle (cos, sin)(2πx/105) is the Archimedean image of x/105 in R/Z. So for Z/N the adelic coordinates are the CRT factors plus a circle. Handing them over hands over N's factorization, by construction. The "Hasse principle" label is still empty: the CRT is an isomorphism, and no local-global problem is being solved | real |
| grokking | the Tate global additive character penalty | wrong. A character identity should vanish on every correct answer. The script's phase x/105 − (r3/3 + r5/5 + r7/7) is zero for only 35 of the 105 residues, with mean 1.000 (a random phase). The real identity needs the CRT weights: x/105 ≡ x·(2/3 + 1/5 − 6/7) mod 1, which is zero for all 105. It is also built from argmax, so it has **no gradient**: it never affected training and only adds ≈ 0.5 to the printed loss | wrong and inert |
| discrete logic | a 2-adic ultrametric over the tree | the tree path length d(u, v) = depth u + depth v − 2·depth LCA, squashed by 1 − 2^-d. **It is not an ultrametric** (34.3% of node triples break the strong triangle inequality), so the label is wrong. **But it is a genuine 2-adic object**, which my first version of this row denied. A finite branch sequence names a 2-adic ball (the 2-adic integers starting with those digits), not a point, and the nodes are the balls of Z_2. On this tree of balls (the Bruhat–Tits tree of Q_2) the natural metric is the path length. The ultrametric lives on the tree's infinite ends, and its trace on the nodes is depth(LCA), the Gromov product from the root (tested in `REAL_ADELIC.md`, part 3) | real, mislabelled |
| subset sum | the adele distance \|x\|_∞ · Π_p \|x\|_p^α | by the product formula the real adelic absolute value of any nonzero rational is 1, and at α = 1 the script gives exactly 1 for every base-smooth difference, i.e. no information. At α = 0.5 it is, in closed form, √\|x\| when every prime of x is in the base, and \|x\| × (smooth part)^-½ otherwise: a **smoothness discount** | a heuristic |

### Results: adelic distance in greedy subset sum (`controls_adelic_search.py`, `runs/controls_adelic.txt`)

| family | Euclidean greedy | adelic greedy | shuffled discount (exploratory) | exhaustive |
|---|---|---|---|---|
| random | 30.0% | 35.8% | 28.9% | 100% (1,024 evals vs 31) |
| structured | 27.4% | 38.7% | 24.6% | 100% |

- **S1 refuted, in the adelic greedy's favour:** +5.8 points on random instances (predicted within ±5).
- **S2 refuted:** +11.3 points on structured instances (predicted ≥ 20).
- **Exploratory:** the shuffled discount has the same discount sizes but no relation to divisibility, and it gives
  nothing (−1 to −3 points). So the gain comes from the divisibility.
- **Reading:** the difference it discounts is the next residual. A residual whose primes are all in the base (the
  candidates' and the target's primes) is more likely to be a candidate, or a sum of candidates sharing factors. So the
  smoothness discount is a cheap proxy for one step of lookahead. It is a real but small effect: the greedy still fails
  on about two thirds of instances, and an exact one-step lookahead (n² evaluations) is the honest comparison, not yet
  run. The four hand-built cases overstate it.

## Results: grokking (2026-10-26; `runs/controls_grok.jsonl`, `runs/controls_grok.txt`; 3 seeds, ~20 s per run)

Epoch at which held-out accuracy first reaches 99% (11 optimizer steps per epoch), and mean held-out accuracy:

| arm | seeds 0 / 1 / 2 | mean epoch | at epoch 10 | epoch 20 | epoch 40 | epoch 120 |
|---|---|---|---|---|---|---|
| as_written (adelic embedding + residue heads) | 6 / 10 / 8 | **8** | 99.8% | 100% | 100% | 100% |
| adelic_ce (adelic embedding only) | 9 / 19 / 9 | **12** | 70.2% | 100% | 100% | 100% |
| plain_heads (residue heads only) | 17 / 43 / 26 | **29** | 5.5% | 66.6% | 98.1% | 100% |
| plain_ce (neither: the plain baseline) | 76 / 107 / 82 | **88** | 1.6% | 25.3% | 68.3% | 99.6% |

- **G1 confirmed:** the original reaches 99% by epoch 8 on average.
- **G2 confirmed:** the embedding alone reaches it by epoch 12, so the embedding carries most of the speed.
- **G3 refuted:** I predicted the plain baseline would stay below 50%. It reaches 99% by epoch 88 and 99.6% at
  epoch 120. This small AttnRes + RoPE transformer groks addition mod 105 within the budget on its own. So no arm
  ends with a train/test gap; the difference between arms is how fast they generalize.
- **G4 (exploratory): the residue targets help on their own.** With a plain embedding, adding heads that predict the
  answer mod 3, 5 and 7 cuts the time to 99% about 3× (88 → 29 epochs). This is the "circular/adelic loss" claim in
  its defensible form: supervising the Chinese-remainder factors of the answer speeds up grokking. The mod-3 head is
  at 78% by epoch 10 against the baseline's 33% (chance), so the factors are learned first and the answer follows.
- **What it does and doesn't show:** both speed-ups hand the network the factorization of 105, as input coordinates
  (≈ 10×) or as targets (≈ 3×). It is a clean demonstration that the right coordinates make generalization fast.
  But the coordinates come from the experimenter, and the trick has nothing to factor for a prime modulus. The open
  question is whether a learner can find such a factorization itself.

## Continual learning without replay (`test_symbolic_thousand_brains.py`; `controls_continual.py`, `runs/controls_continual.txt`)

**T1** (stated before the run; written here after): the original's three tasks read three different input features,
and a column's gating key is the normalized profile of which features are nonzero. So the key is a task label read off
the input. If the same three functions all read feature 0, at least one task fails.

| tasks | x² | relu(x) + 2 | 2x + 3 |
|---|---|---|---|
| disjoint features (as in the original) | pass | pass | pass |
| one shared feature | pass | **fail** (test MSE 211) | **fail** (351) |

**T1 confirmed.** With a shared input, every committed column passes the gate and the vote returns task 1's
answer. "No forgetting" here is parameter isolation: each task's columns are frozen and selected by a label. That is
a known way to avoid forgetting (progressive networks, PackNet and similar). Its price is that tasks never share
anything, so no transfer, and the network grows with the number of tasks. Of course a pure function of the input
can't serve three conflicting tasks without some context cue. The point is that the hard part of continual learning,
sharing structure across tasks without interference, isn't tested here.

## Results: discrete logic (2026-10-26; `runs/controls_logic.jsonl`, `runs/controls_logic.txt`; seed 0, ~5 min per arm)

Majority-class floor 51.7%. Held-out accuracy on the expression's value:

| arm | tree structure | supervised | epoch 10 | epoch 30 | epoch 120 | first ≥ 95% | train at end |
|---|---|---|---|---|---|---|---|
| as_written | given | every node | 68.5% | 100% | **100%** | 22 | 99.8% |
| no_structure | — | every node | 66.0% | 88.5% | **99.2%** | 36 | 99.0% |
| root_only | given | answer only | 66.3% | 69.5% | **72.2%** | — | 92.0% |
| plain_root_only | — | answer only | 65.6% | 66.5% | **66.5%** | — | 90.8% |

- **L1 confirmed:** the original reaches 95% by epoch 22 and 100% by epoch 30.
- **L2 confirmed:** the plain baseline stalls at 66.5% held-out, with 90.8% train, so it overfits.
- **L3 refuted:** I predicted the given tree structure would matter more than the node targets. It is the other way
  round. With the node targets and no structure the network reaches 99.2%; with the structure and no node targets,
  72.2%. The supervised subtree values carry the result. The tree-distance bias adds speed (22 vs 36 epochs to 95%,
  one seed) and +5.7 points without the targets.
- **Reading:** a subtree's value is an intermediate result of the computation. Supervising every one of them turns
  a deep composition into one local step per node, and the transformer then finds which tokens are a node's
  children by itself. This is process supervision against outcome supervision. Without it, both answer-only arms
  memorize: train above 90%, held-out below 73%.
- The arms are 27 or more points apart at the end, so more seeds would not change the verdicts. The epoch-to-95%
  gap between the top two arms is a single seed.
