# The real adelic versions

**Why (the user, 2026-10-26):** the "adelic" math in Gemini's scripts was cargo-culted (`CONTROLS.md`, "The math").
Replace it with the genuine constructions and run the experiments that were meant. The originals are frozen in
`gemini_original/`. The live files keep their names. The shared math is in `adelic.py`, checked against brute force
by `check_adelic_toolkit.py`.

The three claims, and what the real version of each is:

| claim (summary's "Lessons") | Gemini's construction | the real construction |
|---|---|---|
| adelic distance helps search | \|x\| · Π_p \|x\|_p^½, a smoothness discount (the real adelic norm is 1 on Q*) | **local-global counting**: score a move by the estimated number of ways to finish, as the product of the local densities over all places (Hardy–Littlewood / Siegel). A zero local density proves that no completion exists |
| an adelic loss groks modular arithmetic | CE on separate residue heads, plus a "Tate" term that is wrong and has no gradient | **adelic metric losses on the one output distribution**: the expected circle distance (the real place) and the expected p-adic distance at each p dividing N, between the prediction and the answer. The local predictions are marginals of the global one, so they agree by construction. The product formula then holds automatically, and no consistency penalty is needed |
| 2-adic structure helps Boolean logic | an attention bias by tree path length, labelled "ultrametric" | Gemini's bias is genuine but mislabelled: a node's branch sequence names a 2-adic ball, and the path length is the metric on the tree of balls (the Bruhat–Tits tree of Q_2). **The ultrametric the label promised** lives on the tree's ends; on nodes it is the 2-adic valuation of the common part, v = depth(LCA(i, j)) (the Gromov product from the root). The real arm uses the bias β · v, so a node favours its own subtree. Both are tested. The 2^-depth loss weights are the balls' Haar measure, so they stay |

The embedding in the grokking script was already genuine (`CONTROLS.md`), so it stays, generalized to any N.

## Bottom line (2026-10-26; 11 predictions: 7 confirmed, 4 refuted; plus one exploratory arm)

| claim | real version's result |
|---|---|
| adelic distance helps search | **yes, through the p-adic places:** local densities mod small prime powers lift greedy subset sum from 30% to 95% on small instances (Gemini's proxy: 36%), by proving dead ends and measuring the room left. The real place hurts. The gain fades as the numbers outgrow the moduli (25% for numbers up to 1,000, 0% up to 10^6), and at these sizes the exact integer dynamic program is cheaper |
| an adelic loss groks modular arithmetic | **no:** the real adelic loss (circle and p-adic distances on the output) speeds grokking only 1.1–1.5×. **The adelic coordinates do:** 6.7× at N = 105, at least 6.6× at N = 81 (3-adic digits, carries included), and nothing at the prime 97, which has no structure to expose |
| 2-adic structure helps Boolean logic | **only on top of the node targets:** answer-only, both 2-adic geometries memorize (72–74%, against 66% without). With node targets the ultrametric bias is the fastest arm (95% at epoch 14, one seed) |

Across the three, the real math helps when it exposes structure the task actually has: residues of a composite
modulus, local solvability of a sum, the subtree of a node. It does nothing where that structure is absent (a prime
modulus) or when it only reshapes the loss.

## 1. Search: local-global counting (`test_adele_resnet.py`)

Greedy subset sum. One element per layer, overshoot never allowed (as in Gemini's version). Each arm scores a
move c by the residual r' = r − c it leaves, with the pool R' it leaves:

| arm | score of a move |
|---|---|
| euclid | −\|r − c\| (Gemini's baseline) |
| smooth | Gemini's "adele distance" (smoothness discount) |
| arch | the real place: log normal density of a random subset's sum of R' at r' (mean ΣR'/2, variance Σc²/4); r' > ΣR' is impossible |
| padic | the p-adic places: Σ_m log(m · Pr[random subset's sum of R' ≡ r' mod m]), with m = 64, 27, 25, 49, 11, 13, 17, 19, 23, 29, 31 (one power per prime ≤ 31, at most 64). Exact, by a dynamic program over Z/m. Zero means no completion exists |
| adelic | arch + padic: the local-global estimate of the number of completions |

Families (planted: the target is the sum of a random subset):

| family | instances | numbers | target = sum of |
|---|---|---|---|
| small | 2,000 | 10 in 1..100 | 2–5 of them (the instances from `CONTROLS.md`, exactly) |
| structured | 2,000 | 10; the solution's elements share a factor with the target (from `CONTROLS.md`) | 2–5 |
| dense | 500 | 20 in 1..1,000 | 5–15 (hundreds of solutions) |
| hard | 500 | 24 in 1..10^6 | 6–18 (density 1.2: a few solutions) |

Check: euclid and smooth must reproduce `CONTROLS.md` on small and structured (30.0% / 35.8%; 27.4% / 38.7%).

| # | prediction (2026-10-26, before running) |
|---|---|
| Q1 | small: adelic ≥ 60% |
| Q2 | dense: adelic ≥ 90%, euclid ≤ 50% |
| Q3 | hard: adelic beats euclid by ≥ 10 points |
| Q4 | the product beats each factor: adelic > arch and adelic > padic by ≥ 2 points, in at least 3 of the 4 families |
| Q5 | structured: adelic ≥ smooth (the real thing at least matches the proxy on the proxy's home ground) |

    python test_adele_resnet.py

### Results (2026-10-26; `runs/real_search.txt`; CPU, under a minute)

Success rate (exact hit), and in brackets milliseconds per instance:

| family | euclid | smooth (Gemini) | arch | padic | adelic |
|---|---|---|---|---|---|
| small | 30.0% | 35.9% | 20.2% | **94.9%** (6.9) | 93.4% (7.2) |
| structured | 27.4% | 38.6% | 18.1% | **93.7%** (7.2) | 93.1% (7.5) |
| dense | 5.4% | 5.4% | 2.0% | **25.4%** (36) | 20.0% (39) |
| hard | 0% | 0% | 0% | 0.2% (52) | 0% (57) |

The four hand-built cases: padic and adelic solve all four, smooth all four, euclid one, arch none.
Check passed: euclid and smooth reproduce `CONTROLS.md` exactly (600 and 717 of 2,000 on small; 548 and 773 on
structured; the 35.8% there and 35.9% here are both 717/2,000, rounded differently).

- **Q1 confirmed:** small, adelic 93.4% (predicted ≥ 60%).
- **Q2 refuted:** dense, adelic 20.0% (predicted ≥ 90%); euclid 5.4% (≤ 50%, as predicted).
- **Q3 refuted:** hard, every arm about 0%.
- **Q4 refuted, and the reverse holds:** the product never beats the p-adic factor alone. Adding the real place costs
  0.6 to 5.4 points, and the real place alone is worse than plain greedy (20.2% against 30.0% on small).
- **Q5 confirmed:** structured, adelic 93.1% against smooth 38.6%.

**Reading.**
- **The p-adic local densities are what helps:** 30% → 95% on small instances. They work two ways. A zero density
  proves that no completion exists, so dead-end moves are never taken. A non-zero density says how much room is left
  mod 64, 27, 25, 49, …. This is the user's original intuition, that p-adic structure gets greedy out of its trap,
  but the mechanism is local solvability, not smoothness.
- **The real-place factor hurts.** The normal density prefers residuals near half the remaining sum, a balancing
  move. With few elements left the approximation is poor, and greedy has to finish exactly.
- **It fades as the numbers grow:** 95% for numbers up to 100, 25% up to 1,000, 0% up to 10^6. The moduli stop at
  64, and for a large pool each modulus sees a nearly uniform distribution until the pool is small. On hard
  instances (a few solutions among 16 million subsets) no greedy finishes without backtracking.
- **As a practical method it doesn't pay at these sizes.** The exact dynamic program over the integers costs
  O(n · target), cheaper than the scoring on small and dense instances, and it always succeeds. The local-global
  score would matter only for targets too large for that program, and there greedy fails.
- **Not run (a next step):** the obstruction certificates are exact, so they prune soundly inside a backtracking or
  beam search. That is where local solvability should earn its cost on hard instances.

## 2. Grokking: adelic metric losses (`test_transformer_grokking.py`)

Addition mod N with Gemini's model, optimizer, split (half of all pairs) and 120 epochs. Six arms, each on three
moduli: N = 105 (squarefree, three places), N = 81 = 3^4 (one place with four p-adic digits, where the p-adic
filtration is non-trivial), N = 97 (prime: the only finite place is 97 itself, so there is no local structure to
use; this is the falsification check).

| arm | embedding | loss |
|---|---|---|
| ce | plain table | cross-entropy |
| arch | plain | CE + E_P[circle distance] (the real place) |
| padic | plain | CE + Σ_{p\|N} E_P[p-adic distance] (the finite places) |
| adelic | plain | CE + both |
| emb_ce | adelic coordinates (circle + p-adic digits) | CE |
| emb_adelic | adelic coordinates | CE + both |

Distances are rescaled to [0, 1] and weighted 1. The p-adic distance between classes a, b mod N at p is
p^-min(v_p(a − b), k). For N = 97 it is 0 if equal and 1 otherwise. 3 seeds (initialization and batch order). The
measure is the epoch at which held-out accuracy first reaches 99%. A run that never reaches it counts as 120 in a
mean (this understates any speed-up over a slow arm) and is reported as such.

| # | prediction (2026-10-26, before running) |
|---|---|
| Ga | the p-adic loss speeds grokking ≥ 1.5× over ce at N = 105 and N = 81 (real local structure), and < 1.2× at N = 97 (none) |
| Gb | the full adelic loss is at least as fast as the faster of its two parts, at every N (within 10%) |
| Gc | the adelic embedding speeds grokking ≥ 5× over ce at N = 105 and N = 81, and < 2× at N = 97 (where it is a lookup table plus a circle) |
| — | the circle loss alone (arch): no prediction, exploratory |

Sanity check (not a prediction): ce at N = 105 should land in the controls' plain-baseline range (76–107 epochs), and
emb_ce at N = 105 in the adelic-embedding range (9–19).

    python test_transformer_grokking.py --moduli 105 81 97 --seeds 0 1 2

## 3. Boolean logic: the 2-adic ultrametric bias (`test_transformer_discrete_logic.py`)

Gemini's data (checked identical: tokens, targets, depths, paths, path-length matrix), model and training. Padding
is masked in every arm here; Gemini's version gave padding the strongest bias by accident.

| arm | bias | address embedding | supervised |
|---|---|---|---|
| real_root | +\|β\| · depth(LCA(i, j)) (the ultrametric's trace on nodes) | yes | answer only |
| geo_root | −\|β\| · (1 − 2^-path length) (Gemini's tree metric) | yes | answer only |
| none_root | none | no | answer only |
| real_nodes | the ultrametric | yes | every node (weights 2^-depth, the balls' Haar measure) |

The ultrametric trace treats a node's whole subtree alike: a node and all its descendants share v = its depth.
The '=' token sits at the root's address, so its valuation with every node is 0. Seed 0. The controls'
answer-only arms were 72.2% (path-length bias, padding unmasked) and 66.5% (none).

| # | prediction (2026-10-26, before running) |
|---|---|
| La | real_nodes ≥ 95% held-out (with node targets the bias hardly matters) |
| Lb | real_root ≤ geo_root + 3 points at epoch 120: the ultrametric singles out no child, so it helps bottom-up evaluation no more than the tree metric |
| Lc | every answer-only arm < 85% at epoch 120: the node targets, not the geometry, carry the result |

    python test_transformer_discrete_logic.py --arms real_root none_root geo_root real_nodes

### Results (2026-10-26; `runs/real_grok.jsonl`, `runs/real_grok.txt`; 54 runs, 8–15 s each)

Epoch at which held-out accuracy first reaches 99%, per seed (— = not within 120 epochs). The mean counts — as 120;
speed-up is ce's mean over the arm's mean.

| N | ce | arch | padic | adelic | emb_ce | emb_adelic |
|---|---|---|---|---|---|---|
| 105 | 74 / 99 / 74 → 82 | 82 / 96 / 82 → 87 (0.95×) | 52 / 47 / 65 → 55 (**1.51×**) | 43 / 61 / 73 → 59 (1.40×) | 9 / 19 / 9 → 12 (**6.7×**) | 9 / 23 / 12 → 15 (5.6×) |
| 81 | — / — / — → 120 | 113 / — / — → 118 | 113 / — / — → 118 | 90 / — / — → 110 | 15 / 22 / 18 → 18 (**≥ 6.6×**) | 21 / 29 / 26 → 25 (≥ 4.7×) |
| 97 | 107 / 92 / — → 106 | 87 / 69 / 108 → 88 (1.21×) | 92 / 82 / 107 → 94 (1.14×) | 82 / 69 / 98 → 83 (1.28×) | 105 / — / — → 115 (0.92×) | 109 / 98 / — → 109 (0.98×) |

Mean held-out accuracy at epoch 120 for N = 81: ce 95.1%, arch 96.9%, padic 96.7%, adelic 96.3%, emb_ce 99.9%.
Sanity check passed: ce at N = 105 (74–99) sits in the controls' plain range (76–107), and emb_ce (9 / 19 / 9)
matches the controls' adelic-embedding arm exactly.

- **Ga refuted:** the p-adic loss gives 1.51× at N = 105 (just at the bar), but nothing at N = 81 (1.02×, where ce is
  censored and the end accuracies are within 2 points). At N = 97 it gives 1.14×, as predicted (< 1.2×).
- **Gb confirmed:** the full adelic loss is within 10% of, or faster than, its faster part at every N.
- **Gc confirmed:** the adelic embedding gives 6.7× at N = 105 and at least 6.6× at N = 81, and nothing at N = 97
  (0.92×). There it is a lookup table plus a circle.
- **Exploratory, the circle loss alone:** no help at N = 105 or 81 (0.95×, 1.02×), a mild one at N = 97 (1.21×).

**Reading.**
- **The real adelic loss does not make grokking fast.** Scoring the output distribution by its circle and p-adic
  distances to the answer speeds it up 1.1–1.5×. Gemini's separate residue heads did better (3× at N = 105,
  `CONTROLS.md` G4). A separate head forces the hidden state to encode each residue linearly; a metric loss only
  reshapes the output distribution. So the "circular/adelic loss" claim does not survive in its real form.
- **The real adelic coordinates do, wherever N factors.** For N = 81 the coordinates are the four base-3 digits of x
  (the 3-adic expansion): addition with carries, which is genuinely p-adic structure, not the squarefree CRT case.
  The network learns it in 18 epochs, where the plain table does not get there in 120. For the prime 97 the
  coordinates contain nothing beyond the number itself, and the gain vanishes. The speed comes from structure
  in the task that the coordinates expose. A prime modulus has none to expose.
- **Adding the loss on top of the coordinates slows things slightly** (12 → 15, 18 → 25 epochs).

### Results (2026-10-26; `runs/real_logic.jsonl`, `runs/real_logic.txt`; seed 0, ~5 min per arm)

Majority class 51.7%. Held-out accuracy on the expression's value:

| arm | epoch 10 | epoch 30 | epoch 120 | first ≥ 95% | train at end |
|---|---|---|---|---|---|
| real_nodes (ultrametric, every node) | 79.8% | 100% | **100%** | **14** | 99.8% |
| geo_root (tree metric, answer only) | 68.6% | 72.1% | **74.0%** | — | 97.0% |
| real_root (ultrametric, answer only) | 67.1% | 70.9% | **72.5%** | — | 95.4% |
| none_root (answer only) | 66.3% | 66.8% | **66.4%** | — | 92.1% |

- **La confirmed:** with node targets the ultrametric bias reaches 95% by epoch 14 and 100% by epoch 30.
- **Lb confirmed:** answer-only, the ultrametric (72.5%) is no better than the tree metric (74.0%).
- **Lc confirmed:** every answer-only arm stays below 85%; all three memorize (train 92–97%).
- Masking the padding moves the tree-metric arm from 72.2% (`CONTROLS.md`, unmasked) to 74.0%, and the no-bias arm
  from 66.5% to 66.4%. So the accidental padding bias was not doing much.

**Reading.**
- **Neither 2-adic geometry solves the task without the node targets.** The tree metric and the ultrametric each add
  6–8 points over no structure (66% → 72–74%), and both still memorize.
- **With node targets, the ultrametric bias is the fastest arm seen so far:** 95% at epoch 14, against 22 for
  Gemini's tree metric (`CONTROLS.md`, padding unmasked) and 36 for no structure. That is one seed and also differs
  in the padding mask, so it is suggestive only. A plausible mechanism: favouring a node's own subtree, the 2-adic
  ball, is what bottom-up evaluation needs once each node's value is a training target.
- So the Boolean-logic claim holds in the real version too, and for the same reason as in the controls: the
  supervised intermediate values carry it. The real 2-adic structure is a modest accelerator on top.
