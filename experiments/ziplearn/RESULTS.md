# ZipLearner — results log

*The experimental record for `DESIGN.md`. Entries are appended, newest last, and never edited afterwards; a
correction is a new entry. Every number here is traceable to a file under `runs/`. The design document keeps only the
summary table below; the reasoning behind each experiment (what is measured, what passes, what refutes) is
pre-registered in `DESIGN.md` §11 before the run.*

## Summary

| experiment | verdict | in one line |
|---|---|---|
| E0 | measured | attention residuals harmless-to-helpful; PoPE's position-free channels destroy training (withdrawn) |
| E1 | mechanism PASS, criterion mis-set | 36/36 rules kept right; exact after one demonstration; two points can't beat 'identity + one exception' |
| E2 | mechanism PASS, criterion mis-set | every composition written exactly by 6 demonstrations; the plain sweep deadlocks, the restarted one is exact |
| E3 | PASS | 17 recurring permutations became named items: 8.7 bits and 1.15 demonstrations vs 31 bits and 2.8 |
| E4 | PASS | the v1 price was not a valid code (mass 1.45); the rate price is, and is the default |
| E5 | PASS | a written prior of boolean gates: 19.7 → 11.0 demonstrations, no harm off-target |
| E6 | PASS | A→B→C→A with no replay: 100% retention, exactly 3 blocks, 36 bits vs 452 |
| E6b | measured | the transformer on the same sequence forgets each rule to below chance |
| E7 | analysis | the written network is the task's own primitives; the library generates a group of order 36 |
| E8 | A, B PASS; C −0.2 bits | library re-described with 3 generators (161 → 121 bits); unseen group elements in 1.3 demonstrations |
| E8b | measured | under attention residuals the Jacobian is the route weights; early-layer J-lens swaps 72–89% → 13–23% |
| E9 | PASS | planning by search over learned actions: 100% of goals, optimal plans, from 3 observations per action |
| E10 | myopic REFUTED; horizon PASS (tie) | a description is an asset: value = per-plan saving × plans to come |
| E10b | PASS | with no-op and noisy actions: price 92% of goals in 2.3 steps; novelty policies 36%/23%, trapped by noise |
| E11 | PASS after a refutation | many-to-one actions need 'identity + edits'; forward search plans with them; a try erases the goal 48% of the time |
| E12 | PASS | commutativity found from learned maps; abelian planner 8.9 bits vs a 112-word table of 1190 bits |
| E13 | analysis | the replica games: interface complete, 5/16 levels by discovery; goal transfer on MultiKey; failures named |
| E15 | PASS | budget: price forgets the least valuable, keeps the best-evidenced (0.84 vs FIFO 0.52); precision step ∝ 1/√n |
| E14 | mechanism confirmed | order matters exactly as much as earlier compression is reused: gap 28 bits by name, 185 by words, 1193 for the transformer |
| E16 | PASS | continual learning of whole two-layer tasks: 100% retention, exactly 3 blocks — blocks do not multiply across layers (OPEN-7) |
| E17 | REFUTED on the letter | relational goals: identify-then-plan 0.51 (identification limit); words matched to the examples directly 0.99 |
| E18 | PASS | an induction circuit written into a RoPE transformer with no training copies patterns at 100% (OPEN-8 first form) |
| E21 | measured | a near-duplicate rule is minted its own block 6.4 observations after it starts — the wait for a refuting input (OPEN-6) |
| E22 | REFUTED here | annealing the budget vs a hard cap: no difference — nothing fuzzy to crystallise in one-hot rules |
| E23 | REFUTED here | macro-actions raise search nodes 192 → 522 in a 720-state map: a cache needs depth (OPEN-10 stays open) |
| E19 | PASS (outcome) | goal keys priced by outcome (confirmed vs refuted): LockPath's door level solved, CollectAll 3/3, 8/16 levels (E13: 5/16) |
| E24 | PASS | the sleep pass: rules keep 2–4 of 9 context cells, ~10× fewer bits, next levels solved 2–3× faster; lossless of the evidence, or it forgets what the planner needs |
| E25 | measured | the trace diagnostic: discovery is 2/3 revisits; errors are missed changes from the 'unchanged' default; pushes too rare; goal keys too specific |
| E26 | measured | looped transformer: untied growth 2→4 gives 13/17 trained and held-out accuracy 0.12–0.20 (vanilla 8/17, 0.03); held-out solved still 0/8 |
| E27 | REFUTED on discovery, 9/16 levels | exploration valued in bits over whole plans; Tetris L0 solved; discovery is coverage-bound, unchanged |
| E28 | PASS | the rules learned on LockPath, written into one looped attention block, reproduce the planner's rollouts 530/530 (509/509 seed 1); the nearest-key default beats 'unchanged' on unknown windows |
| E29 | PASS | learned search: imagined paths enumerated by bits under a learned strategy = BFS until it learns; then room-20 89 vs 818 calls, room 40 solved at oracle 74 (BFS cannot); the 20-bit strategy reads the wall, not the goal |
| E30 | PASS | the depth loop under attention residuals: held-out 1/8 solved (0.22) with a tied core — the first ever; the learned route lets the anchor fade and reads every earlier pass; windowed sources collapse at 2× passes |
| E33 (a) | measured | the discrete machinery on Latin text = a PPM-style blended-backoff model: 1.87 bits/char frozen, 1.82 online at 2.7M chars (xz 2.26); the E24 sleep price is wrong for text, the prequential one keeps every position; transformer/hybrid arms shelved |

## Format

```
### <date> — <experiment id> — <one-line verdict>
command: <exact command>
files:   <where the numbers are>
numbers: <the pre-registered measurements, as measured>
verdict: PASS / REFUTED / INCONCLUSIVE, against the criteria in DESIGN.md §11
changed: <what this changes in the design, if anything, with the section number>
```

## Entries

### 2026-09-20 — E1 — the mechanism works; the pre-registered two-pair criterion was mis-set (REFUTED on the letter)
command: `python experiments/ziplearn/e1.py` (36 rules, 50 instances each, K = 8 demonstrations of 6 digits, CPU, ~10 s)
files:   `experiments/ziplearn/runs/e1/e1.json`; code `ziplearner.py` (the core loop, library v1), `e1.py`
numbers: kept structure matches the expected one for **36/36 rules** (affine 30, shift 5 for a = 1, identity for 1x+0).
         Exact-match on a fresh 6-digit query after 1, 2, … 8 demonstrations: **1.00** at every k, for all 36 rules and
         for the 12 held-out ones. Unseen rows of the written matrix correct after 1, 2, 3, 4 pairs: 0.17, 0.79, 0.98,
         1.00. After the first two pairs with different x: 0.856 mean, 0.68 worst (5x+8); written matrix equal to the
         true rule on all 11 rows: 0.856. Price at the end: 0.23 bits/digit for the kept structure vs 0.88 for the
         table (guessing 3.46). Controls: a random but consistent function — unseen rows 0.01 correct after 6 pairs (no
         extrapolation), price 0.88 = the table's; a random inconsistent mapping — 0.02, price 3.87 > 3.46 ("learned
         nothing"), query exact 0.00.
verdict: REFUTED against §11's letter (100% unseen rows after two distinct pairs), for a reason that is a property of
         two-point evidence, not a defect: when one of the two pairs is a fixed point of the rule (probability ≈ 2/11
         for a ≠ 1), the identity with ONE exception costs 1 + 4.46 = 5.46 bits, less than the affine's two parameters
         (3.46 + 3.32 = 6.78), so the cheapest description after two pairs is legitimately "copy, with one exception".
         The third pair refutes it (0.98) and the fourth settles it (1.00). Every other pre-registered measure passed.
         The criterion should have said "after three pairs"; §11 is left as written and this entry records the error.
changed: (1) §5's price for the affine structure was wrong — the family has 11·10 members, not 11·11, so the second
         parameter costs log₂10, not log₂11. At the wrong price the general permutation (log₂11 + log₂10) undercut the
         affine by 0.14 bits at two pairs and was kept 30/36 times; at the honest price they tie and the simplicity
         tie-break decides. §5 table corrected. (2) `Permutation` and `Affine` now pay an escape when a pair contradicts
         one-to-one-ness (a repeated output), instead of silently storing it. (3) No change to §4 or the library.

### 2026-09-20 — E2 — every composition is written exactly, but three to four demonstrations are needed, not two (REFUTED on the letter)
command: `python experiments/ziplearn/e2.py --n 100` (17 trained + 8 held-out compositions, 100 instances each, K = 8, CPU, ~2 min)
files:   `experiments/ziplearn/runs/e2/e2.json`; code: `PositionPerm`, `PositionIdentity`, `TwoLayer` in `ziplearner.py`, `e2.py`
numbers: exact-match on a fresh query after 1..8 demonstrations, the sweep restarted from every value-layer description —
         trained 0.18 0.42 0.83 0.95 0.99 1.00 1.00 1.00; held-out 0.02 0.19 0.68 0.90 0.97 0.99 1.00 1.00.
         The plain sweep of §6 from the identity start — trained 0.05 0.25 0.45 0.56 0.59 0.61 0.61 0.61;
         held-out 0.01 0.09 0.30 0.43 0.49 0.50 0.51 0.51: it settles within 2 rounds on most instances
         ({'2': 1481, 'None': 209, '3': 10}) but on a WRONG fixed point for the tasks where both layers act,
         and never above ~0.6. What the learner wrote for the held-out tasks after 8 demonstrations: the correct position
         permutation followed by the correct value map in 100/100 instances of all 8 (e.g. rot_left∘negate → "position
         permutation then affine a=4 b=0"; swap_halves∘inc → "then shift b=1"; reverse∘swap_pairs → "then identity").
verdict: REFUTED against §11's letter (≥ 80% held-out exact after ≤ 2 demonstrations; measured 0.20), PASS on the
         mechanism (held-out 0.68 at 3 demonstrations, 0.90 at 4, 0.99 at 6, 1.00 at 7). The criterion ignored an identification
         limit that can be computed in advance: a position's source is ambiguous while another position has shown the
         same digit in every demonstration so far, probability ≈ 5·(1/5)^k per position at V = 5, so all six positions
         are unambiguous with probability ≈ 0.26 after 2 demonstrations and ≈ 0.78 after 3 — the measured curve, a little
         lower because a wrong-but-cheaper description can also win a tie early. The learner is at the limit of equality evidence; only a smaller hypothesis class (named permutations —
         what §7's library discovery is for, E3) can identify faster. Second lesson for the log: compute the evidence
         limit before setting a criterion.
changed: §6 as written (start from identity, alternate) is insufficient: it deadlocks when both layers act, because
         neither layer can be solved while the other is wrong. Built and adopted: start the sweep from every description
         of the value layer (1 + (V−1) + (V−2)·V = 21 at V = 5) and keep the lowest total price; inversion is exact since
         every value hypothesis is a bijection. §6 gets a note; OPEN-3 (does the sweep settle) is answered for this
         domain: the plain sweep settles, on the wrong answer; the restarted sweep is exact.

### 2026-09-20 — E3 — recurring permutations become named items; a named task costs a quarter of the bits and one demonstration (PASS)
command: `python experiments/ziplearn/e3.py` (a stream of 80 tasks from the 25 compositions, K = 8, CPU, ~20 s)
files:   `experiments/ziplearn/runs/e3/e3.json`; code: `NamedPerm`, `PermLibrary`, `TwoLayerWithLibrary` in `ziplearner.py`, `e3.py`
numbers: the library starts with no named position permutation and ends with **17** (every distinct permutation in
         the domain that occurred twice — the five primitives and their pairwise compositions). Tasks whose permutation was
         already named when they arrived (39): position layer **7.48 bits**, exact after **1.15**
         demonstrations on average. Tasks solved from scratch (34): **30.17 bits**, **2.97**
         demonstrations. Price drop per named permutation, before naming → after: mean 22.9 bits,
         minimum 17.2, over 15 items; the pre-registered bar was 5.4.
verdict: PASS. Every pre-registered measure: recurring items are found; the next task with a named item is cheaper by
         more than the price of its entries; demonstrations-to-exact falls from about 3 to about 1, which answers E2's
         identification limit the way §7 said it would — by a smaller hypothesis class that the learner built itself.
changed: nothing in the design. One number to correct in §5/§11: solving a general permutation from scratch costs about
         30 bits here, not log₂(6!) ≈ 9.5, because the learner pays log₂5 for every digit it cannot yet predict and
         needs ~13 digits of equality evidence to settle six sources; 9.5 is what an oracle that already knew the
         permutation would pay. The gap is the cost of identification, and it is exactly what naming removes.

### 2026-09-20 — E4 — the v1 price was not a valid code; the rate price is, and it is now the default (PASS, with one expectation refuted)
command: `python experiments/ziplearn/e4.py` (shift by 3 with exception rate ε ∈ {0, 0.1, 0.3}, and a random mapping; 200 observations, 50 instances; CPU, ~10 s)
files:   `experiments/ziplearn/runs/e4/e4.json`; code: `flag_bits`, `pay`, `PRICE` in `ziplearner.py`, `e4.py`
numbers: (a) validity — the v1 "flat" price implies probabilities that sum to up to **1.45** per observation (a correct
         prediction at ~0 bits AND an exception at a flat 1 bit cannot both be true): not a code. The "rate" price
         (an adaptive two-symbol code for the exception flag, P(exception) = (n_wrong + ½)/(n + 1)) sums to exactly 1.
         (b) bits per observation at n = 200 vs the bound H(ε) + ε·log₂10: ε = 0.1 — flat 0.53 (below the bound: it
         under-charges), rate 0.84 (bound 0.80); ε = 0.3 — flat 1.46, rate 2.00 (bound 1.88); ε = 0 — both ≈ 0.05 and
         falling. (c) reported exception rate under the rate price: 0.098 for ε = 0.1, 0.316 for ε = 0.3, 0.000 for ε = 0.
         (d) kept: the shift in 46/50 (ε = 0.1) and 36/50 (ε = 0.3); the table wins the rest, when early exceptions
         cluster — the table pays only log₂11 for a first-seen noisy pair where the shift pays an escape, a real
         asymmetry, and it fades with n. Random mapping: rate price 3.44 bits/digit ≈ log₂11 = 3.46 ("learned
         nothing", exactly); flat 3.97 (over-charges, as an invalid code does).
verdict: PASS on the pre-registered criteria (per-observation price flat in n, ε reported ≈ 0.10, the shift kept).
         The expectation that the flat price would "grow and eventually prefer the table" was wrong and is withdrawn:
         the table pays the same exceptions; nothing systematic happens, only the early-clustering effect above.
changed: **PRICE = "rate" is now the default for every structure** (digit-level and position-level, through one `pay`
         helper). E1–E3 re-run under it: E1 unchanged (36/36, same curves, noisy TV 3.39 bits/digit); E2 slightly
         better (held-out exact 0.26 / 0.81 / 0.96 / 0.99 after 2 / 3 / 4 / 5 demonstrations); E3 unchanged (PASS,
         named 8.7 bits vs 31.3 from scratch). §8's "change 1" is BUILT. Changes 2 (parameter precision) and 3 (the
         capacity budget) are not: nothing here has a continuous parameter yet, and a budget with no competition for
         it would test only arithmetic.

### 2026-09-20 — E5 — a written prior halves the demonstrations a boolean formula needs and costs nothing off-target (PASS)
command: `python experiments/ziplearn/e5.py` (formulas over 3 bits; 200 tasks per condition; K = 24 demonstrations; CPU, ~15 s)
files:   `experiments/ziplearn/runs/e5/e5.json`; code `e5.py` (self-contained; uses the rate price through `pay`)
numbers: the written gates AND / OR / XOR / NOT, wired as f = NOT? g2(NOT? g1(v_i, v_j), v_k), give 78 distinct
         formulas of the 256 functions of 3 bits. In-class formulas: the table alone reaches the whole truth table after
         **19.7** demonstrations on average (it must be shown every one of the 8 inputs; accuracy after 1 / 4 / 8 / 16
         demonstrations 0.12 / 0.42 / 0.65 / 0.89); with the prior, **11.0** (0.56 / 0.73 / 0.89 / 0.97). Random
         functions (about 30% of which fall in the class by chance): 19.1 without the prior, 17.9 with it — the prior
         helps rather than hurts, because "the nearest formula plus a few exceptions" is a cheap description of a
         function near the class.
verdict: PASS (gain 8.7 demonstrations in class; harm −1.2, i.e. none). Note what the gain is NOT: naming one of 78
         formulas saves only log₂(256/78) ≈ 1.7 bits against a full table. The gain is in the inputs the learner
         never has to be shown — a formula generalises to unseen inputs from partial evidence, a table cannot — which
         is what a prior is for (§10).
changed: (1) the arithmetic harm check named in §11 was not runnable as written — boolean gates cannot even be offered
         to a digit-level library — and was replaced by the off-target check above; §11 gets a note. (2) A real
         defect found on the way, general to V = 2: under the rate price a description and its complement cost the
         same ("wrong every time" is as compressible as "right every time", and the exception carries no extra digit
         at V = 2), so on the first run the learner kept an always-wrong formula (accuracy 0.43 falling). Fix:
         predictions honour the learned exception rate (flip when wrong more often than right), and ties prefer fewer
         exceptions. The price is untouched; only the read-out. Recorded because it will recur wherever V is small.

### 2026-09-20 — E0 — substrate check: attention residuals are harmless-to-helpful; PoPE's position-free channels break training (measured, decision amended)
command: `python experiments/ziplearn/e0.py` (composition task, 3200 steps, seed 0, one run per cell; ~2–3.5 min per cell on the GPU)
files:   `experiments/ziplearn/runs/e0/*.json`, `summary.json`, per-cell logs; code: `AttnRes`, `Model.routes`, `--res`, `--n_zero`, `--json` in `transformers/h1_lid.py`
numbers: trained compositions solved / mean accuracy / final loss (held-out solved is 0/8 in every cell, as expected):
         RoPE + standard residual **8/17**, 0.52, 0.061 (the recorded run: 9/17) · RoPE + attention residuals **9/17**,
         0.60, 0.054 · PoPE + standard **6/17**, 0.46, 0.078 (recorded: 7/17) · PoPE + attention residuals **9/17**,
         0.56, 0.059 · PoPE with 2 position-free channels + standard **0/17**, 0.07, 0.621 · the same + attention
         residuals 2/17, 0.24, 0.299. Learned routes (mean weight per source; sources = embedding, block 1, block 2,
         then the current partial sum), RoPE + attention residuals: L0 before-MLP 0.19 / 0.81; L1 before-attention
         0.37 / 0.63; L1 before-MLP 0.09 / 0.18 / 0.73; L2 before-attention 0.28 / 0.20 / 0.52; L2 before-MLP 0.04 /
         0.03 / 0.15 / 0.78; output 0.05 / 0.03 / 0.14 / 0.79. PoPE + attention residuals: the same shape within 0.1.
verdict: the substrate does not break the gradient arm; attention residuals help slightly in both positional schemes
         (+1 and +3 trained tasks, one seed — not more than noise allows), and the routes are the paper's picture in
         miniature: each layer reads mostly its immediate predecessor, the embedding keeps weight before attention
         sub-layers (0.28–0.37) and almost none before MLPs, the output reads the last block. The position-free
         channels I added to PoPE are REFUTED as a design: a channel with θ = 0 contributes softplus(q)·softplus(k),
         a non-negative term that rewards large keys regardless of position or identity — a magnitude bias, not a
         content match — and it destroys training (0/17). My claim in §15 that "same digit" is a written identity on
         PoPE's magnitudes was wrong: content matching in PoPE rides on the phase cosines and cannot be separated
         from them by a flat channel.
changed: §15 amended — the θ = 0 channels are dropped; attention residuals (one transformer block per AttnRes block)
         are kept; PoPE stays the intended scheme for the written network and RoPE the gradient arm's default, since
         the two arms are reported side by side and the measured difference between the schemes is within one seed's
         noise. The cost of PoPE's inseparability for writing attention weights is now OPEN-8.

### 2026-09-20 — E6 — three rules in sequence with no labels and no replay: 100% retention, exactly three blocks (PASS)
command: `python experiments/ziplearn/e6.py` (A = shift 3, B = shift 5, C = 2x+1, then A again; 40 pairs each; 20 streams; CPU, ~30 s)
files:   `experiments/ziplearn/runs/e6/e6.json` (and `runs/e6_short/` for the 10-pair stress run); code: `ContinualLayer` in `ziplearner.py`, `e6.py`
numbers: retention — every rule seen so far, tested by giving the layer 3 pairs of it as context (learning nothing) and
         reading the selected block's 11 rows — **1.000 for A, B and C after every stretch**, in all 20 streams. Blocks:
         1 → 2 → 3 → **3** (A's return reuses A's block in every stream). Total bits at the end **36.3** against
         **452** for a single table forced to describe the whole stream. One stream's blocks at the end: shift b=3
         (n=80, exceptions 0.00), shift b=5 (n=41, 0.00), affine a=2 b=1 (n=39, 0.00). Stress run with 10-pair
         stretches (transitions every 10 pairs): retention still 1.000 everywhere, blocks 3.05 (one stream in 20 minted
         a fourth at a transition).
verdict: PASS on every pre-registered measure. No example was stored and nothing was replayed: what carried A across
         80 pairs of other rules was its written matrix and its evidence count.
changed: §9's rules survived, but three mechanisms had to be added to make "mint on refutation" work, each found by a
         failure on the first runs (junk blocks at transitions, a young block swallowing the next rule, duplicate blocks
         for a returning rule): (1) minting is a CHANGE-POINT decision — for every split of the recent window the
         price of "an existing block explains the part before, another existing block or a fresh one explains the part
         after" (a change costs log₂V bits, a fresh block also its name) is compared with "one block explains it all";
         the cheapest wins, a fresh block starts from the part after the change, never from a window straddling two
         rules, and a returning rule is routed to its old block. (2) For routing, an ESTABLISHED block (≥ 3 pairs of
         evidence) is priced under its kept description frozen — it cannot quietly become a table to swallow foreign
         pairs; foreign pairs are exceptions to what it says. (3) A YOUNG block (< 3 pairs) is priced with all its
         structures live, so its own rule's next pairs can still correct an early guess. (2)+(3) is rule 1 literally:
         protection in proportion to evidence. §9 gets these three as its concrete form. OPEN-6 (two rules that agree
         on the window) and OPEN-7 (blocks across layers) remain open; the transformer comparison arm is E6b below.

### 2026-09-20 — E6b — the gradient arm on E6's sequence: it forgets everything but the last rule (the number to beat, measured)
command: `python experiments/ziplearn/e6b.py` (the run-2 transformer, 151,499 weights, 400 steps per stretch on A → B → C → A, batch 64, in-context K = 8; GPU, 19 s)
files:   `experiments/ziplearn/runs/e6b/e6b.json`; code `e6b.py`
numbers: per-digit accuracy on the last demonstration (the rule's own seven demonstrations in the context) — after stretch 1:
         A 1.00; after 2: A **0.00**, B 1.00; after 3: A 0.08, B 0.09, C 1.00; after 4: A 1.00, B **0.00**, C 0.08.
         Forgetting, right after a rule's own stretch → at the end: A 1.00 → 1.00 (it was last), B 1.00 → 0.00,
         C 1.00 → 0.08. Chance is 0.09: the forgotten rules are answered *below* chance — the network recites the
         table it learned last, whatever the demonstrations in front of it say. ZipLearner on the same sequence
         (E6): 1.00 / 1.00 / 1.00, three blocks, 36 bits.
verdict: the comparison arm named in E6 is now run: gradient training on this sequence forgets each rule the moment
         the next one starts, and forgets it into a confident wrong answer, not into ignorance — the same signature
         as the anti-generalisation of the inner-objective runs. Nothing about the transformer was tuned to fail
         (the same model, optimiser and format as run 2); nothing about ZipLearner was tuned to succeed beyond the
         three mechanisms recorded under E6.
changed: nothing.

### 2026-09-20 — E7 — anatomy: the written network is the task's own primitives, and the library has a two-generator description it has not used yet
command: `python experiments/ziplearn/anatomy.py` (CPU, seconds)
files:   `experiments/ziplearn/runs/e7/anatomy.json`, `anatomy.png`; code `anatomy.py`
numbers: (1) E2's network, read off: for rot_left∘negate, layer 1 is the position permutation "output j reads input
         [1, 2, 3, 4, 5, 0]" (rot_left itself, 40 right / 0 wrong) and layer 2 is affine a=4 b=0 (negate itself: −x ≡ 4x
         mod 5); for swap_halves∘inc, [3, 4, 5, 0, 1, 2] then shift b=1. The network became, literally, the two
         primitives the task was made of. (2) E3's 17 named permutations generate a group of order **36** inside the
         720 permutations of six positions — the same group the five primitives generate — so the library can express
         19 permutations it never saw as products of ones it did. **17/17** items are products of two other named
         items. The smallest generating set has **2** items (rot_right and reverse∘swap_pairs); the longest word is 8.
         Priced flat, the library costs 161.4 bits; as two generators plus words, **138.1** bits — 23 bits (14%)
         cheaper, and the learner has not taken them, because no structure in the library says "an item is a word in
         other items". (3) E6's blocks on one stream: shift b=3 (evidence 80, exceptions 0, 9.8 bits; a table would be
         44), shift b=5 (40, 0, 9.3), affine a=2 b=1 (40, 0, 12.6). The two shift blocks as one template with a
         per-context offset would cost 17.5 bits against 22.2 for two blocks — rule 3's merge would pay 4.6 bits, and
         again no structure exists to express it. (4) The gradient arm's routes (figure): every attention sub-layer
         reads the embedding at 0.28–0.37; each layer otherwise reads its immediate predecessor; the output reads the
         last block at 0.79.
verdict: n/a (an analysis). Two findings worth stating plainly. First, nothing in the written network is latent:
         every "region" is a named matrix, every wire is a route, and the regions are the primitives of the domain —
         the transformer's clusters in the inner-objective runs never were. Second, OPEN-5 is now concrete and priced:
         the library's next compression step is to describe items as words in a small generating set (23 bits on the
         table), and E6's next step is a template with a per-context parameter (4.6 bits); both need one new kind of
         structure — a structure whose parameters are OTHER structures — which is the same kind in both places.
changed: OPEN-5 gains its numbers and a concrete proposal (E8 candidate: a "word" structure and a "template with a
         context parameter" structure, priced by the same rule; pass = the library and the blocks re-describe
         themselves and the bits fall by at least the amounts above). OPEN-9 added: the J-space of the gradient arm
         under attention residuals (see the entry's discussion in the chat log of 2026-09-20 and §12).

### 2026-09-20 — E8 — structures of structures: the library compresses itself and generalises one level up (A, B PASS; C misses its own letter by 0.2 bits)
command: `python experiments/ziplearn/e8.py` (CPU, ~1 min)
files:   `experiments/ziplearn/runs/e8/e8.json`; code: `WordLibrary`, `TwoLayerWithWords`, `consolidate` in `ziplearner.py`, `e8.py`
numbers: A — the library, described by a generating set chosen by price plus a word for every other item: **120.8
         bits** (flat 161.4; the anatomy pass's hand-found two-generator description was 138.1) with **3**
         generators — swap_pairs∘rot_left, rot_left, reverse — and a longest word of 4; the group they reach has
         36 permutations. Three generators beat two because shorter words are cheaper than a saved entry.
         B — 18 permutations that are in that group but were never named, each with a value map, 20 instances:
         from scratch 2.80 demonstrations to exact and
         30.6 bits; with the named items only
         2.81 and 31.5
         (no item matches, as expected); with words **1.33** demonstrations
         and **11.8 bits**. The learner identifies a permutation it has never
         seen at a quarter of the cost, because it can be written in items it has.
         C — E6's blocks consolidate into templates: shift ×2 contexts + affine ×1; all blocks 36.3 bits
         separate → 31.3 as templates; the two shift blocks 22.2 → 17.7;
         retention with the template description A 1.000 B 1.000 C 1.000.
verdict: A PASS, B PASS, C REFUTED on the letter: the criterion was 17.5 bits and the measurement is 17.7. The 0.2 bits
         is an accounting difference between the anatomy pass's estimate (which named the template's kind among 3
         kinds) and `consolidate` (which names it among the library's 5); the saving is 4.5 bits of the estimated 4.6,
         retention is unchanged, and the point stands — but the letter is the letter, and I set it from an estimate I
         did not check against the code. Overall REFUTED under the pre-registration's "all three".
changed: OPEN-5 is BUILT in its first form — a description whose parameters are other descriptions: `WordLibrary` (a
         library that chooses its own generators by price and writes every item, seen or not, as a word) and
         `consolidate` (same-kind blocks under one name). What is NOT built: using the word description to grow the
         library further (words of words), and letting the template's per-context parameter be *inferred* for a new
         context rather than stored — both are the next compression steps, and both are again "a structure whose
         parameters are structures".

### 2026-09-20 — E8b — the J-space of the gradient arm: under attention residuals the Jacobian IS the route weights, and early-layer J-lens directions lose their causal power (measured; three predictions checked)
command: `python experiments/ziplearn/jspace.py` on two models retrained with `--save` (RoPE, standard residual vs attention residuals, same seed/task/steps; both solve 9/17 trained compositions); GPU, ~1 min. Jacobians in forward mode, exact.
files:   `experiments/ziplearn/runs/e8b/jspace.json`, `jspace.png`, the two models `rope_std.pt`, `rope_attnres.pt`; code `jspace.py`
numbers: per state after block 0 (embedding) / 1 / 2 / 3, standard vs attention residuals —
         identity share trace(J)/d (same position): **0.732 / 0.776 / 0.911 / 1.000** vs **0.059 / 0.041 / 0.115 / 0.765**.
         The attention-residual numbers are the output mixer's learned routes (E0/E8b: embedding 0.05, block 1 0.03,
         block 2 0.14, block 3 0.79) read back through the Jacobian: the identity highway has become a route weight.
         gain ||J||/√d: 0.96 / 0.98 / 1.07 / 1.00 vs 0.08 / 0.07 / 0.16 / 0.77.
         future share (the part of a perturbation's effect carried to LATER positions, i.e. through attention to other
         tokens): 0.03 / 0.03 / 0.02 / 0.00 vs 0.11 / 0.12 / 0.08 / 0.00 — what early
         sources still influence, they influence through attention, not the stream.
         J-space share of activation variance (span of the 5 J-lens vectors): 0.002 / 0.006 / 0.088 / 0.591 vs
         0.015 / 0.108 / 0.045 / 0.648 — in the standard net the mid states are dominated by non-verbalizable
         content (0.6% at block 1); under attention residuals block 1's source is 10.8% verbalizable, because a source
         holds one block's contribution rather than the accumulated sum.
         persistence of a token's J-lens direction to the next state: 0.981 / 0.947 / 0.647 vs
         0.892 / 0.736 / 0.489 — without the identity map, directions do not persist as well.
         swap rate (add a J-lens direction at the predicting position; fraction of predictions that flip to that
         digit): **0.89 / 0.72 / 0.84 / 0.82** vs **0.22 / 0.13 / 0.23 / 0.76** — in the standard net a
         J-lens direction written at ANY depth flips the output 72–89% of the time; under attention residuals only the
         last block's does (76%); the earlier sources' directions flip it 13–23%.
verdict: an analysis; the three predictions made before running (chat log, 2026-09-20): (i) the J-space localises to
         the sources the output reads — CONFIRMED, quantitatively (identity share = route weight); (ii) a local
         workspace inside each block plus a global one across blocks — NOT TESTABLE here (one transformer layer per
         block); (iii) broadcast becomes readable from the weights — PARTLY: the embedding is read by every attention
         mixer (0.28–0.37) and the future share rises from 0.03 to 0.11, but a 3-block, 5-token model cannot show a
         mid-depth workspace over concepts. Caveats: one seed, 3 blocks, a 5-digit vocabulary; the Jacobian structure
         is exact for these models, the analogy to a 100-layer LM is not.
changed: OPEN-9 gets its first numbers. One design consequence for the written network: in a standard residual net
         "what a layer says" is readable at the output from any depth (the highway), which is why J-lens swaps work
         everywhere; under attention residuals a source is heard only where a route reads it, so a written matrix's
         effect is *addressable* — the same property that made §6's pinned interfaces literal.

### 2026-09-20 — E9 — planning in the written map: goals reached with optimal plans from three observations per action, no value function (PASS)
command: `python experiments/ziplearn/e9.py` (7 actions, 200 goals within 3 actions, k = 1/2/3/8 observations per action; CPU, ~20 s)
files:   `experiments/ziplearn/runs/e9/e9.json`; code `e9.py` (learning by `TwoLayer`, planning by breadth-first search over words in the learned actions)
numbers: goals reached / plans as short as the oracle's / mean plan bits / single-action control —
         k = 1: 0.230 / 0.978 / 3.72 / 0.165;  k = 2: 0.945 / 0.751 / 5.99 / 0.365;  k = 3: **1.000 / 1.000 / 4.83** / 0.490;
         k = 8: 1.000 / 1.000 / 4.83 / 0.490. The oracle's mean shortest plan is 1.72 actions (4.83 bits at log₂7 per
         action); 49% of goals are one action away, which is all the no-composition control can reach.
verdict: PASS on the letter (≥ 95% reached at k = 3 with plans no longer than the oracle's; measured 100% and 100%).
         Nothing was learned from reward and no value function exists: the map (§16 item 1) is the learned action
         descriptions, the inverse (item 2) is search over words, the goal (item 3) is a target state, and the price
         of a plan is its bits. The k = 2 row is the interesting one: 94.5% of goals are still reached, but a quarter
         of the plans are longer than necessary — with two observations some actions have unresolved positions, the
         planner treats them as unusable and *routes around its own ignorance* with longer words (5.99 bits against
         4.83). That number, plan bits lost to an undescribed action, is exactly the price OPEN-11 needs: the value
         of trying an unknown action is the plan bits its description would save.
changed: §16's first form is BUILT for the case of a small map and bijective actions. OPEN-10 (caching when the map
         is large) and OPEN-11 (acting to learn) are untouched; OPEN-11 now has its currency.

### 2026-09-20 — E10 — acting to learn: myopic plan-bit value does not explore (REFUTED); the same value counted over the goals still to come does (PASS, by a margin inside stream noise)
command: `python experiments/ziplearn/e10.py --streams 5 --goals 40` (E9's environment; 4 actions known at start, 3 never seen; up to 2 tries per goal; 6-step budget; CPU, ~6 min, run detached)
files:   `experiments/ziplearn/runs/e10/e10.json`, `full.log`; code `e10.py`
numbers: goals reached / mean steps to goal (failures = 6) / goals until all seven actions are described / tries spent —
         none 0.620 / 3.56 / never / 0;  random 0.850 / 4.50 / 13.2 / 77.8;  least observed (2 tries every goal) 0.830 /
         4.62 / 4.6 / 76.6;  least observed until all are known 0.910 / 2.30 / 3.6 / 6.6;  **price, this goal only
         0.705 / 3.17 / never / 2.6**;  **price, horizon 0.955 / 2.17 / 8.4 / 7.4**. (A 3-stream × 30-goal run beforehand:
         price-horizon 0.878 / 2.56 against until-known 0.933 / 2.23 — the ordering of those two flips between runs.)
verdict: "price (this goal)" — the pre-registered policy — is REFUTED: the expected saving of a never-seen action on
         the goal in front of it, over random-permutation hypotheses, almost never exceeds the try's own 2.8 bits, so it
         tries 2–3 times in 40 goals and never learns the three unknown actions. "price (horizon)" PASSES the letter
         (more goals and fewer steps than random, than fixed-budget novelty and than novelty-with-a-stopping-rule) —
         but the margin over novelty-with-stopping (0.045 goals, 0.13 steps) is inside the run-to-run noise at 5
         streams, and I report it as a tie in substance. The robust findings: (1) a description is an ASSET — its
         value is its per-plan saving times the plans that will use it, and counting only the current plan makes an
         agent that will not learn; (2) in a world where every unknown action is useful, "try what you do not know,
         then stop" is as good as any price, and the price only earns its keep where trying is expensive or some
         actions are useless or noisy — that is the discriminating test, not this one.
changed: OPEN-11 keeps its currency (plan bits) with one correction: multiplied by the horizon. Also found in
         passing: an action never observed must be treated as having NO description, not the cheapest one — the
         first run described never-seen actions as the identity (zero evidence, zero cost) and planned as if they did
         nothing. Next test, not built: E10b with useless (no-op) and noisy (undescribable) actions, where the price
         should stop trying what the rate price has shown to be noise and novelty-with-stopping cannot.

### 2026-09-20 — E10b — with useless and noisy actions, price is the only policy that both learns what matters and leaves noise alone (PASS)
command: `python experiments/ziplearn/e10b.py --streams 5 --goals 40` (E10's environment + 2 no-op + 2 noisy actions; 4 real actions known at start; up to 2 tries per goal; 6-step budget; CPU, ~8 min, detached)
files:   `experiments/ziplearn/runs/e10b/e10b.json`, `full.log`; code `e10b.py`
numbers: goals reached / mean steps (failures = 6) / tries spent on real, no-op, noisy actions / real actions described at the end —
         none 0.620 / 3.56 / 0, 0, 0 / 3.8;  random 0.585 / 5.00 / 48, 13, 16 / 7.0;  least observed until known (any
         description) 0.360 / 5.01 / 7, 2, **57** / 6.6;  least observed until known (exception rate < 0.5) 0.230 /
         5.65 / 9, 2, **69** / 7.0;  **price (horizon) 0.920 / 2.30 / 7, 2, 2.4 / 6.8**.
verdict: PASS on every pre-registered clause, and this time not within noise: the price policy reaches 92% of goals in
         2.3 steps; the novelty policies reach 36% and 23% in ~5 steps because they spend 57–69 tries on the two
         actions whose effect is random (the ε-aware one worst of all, since it can never call noise "known"); random
         tries do worse than not trying (0.585 vs 0.620). The price policy tries each noisy action about once (2.4
         tries across both), sees its description refuted, prices its expected saving at zero and never touches it
         again; it tries each no-op once, describes it as the identity, and never needs it; it learns the three
         useful unknown actions (7 tries) and stops. The mechanism is the rate price of E4 doing double duty: the
         exception rate that says "learned nothing" (noisy TV) is the same number that says "not worth trying".
changed: OPEN-11 is BUILT in its first form: the value of trying an action = its expected plan-bit saving over the
         goals still to come, over the hypotheses its evidence allows — none, once its best description is refuted more
         often than not — minus the try's cost. Together with E9/E10 this closes §16's first loop: describe → plan →
         act → the plan's failures and the price of ignorance decide what to try next. Not built: OPEN-10 (a large
         map), OPEN-12 (goals that are relations), actions that are not bijections (the learned inverse of §16 item 2).

### 2026-09-20 — E11 — non-bijective actions: refuted with a plain map, passed with "identity plus edits"; a try of a many-to-one action erases the goal half the time
command: `python experiments/ziplearn/e11.py` (E9's seven actions + copy01, set0, fill0; 200 goals; 20 learners per (action, k); CPU, ~2 min; run three times, see below)
files:   `experiments/ziplearn/runs/e11/e11.json` (final), `e11_first_run_map_only.json`; code: `PositionMap`, `PositionEdit` in `ziplearner.py`, `e11.py`
numbers: run 1 (library + a many-to-one position MAP only): the many-to-one actions were exactly described after 4
         observations in **0.51** of learners (0.87 at 8); copy01 and set0 were kept as "position identity" — the
         identity with a systematic exception at one slot was cheaper than the map, which pays log₂5 for every slot
         until each is resolved. REFUTED (A 0.51 < 0.8). Run 2, with `PositionEdit` (the identity with per-slot edits: a
         slot costs nothing while it reads itself, one exception when contradicted, then it is solved like a map slot):
         many-to-one exact **0.983** at k = 4, planning reached 0.875 at k = 4 — and fill0 was being kept as a "position
         permutation" that read slot 0 six times: `PositionPerm` never checked that two settled slots cannot share a
         source. Run 3, with that fixed: A — one-to-one 0.29 / 0.64 / 0.93 / 0.97 / 1.00 and many-to-one 0.15 / 0.67 /
         0.89 / **0.98** / 0.99 exact at k = 1 / 2 / 3 / 4 / 8; B — goals reached 0.13 / 0.62 / 0.89 / **1.00** / 1.00 with
         plans as short as the oracle's 1.00 / 0.86 / 0.99 / **1.00** / 1.00 (84 of the 200 goals need a many-to-one
         action in every shortest plan); C — a try of an unknown one-to-one action never makes the goal unreachable
         within the budget; a try of an unknown many-to-one action does so **0.48** of the time.
verdict: PASS on the final run (A 0.983 ≥ 0.95; B 1.000 reached, plans optimal), after a refutation and a bug. k = 4
         is at the edge for this world (0.875 reached on run 2 with the same criteria), so the pass is not a wide one.
         The design claim survives: nothing in the loop needed an inverse — forward search over learned effects plans
         with many-to-one actions as it did with permutations — and the cost of non-bijectivity shows up exactly
         where §16 said it would, in acting to learn: a try that cannot be undone is not worth one step, it is worth
         one step plus the goals it erases (48% here), and E10's price does not yet charge for that.
changed: (1) library: `PositionMap` (any source, repeats allowed, or a constant) and `PositionEdit` (identity with
         per-slot edits) added; the edit structure is the second time "generalise then correct" had to be made
         slot-wise because the rate price treats exceptions as exchangeable — an exception that recurs at the same
         place is structure, and a structure has to exist to say so. (2) `PositionPerm` refuses two settled slots with
         the same source (a real bug; it could not affect E2/E3, whose actions are all permutations, but it labelled
         fill0 wrongly). (3) OPEN-11 gains a term: the price of a try must include the expected plan bits from the
         state it leaves you in, not only the step. Not built.

### 2026-09-20 — E12 — commutativity is found from the learned maps, and it is worth 130× in description: two numbers instead of a table of 112 words (PASS)
command: `python experiments/ziplearn/e12.py` (two sets of four actions, 4 observations each, 200 goals per set; CPU, ~1 min)
files:   `experiments/ziplearn/runs/e12/e12.json`; code `e12.py`
numbers: A — from the learned maps alone, commuting pairs: abelian set **6/6**; non-abelian set **4/6**, the two that
         fail being exactly rot_left/reverse (0.00 agreement) and inc/negate (0.00). B — words of length ≤ 3 with the
         same letter counts have the same effect: abelian **1.00** (22 distinct effects), non-abelian **0.71** (35).
         C — search over words: both sets 100% reached, 100% optimal, 8.8 vs 11.8 states examined per goal. The
         coordinate planner (abelian): identify the rotation and shift the goal needs, then the plan is "i rotations
         one way or 6 − i the other, j shifts or 5 − j": **100% reached, 100% optimal**, planner description
         **8.9 bits** (two generators and two cycle orders, found from the maps: rot_left has order 6, inc order 5).
         The word-table planner (non-abelian): 112 distinct transformations within six letters, each with its
         shortest word: 100% reached, 100% optimal, **1190 bits** of table.
verdict: PASS on every clause. What non-commutativity costs is not the learning of actions (E9/E11: a map is a map)
         and, at this depth, hardly the search (8.8 vs 11.8 states) — it is the *summary*: an abelian set of
         actions is described by counts, a vector that adds, and a plan is arithmetic on it; a non-abelian set has
         no such coordinate, every reachable transformation is its own word, and the planner is a table 130× larger.
         The parallel with E11 holds: non-bijectivity loses invertibility, non-commutativity loses summarisability;
         in both the learner keeps the whole object (map, word) instead of a summary (inverse, counts). Honest
         limit: my coordinate planner still enumerates the 30 (i, j) pairs to identify the needed transformation
         (12.6 checks per goal, more than search's 8.8); the arithmetic saving would need the transformation read
         off the goal directly (where did a marked digit go?), which is not built — the bits claim stands, the
         speed claim is not demonstrated.
changed: "these actions commute" is now a measured, cheap property of a learned library (40 random states per pair)
         and the precondition for a coordinate description — the first structure-of-structures (OPEN-5) that is
         about the *relations between* items rather than words in them. Not built into the library; E12 computes it
         outside. OPEN-10 gains a candidate: cache coordinates where actions commute, words where they do not.

### 2026-09-20 — E13 — first contact with the replica games: the interface runs all six end to end; 5 of 16 levels solved by discovery, with the failure modes named (an analysis, not a verdict)
command: `python experiments/ziplearn/play_games.py --budget 150` (six games, every level, 150 actions per level, no game knowledge; CPU, seconds)
files:   `experiments/ziplearn/runs/e13/play.json`; code `arcgames.py` (the interface and the first player), `play_games.py` (the harness with the oracle)
numbers: levels solved / actions on solved levels (oracle shortest) / world-model prediction accuracy —
         LockPath 1/4 (L0 in 87, oracle 8) / 0.84;  MultiKey **2/2** (77 then **49**, oracle 9 and 13) / 0.59;
         Sokoban 0/3 / 0.47;  CollectAll 1/3 (80, oracle 8) / 0.77;  Toggle 1/1 (71, oracle 7) / 0.56;  Tetris 0/3 / 0.00.
         Total 5/16. Every action of every game was exercised; every model kept radius 1 on price.
verdict: none pre-registered beyond "the interface is complete: the only thing between the agent and a level is its
         ability to learn the mechanics and to solve the level, and the two are reported separately" — that holds.
         What the numbers say: (1) first levels are solved by *discovery* — novelty-directed search over predicted
         frames walks the room until the score rises; 70–90 actions against an oracle of 7–9, which is the cost of not
         knowing what winning is; (2) the goal model transfers: MultiKey's second level was reached in 49 actions with
         a single goal-directed plan, the win keys learned on level 0 ("this local arrangement, this action, scored")
         pointing the planner straight at it; (3) LockPath's key-and-door level fails because the win keys are
         location-free and fire before the door is dealt with — the planner walks to the door and pushes into it
         for the rest of the budget: the goal predicate needs a *condition* (the door is closed) the local window does
         not contain, which is E11's many-to-one lesson again at the goal level; (4) Sokoban's world model is 47%
         right — pushes need the radius-2 rule, and the price never got enough push observations to prefer it;
         (5) Tetris is 0% predicted: the board changes every tick whatever the action, and a window never seen
         predicts "unchanged", the wrong default for a world where everything moves.
changed: the interface exists and is the fixed point from here: `arcgames.py` turns a frame into a state, the game's
         actions into actions, and the score / WIN / GAME_OVER signals into the goal model; the world model is one
         local rule per action (§5's tied blocks, the rate price choosing the window radius); planning is §16's
         search; exploration is the price of ignorance. One correction on the way: goal keys must never be "every
         window of the frame" (the first run's fallback when the winning transition's change was unobservable) —
         they are the windows around the cells that changed, or were predicted to, or changed just before. Next, in
         order of what the numbers point at: a goal predicate with conditions (OPEN-12), the push rule (radius chosen
         by price needs the observations exploration should be buying — E10's price would go looking for them), and
         a default for unknown windows that is learned rather than "unchanged".

### 2026-09-20 — E15 — under a budget the price rule forgets what is worth least and keeps what it has most evidence for; a parameter's precision follows 1/√n (PASS)
command: `python experiments/ziplearn/e15.py` (A: six rules, A seen 60 times, the rest 20, budget 48 bits, 5 streams; B: 200 random offsets × 5 sample sizes; CPU, ~20 s)
files:   `experiments/ziplearn/runs/e15/e15.json`; code: `enforce_capacity`, `block_value`, `precision_bits`, `ContinualLayer.total_bits(consolidated=True)` in `ziplearner.py`, `e15.py`
numbers: A — no budget: 62.9 bits, every rule retained. FIFO under 48 bits: 44.9 bits, retention
         A 0.09 B 0.27 C 0.64 D 1.00 E 1.00 F 1.00, evidence-weighted **0.523** — it drops A first (the oldest, and the
         one with three times the evidence). Price under 48 bits: **41.1 bits**, retention A 1.00 B 0.80 C 0.82 D 0.82 E 0.27 F 1.00,
         evidence-weighted **0.839**; it consolidates the shifts into one template and drops
         one affine block with 21 pairs of evidence. B — the grid step that minimises the expected total code length
         for an offset estimated from n noisy observations: n = 4 → 1.56, 16 → 0.91, 64 → 0.49, 256 → 0.23,
         1024 → 0.11, against σ·√(12/n) = 1.73, 0.87, 0.43, 0.22, 0.11; slope of log step against log n **−0.48**
         (the rule predicts −0.5).
verdict: PASS on both. A's fine print: the kept rules B, C, D retain 0.80–0.82 rather than 1.00 under the price rule —
         the selector occasionally routes a three-pair context to the wrong surviving block once a neighbour is gone,
         a cost of forgetting that is real and small; the dropped rule E keeps 0.27 (it is answered by whichever block
         prices its context least badly). B's constant: the best step is σ·√(12/n), not σ/√n — the √12 is the
         uniform quantisation error — and `precision_bits` now says so (½·log₂n + log₂(span/σ) − ½·log₂12).
changed: §8 changes 2 and 3 and §9 rule 3 are BUILT in their first form. The "annealing" reading of §8 (raise λ
         gradually so structures crystallise) is not built: the budget here is a hard cap enforced after each stretch.

### 2026-09-20 — E14 — the price of a collection is order-independent when reuse is by name, strongly order-dependent when reuse is compositional, and order-dependent through forgetting in the transformer (REFUTED on the letter; the hypothesis confirmed in its mechanism)
command: `python experiments/ziplearn/e14.py`, `... --words 1`, `python experiments/ziplearn/e14b.py` (25 compositions once each in 10 orders; the transformer 100 steps per task in 4 orders; CPU seconds + GPU 2 min)
files:   `experiments/ziplearn/runs/e14/e14.json`, `e14_words.json`, `e14b.json`; code `e14.py`, `e14b.py`; `TwoLayerWithWords.solved_pi` added
numbers: ZipLearner, reuse by NAME only (E3's rule): total bits parts-first 758, wholes-first
         735, eight random orders 730–747
         (spread 2.3%); the same 5 items named in every order; the frozen re-pass 544 bits in
         every order; curriculum gap 28 bits (3.8%). ZipLearner with WORDS (E8's compositional reuse): parts-first
         **532**, wholes-first **716**, random 552–692 (spread
         **22.5%**); 3 or 4 items named depending on the order but the group they generate is the same, so the
         frozen re-pass is 337 bits in every order; curriculum gap **185 bits (35% of the best)**. The
         transformer, prequential bits over the stream: parts-first 3990, wholes-first 3386,
         random 4579 and 4529 (gap 35%); final bits per digit on all 25 tasks 4.5–9.6,
         all far above the 2.3 of guessing: it keeps the last tasks and is confidently wrong on the rest (E6b).
verdict: REFUTED on the letter for the name-only learner (totals agree within 2.3%; parts-first is even the most
         expensive, by a few naming bits, since a full library charges log₂5 per use) and INCONCLUSIVE for the words
         learner (three of four clauses: parts-first cheapest, spread 22.5%, libraries differ; the frozen re-pass does
         not differ, because what the orders change is which items get named, not the group they generate). In
         substance the hypothesis is confirmed with its mechanism exposed: **order matters exactly to the extent that
         earlier compression is reused to compress later information.** Naming alone ends in the same library
         whatever the order and the totals barely move; let products be words in named parts and parts-first is 26%
         cheaper than wholes-first, because a whole met before its parts pays full price. The transformer's 35% gap
         is a different thing — forgetting, not building — and its final model is worse than guessing in every order.
changed: the curriculum gap is now a defined, measured quantity: worst minus best total over orders, zero for an
         ideal compressor, 28 bits for naming, 185 for words, 1193 (of ~3400) for the gradient learner. §8 change 3 gets
         its stated consequence: "how much knowledge fits" has an answer per curriculum, and the best curriculum is
         parts first. Not built: letting the learner *choose* the order (ask for the parts) — the outer objective's
         exploration of §16 applied to what to learn next.

### 2026-09-20 — E16 — blocks across layers: whole two-layer descriptions as blocks keep 100% retention with exactly three blocks (PASS; OPEN-7 answered)
command: `python experiments/ziplearn/e16.py` (rot_left∘inc, reverse∘negate, swap_pairs∘inc, then the first again; 25 demonstrations each; 10 streams; CPU, ~1 min)
files:   `experiments/ziplearn/runs/e16/e16.json`; code `ContinualTwoLayer` in `ziplearner.py`, `e16.py`
numbers: retention (2 demonstrations as context, exact match on a fresh query) **1.00 for every rule after every
         stretch**; blocks 1.0 → 2.1 → 3.1 → **3.1** (exactly three at the end in 90% of streams; one stream minted
         a fourth at a transition); bits 37 → 77 → 114 → 114. One stream's blocks at the end: "position permutation
         then shift b=1 (n=50)", "position permutation then affine a=4 b=0 (n=25)", "position permutation then shift
         b=1 (n=25)" — the returning rule reused its block.
verdict: PASS. OPEN-7 asked whether blocks multiply when a block spans two layers; they do not, because a block is
         priced as one description whatever its depth: the frozen kept description of an established block is the
         pair (value hypothesis, solved position structure) and foreign demonstrations are exceptions to it, exactly
         as at the digit level. Nothing in §9's rules needed changing; the unit of routing became a demonstration.
changed: `ContinualTwoLayer` (§9 for whole tasks). OPEN-7 closed in its first form; a block spanning *more* than two
         layers, or layers with distributed codes, is not tested.

### 2026-09-20 — E17 — goals that are relations: identify-then-plan fails at the identification limit (REFUTED); matching words to the goal's examples directly implements the relation on unseen inputs 99% of the time
command: `python experiments/ziplearn/e17.py` (150 relational goals, each a composition of 1–3 actions given as two demonstration pairs; actions learned from 4 observations; CPU, ~1 min)
files:   `experiments/ziplearn/runs/e17/e17.json`; code `e17.py`
numbers: state planner (E9 on the first pair as a state goal): reaches that pair's output 1.00, implements the relation on
         20 fresh inputs **0.92**. Relation planner as pre-registered (identify the transformation from the
         two pairs with E2's learner, then the shortest word matching it on probes): **0.51**, no word found
         0.41. Relation planner, direct (the shortest word whose learned effect matches both demonstration
         pairs, no identification step): **0.99**, nothing left unfound.
verdict: REFUTED on the letter (0.51 < 0.70). The cause is E2's identification limit: two pairs do not settle a
         position structure (~0.26 of the time at V = 5), so the identify-first planner has no target to match 41% of
         the time. The state planner does well (0.92) because a six-digit state is nearly a fingerprint: a word that
         maps one random input correctly is usually the relation. And the direct planner shows what a relational goal
         needs here: not a *description* of the relation, but the relation's examples used as constraints on the
         word — the plan space is small enough to search, and two pairs constrain it almost completely (0.99).
changed: OPEN-12 gets its first form: a goal given as examples of a relation is planned for by matching plans to the
         examples; identifying the relation as a description is a separate, optional step, worth paying only when it
         buys something the examples cannot (a relation outside the plan space, or one that must transfer to a new
         action set). The pre-registered planner is kept in the code as the refuted arm.

### 2026-09-21 — E18 — attention weights written from a description, no training: the induction circuit copies a pattern with 100% accuracy (PASS; OPEN-8 first form)
command: `python experiments/ziplearn/e18.py` (a 2-layer, 2-head RoPE `h1_lid` model, every weight written; 256 sequences: BOS, a pattern of 8 distinct tokens, the pattern again; CPU, seconds)
files:   `experiments/ziplearn/runs/e18/e18.json`; code `e18.py` (`write`: the circuit as a description; `accuracy`)
numbers: random-initialised model, next-token accuracy inside the second copy 0.116 (chance 0.125); the written model
         **1.000** for patterns of length 4, 6 and 8 (0.438 over all positions, the first copy being unpredictable).
         With repeated tokens in the pattern (length 12 over 8 tokens) 0.396 — the ambiguity any first-order induction
         head has: "the position after an earlier copy of this token" is not unique.
verdict: PASS. The circuit is two written heads: a previous-token head whose query and key come from a constant
         channel so that under RoPE their score depends only on distance, with the key's phases set in the four
         highest-frequency rotary pairs so the peak is one position back; and an induction head whose query is the
         current token and whose key is the previous-token subspace, both in the lowest-frequency pairs where the
         rotation over the sequence is negligible — content match, not position. Values copy token identities; the
         output projections route them between subspaces; the MLPs and the other heads are zero; the unembedding
         reads the output subspace. Three lessons on the way, all recorded because they are what "writing weights"
         actually meets: (1) with V = 5 and patterns of 12 the test itself was ambiguous (0.40) — the vocabulary had to
         be widened to make the circuit's job well-defined; (2) position 0, having no predecessor, attended to itself
         and so became its own "previous token", which made the first token of the pattern match twice (0.91) — a
         BOS token fixed it, as in every real model; (3) the previous-token head must use the HIGH-frequency rotary
         pairs and the induction head the LOW-frequency ones: the same RoPE frequency axis separates "where" from
         "what" by frequency, which is what E0's failed θ = 0 channels were trying to do by hand.
changed: OPEN-8 has its first form: an attention circuit can be written as a description (subspaces, routes, two
         heads' q/k/v with rotary phases as parameters) and works exactly. Not built: the general step — ZipLearner
         *choosing* such a circuit as the cheapest description of in-context data — and PoPE, under which the content
         match is not separable (E0); the written network uses RoPE.

### 2026-09-21 — E21 — a near-duplicate rule gets its own block as soon as an observation refutes the old one: 6.4 observations, the waiting time for a differing input (a measurement; OPEN-6)
command: `python experiments/ziplearn/e21.py` (A = shift 3 for 40 pairs, then D = A on 9 of 11 inputs for 120; 20 streams; CPU, ~20 s)
files:   `experiments/ziplearn/runs/e21/e21.json`; code `e21.py`
numbers: a block of its own for D minted after **6.4** D-observations on average (min 1, max 21; never in
         0 streams); A's block carries an exception rate of 0.000 at the end; D's accuracy on its two differing
         inputs, given a context that contains them, 0.95.
verdict: none (a measurement). OPEN-6's expectation held to the letter: while the window contains only inputs on
         which A and D agree, D is "A" — the correct description of the evidence — and the first differing input is
         an exception; because A's block is established (40 clean pairs), that exception is expensive (≈ 9.7 bits
         under the rate price) and the change-point rule mints within a few more observations. 6.4 is 11/2: the
         expected wait for one of the two differing inputs. The ambiguity is real but short-lived, and its cost is
         one wrong answer per differing input before the mint.
changed: nothing. OPEN-6 closed as "measured": the guaranteed error is the first refuting case, and the recovery is
         a few observations.

### 2026-09-21 — E22 — annealing the budget does nothing here: there is nothing fuzzy to crystallise (REFUTED in this setting; §8's annealing reading)
command: `python experiments/ziplearn/e22.py` (E15's stream; a hard cap of 40 bits from the first stretch against 80 → 40 in steps of 8; 10 streams; CPU, ~20 s)
files:   `experiments/ziplearn/runs/e22/e22.json`; code `e22.py`
numbers: hard cap: 36.1 bits, 3.6 blocks, evidence-weighted retention **0.723**; annealed: 35.6 bits,
         3.4 blocks, **0.701**. Different blocks are dropped (an affine and a shift instead of two affines)
         with no advantage either way.
verdict: REFUTED for this stream. §8's argument for raising λ gradually is that fuzzy descriptions harden into
         discrete ones as the budget tightens; in this stream every description is discrete from the start and the
         budget only chooses which block to drop, so the schedule changes the order of drops and nothing else.
         Annealing has a job only where §7 step 2 (a table re-described by a structure) has fuzzy descriptions to
         harden — not built, and not testable on one-hot rules.
changed: §8's annealing is marked "no effect where descriptions are already discrete; untested where they are not".

### 2026-09-21 — E23 — macro-actions from the library make the search LARGER in a small map (REFUTED here; OPEN-10 first form)
command: `python experiments/ziplearn/e23.py` (E9's environment, 150 goals 4–6 actions away, 17 macro-actions = the library's named permutations as words; CPU, ~1 min)
files:   `experiments/ziplearn/runs/e23/e23.json`; code `e23.py`
numbers: primitives only: 100% reached, **192** nodes per goal, plans 2.92 steps (the oracle's 2.92);
         primitives + macros: 100% reached, **522** nodes per goal, plans 2.92.
verdict: REFUTED. The goals were built from words of 4–6 actions, but the group is small and words cancel: the
         oracle's mean plan is 2.9 steps, breadth-first search with a visited set touches at most the 720 reachable
         states, and adding 17 macros raises the branching factor from 7 to 24 without shortening anything. A cache
         pays only when depth is large relative to branching; in this map nothing is deep. OPEN-10 stays open for the
         reason it was opened: it needs a map too large to search, and we do not have one yet.
changed: nothing; the honest status of OPEN-10 is "not reachable in the current domains".

### 2026-09-21 — E19 — hindsight as exception accounting on the goal model: the door level is solved, 8 of 16 levels (PASS on the outcome; the first form of "learn from experience")
command: `python experiments/ziplearn/play_games.py --budget 150 --out experiments/ziplearn/runs/e19` (all six games, 150 actions per level; CPU, seconds)
files:   `experiments/ziplearn/runs/e19/play.json`; code: `GoalModel` keys with confirmed/refuted counts, `GoalModel.refute`, `Player.expect_win` in `arcgames.py`
numbers: levels solved / actions (oracle): LockPath **2/4** — L0 87 (8), **L1 79 (12)**, L2 not solved; MultiKey 2/2 (77, 49);
         Sokoban 0/3; CollectAll **3/3** — 80, **58 (13)**, **94 (15)**; Toggle 1/1 (71); Tetris 0/3. **Total 8/16** (E13: 5/16). No
         game did worse. LockPath's goal-directed plans fell from 136 to 2 per run: the wrong key fired once, was
         refuted, and exploration resumed.
verdict: PASS on the outcome pre-registered in the OPSD note (the level that failed on a wrong goal is solved within
         the budget), with the mechanism stated exactly: every win key now counts the times it fired and the score
         followed against the times it fired and nothing came; a key refuted more often than confirmed stops
         predicting. That is the rate price of E4 applied to the goal model — a learned goal is a description like
         any other, confirmed by the score and refuted by its absence — and it is the hindsight half of OPSD in this
         framework: the outcome, known afterwards, prices the prediction made before. What the note's items 1–2
         describe — searching in hindsight for the *missing* structure that would have predicted the failure — is
         not built; this pass only withdraws trust from the wrong description, and exploration does the rest. The
         remaining failures are world-model failures, not goal failures: pushes (LockPath L2, Sokoban) need the
         radius-2 rule that exploration never buys enough observations of, and Tetris moves every tick.
changed: the goal model keeps counts, not sets. §16's game paragraph: the goal is learned from the score AND priced
         against it. The user's point 3 (learned goals per the ARC-AGI-3 rules) has its first mechanism; points 1
         (a sleep pass) and the rest of point 2 (finding the missing structure in hindsight) are the next two.

### 2026-09-21 — E24 — the sleep pass: each action's rule keeps the 2–4 context cells it needs, the model shrinks ~10×, and the next levels are solved 2–3× faster (PASS after one correction)
command: `python experiments/ziplearn/play_games.py --budget 150 --sleep 1 --out experiments/ziplearn/runs/e24` (all six games; a sleep pass after every completed level; CPU, ~2 min)
files:   `experiments/ziplearn/runs/e24/play.json`; code: `LocalRule.sleep` (with `full` as the kept evidence and `mask` as the description), `ActionModel.sleep`, `Player.sleep`, the checkpoint in `play` (`arcgames.py`)
numbers: bits of the world model at the first sleep: LockPath 9192 → 660, MultiKey 8800 → 1192, CollectAll 9144 → 752,
         Toggle 7190 → 1754; context cells kept per action out of 9 (radius 1): LockPath [3, 3, 3, 2], MultiKey [4, 3, 3, 3],
         CollectAll [3, 3, 3, 2], Toggle [4, 4, 4, 4] — a move needs its own cell and the cells along its axis, nothing
         else. Levels after a sleep: LockPath L1 **25** actions (79 without sleep; oracle 12), MultiKey L1 **25** (49; 13),
         CollectAll L1 **23** (58; 13) and L2 **34** (94; 15). Levels solved 8/16, the same three failures (LockPath L2,
         Sokoban, Tetris — no level completed, so no sleep ever ran there). Prediction accuracy on LockPath 0.78 (0.70).
verdict: PASS: more compact (≈ 10× fewer bits) and more general (the same rule now predicts windows it never saw,
         because the cells that differed were dropped), measured as faster solutions on the levels that follow. The
         correction: the first version let the price alone choose the mask and it merged away the rare cells that
         change — forgetting a rare change costs fewer bits than an entry — which dropped accuracy to 0.45 and lost a
         level. The planner lives on exactly those cells. The adopted rule: drop context cells while the merged table
         is cheaper AND explains the evidence no worse (no new exceptions): compact without losing what was known.
         That is the tension between the inner objective (bits) and the outer one (plans) in one line, and the
         sleep pass resolves it by refusing lossy compression of the evidence it plans on.
changed: the world-model rule keeps its evidence (`full`) separate from its description (`mask` + merged table) — the
         first place in the design where the two are stored apart; §7 step 2 (re-describing a table by a cheaper
         structure) now has an instance. A per-entry mask (each rule entry keeping only the cells it needs) is the
         obvious next form; E25 says why.

### 2026-09-21 — E25 — what hindsight would have to learn: the diagnostic over the player's traces (a measurement)
command: `python experiments/ziplearn/play_games.py --budget 150 --sleep 1 --trace 1 --out experiments/ziplearn/runs/e25`, then `python experiments/ziplearn/diagnose.py --runs experiments/ziplearn/runs/e25`
files:   `experiments/ziplearn/runs/e25/trace_*.json` (every step: frame, choice and its reason, prediction, outcome), `diagnosis.json`; code `diagnose.py`, the `trace` option in `arcgames.play`
numbers: actions spent, per level — first levels are ALL discovery (80, 77, 87, 71 actions, oracles 7–9); later levels
         are mostly exploration after the goal is known (CollectAll L1: 22 exploratory + 1 goal-directed; LockPath L1:
         23 + 2) — the win keys are radius-1 windows and fire only when the agent is adjacent in a seen arrangement.
         Revisits: **54 of 80, 66 of 87, 57 of 77, 53 of 71** discovery actions return to a frame already seen; Sokoban
         **116 of 150**. Wrong predictions are almost all MISSED CHANGES (a cell changed, predicted unchanged): LockPath
         L0 40 of 30 wrong predictions, Sokoban 106, Tetris 424 (plus 42 spurious and 40 wrong colours), and they occur
         with unknown windows in the frame — the "unchanged" default. Failed levels, read from the examples: LockPath
         L2 — the agent's move is predicted to vacate its cell but not to arrive, because the destination's window
         has the block one cell above and that key was never seen: a rule that does not depend on that cell is paying
         for it; Sokoban — no win found in 150 actions, 116 revisits: pushes happen by accident, the model never
         predicts them, and exploration cannot plan the moves that would make new frames; Tetris — every tick moves
         the piece whatever the action, and a model whose default is "unchanged" is wrong on every cell that moves.
verdict: none (a measurement). What hindsight would have to learn, in order of actions lost: (1) **exploration**, not
         the model: two thirds of discovery actions revisit known frames, because plans are one step long and end at
         the first unknown; a plan that reached the nearest unvisited frame and stayed away from visited ones would
         halve discovery; (2) **the unknown-window default**: "unchanged" is wrong wherever things move; the default
         should be learned (Tetris) or, better, the rule should not depend on cells it does not need — the per-entry
         mask (LockPath L2's block); (3) **rare events**: a push is seen a few times by accident and the price never
         gets the observations it would need to prefer the radius-2 rule — E10's price of ignorance should be buying
         those observations, and is not wired to the games; (4) **goal keys too specific**: a win key is a whole
         radius-1 window, so a goal learned at one spot does not fire at another; the sleep pass's mask applied to
         goal keys is the same fix. None of these is credit assignment over a trajectory; every one is a missing or
         over-specific description, found by comparing what was predicted with what happened — which is what the
         trace now records at every step.
changed: nothing in the design; the next builds are ranked by the numbers above.

### 2026-09-21 — E26 — the looped transformer on the composition task: growth with untied cores is the best gradient arm we have had (13/17 trained, held-out accuracy 0.12–0.20 vs 0.03), still 0/8 held-out solved
command: `python experiments/ziplearn/e26.py` (RoPE, 3200 steps, seed 0, E0's task and budget; the `LoopedModel` in `transformers/h1_lid.py`, built after Chen et al. arXiv:2609.19107 and their MIT-licensed `models/transformer.py`; GPU, ~15 min)
files:   `experiments/ziplearn/runs/e26/summary.json`, per-cell JSON and logs; code `h1_lid.LoopedModel`, `--res loop --loops K --untied --grow_at`, `e26.py`
numbers: trained compositions solved / mean accuracy / held-out solved / held-out mean accuracy —
         vanilla d3 (E0) 8/17 / 0.52 / 0/8 / 0.03;  tied K = 2 (executed depth 4, three blocks' parameters) 10/17 / 0.59 / 0/8 / 0.07;
         tied K = 4 (depth 6) 9/17 / 0.56 / 0/8 / 0.07;  tied K = 6 (depth 8) 9/17 / 0.54 / 0/8 / 0.02;  untied K = 4
         (a six-block stack with the boundary operator) 10/17 / **0.69** / 0/8 / **0.20**;  tied growth 2 → 4 at half
         of training 10/17 / 0.65 / 0/8 / 0.07;  **untied growth 2 → 4: 13/17 / 0.76 / 0/8 / 0.12**, final loss 0.042
         (vanilla 0.061). Wall-clock per cell 95–207 s (executed depth costs).
verdict: consistent with the paper's ordering on our task: growth beats a fixed depth, untied beats tied, and the
         boundary operator alone (tied K = 2 is three blocks with the operator between passes) already beats vanilla.
         Held-out compositions are still never solved to criterion, but their mean accuracy rose from 0.03 to 0.12–0.20
         with untied depth — the first movement on that number since the LID experiments (transformers/NOTES.md). One
         seed, so the ranking among the loop cells (K = 2, 4, 6 within 0.05 of each other) is not established; the gap
         between vanilla and the growth/untied cells (0.52 → 0.69–0.76 trained, 0.03 → 0.12–0.20 held-out) is larger
         than E0's seed-to-seed spread.
changed: the gradient arm gains a looped substrate. Not built: ZipLearner writing a looped block (see the chat note of
         2026-09-21: rules as an attention key/value memory, the loop count as the plan depth, the anchor as the goal).

### 2026-09-21 — E27 — the exploration fix: value every reachable frame in bits, plan whole paths, remember transitions; 9 of 16 levels, Tetris's first level solved, discovery itself unchanged (coverage-bound)
command: `python experiments/ziplearn/play_games.py --budget 150 --sleep 1 --trace 1 --out experiments/ziplearn/runs/e27` then `diagnose.py --runs runs/e27` (all six games; CPU, ~2 min)
files:   `experiments/ziplearn/runs/e27/play.json`, `trace_*.json`, `diagnosis.json`; code: `Player.explore`, `Player.trans`, `ActionModel.exception_rate` in `arcgames.py`; `distinct_frames` in `diagnose.py`
numbers: levels 9/16 (E24/E25: 8/16): LockPath 2/4 (93, 31), MultiKey 2/2 (73, 28), Sokoban 0/3, CollectAll 3/3 (83, 31,
         40), Toggle 1/1 (78), **Tetris 1/3 (level 0 in 92 actions — never solved before)**. Discovery actions on first
         levels 73–93 against E25's 71–87; revisits 55–69 of them against 53–66; distinct frames stood in during
         discovery 17–27 (Tetris 65). Prediction accuracy on the fully explored games rose to 0.95–0.99.
verdict: the fix as pre-registered — fewer discovery actions — is REFUTED, and the reason corrects E25's reading:
         discovery is COVERAGE. Finding an unknown goal in a room of ~25 reachable frames means standing in most of
         them, and a walk over a grid with four moves backtracks about as often as it advances; the old "nearest
         unknown" walk was already near that bound, and so is the new one. What the new explorer changed is elsewhere:
         (1) it plans whole paths and values them in bits — unknown windows to learn, weighted by the rule's learnability
         (the noisy-TV filter of E10b), a frame never visited worth the remaining budget spread over the unvisited
         frames in reach (so a far new frame is worth the walk when nothing nearer is new), untested predictions worth
         the rule's exception rate — which is what got Tetris's first level: multi-step plans through a world that
         moves; (2) it remembers observed transitions and uses them over the model's prediction, which removed an
         attractor the first version had (a mispredicted "new" frame that could never be reached); (3) it never
         goes blind. Three defects were found and fixed on the way, each a lesson about exploration as a price:
         summing a frame's information over all its actions made the agent walk toward "rich" frames without ever
         acting on them (value must be a complete plan: reach, then do the informative thing); requiring a positive
         value left the agent idle when everything near was known (the budget is spent either way — take the best
         plan); and with nothing unknown and nothing unvisited in reach the agent oscillated between two known frames
         (untested predictions carry the model's remaining uncertainty, priced by the rule's exception rate).
changed: exploration is now §16's price of ignorance over frames, not a first-unknown walk; the diagnostic reports
         distinct frames as the coverage measure. What would actually cut discovery is not a better walk but a
         smaller room to cover — a prior over where goals are, learned across levels (E19's keys do this once a
         goal is known, not before) — or a goal that announces itself (a relation, E17). Left as the next question.

### 2026-09-21 — E28 — the written looped block: the rules learned on LockPath written into one attention block, looped once per action, reproduce the planner's rollouts exactly (530/530, 509/509) and beat its default on unknown windows
command: `python experiments/ziplearn/e28.py` (seed 0; `--seed 1 --out runs/e28/seed1`); CPU, about 1 minute each
files:   `experiments/ziplearn/runs/e28/e28.json`, `runs/e28/seed1/e28.json`; code `e28.py` (`WrittenSim`), `Attn.causal` in `transformers/h1_lid.py`, `play(..., return_player=True)` in `arcgames.py`
numbers: the rules after playing levels 0–1 with sleep (levels solved in 93 and 31 actions, both seeds — the player is deterministic):
         ACTION1 3 cells 36 entries, ACTION2 3 cells 32 entries (radius 2), ACTION3 3 cells 36 entries, ACTION4 2 cells 17 entries;
         the union of their masks is the four neighbours. The block: d = 128, 4 gather heads (one per neighbour) + 1 lookup
         head, 121 memory tokens (one per entry) + 2 null tokens, up to 88 cell tokens; every weight written from the tables,
         none trained. Random plans of 1–4 actions from 40 random-walk states per level, judged by snapshot/restore in
         the true game:
         — plans whose every window was known to the table: written block = planner's own rollout **300/300, 230/230**
           (seed 0; 277/277, 232/232 seed 1) at every plan length 1–4; written = truth exactly where the planner = truth
           (level 1: 186/230 — the planner's known-window errors are the hidden key state, and the block reproduces them);
         — plans that met an unknown window (level 1: 70 and 58; level 2, never played: 300 and 300): the block's default,
           the NEAREST stored window, matched the truth 39/70 and 56/58 on level 1 and 246/300 and 276/300 on level 2, against the
           planner's "unchanged" default 23/70, 43/58, 222/300, 264/300 — better in all four;
         — colour codes that were not one-hot before the boundary re-quantisation (ties between equally near entries): 1434, 1475.
verdict: PASS (pre-registered ≥ 0.98: 1.000). ZipLearner's world model runs as a looped transformer block: the
         table is a key/value memory, the frame is the sequence, a neighbour is a written coordinate permutation, one
         pass is one action, the loop count is the plan's length, and the action is the per-pass anchor. What the block
         adds that the table did not have is a DEFAULT for the unseen — attention has no "no entry", it lands on the
         nearest key — and that default is better than the planner's on every level measured, including a level the
         rules were never learned on. Two defects on the way, both leaks of the written scheme rather than of the
         idea: the null token's colour leaked into the memory entries' keys (they now attend to a second, empty null),
         and the re-quantisation read its own zeroed view. Not measured: the block as the planner's search inside
         `Player` (it is a drop-in for `LocalRule.predict` but slower on CPU than the dict), and rules over larger masks
         (the radius-2 rule survived sleep with 3 cells; a 25-cell mask would be 24 heads).
changed: §16's "a plan is an n-layer written network" is now one block looped n times (E26's substrate), and OPEN-8's
         written attention has its second form (E18 the induction head; E28 the world model). Open: (1) the nearest-key
         default is a free generalisation the description-length account does not price — its cost is the exceptions it
         will produce, to be learned by the same rule as everything else; (2) the sleep pass's masks give the heads;
         what gives the loop count when there is no plan — thinking without acting — is the E26 question in the
         written form.

### 2026-09-21 — E29 — learned search, thinking as acting: imagined paths enumerated by their bits under a learned strategy; equal to breadth-first search until it has learned, then 9× fewer calls on a room of 20 and the room of 40 solved at the oracle's length (74) where breadth-first search cannot reach the goal
command: `python experiments/ziplearn/e29.py` (CPU, ~1 min); the run was made against E27's `Player.search` as the baseline arm; that breadth-first search was then DELETED, because the imagination with a strategy that never learns reproduces its numbers exactly (`play(..., strategy=False)`; checked: LockPath 93/31/150, CollectAll 83/31/40)
files:   `experiments/ziplearn/runs/e29/e29.json`, `run.log`; code `PolicyRule`, `locus_of_change`, `Player.locus/imagine` in `arcgames.py` (DESIGN §18); `--strategy` in `play_games.py`; `e29.py`
numbers: (a) the four goal-learnable games, sleep on, 150 actions/level, 4,000 calls/step — learned 611 actions in total, breadth-first 607
         (ratio 1.007): LockPath 93/31/lost, MultiKey 73/28, CollectAll 83/31/44 (BFS 40), Toggle 78; identical except CollectAll's
         third level, where the shortest of the five successes found in bits-order was 4 actions longer than the shortest plan.
         (b) rooms — LockPath's level 0, then empty rooms of side 12, 20, 40, goal in the far corner (oracle 8, 18, 34, 74):
         calls to the first imagined success at each level's first plan, learned vs breadth-first: room 12 **293 vs 290** (the
         strategy was still empty — level 0 was won by exploration, so nothing had been imagined); room 20 **89 vs 818**; room 40
         **165 vs 3,258**. Actions: learned 18, 34, 74 = the oracle on every room; breadth-first 18, 34, and on the room of 40 its
         budget cannot reach the goal (the true plan needs every frame within 74 steps expanded: ~1,444 frames × 4 = 5,800 calls
         > 4,000; the plan both arms found first was a 40-step plan to a spurious win key at the bottom wall, refuted on arrival
         and re-planned — the learned arm's second plan found the remaining 34 steps after 137 calls). The strategy after sleep:
         **2 features, 3 contexts, 20 bits** — it reads only whether the NORTH-EAST sector contains floor and whether it contains
         wall: (wall, no floor) → down, (floor and wall) → right, (floor, no wall) → down. It never read the goal's colour: with the
         goal always in the bottom-right corner, "right until the wall shows to the north-east, then down" is the cheapest
         description of the successful paths, and it carried across room sizes 12 → 20 → 40 unchanged.
verdict: PASS (pre-registered: ratio ≤ 1.1, room-20 calls below BFS's and ≤ 12× oracle, room 40 solved — 89 < 818 and < 408; 165 < 888).
         The search strategy is a learned table and nothing else: `imagine` orders imagined paths by their description
         length under `PolicyRule`, which is breadth-first while the table is empty (E27's numbers to the action) and a
         directed enumeration once it has counted a few successful paths. What was measured is the two claims of §18:
         calls to the first success grow with the room's AREA for breadth-first (290 → 818 → 3,258) and stay flat for the
         strategy (293 → 89 → 165), and the 20-bit strategy transferred to a room four times the side of any it had seen.
         What the strategy learned is a SHORTCUT (the wall, not the goal) — correct for these rooms, wrong the first time
         the goal sits elsewhere, when its exceptions will force it to read a feature that does distinguish (the
         representation-shortcut lesson, generalise-then-correct). Two costs seen: the shortest-of-five rule is not the
         shortest plan (CollectAll +4), and the batch runs to the budget hunting a fifth success when only one pre-win
         frame exists (room 40, second plan: 4,001 calls for a plan found at 137) — both are the "numbers, not prices" of §18.
changed: `Player.search` (breadth-first) deleted; planning = imagination under the strategy; `strategy=False` is the
         baseline. Open: (1) the value of a computation — when to stop imagining (the 5, the 4,000, the ½ are numbers);
         (2) `explore` still enumerates by its own loop: fold it into the imagination with the epistemic value as the goal;
         (3) the strategy over things that are not frames (a hypothesis, a word) — the context is the model's own
         representation, so the retina generalises only where the model's state is a picture; (4) the written form: the
         strategy as memory tokens beside the rules, the heap as the boundary operator choosing the next pass by bits.

### 2026-09-21 — E29 retracted as a design (the numbers stand); the code is deleted
command: none — a review, not a run
files:   deleted from `arcgames.py`: `PolicyRule`, `locus_of_change`, the retina (`NEAR`, `SECTORS`), `Player.locus`, `Player.imagine`, the `strategy`
         flag; deleted `e29.py`; `play_games.py --strategy` removed. `Player.search` (E27's breadth-first search) had been deleted by E29 and is
         not restored. `runs/e29/` is kept as the record. Kept from the E29 work: the per-level budget in `play()`, the vectorised
         `LocalRule._rows`/`predict` (a speed-up, not behaviour), the prediction cache, and `Attn.causal` (E28).
numbers: none new
verdict: the user's review: a thought defined as "one expansion of an imagined frame by the world model" is task-specific and
         against the bitter lesson; code for the thinking process, even adaptive, is the same error; the context is the context
         window, not a retina; a strategy as an enumerated table is wrong; the only primitives are the game's controls; "an
         imagine function is strictly against the rules." Accepted. What E29 measured — calls flat in the path length under a
         strategy, growing with the area under breadth-first search — remains the target the block must hit by itself.
changed: `arcgames.py` has no planner: the games are explored (E27), not solved on purpose, until the looped block plans
         (DESIGN §18, rewritten: the rule, the interpreter frame, the two recurrences, their interaction with attention
         residuals, Coconut's hazards, E30/E31 pre-registered). Ten reference notes added under `refs/`.

### 2026-09-21 — E30 — the depth loop under attention residuals: the first held-out compositions ever solved (1/8, twice), the learned route is NOT the boundary operator, and reading every earlier pass is what extrapolates
command: `python experiments/ziplearn/e30.py` (8 cells, RoPE, 3200 steps, seed 0, E0's task; `h1_lid.LoopedModel` with `--loop_res attnres --mix tied|per_pass --window m`, `--extrap 2`; GPU — seven cells ran while a game shared the card, 383–478 s each; the rerun of the eighth alone took 167 s)
files:   `experiments/ziplearn/runs/e30/summary.json`, per-cell JSON and logs; code: `LoopedModel(res="attnres", mix, window)`, `LoopedModel.routes`, `AttnRes.log`, the extrapolation block in `h1_lid.main`
numbers: trained solved (mean acc) / held-out solved (mean acc) / at 8 passes instead of 4: trained acc, held-out acc —
         tied core, K = 4:   fixed boundary operator 10/17 (0.60) / 0/8 (0.06) / 0.08, 0.00;
                             **attention residuals, tied mixers 13/17 (0.77) / 1/8 (0.22) / 0.60, 0.13** (reverse>rot_left solved at trial 4, acc 0.88; rot_left>negate 0.56);
                             **attention residuals, one query per pass 13/17 (0.79) / 1/8 (0.22) / 0.75, 0.12**;
                             attention residuals windowed (anchor + last 2) 12/17 (0.74) / 0/8 (0.09) / 0.19, 0.02.
         untied growth 2→4:  fixed 13/17 (0.76) / 0/8 (0.13) / 0.69, 0.05 (E26's same cell: 13/17, 0.12);  tied mixers 13/17 (0.77) / 0/8 (0.09) / 0.77, 0.09;
                             per pass 13/17 (0.77) / 0/8 (0.09) / 0.76, 0.08;  windowed 13/17 (0.77) / 0/8 (0.09) / 0.20, 0.00. Final losses 0.040–0.041 (fixed tied 0.056).
         The routes (mean weight per source; sources = [anchor, pass 1, pass 2, …]) of the tied-mixer tied-core cell: pre-attention
         pass 1 [1.00], pass 2 [0.11 0.89], pass 3 [0.03 0.20 0.77], pass 4 [0.01 0.04 0.17 0.78]; pre-MLP pass 4 [0.09 0.10 0.12 0.23 | 0.46 on
         the pass's own attention]; final [0.04 0.02 0.02 0.14 0.79]. One query per pass: the same shape (pass 4 pre-attention
         [0.06 0.07 0.07 0.80], pre-MLP [0.12 0.15 0.12 0.23 | 0.38]). Windowed: [0.08 0.29 0.62] over (anchor, pass k−2, pass k−1).
         Untied core with tied mixers: broader, pass 4 pre-attention [0.07 0.22 0.28 0.42].
verdict: PASS (pre-registered: an attention-residual cell at or above E26's best held-out accuracy 0.20 with non-uniform
         routes: 0.22, twice). The question answered: the learned route is NOT the boundary operator. The fixed operator
         re-injects α·anchor at every pass; the trained mixers let the anchor fade to 0.01–0.06 by the fourth pass, read the
         last pass at ~0.8, and keep a small, decaying read of EVERY earlier pass (0.04–0.20 before attention, ~0.1–0.23 before
         the MLP) — a learned momentum over iterates, as §18(ii) hoped. §18(iv) is REFUTED: restricting the sources to the
         anchor and the last two passes does not help extrapolation, it destroys it (0.19–0.20 at 8 passes, against 0.60–0.77
         with every pass readable), and it loses the held-out gain; the tied-core fixed operator collapses at 8 passes
         (0.08). The held-out gain — the first in the whole line, E0 through E26 were 0/8 — appears only with the TIED core:
         the same block reused with learned routes across its own passes. A stack of distinct cores (untied growth) gets
         nothing from the routes (0.09 against the fixed operator's 0.13) though it extrapolates on its own (0.69). One seed,
         one held-out task solved: the sign is what is established, not the size.
changed: the boundary operator to WRITE for the looped block (§18): the pass's input = mostly the last pass plus a decaying
         read of all earlier passes, the anchor fading rather than re-injected at a constant weight; a windowed state is out.
         E31 (Coconut) runs next on the same substrate. The E30 cells ran while a game shared the GPU — wall-clock per cell
         is not comparable to E26's.

### 2026-09-21 — E33 (a) — the ZipLearner language model on the Latin corpus: the discrete machinery on text is a PPM-style blended-backoff character model; the E24 sleep price was wrong for text and the corrected price keeps every context position; 1.87 bits/char frozen and 1.82 online at 2.7M characters against xz's 2.26; the transformer and hybrid arms are shelved
command: `python experiments/ziplearn/textlm.py --sizes 1000 10000 100000 1000000 all` (CPU, ~4 min)
files:   `experiments/ziplearn/runs/e33/e33.json`, `run.log`; code `textlm.py` (`ContextLM`, `xz_bits_per_char`)
numbers: `corpora/latin books`: 15 training books (2,726,046 characters), held out Caesar's *De Bello Civili* (240,588
         characters; his *De Bello Gallico* is in training), alphabet 70 (characters rarer than 200 occurrences as one symbol).
         Bits per character on the held-out book, FROZEN (trained on N characters, then predicting) and ONLINE (prequential:
         the counts keep growing through the book, which is what xz does):
         | training characters | ZipLM frozen | n-gram order 2 / 4 / 6 frozen | ZipLM online | n-gram 2 / 4 / 6 online | xz (online) |
         | 990 | 4.504 | 4.534 / 4.508 / 4.505 | 2.129 | 2.784 / 2.237 / 2.144 | 2.535 |
         | 9,990 | 2.983 | 3.182 / 3.008 / 2.987 | 2.097 | 2.771 / 2.209 / 2.113 | 2.508 |
         | 99,990 | 2.346 | 2.881 / 2.422 / 2.354 | 2.007 | 2.763 / 2.131 / 2.022 | 2.437 |
         | 826,605 | 1.980 | 2.809 / 2.113 / 1.987 | 1.882 | 2.780 / 2.041 / 1.897 | 2.324 |
         | 2,726,046 | **1.873** | 2.810 / 2.048 / 1.886 | **1.822** | 2.798 / 2.014 / 1.842 | 2.263 |
         The ZipLM's mask after sleep was all eight positions at every size, so its numbers equal the order-8 n-gram's; the
         prequential code of the whole training text under it is 850 KiB, 2.6 bits/char — the price of learning the corpus
         from nothing, sequentially. Runtime 16–72 s per size.
verdict: measured (the pre-registered arms (b) the gradient transformer and (c) the hybrid are SHELVED by the user's
         instruction of 2026-09-21 — no more gradient-arm work unless stated — so the crossover the pre-registration asked
         for is not measured; the apparatus test passes trivially, the ZipLM being the order-8 control itself). Two findings
         that matter: (1) E24's sleep price — the per-context two-part code — is WRONG for a predictor that backs off: it
         priced sparse high-order contexts as waste and kept only the previous one or two characters (frozen 2.88 at 100k
         characters, against 2.35 with all eight); the honest price is the PREQUENTIAL code the predictor actually pays
         on the training text, and under it no position is ever dropped. The sleep pass is a no-op on text: what the
         "modified ZipLearner" brings to language is the blended-backoff table itself — E28's memory tokens with the
         nearest-key default as recency-ordered backoff — which is PPM (Cleary & Witten 1984) in a new costume. (2) Its
         data efficiency: 4.5 → 1.87 bits/char frozen from 1k to 2.7M characters, and online it reaches 2.13 at 1k, learning
         the book from the book; better than xz at every size. Against neural models the literature (from memory) puts
         PPM-class models level with small character transformers up to roughly a million characters and far behind them
         at scale (enwik8 is near 1.0 bits/char for large models against PPM's ~2); that is the crossover the shelved arm
         would have measured, and it is where §19's thesis says the gains beyond this table must come from.
changed: `textlm.py` is the discrete machinery's language model; its ceiling on text is PPM's. The two evaluations
         (frozen, online) are both reported from here on for sequential learners.
