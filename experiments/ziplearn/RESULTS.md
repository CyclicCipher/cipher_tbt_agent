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
