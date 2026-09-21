# ZipLearner — design document

*v0.6, 2026-09-21 (v0.1–v0.5 on 2026-09-20; v0.6 records E14–E18: capacity, precision, order dependence, blocks across
layers, relational goals, written attention). E0–E27 have been run; one line each in §13, the full entries in `RESULTS.md`. Code:
`experiments/ziplearn/ziplearner.py` (the weight-writing learner: structures, one matrix, two layers, the cross-task
library, the continual layer, the word library) and `e0.py` … `e11.py`; the earlier arithmetic-only version is
`experiments/inner_objective/ziplearn.py`. This document is the source of truth for the ZipLearner project;
experimental results are reported in `RESULTS.md` and nowhere else.*

## 0. How to read this

Every technical word is defined the first time it is used, and again in the glossary (§14). Each section carries a
status:

- **BUILT** — code exists and has been run; the numbers are in `RESULTS.md`.
- **DESIGNED** — the steps are written down concretely enough to code; no code yet.
- **OPEN** — I do not know how to do this yet, and I say so instead of using a vague verb.

When an experiment runs, its numbers go into `RESULTS.md` under its ID (E1, E2, …) with the command that produced
them, so every number in this document can be traced to a file on disk.

## 1. The idea in one paragraph

A ZIP file learns without gradients. It counts what recurs and stores it once (a dictionary). PNG goes one step
further: for each block of pixels it tries a few simple filters — "this pixel minus the one to its left" — and keeps
whichever makes the block smallest. Learning is counting; choosing a model is "which filter makes this smallest";
predicting is decompressing. ZipLearner is that recipe made into a learner. Its hypotheses are ways of describing the
data; its score is the number of bits the description takes; it keeps the cheapest; it predicts by decompressing. The
version that exists today does this for arithmetic tables. The version this document designs does it for the weight
matrices of a neural network: it writes the matrices directly — no backprop — and the matrices stay readable.

## 2. What exists today — BUILT

`ziplearn.py` learns one task at a time from demonstrations. A demonstration is six input digits (each 0–10) and the six
output digits some hidden rule produced. It keeps three hypotheses running side by side:

| hypothesis | what it stores | how it predicts a new input digit x |
|---|---|---|
| H0, the dictionary | a count of every (x → y) pair seen | the y seen most often with x; nothing if x is new |
| H1, "first difference is constant" | the first two distinct (x → y) points | a straight line through them (mod 11) |
| H2, "second difference is constant" | the first three distinct points | a parabola through them (mod 11) |

Each hypothesis pays, in bits, for every observation as it arrives: log₂11 ≈ 3.46 bits when it had no prediction (a
parameter it had to store), about 0 bits when its prediction was right, log₂11 + 1 bits when wrong (an "escape": say
"I was wrong" then say the digit). After each demonstration the hypothesis with the lowest total, plus 2 bits per
level of complexity, answers the query.

Results on x → (a·x + b) mod 11 with 24 trained (a, b) pairs and 12 held out (`runs/ziplearn_affine.json`): every
held-out rule predicted exactly after **one** demonstration; the learner's own cost 0.19 bits per digit. On random
rules: 0% and a cost of 3.86 bits per digit, above the 3.46 of guessing — it reports that it learned nothing. The
dictionary alone reaches 96% only after 8 demonstrations, because it must see every input digit before it can answer
it. On the composition task (reverse, rotate, swap digits, then change values) it fails — 18% trained, 0% held out —
because it has no hypothesis about digits changing *position*.

The honest caveat: the "first difference is constant" hypothesis *is* the family of straight lines, and every affine
rule is one, so the arithmetic pass is largely built into the hypothesis I chose. What is not built in: the bit prices
prefer the line over the parabola on their own; noise is rejected without any special mechanism; the dictionary ablation
shows that a ZIP with no filters cannot extrapolate at all.

## 3. Vocabulary

- **Weight** — one number `w[i, j]`: how much unit i's activity adds to unit j's input.
- **Connection** — a non-zero weight.
- **Layer** — a table of weights, the matrix **W**, plus a fixed nonlinearity. The layer computes `W · input`.
- **One-hot** — representing the digit x as eleven zeros with a single 1 in slot x.
- **Row = lookup** — for a one-hot input x, `W · input` is simply *row x of W*. So over one-hot inputs a weight
  matrix is a lookup table: row x is what the layer says about x.
- **Permutation matrix** — a matrix with exactly one 1 in each row and column. Over one-hot codes it maps every input
  digit to exactly one output digit. The rule y = (a·x + b) mod 11 is a permutation matrix; the shift by 3 is the one
  with the 1 of row x in column x + 3.
- **Structure** (also "family") — a named set of matrices described by a few numbers: "all shifts" (one number),
  "all affine permutations" (two numbers), "any table" (one number per entry).
- **Parameters** — the few numbers that pick one matrix out of a structure.
- **Price** — the number of bits a description costs: bits to name the structure, bits to write its parameters, bits
  for every observation the written matrix gets wrong.
- **Exception** — an observation the written matrix gets wrong.
- **Fit / write / price / keep** — the four operations of the core loop (§4).
- **Interface** — where one layer's output becomes the next layer's input.
- **Source** — something a layer may read as its input: the token embedding, or an earlier layer's output.
- **Route** — which source a layer reads. A fixed choice in the written network; a per-token softmax over sources in
  the gradient-trained comparison arm (attention residuals, §15).
- **Target** — what a layer's output *should* be for a given input, according to the demonstrations (for the last
  layer) or according to the layers above it (for hidden layers, §6).

## 4. The core loop for one matrix — DESIGNED

Running example: the hidden rule is "shift by 3" (y = x + 3 mod 11). The layer is an 11×11 matrix W over one-hot
digits.

1. **Fit.** A demonstration pair arrives: (x = 2 → y = 5). Every structure in the library that can solve its
   parameters from the pairs seen so far does so. The shift structure needs one pair: b = 5 − 2 = 3. The affine
   structure needs two pairs with different x; it waits. The table structure never "solves": it just records the entry.
2. **Write.** Each structure that has its parameters writes its *whole* matrix. The shift writes all eleven rows:
   `W[x, x + 3 mod 11] = 1` for x = 0 … 10 — including the ten rows for digits it has never seen. This step is what
   "instilling structure into the weights" means concretely: 121 numbers written from 1.
3. **Price.** Each structure's running total is: bits to name the structure (log₂ of the library size) + bits for its
   parameters (log₂11 per digit-valued parameter) + bits for exceptions so far (log₂11 + 1 each). A pair its written
   matrix predicts correctly costs ≈ 0.
4. **Keep.** The structure with the lowest total is the layer's current description. The layer's matrix *is* that
   structure's written matrix. A query is answered by reading a row.

This is exactly what `ziplearn.py` does today, with two changes: a hypothesis is a written matrix rather than a
formula, and prediction is reading a row rather than evaluating the formula. Pseudo-code:

```
for each demonstration pair (x, y):
    for each structure S in the library:
        if S can now solve its parameters: solve them; write W_S in full
        price[S] += cost of (x, y) under W_S          # 0 if W_S already predicts y; otherwise an exception
    keep = the S with the smallest (bits(S) + bits(params_S) + price[S])
answer to a query x  =  row x of W_keep
```

## 5. The library of structures, version 1 — DESIGNED

| structure | parameters | solved from | price of parameters | what it captures |
|---|---|---|---|---|
| table | one output per input seen | each pair, as it arrives | log₂11 per entry | anything; cannot extrapolate |
| identity | none | nothing | 0 | "this layer copies its input" |
| shift | b | 1 pair | log₂11 | y = x + b |
| affine permutation | a, b | 2 pairs with different x | log₂11 + log₂10 (a ≠ 0) | y = a·x + b |
| general permutation | a permutation of 11 things | 11 pairs | log₂(11!) ≈ 25 | any one-to-one relabelling |
| tied blocks | one small matrix reused at every position | pairs at any position | the small matrix's price | "the same rule at every digit position" |
| position permutation | a permutation of the 6 positions | 6 pairs of positions | log₂(6!) ≈ 9.5 | reverse, rotate, swap — digits moving between slots |
| read-from (route) | which earlier source this layer reads | try each source, keep the cheapest description | log₂(number of sources) | the wiring between layers (§6, §15) |
| position map | for each output: a source (repeats allowed) or a constant | candidate intersection, no one-to-one rule | log₂5 per unresolved digit | many-to-one actions: copies and writes (E11) |
| position edit | the identity, with per-slot edits | a slot is free while it reads itself; contradicted, it becomes a map slot | one exception per edited slot, then as a map | "everything stays except…" (E11) |

Two structures for later, when inputs are no longer one-hot digits but distributed activity (many units partly active):

| structure | parameters | solved from | price of parameters | what it captures |
|---|---|---|---|---|
| low-rank | two thin matrices | least squares on observed pairs | (rows + cols) × rank numbers at finite precision (§8) | "the map only uses a few directions" |
| repeated diagonal | k numbers | least squares | k numbers at finite precision | "the same weights slide along the input" — a convolution; translation invariance |

Naming a structure costs log₂(library size) bits — the generalisation of the "+2 bits per level" in today's code.
This list is the hand-chosen part of the design. It is a better place for the hand-chosen part than today's
arithmetic filters, because every entry is a generic fact about matrices, not about any task.

## 6. Depth: more than one layer — DESIGNED, with one OPEN part

The composition task needs two layers: first move digits between positions (a position permutation), then change
their values (a value permutation). Demonstrations only give the network's input and its *final* output. Layer 2's
input is layer 1's output, which depends on what layer 1 currently is; layer 1's target is not given.

The rule: **pin the interfaces, solve each layer alone, sweep.**

1. Freeze layer 1. Layer 2's inputs are what layer 1 currently produces; its targets are the demonstrations' outputs.
   Run the core loop (§4) on layer 2.
2. Freeze layer 2. Layer 1's targets are obtained by running layer 2 *backwards*: if layer 2 is the permutation P and
   must output y, its input must be P⁻¹y. That is layer 1's target. Run the core loop on layer 1.
3. Repeat until neither layer's kept structure changes.

Running a layer backwards is exact when its written matrix is a permutation (or any invertible structure). No gradient
is involved anywhere; a target is decompression run in reverse. With attention residuals (§15) a layer's input *is* a
chosen earlier source rather than a running sum, so the interface being pinned here is literal: the read-from route
(§5) says which source, and the inverted target goes to that source's layer and nowhere else.

The total price is the sum over layers, so "the cheapest description of the whole network" splits into per-layer
choices as long as the interfaces are pinned during each solve. This is the DiffusionBlocks shape: blocks solved
independently against fixed interfaces. Their interfaces are pinned by the data (noise levels); ours are pinned by the
current state of the neighbouring layers, which moves between sweeps.

- **OPEN → measured (E2, results log):** the plain sweep from the identity start settles within 2 rounds but on
  a wrong fixed point whenever both layers act — neither layer can be solved while the other is wrong. Adopted:
  restart the sweep from every description of the value layer (21 at V = 5) and keep the lowest total price.
- **OPEN:** running a *nonlinearity* backwards. With one-hot codes and permutation structures no nonlinearity is
  needed between layers (a permutation followed by a permutation is a permutation), so E1–E3 do not hit this. It
  becomes real when codes are distributed (§5's "later" structures). Known weak point of every target-propagation
  method; no fix designed.

## 7. Where new library items come from — DESIGNED (this is the bet)

Today the library is a list I typed. The reason to put ZipLearner on weight matrices is that a matrix is a
continuous space that already contains every possible structure, so the learner can **fit first and name later**:

1. **Fit fuzzily.** A layer whose cheapest description is the table (no structure fits yet) still has a written
   matrix: the counts. Nothing is lost by not having a name for it.
2. **Look for a name.** Periodically, for each structure in the library, solve its parameters from a few rows of the
   table and check whether the remaining rows agree. If they do, the table is re-described by the structure —
   cheaper — and rewritten, which fills in the rows never seen. This is how a discrete item *arises* from a
   continuous fit: the discrete description is adopted the moment it is cheaper than the entries, and not before.
3. **Compare across tasks.** After several tasks there are several written matrices. If matrix B equals matrix A
   followed by some fixed matrix Q, and the same Q shows up for other pairs of tasks, then Q is a new item: it is
   added to the library with a price of its own, and every task that uses it pays for Q once instead of for its
   entries. This is exactly ZIP's dictionary rule — a substring recurring across files becomes an entry — applied to
   matrices.

The bet, stated so it can lose: *the library of generic matrix structures is small and finite, and task-specific
items are points inside those structures that fitting finds and step 2–3 name.* It loses if the items a task needs are
not inside any generic structure, or if step 3 finds nothing that recurs. E3 tests it.

## 8. Lossy compression for a finite network — changes 1–3 BUILT (E4, E15); annealing DESIGNED

The price in §4 is lossless: every exception is stored individually at log₂11 + 1 bits, so a rule with a few
exceptions carries every one of them forever, and the total grows with the number of observations. A finite network
cannot hold that, and should not want to: the useful content of "a shift with 10% exceptions" is the shift and the
number 10%, not the list of exceptions. The right theory is rate–distortion coding: *rate* is the bits you spend,
*distortion* is the error you accept, and one number λ (bits per unit of error) sets the trade-off. Three changes,
smallest first:

1. **Exceptions become a rate, not a list.** Each structure carries one extra parameter ε, the fraction of
   observations it gets wrong. The price of n observations is then n·H(ε) + (number of exceptions)·log₂10, where
   H is the binary entropy — the cost of saying *which* observations are exceptions, without remembering them
   individually. For ε = 0 this is today's lossless price; for ε = 10% it is flat per observation instead of growing.
   The written matrix is unchanged; only the price changes. The learner now remembers the rule and the rate of
   exceptions, and forgets the exceptions themselves.
2. **Parameters at justified precision.** A parameter estimated from n observations is written with about ½·log₂n
   bits of precision — coarser when it rests on few observations, finer when on many (the MDL/MML precision rule).
   For digit-valued parameters (a, b, a permutation) this changes nothing; for continuous parameters (§5's "later"
   structures) it is what stops a low-rank fit from spending 32 bits on each of its numbers.
3. **The capacity budget.** A network of a fixed size can hold at most C bits of description (its weights' total
   precision). The learner then minimises *error* subject to *total price ≤ C*, which is the same as minimising
   price + λ·error with λ chosen so the budget is met. When C is generous, λ is small and fuzzy tables are tolerated;
   as C tightens, λ rises and only structures survive. Raising λ gradually is deterministic annealing; it is the
   mechanism by which discrete items crystallise out of continuous fits (§7, step 2) *because the budget forces it*,
   not because a threshold was typed.

Named algorithms, for reference: rate–distortion theory (Shannon 1959; Blahut–Arimoto for computing the curve);
transform coding — transform, quantise, entropy-code — is JPEG's shape and matches "choose the structure, round the
residual, price what is left"; finite-precision parameters (Wallace & Boulton 1968; Rissanen 1978); description length
of network weights via noisy weights (Hinton & van Camp 1993); deterministic annealing (Rose 1998); the information
bottleneck (Tishby, Pereira & Bialek 1999) for what a hidden layer should keep when the layers above are known.

The risk to watch: a lossy price can excuse the wrong thing. With λ small enough, "a table with some errors" is as
cheap as "a shift", and the learner stops looking for structure. E4 measures the ε the learner reports against the true
exception rate and checks that the structure is still chosen.

## 9. Continual learning without a replay buffer — rules 1–3 BUILT (E6, E15)

The problem: a network trained by gradient on task A and then on task B forgets A, because B's gradient moves the
same weights A used, indiscriminately. The usual remedies keep A's examples (a replay buffer), keep A's weights in a
copy (an archive), or penalise moving weights A found important. ZipLearner needs none of these, for one reason:

**The description is the archive.** A compressor never keeps the original file once it has the compressed one; the
compressed form regenerates it. A's written matrix, plus two numbers kept beside it — its **evidence count** n_A (how
many observations it has explained) and its exception rate ε_A — is a complete record of what A's data said. To ask
"would changing this matrix hurt A?" the learner does not need A's examples: it knows from the matrix itself which
rows would change, and from n_A how many observations stood on them. The price of the change is computable in closed
form. Nothing is stored beyond the model and two numbers per description.

Three rules follow.

**Rule 1 — never overwrite; add, and let the evidence decide.** When B's observations contradict A's kept
description, two descriptions of everything seen so far compete on price: (a) change A's parameters to fit B — which
turns (1 − ε_A)·n_A of A's past observations into exceptions, at log₂11 + 1 bits each, all charged now; (b) keep A and
add a second description for B, at the fixed cost of naming a structure and its parameters (a few bits). Option (b)
wins as soon as n_A is more than a handful. Old knowledge is protected in exact proportion to the evidence behind it,
and nothing is ever moved by accident. (This is what the "weight importance" of EWC estimates; here it is exact, free,
and never touches a gradient.)

**Rule 2 — context selects; a new context is minted only on refutation.** A layer holding several descriptions is
*one* matrix whose rows are indexed by (context, input): a block of rows per context (§5's tied-blocks structure,
untied). Adding B adds a block; A's block is rows nobody writes to. No sub-networks. (In the attention-residual substrate of
§15 a block's summed output is a selectable source, which is what "context selects the block" needs.) The active context is inferred,
not labelled: it is the block that prices the last few observations cheapest — the recent window is the layer's
input, not a store of old tasks. A new block is created only when every existing block would have to pay more in
exceptions than a new block's parameters cost; until then an observation is filed as an exception of the cheapest
block (§8's rate). Mint on refutation, never on incompleteness.

**The concrete form of rules 1–2 as built (E6):** minting is a change-point decision over the recent window; an established block (≥ 3 pairs of evidence) is priced for routing under its kept description, frozen; a young block is priced with all its structures live; a returning rule is routed to its old block. See the log.

**Rule 3 — capacity forces consolidation, and forgetting is chosen by price.** When the total description exceeds
the capacity C (§8), the learner first *merges*: two blocks that differ by a recurring matrix Q become one block plus
Q (§7 step 3 — library discovery and consolidation are the same operation seen from two sides). If merging is not
enough, it *drops* the block with the least (evidence × bits it saves). Forgetting happens, but the cheapest-to-lose
goes first, which is the opposite of gradient forgetting, where whatever overlaps the new task's gradient goes first.
An optional age discount on n (an evidence count that decays if a block is never selected) gives a forgetting curve;
not in version 1.

What this does not solve — **OPEN-6:** if two rules agree on every observation in the recent window and differ
elsewhere, no selector can tell them apart; the price treats them as one description until an observation refutes
it, which is the correct behaviour and also a guaranteed error on the first refuting case. *E21 measured it: the mint
comes 6.4 observations after the near-duplicate starts — the wait for a refuting input — at the cost of one wrong answer each.* **OPEN-7:** in a
multi-layer network a new block in layer 1 changes what layer 2 sees for that context; the sweep of §6 handles it,
but whether block counts stay small across layers, or multiply, is not known. E6 measures both.

## 10. Prior knowledge written in — DESIGNED

A rule of logic is a small table: AND, OR, NOT, XOR over one-hot truth values. Writing one into a layer is §4's write
step with the parameters given instead of solved. "Not both A and not-A" is not a table but a nonlinearity: a
winner-take-all over the two units. Modus ponens (from A and "A implies B", get B) is a matrix applied to a state
vector: the rule is the matrix, applying it is the multiply.

The priors most worth writing in are not tables but *structures offered cheaply*: identity at 0 bits ("things stay as
they are unless acted on"), shift and rotation ("moving does not change the object"), repeated diagonal (translation
invariance). They cost nothing on tasks that do not use them, which is the test of a good prior.

The caveat that decides whether any of this works: a written law only helps if the network's code for "A" and "B"
lands in the slots the written matrix reads. That is the interface problem of §6, in another costume; priors and pinned
interfaces stand or fall together. E5 tests one prior on one task, with the harm check on another.

## 11. Experiments, pre-registered — each ≤ 20 minutes, most run in seconds

For each: what is measured, what counts as a pass, what would refute the design. Numbers go to `RESULTS.md`.

**E0 — substrate check (5 min, gradient arm only).** The composition task of `h1_lid`, one seed each, four cells:
{RoPE, PoPE with θ = 0 channels} × {standard residual, attention residuals}. Measure: trained-task solve rate and
held-out solve rate, as in `transformers/NOTES.md` (RoPE 9/17, PoPE 7/17, both 0/8). Purpose: confirm the substrate
of §15 does not break the comparison arm, and read the learned routes. Not a verdict on the design; a 0/8 held-out
result is expected and does not fail anything.

**E1 — one layer, arithmetic in weights.** The 36 (a, b) rules, one 11×11 matrix, library of §5 (table, identity,
shift, affine). Measure: (i) the fraction of *unseen* rows that are correct after 1 and after 2 demonstration pairs;
(ii) which structure is kept; (iii) whether the written matrix equals the true permutation. Pass: 100% unseen rows
after 2 pairs for every rule, affine (or shift when a = 1) kept in every case. Refute: the table is kept, or unseen
rows are wrong.

**E2 — two layers, composition.** The 49 two-step tasks of the composition domain (17 trained, 8 held out, as
before). Layer 1 = position permutation, layer 2 = value structure, solved by the sweep of §6. Measure: held-out
exact-match after k demonstrations; rounds of the sweep until stable. Pass: ≥ 80% of held-out tasks exact after ≤ 2
demonstrations (today's ZipLearner: 0%) and the sweep stable within 3 rounds. Refute: < 50%, or oscillation.

**E3 — library discovery.** Generate tasks from a hidden item not in the library (for instance "swap adjacent
digits" is deliberately removed from the position structures). After k tasks, measure: whether §7 step 3 finds a
recurring Q; the bits the *next* task costs before and after Q is added. Pass: Q is found and the next task's price
drops by at least the price of Q's entries. Refute: no recurring Q, or no drop.

**E4 — lossy price.** Tasks = a shift with a 10% random exception rate. Compare the lossless price (§4) with the
rate price (§8, change 1) as the number of observations grows. Measure: kept structure; reported ε; price per
observation. Pass: the rate price stays flat, reports ε ≈ 0.10, and keeps the shift; the lossless price grows and
eventually prefers the table. Then, under a capacity budget C smaller than the table's size: the shift is kept. Refute:
the rate price abandons the shift, or reports ε far from 0.10.

**E5 — a written prior.** A boolean-formula task (demonstrations of an unknown formula over 3 variables), with and
without AND/OR/NOT + winner-take-all written into layer 1. Measure: demonstrations to criterion; and E1's result with
the prior present (harm check). Pass: fewer demonstrations with the prior and no change on E1. Refute: no gain, or
harm on E1. *(Run 2026-09-20: the E1 check is not runnable — gates cannot be offered to a digit library — and was replaced by the same prior on random functions; see the log.)*

**E6 — sequential learning, no replay.** Rules arrive one at a time with no task labels — A (shift 3), B (shift 5),
C (affine 2x+1), then A again — each for a stretch of observations, then the next. Nothing from a finished stretch is
kept except the descriptions and their evidence counts. After every stretch, test *all* earlier rules (each given its
own demonstrations as context). Measure: retention (accuracy on every earlier rule), number of blocks, total bits.
Pass: 100% retention throughout; blocks = 3 (A's second visit reuses its block, no duplicate); total bits below the
lossless table's. Refute: retention drops after any stretch, or a fourth block appears for A's return. Comparison arm:
the run-2 transformer trained on the same sequence by gradient, where forgetting is expected — the number to beat.

**E7 — the anatomy pass (added 2026-09-20 evening, after E0–E6; an analysis, not a pass/refute test).** Read the
network ZipLearner wrote: every written matrix with its structure, parameters, evidence and exceptions (E2's two
layers, E6's blocks); the algebra of E3's library — the group the named permutations generate, which items are
products of others, the smallest generating set, and the library priced flat against generators + words; the merge
arithmetic for E6's blocks; the gradient arm's learned routes as a figure. What would be surprising: a generated group
much larger than the library, or no item being a product of others.

**E8 — structures of structures (pre-registered 2026-09-20 evening from E7's numbers).** One new kind of description whose
parameters are other descriptions, in two places. A: the library re-describes itself as a generating set chosen by price
plus a word for every other item; pass: ≤ 138.1 bits (flat 161.4); refute: never below flat. B: tasks whose permutation
is in the group the library generates but was never named (19 of 36), three learners on the same tasks — from scratch,
named items only, words; pass: with words, exact after ≤ 2 demonstrations on average and fewer bits than from scratch;
refute: no better than from scratch. C: E6's blocks consolidate into templates (same-kind blocks share one structure
name); pass: the two shift blocks ≤ 17.5 bits (22.2 separate) with retention unchanged; refute: no saving.

**E9 — planning in the written map (pre-registered).** The composition domain's seven primitives as the actions of an
environment over 6-digit sequences (V = 5). Phase 1: the agent observes each action k times on random states and
describes it (§4, one matrix per action; the position primitives as position permutations, inc/negate as value maps).
Phase 2: goals — a target sequence reachable from a random start in ≤ 3 actions — and the agent plans by
breadth-first search over words in its *learned* action descriptions, then the plan is executed on the true
environment. Measures: fraction of goals reached; plan length against the shortest possible (an oracle search on the
true actions); plan bits; all as a function of k = 1, 2, 3, 8 demonstrations per action. Control: a planner with the
same learned actions but no composition (single actions only) — it can only reach one-step goals. Pass: ≥ 95% of
goals reached at k = 3 with plans no longer than the oracle's; refute: < 80% at k = 3, or plans systematically longer
than the oracle's. Nothing is learned from reward; there is no value function.

**E10 — acting to learn (pre-registered 2026-09-20, late; OPEN-11).** E9's environment; the agent starts knowing four
actions (3 observations each) and never having seen three; a stream of goals reachable with all seven, some unreachable
with the known four. Before planning each goal it may spend up to 2 tries (execute an action to see what it does;
every executed action, try or plan step, is an observation). Policies for the tries: none; random; least observed
(count-based novelty); **price** — the action with the largest expected plan-bit saving for the current goal over the
hypotheses its partial description still allows, minus the try's own cost, and only if positive. Measures: goals
reached within a 6-step budget; mean steps to goal (failures count as the budget); goals until all seven actions are
described. Pass: price reaches at least as many goals as random and least-observed with fewer mean steps. Refute:
price worse than random.

**E10b — acting to learn with useless and noisy actions (pre-registered 2026-09-20, late).** E10's environment plus
two no-op actions and two noisy ones (a fresh random permutation every use; no description ever fits). Goals use the
seven real actions only. Policies: none; random; least observed until known, "known" = any description (it calls a
noisy action "identity with exceptions" after one try and stops); least observed until known, "known" = a description
with exception rate < 0.5 (it refuses to call noise known and keeps trying it — the trap); price (horizon), whose
hypotheses for an action refuted more often than not are empty, so its expected saving is zero. Pass: price reaches at
least as many goals as every novelty policy with fewer mean steps, and spends fewer tries on the noisy actions than
the ε-aware novelty policy. Refute: price worse than random.

**E11 — non-bijective actions (pre-registered 2026-09-20, late; §16 item 2).** E9's environment plus three many-to-one
actions — copy01 (position 1 := position 0), set0 (position 0 := 0), fill0 (every position := position 0) — and one new
structure, a position MAP (an output reads any input position, repeats allowed, or is written a constant). A: the
effect of each action learned from k observations, tested on 200 random states, one-to-one vs many-to-one, k = 1..4, 8.
B: planning with all ten actions by forward search, as E9. C (reported, not judged): how often a try of an unknown action
makes the goal unreachable within the budget — one-to-one unknowns vs many-to-one. Pass: A ≥ 0.95 exact at k = 4 for
the many-to-one actions and B ≥ 0.95 reached at k = 4 with plans no longer than the oracle's; refute: either < 0.8.

**E12 — abelian vs non-abelian action sets (pre-registered 2026-09-20, late).** Two sets of four actions learned from
4 observations each: abelian (rot_left, rot_right, inc, dec — rotations and value shifts commute) and non-abelian
(rot_left, reverse, inc, negate). A: from the learned maps alone, which pairs commute. B: do words of length ≤ 3 with the
same letter counts have the same effect (abelian: the effect of a word is its counts). C: planning by search (E9), by
COORDINATES for the abelian set (identify the rotation and shift needed, then arithmetic), by a WORD TABLE for the
non-abelian set (every reachable transformation with its shortest word); goals reached, optimality, states examined,
and the bits describing each planner. Pass: A 6/6 abelian pairs commute and ≤ 4/6 non-abelian; B 1.00 vs < 1.00; C the
coordinate planner ≥ 95% of goals with optimal plans and fewer planner bits than the word table. Refute: A wrong, or
the coordinate planner < 80%.

**E13 — the replica games (added 2026-09-20, late; an interface, then an analysis).** `arcgames.py` puts the loop of
§16 on the ARC-AGI-3 replica games in `src/tasks/games` (LockPath, MultiKey, Sokoban, CollectAll, Toggle, Tetris): the
frame is the state, the game's actions are the actions, the score and WIN / GAME_OVER are the only feedback, nothing
about any game is known. World model: one local rule per action — the new colour of a cell as a table over the
window around it, the same table at every cell, radius chosen by price. Goal model: the windows around the cells
that changed at a scoring or fatal transition, per action. Planner: search over predicted frames to a predicted win,
avoiding predicted deaths; with no known goal, to the nearest unknown or unvisited frame. Harness: `play_games.py`,
every level, a fixed action budget, the oracle's shortest solution beside it, and the world model's prediction
accuracy reported separately from the levels solved — so "could not learn" and "could not solve" are distinguishable.
Criterion for the interface: every game runs end to end and every action is exercised. No claim about solving.

**E14 — the order dependence of the price of a collection (pre-registered 2026-09-20, late; §8 change 3, §7).** The
25 compositions as a stream in several orders — primitives-first, compositions-first, and eight random orders — each
from an empty library. Measured: total bits paid over the stream; the final library (named items, the generating set
E8's word description chooses); the price of a fixed held-out set under each final library. The curriculum gap is best
minus worst total; an ideal compressor's gap is zero (symmetry of information), so the gap measures the learner's
distance from ideal. Predictions: primitives-first is cheapest; random orders spread by more than 10% of the mean;
final libraries differ across orders (hysteresis) and so do held-out prices; the run-2 transformer on the same orders
(held-out bits per digit) is order-dependent in the extreme. Refute: totals agree within noise.

**E15 — the capacity budget and parameter precision (pre-registered 2026-09-20, late; §8 changes 2–3, §9 rule 3).**
A: six rules in sequence, A seen 60 times and the others 20, under a budget of 48 bits (all six need ≈ 70); the layer
consolidates same-kind blocks into templates and then drops the block worth least (evidence × bits saved over a
table); compared with FIFO (drop the oldest) and no budget. Pass: the price rule ends under budget, keeps A where
FIFO drops it, and its evidence-weighted retention is at least FIFO's; refute: over budget or A dropped. B: a
real-valued offset from n noisy observations, written at the grid step that minimises the total code length; pass:
the best step shrinks as 1/√n (slope −0.5 ± 0.15 on log–log), which is what `precision_bits` assumes.

**E16 — blocks across layers (pre-registered 2026-09-20, late; OPEN-7).** E6's protocol on two-layer rules —
rot_left∘inc, reverse∘negate, swap_pairs∘inc, then the first again — a block being a whole TwoLayer description,
a demonstration the unit of routing and minting. Pass: retention ≥ 0.95 for every rule after every stretch and
exactly 3 blocks at the end in ≥ 90% of streams; refute: retention < 0.8 or more than 4 blocks on average.

**E17 — goals that are relations (pre-registered 2026-09-20, late; OPEN-12).** In E9's environment a goal is two
demonstration pairs of an unknown transformation (a composition of 1–3 actions); the plan must implement it on inputs
never seen. State planner (E9: any word taking the first input to its output) against a relation planner (identify
the transformation from the pairs with E2's learner, then the shortest word whose effect equals it on probe states).
Pass: the relation planner ≥ 90% on 20 fresh inputs and the state planner lower; refute: the relation planner < 70%.

**E18 — writing attention weights (pre-registered 2026-09-21; OPEN-8).** An induction circuit written into the
2-layer RoPE model from a description — a previous-token head (constant-channel query/key, phases set for "one back"
in the high-frequency rotary pairs) and an induction head (content match in the low-frequency pairs) — with no
training; test: a pattern of distinct tokens repeated, predict the second copy. Pass: ≥ 0.95 with a random model at
chance; refute: < 0.6.

**E21 — near-duplicate rules (OPEN-6; a measurement).** A, then D = A on 9 of 11 inputs: observations until D is minted its own
block; A's exception rate; D's accuracy on its differing inputs.
**E22 — annealing the budget (§8).** E15's stream under a hard cap of 40 bits against 80 → 40 in steps; pass: annealing
retains at least as much evidence-weighted accuracy at no more bits; refute: the hard cap retains more.
**E23 — macro-actions as the cache (OPEN-10).** E9's environment, goals 4–6 actions away, the library's named permutations
as macro-actions; pass: nodes expanded fall by ≥ 2× at no loss; refute: no reduction.

**E19 — hindsight by description (pre-registered in `notes/opsd_and_learning_from_experience.md`).** After an episode,
price the goal model's predictions against the outcomes: a win key that fired without a score is refuted; keys refuted
more often than confirmed stop predicting. Test on E13's LockPath door level. Pass: solved within the budget with no
game worse; refute: the same failure.

**E24 — the sleep pass (pre-registered 2026-09-21).** After every completed level, each action's rule drops the context
cells it does not need — greedy, while the merged table is cheaper and explains the evidence with no new exceptions.
Pass: bits fall by ≥ 20% and the levels that follow are solved in no more actions; refute: accuracy or levels lost.
**E25 — the trace diagnostic (a measurement).** Every step's frame, choice, prediction and outcome recorded; actions
decomposed into discovery / goal-directed / exploratory / blind; wrong predictions into missed, spurious and
wrong-colour changes; revisits, no-effect actions, refuted goal keys; "what was it thinking" examples on failed levels.

**E26 — the looped transformer (pre-registered 2026-09-21).** The gradient arm as prelude → core × K with the boundary
operator (RMSNorm + α·anchor between passes) → coda, after Chen et al. 2609.19107; tied K = 2/4/6, untied K = 4, tied and
untied growth 2 → 4; E0's task and budget. Pass: more executed depth raises the trained solve rate; any held-out
composition solved is the result to look for. Refute: no cell better than vanilla.
**E27 — the exploration fix (pre-registered 2026-09-21).** Exploration as the price of ignorance over every reachable
frame the model can predict: whole plans valued in bits (unknown windows to learn, filtered by learnability; a never-
visited frame worth the remaining budget spread over the unvisited frames in reach; untested predictions worth the
rule's exception rate), less the plan's cost. Pass: fewer discovery actions than E25 at no loss of levels; refute: more.

## 12. Open questions

**OPEN-1 — the outer objective (deferred by request, recorded here verbatim in substance).** ZipLearner learns a
generalisable world model — a model of everything put into its "world". It does not, by itself, say what to *do* with
that model, or in what order, to achieve an objective. In the language of the earlier experiments: ZipLearner's inner
objective may be very good, but the outer objective — using the world model to behave toward a goal — is unsolved. The
ideal is a general way of implementing an outer objective, so that any outer objective can be plugged in. Addressed
in §16 (first form, designed 2026-09-20 evening) with E9 as its first test.

**OPEN-2** — inverting a nonlinearity for hidden-layer targets (§6).
**OPEN-3** — whether the per-layer sweep settles (§6; measured in E2).
**OPEN-4** — what the capacity C of a given network is, numerically, in the price's units (§8).
**OPEN-5 → BUILT in its first form (E8):** structures of structures — `WordLibrary`, `consolidate`. Remaining: words of words; inferred template parameters. Originally: whether §7's discovery is enough, or the library itself needs a library. *E7 priced it: E3's library is 23 bits cheaper as two generators + words, E6's shift blocks 4.6 bits cheaper as one template with a per-context offset; the missing piece is one structure kind whose parameters are other structures.*
**OPEN-8** — writing attention weights under PoPE: content matching is not separable from the phase cosines (E0), so "same digit anywhere" is not one written parameter; what the written form of an induction head is, is not designed. *E18: written, under RoPE — a previous-token head in the high-frequency rotary pairs and an induction head in the low-frequency ones, 100% with no training. Open: ZipLearner choosing such a circuit as the cheapest description; PoPE.*
**OPEN-9** — the J-space of the gradient arm (Gurnee et al., Anthropic, July 2026: J-lens = the average Jacobian of the final residual with respect to a layer's residual, pulled back through the unembedding; J-space = sparse non-negative combinations of the per-token J-lens directions; a mid-layer global workspace, ≤10% of variance but causal for flexible reasoning). Under attention residuals the Jacobian's identity highway becomes a content-dependent route weight, so the J-space should localise to the sources later mixers read. *Measured (E8b): it does — identity share 0.06/0.04/0.12/0.77 = the routes; early-layer J-lens swaps drop from 72–89% to 13–23%.*

## 13. Results — the log is `RESULTS.md`

One line per experiment, in the order they were run; the full entries (command, files, numbers, verdict, what
changed) are in `RESULTS.md`, appended and never edited.

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

## 14. Glossary

- **bits** — the unit of price; log₂11 ≈ 3.46 bits is the cost of naming one digit out of eleven with no information.
- **block** — the rows of a matrix belonging to one context; one matrix, several blocks.
- **capacity (C)** — the most bits of description a given network can hold.
- **evidence count (n)** — how many observations a description has explained; kept beside it, replaces a replay buffer.
- **exception** — an observation a written matrix predicts wrongly.
- **interface** — the boundary where one layer's output is the next layer's input.
- **mint on refutation** — create a new description only when every existing one is refuted, never merely because data is incomplete.
- **lossless / lossy** — a description that reproduces every observation exactly / one that accepts some error to save bits.
- **one-hot** — a digit as a vector of zeros with a single 1.
- **permutation matrix** — one 1 per row and column; a one-to-one relabelling.
- **price** — total bits: structure name + parameters + exceptions.
- **rate–distortion** — the theory of how many bits (rate) a given tolerated error (distortion) costs.
- **route / read-from** — which source a layer reads; fixed in the written network, a softmax in the gradient arm.
- **source** — the embedding or an earlier layer's output, as something a layer may read.
- **structure / family** — a named set of matrices picked out by a few parameters.
- **sweep** — solving each layer in turn with its neighbours frozen, then repeating.
- **target** — what a layer's output should be.
- **write** — setting every entry of a matrix from a structure's parameters, including entries for inputs never seen.
- **λ (lambda)** — the exchange rate between bits and error in a lossy price.

## 15. Substrate decisions — DECIDED 2026-09-20 (reversible)

The transformer used as the gradient-trained comparison arm, and later as the network whose weights ZipLearner writes,
is the `h1_lid` model (3 transformer blocks, d = 64, 4 heads, about 150k weights) with two changes.

**Positional scheme: PoPE for the written network, RoPE for the gradient arm; the θ = 0 channels are WITHDRAWN (E0, results log: they destroy training — a flat channel is a magnitude bias, not a content match).** *The rest of this paragraph is the original reasoning, kept for the record.* PoPE (polar coordinate positional embeddings,
arXiv 2509.10534; already implemented in `transformers/h1_lid.py`) puts *content* in the magnitude of each query and key
and *position* in the phase, with a learnable per-channel offset δ_c saying which relative position the channel
prefers. RoPE mixes the two inside one cosine. What ZipLearner needs is position as a separable, writable parameter:
"attend to position s − 6" is the single number δ_c = −6·θ_c under PoPE, and "same digit" is a written identity on the
magnitudes — so the demonstration-reading circuit (find the earlier copy of the query's digit; copy what followed it)
is two heads of one pure kind each. The θ = 0 channels were meant to give content-only matching a home; E0 showed they do the opposite (see the log). The recorded one-seed result (RoPE 9/17 vs PoPE 7/17 trained tasks, both 0/8 held out) is not
evidence either way.

**Residual scheme: attention residuals, one transformer block per AttnRes block.** Reference notes and the paper are
in `refs/attention_residuals.md` and `refs/attention_residuals_2603.15031.pdf`. Instead of every layer reading the
same running sum of all earlier outputs, each layer's input is a softmax-weighted mix of *sources* — the embedding and
the earlier blocks' outputs — with one learned vector per layer as the query, RMSNorm on the keys, and every query
initialised to zero (the paper's own requirements). With our three blocks this is at most four sources per layer, so
memory is not a concern and the block size is a flag (`S = 1`, every attention and MLP output its own source, is the
paper's "Full AttnRes"). Reasons, in order: (1) the routing weights are the network's wiring diagram, readable
directly; (2) §6's pinned interface becomes literal (a layer reads a chosen source, not a sum), so inverted targets have
a single destination; (3) the route is a structure with a price (§5, read-from), chosen in the written network by
trying each source and keeping the cheapest description; (4) the paper's ablations say what to keep — softmax, not
sigmoid; RMSNorm on keys; one mix for all channels, not per head. Expected effect on composition in the gradient arm:
easier plumbing (layer 2 can read layer 1's output without the embedding still sitting in a sum), but plumbing does not
create the parts — our transformer's failure was "recognise the table and recite it", and E0 measures rather than
assumes any change.

What is *not* decided: whether the gradient arm's per-token softmax routing or the written network's fixed routing is
the right comparison when both exist; E2 will run the written network, E0 the gradient arm, and the two are reported
side by side, not merged.

## 16. The outer objective — the loop BUILT (E9–E12, E17), the games first contact (E13), the rest OPEN

**The problem.** Everything before this section is a world model: descriptions that say what comes out when something
goes in. The outer objective is to *act*: given a goal, choose actions that reach it. The ideal is that any goal can
be plugged in without changing the machinery.

**What acting needs, and what compression already built.**

1. *A map — states joined by actions.* Each action is a structure (what it does to a state), and actions chain. E7/E8
   found that the library already holds this map: the named permutations form a group reachable from three
   generators, and a word in the generators is a path through it. **A plan is a word.** Read as a network, a plan of n
   actions is an n-layer written network whose routes are the order of the actions; executing it is the forward pass.
2. *An inverse — from a wanted change to the action that causes it.* Free for bijective structures (apply S⁻¹); for
   actions that are not bijections, one more matrix from state-difference to action, learned by counting (§4).
3. *A goal slot and a price for plans.* A goal is a description in the model's own language — a target state, or a
   relation. A plan costs its bits (word length × log₂ actions). A goal's **worth** is the one external number; it is
   what makes the objective pluggable. Selection = cheapest (plan bits − worth) wins: the same rule as everywhere.
4. *Not required, until shown otherwise:* a value learned by reward for every state. With a map and an inverse,
   cost-to-go is computed by search on demand and cached only where search is too expensive (OPEN-10 below).

**The loop (first form).** Observe (state, action, next state) triples → each action gets a description by the core
loop of §4 (one matrix per action; the library of §5 supplies the structures) → given a goal, search the words in the
learned actions for the cheapest one that reaches it (breadth-first over depth, so the shortest word is found
first; bits = length × log₂ actions) → execute → the next states are new observations for the inner objective.

**Where the two objectives meet.** An action whose effect is not yet described cannot be planned with; a plan that
would be cheaper through such an action is a reason to *try* it. That is exploration as the outer objective's own
demand, with the inner objective paying it back in bits: OPEN-11.

**The games (E13).** The loop above runs on the ARC-AGI-3 replica games through `arcgames.py`: the state is the frame, an action's description is a local rule (a window → the centre's new colour, tied across cells), the goal is a set of win keys learned from the score, and the planner is the search above over predicted frames. First contact solved 5 of 16 levels by discovery (E13); with the goal model priced against the score (E19) 8 of 16; see the log for what fails and why.

**OPEN-10** — when the map is too large to search: what to cache (a cost-to-go per state, or per description?), and
whether the cache is itself a written structure. *E12: coordinates where the actions commute (8.9 bits), a word table where they do not (E8's `WordLibrary`, a written structure); a map too large for either is not tested.* **OPEN-11** — acting to learn: the price of trying an unknown
action against the bits its description would save. *E10: the saving must be counted over the goals still to come (a description is an asset); the myopic version does not explore. E10b (useless + noisy actions): price 92% of goals in 2.3 steps, novelty policies 36%/23% — trapped by the noisy actions; BUILT in its first form.* **OPEN-12** — goals that are relations, not states ("make the
output the reverse of the input"), and goals over the library ("find a shorter word"). *E17: a relation given as examples is planned for by matching plans to the examples (0.99); identifying it first fails at the identification limit (0.51). Goals over the library: not built.*

## 17. Files

- `experiments/ziplearn/ziplearner.py` — the weight-writing ZipLearner: `Structure` and the library v1 (`Table`, `Identity`,
  `Shift`, `Affine`, `Permutation`), `Layer` (one matrix, §4), the rate price (`flag_bits`, `pay`, §8), `PositionPerm` /
  `PositionIdentity` / `TwoLayer` (two layers, §6), `NamedPerm` / `PermLibrary` / `TwoLayerWithLibrary` (§7), `ContinualLayer` (§9).
- `experiments/ziplearn/RESULTS.md` — the results log (append-only), with the summary table mirrored in §13.
- `experiments/ziplearn/arcgames.py`, `play_games.py` — the interface to the replica games (`src/tasks/games`) and the harness (E13).
- `experiments/ziplearn/e0.py` … `e11.py`, `anatomy.py`, `jspace.py` — the experiments of §11, one file each; outputs in `runs/e*/`.
- `experiments/ziplearn/refs/` — reference notes (`attention_residuals.md`) and the paper PDF.
- `experiments/transformers/h1_lid.py` — the transformer substrate: PoPE (+ the withdrawn `--n_zero`), attention residuals
  (`AttnRes`, `--res attnres|attnres_full`), `Model.routes`, `--json`.
- `experiments/inner_objective/tasks.py` — the two task families (arithmetic, composition), reused by every experiment.
- `experiments/inner_objective/ziplearn.py`, `runs/ziplearn_*.json` — the earlier arithmetic-only version and its numbers (§2).
