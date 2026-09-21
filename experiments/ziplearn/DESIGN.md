# ZipLearner — design document

*v0.8, 2026-09-21 (v0.1–v0.5 on 2026-09-20; v0.6 records E14–E18; v0.7 adds §18 thinking inside the block, §19 the library problem and the continuous thesis; v0.8 adds §20 credit assignment without backpropagation; earlier: capacity, precision, order dependence, blocks across
layers, relational goals, written attention). E0–E30 and E33(a) have been run (E29 retracted as a design; E31 running; E32 and E33(b,c) shelved; E35 then E34 next); one line each in §13, the full entries in `RESULTS.md`. Code:
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

**E28 — the written looped block (pre-registered 2026-09-21; OPEN-8, §16 item 1).** The `LocalRule` tables learned on
LockPath (levels 0–1, with sleep) written into one attention block with no training: the entries as memory tokens
(window colours per offset, action, outcome), the frame's cells as tokens with one-hot coordinates; a gather layer
(one head per mask offset, query = the cell's coordinates permuted by the offset) and a lookup layer (query = the
window and the action, keys = the entries, value = new colour minus old); looped once per action of a plan, the action
as the per-pass anchor, the boundary operator clears the gathered subspaces and re-quantises the colour. Pass: the
block's rollouts equal the planner's on ≥ 0.98 of random 1–4-action plans whose windows were all known; refute:
< 0.9. Reported: agreement with the true game, and the block's nearest-key default against the planner's
"unchanged" on plans that met unknown windows.

**E29 — learned search: thinking as acting (pre-registered 2026-09-21; §18).** The imagination of §18 against E27's
breadth-first search, both with the sleep pass, 150 actions per level, 4000 model calls per real step, on (a) the
four games whose goal is learnable on their first level — LockPath, MultiKey, CollectAll, Toggle — and (b) a LockPath
made of LockPath's level 0 followed by three empty rooms of side 12, 20 and 40 with the goal in the far corner
(oracle 8, 20, 36, 76), the search depth unlimited for both arms (the BFS arm gets 3 actions on the room of 40, whose
1,600 positions exceed the 1,000 frames its budget can expand). Measured: real actions per level; model calls to the
first imagined success at each level's first plan; the strategy's features and contexts after each sleep. Pass:
(a) the learned arm's total actions over the four games within 1.1× of BFS's (it IS breadth-first until it has
learned, so it must not lose), (b) calls to the first success on the rooms of 20 and 40 below BFS's on the room of
20 and at most 3× the path length × 4, and (c) the room of 40 solved by the learned arm. Refute: (a) above 1.25×,
or (c) not solved.
*Result: PASS on these criteria (`RESULTS.md`). REJECTED as a design the same day and deleted — a search procedure
written around the model, with a thought defined in the task's terms and a hand-built context (§18, the rule).*
**E30 — the depth loop under attention residuals (pre-registered 2026-09-21; §18).** `LoopedModel` with attention
residuals across passes: the sources at pass k are the anchor and the outputs of passes 1..k−1. Cells, on E0's task and
budget (3200 steps, seed 0): (a) the fixed boundary operator — E26's tied K = 4 and untied growth 2 → 4; (b) attention
residuals with the mixers tied to the core (routing by content); (c) one mixer query per pass; (d) windowed sources,
the anchor + the last 2 passes. Measured: trained and held-out compositions solved and mean accuracy (E26: 13/17,
held-out 0.12–0.20); the routes (mean weight per source at each pass); accuracy when run at twice the trained loop
count, for (a), (b) and (d). Pass: some attention-residual cell at or above E26's best held-out accuracy with routes
that are not uniform; the question answered either way: does the learned route reproduce the boundary operator
(anchor + last pass) or read further back. Refute: every attention-residual cell below the fixed operator.
*Result: PASS — held-out 1/8 (0.22) in both tied-core attention-residual cells, the first held-out composition solved in
the line; the route lets the anchor fade (1.00 → 0.01) and reads every earlier pass with decaying weight; the windowed
variant loses the gain and collapses at twice the passes (0.19), so §18(iv) is refuted; untied stacks gain nothing.*
**E31 — continuous thoughts under attention residuals (pre-registered 2026-09-21; §18).** Coconut's recurrence in the
gradient arm on E0's composition task: c thought positions between the demonstrations and the answer, each fed the
previous position's final state; trained with Coconut's curriculum from a chain-of-thought form of the task (the
intermediate result written as tokens, then replaced by thoughts stage by stage). Cells: c ∈ {1, 2, 3} × {plain
residuals, attention residuals} × {raw feedback, the boundary operator on the fed-back state: RMSNorm + anchor}.
Measured: held-out composition accuracy (E26's best 0.20); training-loss spikes at c = 3 (Coconut's instability) with
and without the operator; a probe of the thought for the intermediate result and for more than one candidate at once
(the breadth-first claim). Pass: held-out accuracy above 0.20 in some cell, and the operator removing the c = 3
instability where it appears. Refute: no cell above the looped model.
**E32 — "look for a name" on a continuous representation (pre-registered 2026-09-21; §19).** The gradient-trained
tied-core looped model of E30 (attention residuals, tied mixers) — a net whose contents E7's anatomy can check. A
sleep pass over its activations and weights: (1) a sparse dictionary over the residual stream at each pass (features as
directions, Bricken et al.'s form), priced in bits at §8/E15's precision; (2) low-rank factors of the core's matrices
at the same price; (3) where a written table or program is cheaper (the digit primitives), that. Measured: the
description length before and after; the behaviour under the compressed description (trained and held-out accuracy
within 0.02 of the original); whether the dictionary's features coincide with the task's primitives (each feature's
effect matched to a primitive, as E7 did for the written net); and continual learning — a second task family trained
by gradient descent after consolidation, the first family's accuracy retained (E6's measure) against plain fine-tuning.
Pass: lossless within 0.02 at no more than half the bits, features that match primitives, retention above fine-tuning.
Refute: the compressed description loses more than 0.1 accuracy, or nothing structured is found.
**E33 — a language model on the Latin corpus: data efficiency (pre-registered 2026-09-21; §19; asked by the user).**
`corpora/latin books` (16 books, 3.0 MB, 404k words), one book held out, character level. Arms: (a) the ZipLearner
language model — the discrete machinery of E24/E28 on text: a table over the last R = 8 characters with the sleep
pass choosing which context positions matter, the count-based two-part code, prediction by the KT rate and the
nearest-context default; it is E28's block with the corpus as the memory tokens; (b) the gradient transformer
(`h1_lid.Model`, 3 blocks, d = 96, RoPE, a 128-character context) trained at each data size; (c) the hybrid the
continuous thesis proposes: (b) consolidated by E32's sleep, then trained on. Measured: bits per character on the
held-out book at 10³, 10⁴, 10⁵, 10⁶ and 3×10⁶ training characters — the data-efficiency curve — and the crossover
size at which (b) overtakes (a); a general compressor's bits per character on the same split (xz) as the classical
reference; a plain order-8 n-gram with KT smoothing as the control for (a). Expectation, pre-registered: (a) wins below
the crossover, (b) above, and (c) shifts the crossover left if the thesis is right. Refute the apparatus: (a) no better
than the plain n-gram. (a) and (b) can run before E32; (c) needs it.
*Result (a): measured, `RESULTS.md` — frozen 1.87 and online 1.82 bits/char at 2.7M characters (xz 2.26); the sleep pass keeps
every position once its price is the prequential code (E24's per-context price was wrong for a backing-off predictor);
the discrete machinery on text is PPM. Arms (b) and (c) shelved by the user's instruction of 2026-09-21: no further
gradient-arm work unless stated. E32 is shelved by the same rule (it needs a gradient-trained net).*

**E35 — the written denoising chain (pre-registered 2026-09-21; §20; first in the approved order).** The looped block
fitted BLOCKWISE by counting, no gradient anywhere: K blocks on a noise schedule 1 = t_K > … > t_1 > t_0 = 0; block k
maps the data at t_k to the data at t_{k−1}; a block is a `LocalRule`-style table from the window around a cell at t_k
to the cell's value at t_{k−1}, learned from (noisier, cleaner) pairs generated from clean data, priced by the two-part
code, compressed by the sleep pass; running the chain from pure noise is generation, from level k is repair. Domain 1,
the replica games' frames (the frames stood in during exploration of LockPath and CollectAll levels, split into training
and held-out frames), corruption = each cell replaced by a uniform random colour with probability t, K = 8. Measured:
(a) REPAIR — a held-out frame corrupted at t = 0.25 / 0.5 / 0.75, cell accuracy of the chain from that level against the
ONE-SHOT control (a single table from the corrupted window straight to the clean cell — the "no depth" model at the
same window size); (b) GENERATION from pure noise — per-cell agreement with the nearest training frame, and the number
of distinct frames among 100 samples; (c) each block's table before and after sleep (entries, bits, cells kept).
Domain 2, text (`corpora/latin books`, the E33 split): masked-character diffusion — each character masked with
probability t (absorbing: masked stays masked until a block fills it), a block fills masked characters from a window of
±4 characters in which masks are visible; measured: reconstruction accuracy of the masked characters on the held-out
book at t = 0.15 / 0.5, chain against one shot. Pass: repair at t = 0.5 with the chain at least 0.05 above one shot on
the frames, and the same sign on text; sleep shrinking the per-block tables. Refute: the chain no better than one shot
anywhere.
**E34 — gradient-free context mixing on the Latin stream (pre-registered 2026-09-21; §19–§20; second).** A library of
context functions — character orders 1–8, skip pairs at distances 2–4, the previous word (the characters since the last
space), the position in the line, and a match model (E18's induction written: the character that followed the last
occurrence of the current 6-gram) — each a count table with the KT rate; selection by the prequential code (greedy
add and drop, E33's price); mixing by Bayesian weights in closed form (exponential weights per context class, with
fixed-share switching for non-stationarity), no trained mixer. Measured: online bits per character on the held-out book
against the table's 1.822 and xz's 2.263; which contexts survive selection. Pass: at or below 1.6 (the literature's
fraction would put it near 1.3–1.4); refute: no gain over the table.

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
| E28 | PASS | the rules learned on LockPath, written into one looped attention block, reproduce the planner's rollouts 530/530 (509/509 seed 1); the nearest-key default beats 'unchanged' on unknown windows |
| E29 | PASS, then RETRACTED | learned search as a procedure: room-20 89 vs 818 calls, room 40 solved at oracle 74; rejected as a design (a search written around the model) and deleted; the rooms stay as the test bed for §18 |
| E30 | PASS | the depth loop under attention residuals: held-out 1/8 solved (0.22) with a tied core — the first ever; the learned route lets the anchor fade and reads every earlier pass; windowed sources collapse at 2× passes |
| E33 (a) | measured | the discrete machinery on Latin text = a PPM-style blended-backoff model: 1.87 bits/char frozen, 1.82 online at 2.7M chars (xz 2.26); the E24 sleep price is wrong for text, the prequential one keeps every position; transformer/hybrid arms shelved |

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

**What the attention residuals learn, and what that means for the written route (paper re-read 2026-09-21; asked by
the user).** The paper's ablations on its 16-layer model (validation loss): a fixed per-layer scalar route over all
earlier layers (DenseFormer) 1.767 = the baseline 1.766; its own mix with the query and key removed and learned scalars
instead 1.749; full AttnRes with the content keys 1.737; a sliding window over the last 8 layers 1.764. So a route that
does not depend on the token's content is worth nothing, and what is worth something is selective, content-dependent
access to DISTANT layers. What the trained routes look like (their Fig. 8): "each layer attends most strongly to its
immediate predecessor, yet selective off-diagonal concentrations emerge … indicating learned skip connections"; "the
embedding h₁ retains non-trivial weight throughout, especially in pre-attention layers"; pre-MLP mixes sharper and
local, pre-attention mixes broader; and (§6.2) "depth-wise attention sinks, where certain layers consistently attract
high weight regardless of input". Their frame (§6.1): the standard residual is a linear recurrence over depth; AttnRes
is softmax attention over depth, "just as Transformers replaced temporal recurrence with self-attention".
Consequences here. (1) The earlier note that the written route is "a fixed choice per layer priced at log₂(sources)"
is withdrawn: a fixed choice is what the ablation shows to be worthless. The written route is a pseudo-query over
CONTENT — in our block the content that should decide a route is explicit (the token-class flags cell / entry / null,
and state flags such as "the lookup found an entry"), so a written query on those dims is a hard, per-token route: a
branch, which is also how Giannou et al. get conditional branching; its price is a small table over the flags. (2) The
paper's sink layer is our anchor: kept as a source at every pass. (3) The structure to expect from a trained loop, and
so the shape to write, is local + anchor + a few skips; E30 measures it across passes (its first attention-residual
cell: the pre-attention mixer reads the last pass at 0.78–0.89 with a decaying tail to earlier passes and the anchor
fading from 1.00 to 0.01; the pre-MLP mixer keeps 0.23 on the previous pass and ~0.1 on each older one beside 0.46 on
the current attention — an Anderson-like blend of iterates; the final mixer 0.79 on the last pass, 0.14 on the one
before). (4) Attention weights in a frontier model, for the same question (from memory of Olsson et al. 2022, Elhage
et al. 2021, Wang et al. 2022, Geva et al. 2021, Todd et al. 2023): attention is the ROUTING — previous-token and
induction heads, retrieval heads, binding/name-mover heads, function-vector heads that identify the task in context —
while knowledge sits mostly in the MLPs as key-value memories; the clean algorithmic heads are few, composed across
layers (Q/K-composition), and gradient descent finds them in phase transitions. In ZipLearner's terms a head is a
PREDICATE on earlier positions (relative offset; same content; a class flag → q/k) and a TRANSFORM of what is copied
(→ v/o); predicates are already learned as position structures (`PositionPerm`/`PositionMap`, E9–E12) and transforms
as content structures, and E28's gather head / E18's induction head are what two of them compile to. So "ZipLearner
writes attention weights" means: the head KINDS are the instruction set, a small library of predicate/transform pairs
with written weight templates; ZipLearner chooses which instances exist by description length (the sleep pass does
this for offsets), writes the programs they execute, and mints a new kind only when no program in the existing set
compresses the data (§7's bet). Not designed: the minting, and the compiler from a learned predicate to q/k phases in
general (E18 did one case by hand).

**Recurrence scheme — DECIDED in outline 2026-09-21, the cells in E30/E31 (§18).** Two recurrences: depth (the core
looped over the same positions, `LoopedModel`, E26) and sequence (Coconut's continuous thoughts: a position's final
state as the next position's input). One boundary operator for both axes: normalise the carried state, re-inject the
anchor (Geiping et al.: a recurrence that does not re-read its input each step is unstable), and let attention
residuals choose the rest — the fixed `rms_norm(x) + α·anchor` is the one-source special case. The loop count and the
thought count are a convergence test on the state or a written halting head, never a pad or a budget. The block is
read as an interpreter (Giannou et al.): weights written once, programs — knowledge and behaviour — as tokens in the
context. The interactions with attention residuals are argued in §18 and measured in E30 (depth) and E31 (sequence).

## 16. The outer objective — the loop BUILT (E9–E12, E17), the games first contact (E13), the rest OPEN

**The problem.** Everything before this section is a world model: descriptions that say what comes out when something
goes in. The outer objective is to *act*: given a goal, choose actions that reach it. The ideal is that any goal can
be plugged in without changing the machinery.

**What acting needs, and what compression already built.**

1. *A map — states joined by actions.* Each action is a structure (what it does to a state), and actions chain. E7/E8
   found that the library already holds this map: the named permutations form a group reachable from three
   generators, and a word in the generators is a path through it. **A plan is a word.** Read as a network, a plan of n
   actions is an n-layer written network whose routes are the order of the actions; executing it is the forward pass. *E28
   built it as ONE block looped n times (E26's substrate): the rule table as memory tokens, the frame as the sequence,
   a neighbour as a written coordinate permutation, the action as the per-pass anchor — exact agreement with the
   planner's rollouts, and a nearest-key default for windows never seen that beats "unchanged".*
2. *An inverse — from a wanted change to the action that causes it.* Free for bijective structures (apply S⁻¹); for
   actions that are not bijections, one more matrix from state-difference to action, learned by counting (§4).
3. *A goal slot and a price for plans.* A goal is a description in the model's own language — a target state, or a
   relation. A plan costs its bits (word length × log₂ actions). A goal's **worth** is the one external number; it is
   what makes the objective pluggable. Selection = cheapest (plan bits − worth) wins: the same rule as everywhere.
4. *Not required, until shown otherwise:* a value learned by reward for every state. With a map and an inverse,
   cost-to-go is computed by search on demand and cached only where search is too expensive (OPEN-10 below).

**The loop (first form).** Observe (state, action, next state) triples → each action gets a description by the core
loop of §4 (one matrix per action; the library of §5 supplies the structures) → given a goal, search the words in the
learned actions for the cheapest one that reaches it (breadth-first over depth *— the first form; superseded by §18: the search is not a procedure —*, so the shortest word is found
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
- `experiments/ziplearn/e28.py` — `WrittenSim`: the learned rules of a game written into one looped attention block (E28).
- `experiments/ziplearn/refs/` — reference notes: `attention_residuals.md` (+ PDF), `scaling_exponents_recursion.md`,
  `latent_reasoning_and_looped_planning.md` (overview) and one file per paper of §18 (Coconut, recurrent depth, looped latent
  thoughts, looped transformers as computers, VIN, Universal Transformers, ACT, Searchformer, Stream of Search, Thinker).
- `experiments/transformers/h1_lid.py` — the transformer substrate: PoPE (+ the withdrawn `--n_zero`), attention residuals
  (`AttnRes`, `--res attnres|attnres_full`), `Model.routes`, `--json`; `LoopedModel` (`--res loop`, with `--loop_res attnres
  --mix --window --extrap`, routes per pass); a key/value cache for incremental forwards (`forward_embedded`).
- `experiments/transformers/coconut.py` — continuous thoughts (E31): the curriculum, raw vs operator feedback, the probe.
- `experiments/ziplearn/e30.py`, `e31.py` — the recurrence-scheme experiments of §18 (drivers over the two files above).
- `experiments/ziplearn/textlm.py` — the discrete machinery as a character language model on `corpora/latin books` (E33 a).
- `experiments/ziplearn/refs/diffusionblocks_2506.14202.md` — the reference behind §20.
- `experiments/inner_objective/tasks.py` — the two task families (arithmetic, composition), reused by every experiment.
- `experiments/inner_objective/ziplearn.py`, `runs/ziplearn_*.json` — the earlier arithmetic-only version and its numbers (§2).

---

## 18. Thinking inside the block — the recurrence scheme (DESIGNED 2026-09-21; E29's procedure REJECTED and deleted)

**The rule (2026-09-21).** Four constraints, set by the review of E29 and binding on everything behavioural:
1. *The only primitives the model reaches for are the game's controls.* No meta-actions (imagine / reset / commit), no
   policy over thoughts.
2. *The only context is the context window*: the history of frames (cells as tokens, E28's layout), the actions
   taken, and the memory tokens ZipLearner writes. No retina, no locus, no feature extractor outside the model. "The
   cell at offset (di, dj) from the thing that moves" is a gather head with a written coordinate permutation (E28);
   "colour c anywhere to the east" is a head with a directional positional phase (E18); which such heads exist is the
   sleep pass's choice, written as weights.
3. *A thought is the block's own computation* — one pass of the looped core over its latent state. It assumes
   nothing about frames; what a pass computes is whatever the written weights compute. Code for the thinking process
   — an expansion, an enumeration, a heap, a budget, even an adaptive one — is forbidden, and so is defining a thought
   in the task's terms ("one expansion of an imagined frame").
4. *What the block does when it loops is written by ZipLearner from experience*, or it does not exist yet.

What this removed: `Player.search` (E27's breadth-first search) and `Player.imagine` / `PolicyRule` / the retina and
locus (E29), all deleted from `arcgames.py`. The games are explored (E27's price of ignorance) and not solved on
purpose until the block plans. E29's numbers stand in `RESULTS.md` as a measurement of what a procedure could do; its
design is retracted.

**The frame: the block is an interpreter; knowledge and behaviour are programs in the context.** Giannou et al.
(ICML 2023) loop a 13-layer transformer with WRITTEN weights and get a general-purpose computer: "our input sequence
acts as a punchcard, consisting of instructions and memory for data read/writes" — a program counter, conditional
branches, function calls, all as attention over the input. E28 is this in miniature: the lookup head is one
instruction (match the window, write the colour), the memory tokens are the program, a loop executes it once per
action. Under the rule, planning, valuing and exploring must be programs too — tokens ZipLearner writes by compressing
experience — executed by the same loop. The search for programs is compression, priced in bits; the execution is
looping. This is exactly the idea. What no paper gives is how the programs are found: theirs are written by the
authors; ours must be the cheapest description of the agent's own successful behaviour, and that is the open
question of this section.

**Two recurrences, and what each is for.**
- *Depth*: the core applied K times to the same positions (Chen et al. 2609.19107, our `LoopedModel`; Geiping et al.
  2502.05171, 3.5B parameters, "iterating a recurrent block, thereby unrolling to arbitrary depth at test-time";
  Saunshi et al., ICLR 2025: T loops simulate T steps of chain-of-thought). Its state is the residual stream of a
  fixed set of positions; a pass refines them all toward a fixed point — Geiping et al. see most tokens "converge to a
  fixed point", some fall into "an orbit pattern", some drift "in a single direction".
- *Sequence*: Coconut (Hao et al. 2412.06769) — the last hidden state of position t is fed back "as the subsequent
  input embedding directly in the continuous space": a continuous thought is a new position that attends to the
  question and the earlier thoughts. Their finding: "the continuous thought can encode multiple alternative next
  reasoning steps, allowing the model to perform a breadth-first search" — a frontier in superposition, no search
  written.
They are not substitutes. Depth recurrence refines a fixed state; sequence recurrence ADDS state — positions the loop
can write to and read back, memory for a plan of unknown length. A fixed-point computation (a value over the frame,
VIN below) wants depth; a rollout wants sequence — E28's rollout was the sequence recurrence done by hand, one pass
per action; under Coconut's form the same computation is the block adding a thought position per step. The block gets
both: K passes per thought position until the state converges, then the converged state becomes the next thought.

**How the depth loop interacts with attention residuals (the question to settle first; E30).** As built,
`LoopedModel` ignores `res` — its blocks are always "std" — and uses the fixed boundary operator
`x ← rms_norm(x) + α·anchor`: a two-source mix, the last pass normalised plus the prelude's output. Attention
residuals generalise it. At pass k the sources are {anchor, out₁, …, out_{k−1}} (each pass's output, in the paper's
Block form) and the core's mixers read them with a learned query per mixer, softmax over sources, RMSNorm on the keys,
the raw sources as values. What follows: (i) Geiping et al.'s stability condition — the input must enter EVERY pass, or
"the iterative process would not be stable" — holds by construction, with the amount learned per token instead of
the constant α; (ii) a pass can read a state from several passes back, not only the last: a learned combination of
previous iterates, which is what Anderson mixing / momentum does for a fixed-point iteration — the concrete reason to
expect help, not harm; (iii) with tied cores the mixers are tied too, so the routing is by content (the query matches a
source's normalised content), unless one small query vector per pass is added (untied mixers on a tied core); both
are cells; (iv) the source set grows with K, so at a test depth beyond the trained one the softmax spreads over more
sources — a WINDOWED source set (the anchor + the last m passes) keeps the pass a function of a fixed-size state, which
is also what a fixed point and a convergence test require; m = 1 is the boundary operator itself; (v) the paper's
mix is one weight per source for all channels, but the written block's boundary operator clears SOME subspaces (the
gathered colours) and keeps others — a per-source scalar cannot; in the written form the clearing must happen inside
the heads (write new − old; E28's lookup value already has that form). Not decided, noted.

**How Coconut interacts with attention residuals (E31).** A thought position's "embedding" source is the previous
position's final state (under Full AttnRes, the final mixer's output that feeds the head). With plain residuals that
vector re-enters at layer 0 and is re-processed upward; with attention residuals every layer reads the embedding
source directly, so depth-j content of the previous thought reaches depth j of the next — a recurrence between
matching layers across positions (the Feedback Transformer of Fan et al. 2020 built this with a fixed mix, from
memory). The reason to expect it positive: a late-layer plan state carries to the layer that continues it without a
round trip through the early layers. The hazards, verified in the papers: Coconut needs a curriculum — trained
without it "the models trained this way do not perform any better than no-CoT" — and gets unstable as thoughts per
step grow ("c=3 … a sharp spike in training loss"); its number of thoughts is a pad, not a decision; its arithmetic
results trail chain-of-thought (GSM8k 34.1% vs 42.9%). The user's recollection of a drift problem matches the closest
verified statements: Coconut feeds the raw last hidden state back with no normalisation between thoughts, and
Geiping et al. prove a recurrence that does not re-read its input each step is unstable. Consequence: ONE boundary
operator for both axes — normalise the carried state, re-inject the anchor (the question, the goal), let attention
residuals choose the rest — between passes (depth) and between thoughts (sequence). The thought count: not a pad but
the same convergence test as the depth loop (Geiping's KL threshold on successive states) or a written halting head
(Graves's ACT, the Universal Transformer).

**What the loop can compute that E29 wrote as a procedure.** Value Iteration Networks (Tamar et al. 2016): value
iteration on a grid is K iterations of one layer — a convolution per action and a max — so the planner IS a looped
layer over the frame; the E28 block has the convolution (gather heads) and the transition (lookup head), and a backup
is a max over the anchor slot. Coconut: a superposed frontier in the state, pruned by the model. E29's rooms
(`RESULTS.md`) stay as the test bed: calls to the goal grew with the room's area under breadth-first search and stayed
flat under a strategy; the block, looped, must show that flatness with nothing around it but the game.

**Search learned from one's own traces.** Searchformer (Lehnert et al. 2024) and Stream of Search (Gandhi et al.
2024): the second stage of both — compress the model's own successful traces — beats the teacher and solves what no
hand-written strategy could. That stage is the ZipLearner loop. The first stage (imitate a written search) is
forbidden by the rule and unnecessary when the first successes come from exploration (E27).

**E30's answer (2026-09-21, `RESULTS.md`).** The trained routes are not the boundary operator. Before attention a
pass reads the last pass at ~0.8, keeps a decaying read of every earlier pass (0.04–0.20), and lets the anchor fade
from 1.00 to 0.01 by the fourth pass; before the MLP the mix is broader (0.1–0.23 on each earlier pass beside 0.46 on
the pass's own attention) — the momentum over iterates of (ii). Restricting the sources to the anchor and the last two
passes (iv) is refuted: it loses the held-out gain and collapses at twice the passes (0.19), where every-pass access
keeps 0.60–0.77. The held-out compositions solved (1/8 in both tied-core attention-residual cells, the first in the
line) came only with the TIED core — the same block, learned routes across its own passes; a stack of distinct blocks
gained nothing from the routes. What to write for the block, then: the pass's input is the last pass plus a decaying
memory of all earlier passes, the anchor fading rather than re-injected at a fixed weight, no window — and in the
written form the weights of that mix are flag-conditioned (§15), with the decay as the default.

**Order of work (set 2026-09-21).** First the recurrence scheme, measured on the gradient arm where it is cheap: E30
(the depth loop under attention residuals) and E31 (continuous thoughts under attention residuals, with the hazards
above as the things to watch). Then the written block adopts the scheme they pick. Not designed yet, and stated as
such: what compressing behaviour writes into the block.

**References** (one file per paper in `refs/`, abstract verbatim plus notes; the overview is
`refs/latent_reasoning_and_looped_planning.md`): `coconut_2412.06769.md`, `recurrent_depth_2502.05171.md`,
`looped_latent_thoughts_2502.17416.md` (+ PDF, CC BY), `looped_transformers_programmable_computers_2301.13196.md`,
`value_iteration_networks_1602.02867.md`, `universal_transformers_1807.03819.md`, `adaptive_computation_time_1603.08983.md`,
`searchformer_2402.14083.md`, `stream_of_search_2404.03683.md` (+ PDF, CC BY), `thinker_2307.14993.md` (+ PDF, CC BY).

---

## 19. The library problem, and the continuous thesis (DESIGNED 2026-09-21)

**The library problem, stated.** §5's library is a list I typed, and §7 grows it by three steps that have all been
run: fit fuzzily (the table), look for a name (adopt a structure when cheaper — E3, E8), compare across tasks (a
recurring matrix becomes an item — E8, E12). Every item those steps have ever minted is an INSTANCE of a kind already
in the list: a particular permutation, a word in the generators. They have never minted a KIND — a new form of
predicate or feature. The games interface never used this library at all; `arcgames.py` has a second, smaller one
(windows over cells, masks from sleep, goal keys), also typed. E17 met the wall exactly (a relation is not in any kind:
identify-then-plan 0.51), and E29's only real generalisation came from a kind typed for it (the sector features). §7's
own loss condition — "the items a task needs are not inside any generic structure" — is the one that bites on any game
the library was not written for: counting, ordering, "the same colour as the key", a piece that rotates. The plan as
written cannot scale.

**The resolution, in the interpreter frame (§18).** The library is not a list of structures but the INSTRUCTION SET of
the written block — a fixed, small, complete set: what one attention head, the residual and a threshold can compute
(read by offset, read by content, read by direction, pool, write, compare, branch). Everything else is a program over
it, data in the context; a new kind is a program, never a new instruction; and §7's third step becomes the right rule
at the right level — a sub-program recurring across programs is compressed into a named macro. That is library
learning as DreamCoder does it (Ellis et al. 2021: wake = solve tasks by search with the current library, sleep =
abstract recurring sub-programs into new primitives; Stitch and LILO are the faster and language-guided successors,
from memory), and it is the only form known to grow kinds by compression. Completeness comes from the instruction set
(Giannou et al.: a one-instruction computer suffices); the cost is the search, exponential in the length of what is
new and tamed only by the ladder of macros. The rule from E29 applies to the instruction set itself: it must be the
substrate's own primitives, not features I like.

**The continuous thesis (the rethink, 2026-09-21).** Transformers trained by gradient descent are the one design that
has compressed a messy, high-dimensional world — the frontier's robot models generalise to an unknown house through
in-context learning — and they keep getting better at it. So the substrate is a continuous transformer whose
representations are learned by gradient descent, and ZipLearner is the COMPRESSION THAT ACTS ON IT. The neat discrete
world is the case where the cheapest description is exact — E28's block is the M → ∞ limit of soft attention over
sparse features — and is not special-cased. What is known about how gradient descent uncovers continuous
representations, and it is a fairly complete picture (from memory of the papers; notes to follow):
1. *Features are directions, more of them than dimensions* (Elhage et al. 2022, superposition; Bricken/Templeton et
   al. 2023–24, sparse dictionaries): a trained net stores a feature as a direction, packing sparse features into
   nearly-orthogonal directions; composition is roughly addition. Our one-hot code is this with orthogonality exact.
2. *Gradient descent extracts the data's statistical modes, strongest first, each in a burst* (Saxe, McClelland &
   Ganguli 2014, 2019): a deep linear net learns the singular modes of the input–output correlation in order of
   singular value, each with a sigmoidal transition, and in a hierarchical domain those modes are the hierarchy of
   concepts, coarse to fine. Rare compositional structure comes late or never — E0–E26's 0/8.
3. *The parameter-to-function map is biased toward simple functions* (Valle-Pérez, Camargo & Louis 2018; Mingard et
   al. 2021; Hochreiter & Schmidhuber's flat minima 1997; Hinton & van Camp's MDL training 1993): the volume of
   parameters mapping to a function falls off with its complexity, so init plus SGD samples from a Solomonoff-like
   prior. Gradient descent in an overparameterised net is already a soft, approximate search biased toward short
   descriptions; ZipLearner's principle is what it approximates, made exact where exactness pays.
4. *In-context learning is learned inference over a task family* (Xie et al. 2021; von Oswald et al. 2023; Garg et
   al. 2022): trained across many tasks, the net's features are the family's parameters and the forward pass infers
   them — "fit first, name later" inside the forward pass.
5. *What it does not do*: exact algorithms and their length generalisation; compositions that are not statistical
   modes; learning without forgetting (E6b below chance vs E6's 100%); readability. Those are where description length
   must be enforced rather than approximated.
Hence the division of labour: (a) SLEEP = consolidate the learned representation into the cheapest description — a
sparse dictionary, low-rank factors, and, where cheaper, exact tables and programs; §7's "look for a name" on
continuous activations, which is what a sparse autoencoder is mechanically; (b) THINKING = §18, the looped block
running programs over those features; (c) CONTINUAL LEARNING = consolidation by price instead of replay (E6's win,
on continuous representations). The library of §5 becomes the forms a compression can take — dictionary, low-rank,
convolution, program — priced.

**The reach, asked 2026-09-21 (mathematics; goodness and justice; a chair and its 3D form).** Mathematics: in
principle everything with a procedural or formal description, in time exponential in the length of what is new under
the current library — the ladder of macros is the mechanism; the general instruction set is still to be written.
Abstract normative concepts: the MECHANISM of abstraction (a shared description with the varying parts as arguments,
adopted when cheaper) and abstract THOUGHT (programs over programs) are in scope; the concepts are not — they have no
short program, are graded, relational and normative, and need the continuous regime, language as the domain, and a
source of value beyond one number; language models show they are learnable from text as distributed features, which
discrete programs are not. Vision: the structural half is near — features at relative positions in an object-centred
frame under a transformation group (E28's frame in 2D, E12's group, SE(3) for 3D), and "a chair without the length of
its legs" is exactly what compression yields (the macro is what recurs, the arguments what varies; affordances are
programs too); what is missing is the front end from pixels to a code with parts, deep learning's home ground, where
§5's repeated diagonal (a convolution) is the unbuilt first step. The two bounds on all three: the SEARCH and the CODE
— a concept is short only in the right representation, the seam of E29's retina and of "what attends to what".

**Where to start.** With the claim everything depends on: that "look for a name" works on a continuous representation
— E32, on a net whose contents we can read. Then the user's test case: a language model on the Latin corpus (E33),
where the discrete machinery is at home, the transformer is the reference, and the hybrid is the thesis; the number of
interest is data efficiency — bits per character against training characters, and where the curves cross. Then a
messy, continuous version of what we have (shapes with continuous variation, where "macro + arguments" can be checked
against ground truth), and only then pixels. Not to be done: throwing away the discrete machinery (it is the exact
edge case), or starting from real vision before the compression step is shown to work on something readable.

---

## 20. Credit assignment without backpropagation — the diffusion interpretation (DECIDED 2026-09-21; the vital clue)

**The objection this answers.** §19 left one thing to gradient descent: the last third of the way to a frontier
language model, the *nonlinear composition of learned features across layers*, because backpropagation is credit
assignment through a differentiable composition and the gradient-free alternatives — greedy layer-wise fits, local
rules, search over compositions — had lost to it or were blind. The user's question (2026-09-21): if the data is
compressed into weights, can the *compositions* be compressed too? The clue is DiffusionBlocks.

**The paper.** Shing, Koyama & Akiba (Sakana AI), *DiffusionBlocks: Block-wise Neural Network Training via Diffusion
Interpretation*, arXiv:2506.14202 (v4, June 2026); abstract verbatim in `refs/diffusionblocks_2506.14202.md`. Its
key sentence: "residual connections naturally correspond to updates in a dynamical system. With minimal modifications
to this system, we can convert the updates to those of a denoising process, where each block can be learned
independently by leveraging the score matching objective." Trained one block at a time, it "matches the performance of
end-to-end training" on vision, diffusion, autoregressive, masked-diffusion and RECURRENT-DEPTH transformers — the last
being our looped block's family (Geiping et al., §18).

**What it means.** Credit assignment through depth is needed only when the intermediate representations are unknown.
The diffusion interpretation DEFINES them: block k's input is the data at noise level t_k and its target is the data at
t_{k−1}; every block's (input, target) pairs are observable from the data and the noise, so every block can be fitted
LOCALLY by any method — including ZipLearner's — with no gradient through the stack. For refinement-type depth (a
state improved pass by pass toward the data) the problem dissolves. Our loop was already this shape without the name:
Geiping et al. start the recurrence from noise; the anchor is the conditioning; E30's decaying reads of earlier passes
are what multi-step ODE solvers do with previous steps (Adams–Bashforth, DPM-Solver++). Two corrections to §19 follow:
(1) *nonlinearity is not the obstacle* — a denoising block may be as nonlinear as we like (a table, a piecewise fit),
because its target is known; what had no closed form was the composition, and the schedule supplies it; (2) *size is
not the obstacle* — the limit on end-to-end training was memory across the stack, which blockwise fitting removes, so a
written network can be as deep as its schedule.

**"Compress the compositions" — two levels.**
- *Composition fixed by a schedule.* Each block's weights are the cheapest description of its own (noisier in, cleaner
  out) pairs: DiffusionBlocks written rather than trained — E24's counting and sleep pass per block, or §5's continuous
  structures fitted by least squares when the continuous rework comes, the discrete case as the exact limit.
- *Composition not given by a schedule* — programs whose intermediate states are not noisy versions of the answer. Then
  the composition itself is the description to compress: which block reads what (routes as flag-conditioned tables,
  priced, §15), how many passes (convergence, §18), which macro (library growth, §19). That is DreamCoder's abstraction
  step in our words. Their intermediate targets come from tools already here: §6/E2's inversion of the next layer;
  Coconut's curriculum (E31: the intermediate written as tokens, then latent).
What the interpretation does not give: a schedule for arbitrary programs. Its class is refinement — and that class is
large: generation, repair, planning as iteration on the frame (VIN, §18), and the masked-diffusion form of language.

**The target, restated (2026-09-21).** 1.0 bits/character is a number of enwik8's scale (10⁸ characters), unreachable
on 2.7M characters of Latin by any method. On Latin the goal is the fraction: the compression literature's step from a
table (~1.9 on enwik8) to context mixing (~1.3) is a third; the Latin analogue is 1.82 → ~1.3, and it is gradient-free
territory (E34). What remains beyond it, if E34 and E35 land, is measured rather than assumed.

**Order of work (approved 2026-09-21): E35, then E34; E31 finishes on its own; no gradient-arm work otherwise.**
