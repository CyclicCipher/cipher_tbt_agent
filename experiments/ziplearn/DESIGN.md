# ZipLearner — design document

*v0.4, 2026-09-20 (v0.1–v0.3 the same day; v0.4 adds §16, the outer objective). E0–E6 and E6b have been RUN — every verdict, with the numbers and
what each one changed, is in §13. Code: `experiments/ziplearn/ziplearner.py` (the weight-writing learner: one matrix,
two layers, the cross-task library, the continual layer) and `e0.py` … `e6b.py`; the earlier arithmetic-only
version is still `experiments/inner_objective/ziplearn.py`. This document is the source of truth for the ZipLearner project; experimental results are reported in §13
and nowhere else.*

## 0. How to read this

Every technical word is defined the first time it is used, and again in the glossary (§14). Each section carries a
status:

- **BUILT** — code exists and has been run; the numbers are in §13.
- **DESIGNED** — the steps are written down concretely enough to code; no code yet.
- **OPEN** — I do not know how to do this yet, and I say so instead of using a vague verb.

When an experiment runs, its numbers go into §13 under its ID (E1, E2, …) with the command that produced them, so every
number in this document can be traced to a file on disk.

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

## 8. Lossy compression for a finite network — change 1 BUILT (E4), changes 2–3 DESIGNED

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

## 9. Continual learning without a replay buffer — rules 1–2 BUILT (E6), rule 3 DESIGNED

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
it, which is the correct behaviour and also a guaranteed error on the first refuting case. **OPEN-7:** in a
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

For each: what is measured, what counts as a pass, what would refute the design. Numbers go to §13.

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
**OPEN-8** — writing attention weights under PoPE: content matching is not separable from the phase cosines (E0), so "same digit anywhere" is not one written parameter; what the written form of an induction head is, is not designed.
**OPEN-9** — the J-space of the gradient arm (Gurnee et al., Anthropic, July 2026: J-lens = the average Jacobian of the final residual with respect to a layer's residual, pulled back through the unembedding; J-space = sparse non-negative combinations of the per-token J-lens directions; a mid-layer global workspace, ≤10% of variance but causal for flexible reasoning). Under attention residuals the Jacobian's identity highway becomes a content-dependent route weight, so the J-space should localise to the sources later mixers read. *Measured (E8b): it does — identity share 0.06/0.04/0.12/0.77 = the routes; early-layer J-lens swaps drop from 72–89% to 13–23%.*

## 13. Results log

Entries are appended, newest last, and never edited afterwards; a correction is a new entry. Format:

```
### <date> — <experiment id> — <one-line verdict>
command: <exact command>
files:   <where the numbers are>
numbers: <the pre-registered measurements, as measured>
verdict: PASS / REFUTED / INCONCLUSIVE, against §11's criteria
changed: <what this changes in the design, if anything, with the section number>
```

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

## 16. The outer objective — first form BUILT (E9), the rest DESIGNED / OPEN

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

**E10 — acting to learn (pre-registered 2026-09-20, late; OPEN-11).** E9's environment; the agent starts knowing four
actions (3 observations each) and never having seen three; a stream of goals reachable with all seven, some unreachable
with the known four. Before planning each goal it may spend up to 2 tries (execute an action to see what it does;
every executed action, try or plan step, is an observation). Policies for the tries: none; random; least observed
(count-based novelty); **price** — the action with the largest expected plan-bit saving for the current goal over the
hypotheses its partial description still allows, minus the try's own cost, and only if positive. Measures: goals
reached within a 6-step budget; mean steps to goal (failures count as the budget); goals until all seven actions are
described. Pass: price reaches at least as many goals as random and least-observed with fewer mean steps. Refute:
price worse than random.

**OPEN-10** — when the map is too large to search: what to cache (a cost-to-go per state, or per description?), and
whether the cache is itself a written structure. **OPEN-11** — acting to learn: the price of trying an unknown
action against the bits its description would save. **OPEN-12** — goals that are relations, not states ("make the
output the reverse of the input"), and goals over the library ("find a shorter word").

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

## 17. Files

- `experiments/ziplearn/ziplearner.py` — the weight-writing ZipLearner: `Structure` and the library v1 (`Table`, `Identity`,
  `Shift`, `Affine`, `Permutation`), `Layer` (one matrix, §4), the rate price (`flag_bits`, `pay`, §8), `PositionPerm` /
  `PositionIdentity` / `TwoLayer` (two layers, §6), `NamedPerm` / `PermLibrary` / `TwoLayerWithLibrary` (§7), `ContinualLayer` (§9).
- `experiments/ziplearn/e0.py` … `e6.py`, `e6b.py` — the experiments of §11, one file each; results in `runs/e*/` and §13.
- `experiments/ziplearn/refs/` — reference notes (`attention_residuals.md`) and the paper PDF.
- `experiments/transformers/h1_lid.py` — the transformer substrate: PoPE (+ the withdrawn `--n_zero`), attention residuals
  (`AttnRes`, `--res attnres|attnres_full`), `Model.routes`, `--json`.
- `experiments/inner_objective/tasks.py` — the two task families (arithmetic, composition), reused by every experiment.
- `experiments/inner_objective/ziplearn.py`, `runs/ziplearn_*.json` — the earlier arithmetic-only version and its numbers (§2).
