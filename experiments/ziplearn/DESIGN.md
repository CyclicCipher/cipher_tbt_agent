# ZipLearner — design document

*v1.0, 2026-09-27. Section numbers are stable: §0–§21 are the same sections as v0.9, compressed in place (the
pre-registrations of experiments that have run, the design bake-off post-mortems and the kept-for-the-record
paragraphs were cut — they are in git history and, for every number, in `RESULTS.md`). New in v1.0: §0's purpose,
§22 legibility at scale, §23 the semantic landscape without gradients, §24 the gap to a neural language model.
E0–E30, E33(a), E34 and E35 have run (E29 retracted as a design; E31 stopped after 2 cells; E32 and E33(b,c)
shelved); one line each in §13, the full entries in `RESULTS.md`, which is append-only and is the only place
numbers live.*

## 0. Purpose, and how to read this

**The two questions this project exists for** (the user's framing, 2026-09-27). (1) *What is inside the black box?*
(2) *How do we know it is safe?* Post-hoc interpretability attacks both by inverting a trained network — a search
problem under adversarial conditions (superposition, non-modularity, no ground truth), against a target that grows
faster than the tooling. This project attacks them in reverse: **start from a program that was written, so there is
nothing to invert.** Construction faces an expressivity problem instead of an inverse problem, and expressivity
problems come with verification — `brainbuilder.compile` takes 0.037 s and B0 checks bit-for-bit equality.

**What construction actually buys — stated honestly, because the strong version is false.** A compiled brain is
transparent at the level of MECHANISM and not automatically at the level of BEHAVIOUR: behaviour is mechanism
composed with content, and E28's content is a table of 12,804–30,629 entries (E35), any one of which is readable and
all of which together are not. The black box MOVES from the weights to the learned content; it does not vanish.
Three properties survive that correction, and they are the safety case:

1. **Auditability.** Every surprise is localisable — which instruction, which pass, which table entry. That is a
   different thing from transparency, and it is the one that holds at scale.
2. **Calibrated belief with an audit trail.** A count-based store's codelength IS its confidence
   (`ziplib.price.kt_bits`), so the brain can answer "how sure, and on what evidence" with the counts that produced
   the estimate. No post-hoc method can add this to a gradient-trained model: the information was never stored.
3. **Exact editing and unlearning.** ROME and MEMIT exist because editing gradient-descent weights is approximate
   and leaky. Editing a `Store` entry is a write, unlearning is a delete, and both are priced.

**Cheap training is not a side benefit; it is the load-bearing test.** An interpretable model nobody uses makes
nothing safe. So the number to track is the **legibility tax**: the bits the constructive route pays over a
gradient-trained model of equal budget. It is a PROFILE, not a scalar (§24.1, E37) — already negative on adaptation
under distribution shift (−6.04 bits/char), continual learning, exact composition and storage density on structured
content, and positive only on bits per character over stationary bulk text, where it is ≈ 0.4. The question that
decides the programme is not whether that one number reaches zero but whether it is fixed, shrinking or growing with
scale (§22), and which axis the target application actually lives on.

**Status words.** **BUILT** — code exists and has been run, numbers in `RESULTS.md`. **DESIGNED** — concrete enough
to code, no code yet. **OPEN** / **NOT DESIGNED** — I do not know how, and say so instead of using a vague verb.
Every technical word is defined at first use and again in §14.

## 1. The idea in one paragraph

A ZIP file learns without gradients. It counts what recurs and stores it once. PNG goes further: for each block of
pixels it tries a few simple filters — "this pixel minus the one to its left" — and keeps whichever makes the block
smallest. Learning is counting; choosing a model is "which filter makes this smallest"; predicting is decompressing.
ZipLearner is that recipe made into a learner: its hypotheses are ways of describing the data, its score is the bits
the description takes, it keeps the cheapest, it predicts by decompressing. Applied to the weight matrices of a
neural network it writes the matrices directly — no backprop — and the matrices stay readable.

## 2. What exists today — BUILT

The line's origin is `experiments/inner_objective/ziplearn.py`: three hypotheses over arithmetic demonstrations (a
dictionary of (x → y) counts; "first difference constant"; "second difference constant"), each paying bits per
observation, the cheapest answering. On x → (a·x + b) mod 11 with 12 held-out rules it predicted every held-out rule
exactly after ONE demonstration at 0.19 bits/digit; on random rules it reported having learned nothing (3.86 bits,
above the 3.46 of guessing); on compositions it scored 0% because it had no hypothesis about digits changing
POSITION. The honest caveat that motivated everything after: "first difference constant" IS the family of straight
lines, so the arithmetic pass was largely built into the chosen hypothesis. What was not built in: the prices prefer
the line over the parabola on their own, and noise is rejected with no special mechanism.

Everything since is in `ziplearner.py` (structures, one matrix, two layers, the cross-task library, the continual
layer), `arcgames.py` (the games), `textlm.py` (text), `e0.py`…`e35.py`, and, from 2026-09-22, the `ziplib/` package
plus `brainbuilder.py` (§21). §17 is the file map.

## 3. Vocabulary

Terms needed to read §4–§10; the full glossary is §14.

- **Weight / connection / layer** — `w[i, j]`; a non-zero weight; the matrix **W** plus a fixed nonlinearity.
- **One-hot, row = lookup** — a digit as eleven zeros with one 1; for a one-hot input, `W · input` is *row x of W*,
  so over one-hot codes a weight matrix IS a lookup table.
- **Structure** (family) — a named set of matrices described by a few numbers ("all shifts", "all affine
  permutations", "any table"). **Parameters** pick one matrix out of it.
- **Price** — bits: naming the structure + writing its parameters + every observation the written matrix gets wrong
  (an **exception**).
- **Interface / source / route / target** — where one layer's output becomes the next's input; what a layer may read
  (the embedding, or an earlier layer's output); which source it reads; what its output should be.

## 4. The core loop for one matrix — DESIGNED

Running example: the hidden rule is y = x + 3 mod 11; the layer is an 11×11 matrix over one-hot digits.

1. **Fit.** A pair (x = 2 → y = 5) arrives. Every structure that can solve its parameters does so — shift needs one
   pair (b = 3), affine waits for two, the table just records the entry.
2. **Write.** Each solved structure writes its WHOLE matrix: the shift writes all eleven rows including the ten for
   digits never seen. This is what "instilling structure into the weights" means concretely — 121 numbers from 1.
3. **Price.** Structure name (log₂ library size) + parameters (log₂11 per digit-valued parameter) + exceptions so far
   (log₂11 + 1 each). A correctly predicted pair costs ≈ 0.
4. **Keep.** The lowest total is the layer's description; the layer's matrix IS that structure's written matrix; a
   query is answered by reading a row.

```
for each demonstration pair (x, y):
    for each structure S in the library:
        if S can now solve its parameters: solve them; write W_S in full
        price[S] += cost of (x, y) under W_S          # 0 if W_S already predicts y; otherwise an exception
    keep = the S with the smallest (bits(S) + bits(params_S) + price[S])
answer to a query x  =  row x of W_keep
```

## 5. The library of structures, version 1 — BUILT, and SUPERSEDED by §21.3

| structure | parameters | solved from | price of parameters | what it captures |
|---|---|---|---|---|
| table | one output per input seen | each pair | log₂11 per entry | anything; cannot extrapolate |
| identity | none | nothing | 0 | "this layer copies its input" |
| shift | b | 1 pair | log₂11 | y = x + b |
| affine permutation | a, b | 2 pairs, different x | log₂11 + log₂10 | y = a·x + b |
| general permutation | a permutation of 11 | 11 pairs | log₂(11!) ≈ 25 | any one-to-one relabelling |
| tied blocks | one small matrix reused per position | pairs at any position | the small matrix's price | the same rule at every digit |
| position permutation | a permutation of 6 positions | 6 pairs | log₂(6!) ≈ 9.5 | reverse, rotate, swap |
| read-from (route) | which source this layer reads | try each, keep cheapest | log₂(sources) | the wiring (§6, §15) |
| position map | per output: a source or a constant | candidate intersection | log₂5 per unresolved digit | many-to-one actions (E11) |
| position edit | identity with per-slot edits | a slot is free while it reads itself | one exception per edit | "everything stays except…" (E11) |
| low-rank | two thin matrices | least squares | (rows + cols) × rank at finite precision (§8) | "the map uses few directions" |
| repeated diagonal | k numbers | least squares | k numbers at finite precision | a convolution; translation invariance |

**Superseded.** §19 found the flaw: every item these steps ever minted was an INSTANCE of a kind already on the
list, never a new KIND. The library's successor is the INSTRUCTION SET of the written block (§21.3) — a small fixed
set of what one attention head, the residual and a threshold can compute — with everything else a program over it.
This table remains the description of what `ziplearner.py` runs on the digit tasks.

## 6. Depth: more than one layer — BUILT (E2), one part OPEN

Demonstrations give the network's input and its FINAL output; a hidden layer's target is not given. The rule: **pin
the interfaces, solve each layer alone, sweep.** Freeze layer 1, run §4 on layer 2 against the demonstrations'
outputs; freeze layer 2, obtain layer 1's target by running layer 2 BACKWARDS (if layer 2 is the permutation P and
must output y, its input must be P⁻¹y); repeat until neither changes. Exact whenever the written matrix is
invertible — a target is decompression run in reverse, no gradient anywhere. The total price splits per layer as long
as interfaces are pinned during each solve. This is the DiffusionBlocks shape (§20): blocks solved independently
against fixed interfaces; theirs are pinned by the noise schedule, ours by the neighbours' current state.

- **Measured (E2):** the plain sweep settles in 2 rounds but on a WRONG fixed point whenever both layers act —
  neither can be solved while the other is wrong. Adopted: restart from every description of the value layer and keep
  the lowest total price.
- **OPEN-2:** running a NONLINEARITY backwards. One-hot codes and permutations need none (a permutation after a
  permutation is a permutation), so E1–E3 never hit it; it becomes real with distributed codes. The known weak point
  of every target-propagation method; no fix designed.

## 7. Where new library items come from — BUILT (E3, E8); the bet, and how it lost

A matrix is a continuous space that already contains every structure, so the learner can **fit first and name
later**: (1) *fit fuzzily* — a layer with no structure yet still has a written matrix, the counts; (2) *look for a
name* — periodically solve each structure's parameters from a few rows and check the rest, adopting the structure the
moment it is cheaper than the entries, which fills in rows never seen; (3) *compare across tasks* — if matrix B
equals matrix A followed by a fixed Q, and the same Q recurs, Q becomes a library item priced once. Step 3 is ZIP's
dictionary rule applied to matrices.

The bet, stated so it can lose: *the library of generic matrix structures is small and finite, and task-specific
items are points inside those structures.* **It lost in the way §19 names:** steps 2–3 mint instances, never kinds.
E3 and E8 confirm the mechanism (17 recurring permutations became named items; the library re-described with 3
generators, 161 → 121 bits); E17 met the wall (a relation is in no kind: 0.51).

## 8. Lossy compression for a finite network — BUILT (E4, E15); annealing REFUTED here (E22)

The §4 price is lossless: every exception is stored forever and the total grows with the observations. The right
theory is rate–distortion — *rate* is bits spent, *distortion* is error accepted, λ is the exchange rate. Three
changes:

1. **Exceptions become a rate, not a list.** Each structure carries ε, the fraction it gets wrong; n observations
   cost n·H(ε) + (exceptions)·log₂10. For ε = 0 this is the lossless price; for ε = 10% it is flat per observation.
   The learner remembers the rule and the rate, and forgets the exceptions. *(E4 also found the v1 price was not a
   valid code — mass 1.45; the rate price is, and is the default.)*
2. **Parameters at justified precision.** A parameter from n observations is written with ≈ ½·log₂n bits (the MDL/MML
   rule; `precision_bits`). *(E15: the best grid step shrinks as 1/√n, slope −0.5 ± 0.15.)*
3. **The capacity budget.** A network holds at most C bits; minimise error subject to price ≤ C, equivalently
   price + λ·error. Raising λ gradually is deterministic annealing — *refuted here (E22): no difference against a hard
   cap, because one-hot rules have nothing fuzzy to crystallise. Expect it to matter only in the continuous regime.*

Reference: Shannon 1959; Blahut–Arimoto; Wallace & Boulton 1968; Rissanen 1978; Hinton & van Camp 1993 (description
length of weights); Rose 1998; Tishby, Pereira & Bialek 1999. The risk the price must not excuse: with λ small
enough, "a table with some errors" is as cheap as "a shift" and the learner stops looking for structure (E4 checks).

## 9. Continual learning without a replay buffer — BUILT (E6, E15, E16)

**The description is the archive.** A compressor never keeps the original once it has the compressed form. A's
written matrix plus two numbers — its **evidence count** n_A and its exception rate ε_A — is a complete record of what
A's data said, so "would changing this hurt A?" is answerable in closed form with no examples stored.

**Rule 1 — never overwrite; add, and let the evidence decide.** When B contradicts A, two descriptions compete on
price: change A's parameters (turning (1 − ε_A)·n_A past observations into exceptions, charged now) or keep A and add
a second description (a few bits). The second wins as soon as n_A is more than a handful. Old knowledge is protected
in exact proportion to its evidence. (This is what EWC's weight importance estimates; here it is exact and free.)

**Rule 2 — context selects; a new context is minted only on refutation.** A layer holding several descriptions is ONE
matrix whose rows are indexed by (context, input) — a block per context; adding B adds a block, A's block is rows
nobody writes to. The active context is inferred, not labelled: the block that prices the recent window cheapest. A
new block is created only when every existing block would pay more in exceptions than a new block's parameters cost.
*Mint on refutation, never on incompleteness.* As built (E6): minting is a change-point decision over the recent
window; an established block (≥ 3 pairs) is priced frozen under its kept description; a returning rule routes to its
old block.

**Rule 3 — capacity forces consolidation; forgetting is chosen by price.** Over budget, first MERGE (two blocks
differing by a recurring Q become one block plus Q — §7 step 3 and consolidation are one operation seen twice), then
DROP the block with the least (evidence × bits saved). The cheapest-to-lose goes first, the opposite of gradient
forgetting.

- **OPEN-6 (measured, E21):** two rules agreeing on the recent window and differing elsewhere cannot be told apart;
  the mint comes 6.4 observations after the near-duplicate starts — the wait for a refuting input — at one wrong
  answer each. Correct behaviour, and a guaranteed error on the first refuting case.
- **OPEN-7 (closed, E16):** blocks do NOT multiply across layers — 100% retention, exactly 3 blocks on two-layer rules.

## 10. Prior knowledge written in — BUILT (E5)

A rule of logic is a small table (AND, OR, NOT, XOR over one-hot truth values); writing one is §4's write step with
the parameters given instead of solved. The priors most worth writing are not tables but *structures offered
cheaply*: identity at 0 bits ("things stay as they are unless acted on"), shift and rotation ("moving does not change
the object"), repeated diagonal (translation invariance). They cost nothing on tasks that do not use them, which is
the test of a good prior. *E5: written boolean gates cut demonstrations to criterion 19.7 → 11.0 with no harm
off-target.* The caveat that decides whether any of it works: a written law helps only if the network's code for "A"
lands in the slots the written matrix reads — the interface problem of §6 in another costume. Priors and pinned
interfaces stand or fall together.

## 11. Pre-registration — the discipline, and where the live ones are

Every experiment is pre-registered before it runs: what is measured, what counts as a pass, what would refute the
design; each ≤ 20 minutes; numbers go to `RESULTS.md` under an ID with the command that produced them, and nowhere
else. E0–E35's pre-registrations were written in this section and have all run; they are preserved in `RESULTS.md`
(each entry restates what was measured) and in git history, and are cut from this document as post-mortems. §13 is
the index.

**Live pre-registrations, not yet run:** §21.9 (B0–B8, G1–G3 — the blueprint and gradient-compatibility programme),
§22.6 (S1–S2 — superposition with a codebook; S2 RUN), §23.5 (S3 — the counted metric; RUN and refuted on Latin for
a data reason), §24.6 (D1–D3 — the gap, the projection test and decompilation; D1's gradient-free half RUN),
§19.1 (S4 — level-3 naming tested on morphology). Nothing else is outstanding.

## 12. Open questions

**OPEN-1 — the outer objective.** ZipLearner learns a generalisable world model. It does not, by itself, say what to
*do* with that model, or in what order, to achieve an objective. The ideal is a general way of implementing an outer
objective so that any objective can be plugged in. Addressed in §16, first test E9.

- **OPEN-2** — inverting a nonlinearity for hidden-layer targets (§6). Still open.
- **OPEN-4** — what the capacity C of a given network is, numerically, in the price's units (§8).
- **OPEN-6** — near-duplicate rules (§9; measured, E21: 6.4 observations).
- **OPEN-8** — writing attention weights. *E18 closed the first form under RoPE (a previous-token head in the
  high-frequency rotary pairs, an induction head in the low-frequency ones, 100% with no training); E28 and the
  compiler closed the general case for the instruction set of §21.3. Still open: ZipLearner CHOOSING such a circuit
  as the cheapest description, and the same under PoPE.*
- **OPEN-10** — when the map is too large to search: what to cache, and whether the cache is itself a written
  structure. *E12: coordinates where the actions commute (8.9 bits), a word table where they do not (1,190 bits);
  E23 refuted macro-actions as the cache (nodes 192 → 522 — a cache needs depth). A map too large for either is not
  tested.*
- **OPEN-12** — goals that are relations, not states. *E17: a relation given as examples is planned for by matching
  plans to the examples (0.99); identifying it first fails at the identification limit (0.51). Goals over the
  library: not built.*
- **Closed:** OPEN-3 (the sweep settles — E2, with restarts), OPEN-5 (structures of structures — E8's `WordLibrary`),
  OPEN-7 (blocks do not multiply across layers — E16), OPEN-9 (the J-space localises to the routes — E8b: identity
  share 0.06/0.04/0.12/0.77, early-layer J-lens swaps 72–89% → 13–23%), OPEN-11 (acting to learn — E10/E10b: a
  description is an asset, valued over the goals still to come; price 92% of goals in 2.3 steps against novelty's
  36%/23%).

## 13. Results — the index; the log is `RESULTS.md`

One line per experiment in the order run; the full entries (command, files, numbers, verdict, what changed) are in
`RESULTS.md`, appended and never edited.

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
| E35 | REFUTED on the letter | written denoising chain: memory blocks beat one shot by +0.07 at t = 0.75 and generate rooms from noise; the MDL price collapses denoisers to the identity; text ±3: no gain |
| E34 | INCONCLUSIVE | gradient-free context mixing: 18 experts, Bayesian mixing 1.820 vs the table 1.823; geometric with Bayesian exponents 1.801; the raw product 7.7 — the learned multiplicative mixer is the first irreducible gradient |
| B0 | PARTIAL | the compiler reproduces the hand-written brains: E18 29/29 tensors equal, score 1.000; `gridworld.json` = `WrittenSim` bit-for-bit (530/530 and 509/509, ties 1434/1475, 900/900 plans) and at d_layout + 64; compile 0.037 s. The `rope2d` cell is UNRUN, not failing — the recorded CompileError is stale, the same call compiles today (§21.9, §22.3) |
| E36 | measured | where the bits go on Latin: word-initial = 13.4% of positions and 32.1% of the code length (4.52 vs 1.48); R = 8→24 changes nothing; the morphology hypothesis REFUTED (backoff already shares stems); the counted metric unusable at 5.2 tokens/type — §24.4 retracted, §23 re-aimed at the games |
| E37 | measured | the legibility tax is a PROFILE: adaptation is worth +0.049 bits/char in-distribution and **+6.040** on Middle High German — negative on four axes (adaptation under shift, continual learning, exact composition, R_h), positive only on stationary bulk text |
| S2 | PASS with a correction | a designed codebook's interference is predictable: coherence 4–6× the Welch bound, the k < ½(1+1/μ) guarantee conservative by ~2×, usable packing m/d · k ≈ 32; but §22.5's argmin needs the workload's read/write ratio — it is not self-contained |
| S4 | SPLIT: PASS on bits, REFUTED on evidence | level-3 naming pays: signatures 0.625 of the literal code vs BPE's 0.711 at matched vocabulary, and unsupervised MDL recovers Latin's real inflectional endings (+ the enclitic -que); but tokens/unit only 1.66× (pre-reg ≥ 3×) because ONE cut per word catches only the outermost layer — the operation needed is RECURSIVE anti-unification |
| E38 | REFUTED | the crossover on 35M: deficit +0.646/+0.559/+0.470/+0.390 at 2.7/5/10/20M, slope −0.296 per decade = E36's, but the INTERCEPT is setup-dependent — extrapolated crossover 4×10⁸ characters. Homogeneity of train/test sets it, not volume. Plus `ziplib/chain.py`: same model, verified to 3.6e−15 bits, 35M in 449 s |
| E39 | REFUTED on the letter | §9 rule 3 on the text store: a threshold destroys the model (134× fewer contexts, +0.195), a RANKED capacity budget gives a clean curve (10× fewer for +0.105) — but a +0.089-bit floor at 50% shows the price scores worth on seen data only. Third sighting, with E28 and E35, of one defect |
| E40 | measured | why 1.8: the chain is CONTEXT-bound, not data-bound — it matches its full 8-character context at 91.7% of positions and still pays 1.78 bits there, converging to H(c | 8 chars) ≈ 1.75. Beyond depth 8 the evidence vanishes (order 12: 46.5% coverage, median 3 observations), so no counting model has both context and evidence. The n-gram trap, measured |
| E41 | PASS (one clause missed) | in-context learning in a WRITTEN block: byte-identical weights, world model entirely in the context (empty = 0.000); one demonstration per action = 0.570, eight = 0.743 matching the counted store's 0.732 at a quarter the tokens, 128 = 0.830 (+0.098 over it). All 358 test items are unseen windows, so it is all nearest-key generalisation. Fourth sighting that the price destroys what memorised examples retain |
| E42 | REFUTED | in-context transfer over 5 games, one set of weights: the UNCHANGED baseline scores 0.87–0.95 and beats every arm on aggregate, which retracts E41's headline. On changed windows (where the constant scores 0.000) the block reaches 0.23–0.86 from context alone — but on 21–34 items per game, too few to conclude. Toggle 0.227 = the r=1 field cannot express its rule; cross-game > native twice = the action channel is doing little |
| E43 | measured | how much of the board the RULE should see: widening the window makes 4 of 5 games WORSE (LockPath 0.833→0.537) because observations per window collapse ~4× per radius step — E40's trap in 2-D. Toggle is the exception and doubles (0.375→0.792) because its rule is genuinely non-local. The fix is not a bigger window but a non-local predicate KIND (content-match, not offset) — §19's kind problem again |
| E44 | PASS | the evidence collapse IS a factorization issue: unfactored, LockPath falls 31.7→3.7 obs/key with radius; MASKED it holds 184→214→185 keeping 3 cells of 49 (×49.5). Corrects E43 — the trap belongs to the unfactorized reader the in-context arms use, not to the task. Sokoban is the exception (mask keeps every cell, ×1.0). Colour equivalence is a second axis worth ×1.2–2.3/cell, unbuilt. Explains the text line too: E33's mask was a no-op because language is DENSE in relevance where grids are sparse |
| E41 | headline WITHDRAWN | the mechanism stands (byte-identical weights; empty context 0.000, so the world model is in the context) but the performance claim does not: E42 shows a constant predictor scores 0.922 on the metric E41 used |

## 14. Glossary

- **bits** — the unit of price; log₂11 ≈ 3.46 bits names one digit out of eleven with no information.
- **block** — the rows of a matrix belonging to one context; one matrix, several blocks.
- **capacity (C)** — the most bits of description a given network can hold.
- **coherence (μ)** — the largest inner product between two distinct codebook directions; the interference constant of
  a superposed code (§22).
- **evidence count (n)** — how many observations a description has explained; kept beside it, replaces a replay buffer.
- **exception** — an observation a written matrix predicts wrongly.
- **interface** — the boundary where one layer's output is the next layer's input.
- **legibility tax** — the bits a written/constructed model pays over a gradient-trained model of equal budget (§0, §24).
- **mint on refutation** — create a new description only when every existing one is refuted, never because data is incomplete.
- **lossless / lossy** — a description reproducing every observation exactly / one accepting error to save bits.
- **one-hot** — a digit as a vector of zeros with a single 1.
- **permutation matrix** — one 1 per row and column; a one-to-one relabelling.
- **price** — total bits: structure name + parameters + exceptions.
- **rate–distortion** — the theory of how many bits (rate) a tolerated error (distortion) costs.
- **route / read-from** — which source a layer reads; fixed in the written network, a softmax in the gradient arm.
- **source** — the embedding or an earlier layer's output, as something a layer may read.
- **structure / family** — a named set of matrices picked out by a few parameters.
- **sweep** — solving each layer in turn with its neighbours frozen, then repeating.
- **target** — what a layer's output should be.
- **write** — setting every entry of a matrix from a structure's parameters, including entries for inputs never seen.
- **λ (lambda)** — the exchange rate between bits and error in a lossy price.

## 15. Substrate decisions — DECIDED 2026-09-20/21 (reversible)

The transformer used as the gradient-trained comparison arm, and as the network whose weights ZipLearner writes, is
`experiments/transformers/h1_lid.py` (3 blocks, d = 64, 4 heads, ~150k weights) with the changes below. §21.4 lists
the five flag-edits BrainBuilder added on 2026-09-22 (defaults unchanged).

**Positional scheme: RoPE, extended to `rope2d` (§21.3).** PoPE's θ = 0 position-free channels are WITHDRAWN — E0
showed they destroy training; a flat channel is a magnitude bias, not a content match. The recorded one-seed result
(RoPE 9/17 vs PoPE 7/17 trained, both 0/8 held out) is not evidence either way.

**Residual scheme: attention residuals, one transformer block per AttnRes block** (Kimi, arXiv 2603.15031;
`refs/attention_residuals.md` + PDF). Each layer's input is a softmax-weighted mix of SOURCES — the embedding and the
earlier blocks' outputs — with one learned query vector per layer, RMSNorm on the keys, every query initialised to
zero. Reasons, in order: (1) the routing weights are the wiring diagram, readable directly; (2) §6's pinned interface
becomes literal, so an inverted target has a single destination; (3) the route is a structure with a price; (4) the
paper's ablations say what to keep — softmax not sigmoid, RMSNorm on keys, one mix for all channels.

**What the attention residuals learn** (paper re-read 2026-09-21). Their 16-layer ablations: a fixed per-layer scalar
route over all earlier layers (DenseFormer) 1.767 = baseline 1.766; learned scalars without query/key 1.749; full
AttnRes with content keys 1.737; a sliding window over the last 8 layers 1.764. **A route that does not depend on the
token's content is worth nothing; what is worth something is selective, content-dependent access to DISTANT layers.**
Their Fig. 8: each layer attends most strongly to its immediate predecessor with selective off-diagonal
concentrations (learned skips); the embedding retains weight throughout; pre-MLP mixes sharper and local, pre-attention
broader; §6.2 depth-wise attention sinks; §6.1 frames the standard residual as a linear recurrence over depth and
AttnRes as attention over depth. Consequences here: (1) the earlier claim that the written route is "a fixed choice
per layer priced at log₂(sources)" is WITHDRAWN — a fixed choice is what the ablation shows to be worthless; the
written route is a pseudo-query over CONTENT, which in our block is explicit (the token-class flags, and state flags
such as "the lookup found an entry"), so a written query on those dims is a hard per-token route — a `Branch`
(§21.3), which is also how Giannou et al. get conditional branching. (2) The paper's sink layer is our anchor. (3)
The shape to write is local + anchor + a few skips; E30 measured it across passes.

**What attention contributes at frontier scale, and what "ZipLearn writes attention weights" therefore means.**
Attention is the ROUTING — previous-token and induction heads, retrieval heads, binding/name-mover heads,
function-vector heads that identify the task in context — while knowledge sits mostly in the MLPs as key-value
memories; the clean algorithmic heads are few, composed across layers, and gradient descent finds them in phase
transitions (Olsson et al. 2022; Elhage et al. 2021; Wang et al. 2022; Geva et al. 2021; Todd et al. 2023). In
ZipLearner's terms a head is a PREDICATE on earlier positions (relative offset; same content; a class flag → q/k) and
a TRANSFORM of what is copied (→ v/o). So: the head KINDS are the instruction set (§21.3), a small library of
predicate/transform pairs with written templates; ZipLearner chooses which instances exist by description length (the
sleep pass does this for offsets) and writes the programs they execute; a new kind is minted only when no program in
the existing set compresses the data (§7's bet, §19's correction). **What §23 adds:** the predicate "same content" is
EQUALITY in our set and SIMILARITY in a trained model, and the similarity metric is a counted second-moment object,
not necessarily a gradient one.

**Recurrence scheme — DECIDED 2026-09-21.** Two recurrences: depth (the core looped over the same positions,
`LoopedModel`, E26) and sequence (Coconut's continuous thoughts: a position's final state as the next position's
input). ONE boundary operator for both axes: normalise the carried state, re-inject the anchor (Geiping et al.: a
recurrence that does not re-read its input each step is unstable), and let attention residuals choose the rest — the
fixed `rms_norm(x) + α·anchor` is the one-source special case. The loop count and the thought count are a convergence
test or a written halting head, never a pad or a budget. The block is read as an interpreter (Giannou et al.): weights
written once, programs — knowledge and behaviour — as tokens in the context.

## 16. The outer objective — BUILT in its first form (E9–E12, E17, E19, E27), the rest OPEN

**The problem.** Everything before this is a world model: what comes out when something goes in. The outer objective
is to ACT — given a goal, choose actions that reach it — and the ideal is that any goal plugs in without changing the
machinery.

**What acting needs, and what compression already built.**

1. *A map — states joined by actions.* Each action is a structure; actions chain. E7/E8 found the library already
   holds the map: the named permutations form a group reachable from three generators, and a word in the generators
   is a path. **A plan is a word.** Read as a network, a plan of n actions is an n-layer written network; executing
   it is the forward pass. *E28 built it as ONE block looped n times: the rule table as memory tokens, the frame as
   the sequence, a neighbour as a written coordinate permutation, the action as the per-pass anchor — exact agreement
   with the planner, and a nearest-key default for unseen windows that beats "unchanged".*
2. *An inverse — from a wanted change to the action that causes it.* Free for bijective structures (apply S⁻¹); for
   the rest, one more matrix from state-difference to action, learned by counting (`ziplearn.count_inverse`, GCML's W).
3. *A goal slot and a price for plans.* A goal is a description in the model's own language. A plan costs its bits; a
   goal's **worth** is the one external number, which is what makes the objective pluggable. Selection = cheapest
   (plan bits − worth) wins: the same rule as everywhere.
4. *Not required, until shown otherwise:* a value learned by reward for every state. With a map and an inverse,
   cost-to-go is computed on demand and cached only where search is too expensive (OPEN-10).

**Where the two objectives meet.** An action whose effect is not yet described cannot be planned with; a plan that
would be cheaper through such an action is a reason to TRY it — exploration as the outer objective's own demand, with
the inner objective paying it back in bits (E10/E10b).

**The games.** `arcgames.py` runs the loop on the ARC-AGI-3 replica games in `src/tasks/games`: the state is the
frame, an action's description is a local rule (a window → the centre's new colour, tied across cells), the goal is a
set of win keys learned from the score, exploration is the price of ignorance over whole plans (E27). Progress by
experiment: 5/16 levels (E13) → 8/16 with goal keys priced by outcome (E19) → 9/16 with the exploration fix (E27);
discovery is coverage-bound. **Planning is NOT DESIGNED here** — `Player.search` was deleted by §18's rule and the
block must do it (§21.9 B3).

## 17. Files

- `experiments/ziplearn/DESIGN.md` — this document (the source of truth for the design); `RESULTS.md` — the
  append-only results log, the only place numbers live; §13 mirrors its one-line index.
- `experiments/ziplearn/ziplib/` — the shared library (§21.2): `price.py`, `layout.py`, `codec.py`, `store.py`,
  `instructions.py`, `blueprint.py`, `brain.py`.
- `experiments/ziplearn/brainbuilder.py` — `compile`, `verify`, `anatomy`, `ablate`, `b0_regressions`, B6–B8;
  `blueprints/gridworld.json` (the E28 brain), `blueprints/induction.json` (E18).
- `experiments/ziplearn/ziplearn.py` — the learner over a verified brain (NOT WRITTEN YET; §21.2 gives its API).
- `experiments/ziplearn/ziplearner.py` — the weight-writing learner on the digit tasks: `Structure` and library v1,
  `Layer`, `PositionPerm`/`TwoLayer`, `NamedPerm`/`PermLibrary`, `ContinualLayer`.
- `experiments/ziplearn/arcgames.py`, `play_games.py` — the games adapter and harness; `LocalRule`, `ActionModel`,
  `GoalModel`, the exploring `Player`.
- `experiments/ziplearn/e0.py`…`e35.py`, `anatomy.py`, `jspace.py` — one file per experiment; outputs in `runs/e*/`.
  `e28.py` (`WrittenSim`) is the regression reference for B0 until the cutover.
- `experiments/ziplearn/textlm.py` — the discrete machinery as a character language model (E33a).
- `experiments/transformers/h1_lid.py` — the substrate: RoPE + `rope2d` coords, attention residuals (`AttnRes`,
  `Model.routes`), `LoopedModel` (`core_layers`, `boundary`, per-pass routes, `--extrap`), `BoundaryOp`,
  `Block(norm=)`, a k/v cache (`forward_embedded`). `coconut.py` — continuous thoughts (E31).
- `experiments/ziplearn/refs/` — one note per paper (abstract verbatim + notes), several with PDFs: attention
  residuals, Coconut, recurrent depth, looped latent thoughts, looped transformers as computers, VIN, Universal
  Transformers, ACT, Searchformer, Stream of Search, Thinker, DiffusionBlocks, opinion pools, PAQ/Mattern context
  mixing, Bayesian alternatives to learned mixing, gradient-free features and denoisers.
- `experiments/ziplearn/notes/` — `gradient_free_mixing_and_features.md` (the two research rounds' synthesis; **a
  24-item errata block at its head overrides the text below it**), `brainbuilder_module_spec.md` (the file-by-file
  implementation skeleton), `opsd_and_learning_from_experience.md`.
- `experiments/ziplearn/research/` — the feel-experiment scripts, their JSON and logs (rounds 1 and 2).
- `experiments/inner_objective/tasks.py` — the two task families (arithmetic, composition), reused everywhere;
  `experiments/inner_objective/ziplearn.py` — the arithmetic-only origin (§2).

---

## 18. Thinking inside the block — the recurrence scheme (DESIGNED 2026-09-21)

**The rule.** Four constraints, set by the review that rejected E29 and binding on everything behavioural:

1. *The only primitives the model reaches for are the game's controls.* No meta-actions (imagine / reset / commit),
   no policy over thoughts.
2. *The only context is the context window* — the history of frames (cells as tokens, E28's layout), the actions
   taken, and the memory tokens ZipLearner writes. No retina, no locus, no feature extractor outside the model. "The
   cell at offset (di, dj)" is a gather head with a written coordinate permutation (E28); "colour c anywhere to the
   east" is a head with a directional positional phase (E18); which heads exist is the sleep pass's choice, written
   as weights.
3. *A thought is the block's own computation* — one pass of the looped core over its latent state. Code for the
   thinking process (an expansion, an enumeration, a heap, a budget, even an adaptive one) is forbidden, and so is
   defining a thought in the task's terms ("one expansion of an imagined frame").
4. *What the block does when it loops is written by ZipLearner from experience*, or it does not exist yet.

This deleted `Player.search`, `Player.imagine`, `PolicyRule`, the retina and the locus from `arcgames.py`. E29's
numbers stand in `RESULTS.md` as a measurement of what a procedure could do; its design is retracted.

**The frame: the block is an interpreter; knowledge and behaviour are programs in the context.** Giannou et al.
(ICML 2023, arXiv 2301.13196) loop a 13-layer transformer with WRITTEN weights and get a general-purpose computer:
the input sequence is a punchcard of instructions and memory, with a program counter, conditional branches and
function calls all as attention. E28 is this in miniature: the lookup head is one instruction, the memory tokens are
the program, the loop executes it once per action. Under the rule, planning, valuing and exploring must be programs
too — tokens ZipLearner writes by compressing experience, executed by the same loop. The search for programs is
compression, priced in bits; the execution is looping. **What no paper gives is how the programs are found:** theirs
are written by the authors; ours must be the cheapest description of the agent's own successful behaviour, and that
is the open question of this section.

**Two recurrences, and what each is for.** *Depth* — the core applied K times to the same positions (Chen et al.
2609.19107, our `LoopedModel`; Geiping et al. 2502.05171 at 3.5B parameters; Saunshi et al. ICLR 2025: T loops
simulate T chain-of-thought steps). Its state is the residual stream of a fixed set of positions; a pass refines them
toward a fixed point (Geiping et al. observe convergence, orbits and drift). *Sequence* — Coconut (Hao et al.
2412.06769): the last hidden state of position t is fed back as the next input embedding in continuous space; their
finding is that "the continuous thought can encode multiple alternative next reasoning steps", a frontier in
superposition with no search written. **They are not substitutes:** depth refines a fixed state, sequence ADDS state
— positions the loop can write to and read back, memory for a plan of unknown length. E28's rollout was the sequence
recurrence done by hand, one pass per action. The block gets both: K passes per thought position until the state
converges, then the converged state becomes the next thought.

**Depth × attention residuals (E30).** At pass k the sources are {anchor, out₁, …, out_{k−1}} and the core's mixers
read them with a learned query, softmax over sources, RMSNorm on keys. (i) Geiping's stability condition — the input
must enter every pass — holds by construction, with the amount learned per token instead of the constant α; (ii) a
pass can read several passes back: a learned combination of previous iterates, which is what Anderson mixing does for
a fixed-point iteration, and the concrete reason to expect help; (iii) with tied cores the mixers are tied, so routing
is by content; (iv) a WINDOWED source set (anchor + last m passes) would keep a pass a function of fixed-size state —
**refuted by E30**; (v) the paper's mix is one weight per source for all channels, but the written boundary operator
clears SOME subspaces and keeps others, so in the written form the clearing happens inside the heads (E28's lookup
value `OUT − T` already has that form).

**E30's answer.** The trained routes are NOT the boundary operator. Before attention a pass reads the last pass at
~0.8, keeps a decaying read of every earlier pass (0.04–0.20), and lets the anchor fade 1.00 → 0.01 by the fourth
pass; before the MLP the mix is broader (0.1–0.23 on each earlier pass beside 0.46 on its own attention) — the
momentum over iterates of (ii). Restricting sources to the anchor and the last two passes loses the held-out gain and
collapses at twice the passes (0.19) where every-pass access keeps 0.60–0.77. The held-out compositions solved (1/8,
the first in the line) came only with the TIED core; a stack of distinct blocks gained nothing. **What to write:** the
pass's input is the last pass plus a decaying memory of all earlier passes, the anchor fading rather than re-injected
at fixed weight, no window — flag-conditioned weights with the decay as the default.

**Coconut × attention residuals (E31, stopped after 2 cells).** A thought position's embedding source is the previous
position's final state; with attention residuals every layer reads that source directly, so depth-j content of the
previous thought reaches depth j of the next — a recurrence between matching layers across positions. Hazards
verified in the papers: Coconut needs a curriculum ("models trained this way do not perform any better than no-CoT"
without it), gets unstable as thoughts per step grow ("c=3 … a sharp spike in training loss"), its thought count is a
pad rather than a decision, and its arithmetic trails chain-of-thought (GSM8k 34.1% vs 42.9%). The user's recollected
drift problem matches: Coconut feeds the raw last hidden state back with no normalisation, and Geiping et al. prove
that recurrence unstable. Hence ONE boundary operator for both axes, and a convergence test (Geiping's KL threshold)
or a written halting head (ACT, the Universal Transformer) for the count.

**What the loop can compute that E29 wrote as a procedure.** Value Iteration Networks (Tamar et al. 2016): value
iteration on a grid is K iterations of one layer — a convolution per action and a max — so the planner IS a looped
layer over the frame; E28's block has the convolution (gather heads) and the transition (lookup head), and a backup
is a max over the anchor slot. Coconut: a superposed frontier in the state. Searchformer (Lehnert et al. 2024) and
Stream of Search (Gandhi et al. 2024): the second stage of both — compress the model's own successful traces — beats
the teacher and solves what no hand-written strategy could, and that stage IS the ZipLearner loop; the first stage
(imitate a written search) is forbidden by the rule and unnecessary when the first successes come from exploration
(E27). E29's rooms stay as the test bed: calls to the goal grew with the room's area under breadth-first search and
stayed flat under a strategy; the block, looped, must show that flatness with nothing around it but the game.

**References** (one file per paper in `refs/`; overview `refs/latent_reasoning_and_looped_planning.md`):
`coconut_2412.06769.md`, `recurrent_depth_2502.05171.md`, `looped_latent_thoughts_2502.17416.md`,
`looped_transformers_programmable_computers_2301.13196.md`, `value_iteration_networks_1602.02867.md`,
`universal_transformers_1807.03819.md`, `adaptive_computation_time_1603.08983.md`, `searchformer_2402.14083.md`,
`stream_of_search_2404.03683.md`, `thinker_2307.14993.md` (recorded as what we will NOT build).

### 18.1 The course correction of 2026-09-27, and E41 — in-context learning in a written block

**The drift, named.** E33 was pre-registered with three arms: (a) the discrete machinery on text, (b) a gradient
transformer, (c) the hybrid — the hybrid being the thesis. Arms (b) and (c) were shelved under "no gradient arm"
and arm (a) then expanded to fill the space: E34, E36, E38, E39 and E40 are all refinements of a character n-gram
model. That was never the goal. §19 states it: *the substrate is a continuous transformer whose representations
are learned by gradient descent, and ZipLearner is the COMPRESSION THAT ACTS ON IT*, and §19's "Where to start"
names E32 — does "look for a name" work on a continuous representation — as the crux. The no-gradient-arm
instruction was about the COMPARISON arm; E32 is not a comparison, it is the subject, and applied literally the
instruction removed the substrate the plan depends on.
What those five experiments are worth is the case for leaving the discrete route, not a language model: E40 proves
counting converges to H(c | 8 characters) ≈ 1.75 and cannot obtain the partition; E36 localises the missing
information at word boundaries; E37 names the one axis where counting wins.

**Order set by the user 2026-09-27: (2) then (1).** (2) Move the line to the games, where the interpreter can be
WRITTEN (E28), the store COUNTED, and in-context learning tested with no transformer trained anywhere. (3) Then
unshelve E32 on a small trained net, informed by what (2) finds.

**Self-Play Pretraining with Zero Data (arXiv 2609.30063; `refs/self_play_pretraining_zero_data_2609.30063.md`,
abstract only).** A generator proposes programs interpreted by a universal Turing machine; a learner predicts the
resulting byte sequences; the generator is trained by RL to sit at the frontier of the learner's capability.
Neither model ever sees natural data, and zero-shot loss on natural datasets is reported to scale predictably with
self-play compute. **This is §19 point 3's premise — the world is generated by short programs, SGD samples a
Solomonoff-like prior — turned into a training procedure and tested from a direction we cannot test ourselves.**
Three consequences for this line: (i) their generator/learner split is our BrainBuilder/ZipLearn split with RL
where we would use a PRICE, and a generator scored by how much the learner's description length improves is the
same object, gradient-free (`ziplib/price.py` already computes it); (ii) the learner acquires ICL from programs
alone, so ICL is not about language or the world but about being trained on a mixture over GENERATORS where the
only way to predict is to infer which one you are in — and a UTM's program distribution is Zipfian and bursty by
construction, which is the Chan et al. recipe; (iii) it is the answer to E42/E43's sample-size death — GENERATE
the transitions. See the ref note's last two sections for the experiment this unblocks and for why Perlin noise
is the sharp probe (it splits into a short program plus an incompressible 256-byte seed, so in-context learning it
requires FACTORING generator from parameter — E44's question one level up).

**Why in-context learning is the right target and not a detour.** ICL is learning without weight changes:
pretraining builds the partition, the prompt selects a cell. That is §18's interpreter frame in the field's
vocabulary — *the block is an interpreter; knowledge and behaviour are programs in the context*. And the
distributional properties the literature finds behind emergent ICL (burstiness; a Zipfian long tail — and the
robotics reviews of 2026 now hypothesise the same for physical observation/action streams) are **exactly the
regime in which counting fails**: our own corpus has 5.2 tokens per word type and a median of 3 observations per
order-12 context (E36, E40). The property that breaks the count table is the property that makes ICL emerge.
E28 already contains a latent result here that was never framed as one: the rules live in MEMORY TOKENS, so
swapping the tokens changes the world model with the weights untouched.

**E41 — in-context learning in a written block (pre-registered 2026-09-27; §18 rules 2–4).** One `WrittenSim`
compiled ONCE over the full radius-1 neighbourhood, so the layout and every weight are fixed; only the memory
tokens differ between arms. Four arms on LockPath, held-out transitions:
- **store** — tokens from the counted, sleep-swept `LocalRule` table (E28's baseline; dropped cells are wildcards);
- **substitution** — the same weights with ANOTHER game's counted rules as tokens, to show the weights are not
  the world model;
- **icl(k)** — tokens built from k RAW observed transitions: whole windows, no counting, no majority vote, no
  sleep pass, no write to any store. This is the in-context arm: the only thing that changed is what is in the
  context;
- **empty** — no entry tokens, so the block has nothing to read.
Measured: cell accuracy on held-out transitions against k; the k at which icl reaches the counted store's
accuracy; and, separately, accuracy on windows that appear VERBATIM among the demonstrations against windows that
do not — the second is where the nearest-key default has to do the generalising, which is §21.10's first
NOT-DESIGNED item and the thing E28 flagged as "a free generalisation the price does not account for".
**Pass:** accuracy rises monotonically with k and icl reaches ≥ 0.9 of the store arm's accuracy at some k ≤ the
store's entry count, AND substitution is exact. **Refute:** icl flat in k (the block cannot read demonstrations),
or substitution wrong (the weights encode the game).

**RAN 2026-09-27 — PASS, one clause missed (`RESULTS.md`).** Weights byte-identical across every arm; with an
empty context the block scores **0.000**, so it holds no world model and everything it knows is in the context.
One demonstration per action is worth 0.570; eight (32 tokens) reach 0.743, matching the counted store's 0.732 at
a quarter of the tokens; 128 reach **0.830, beating the store by +0.098**. All 358 held-out items are windows that
appear nowhere in the context, so the entire result is the nearest-key default generalising. Missed: monotonicity
(a dip at k = 2). Withdrawn as a test: the substitution arm — LockPath's four actions are movement in four
directions and agree too much for it to discriminate; `empty` at 0.000 is the control that carries the claim.
**RETRACTED THE SAME DAY by E42.** A constant "the cell keeps its colour" predictor scores **0.922** on the metric
E41 used, above both of its numbers, and E41 did not measure it. So "raw demonstrations beat the counted store" is
withdrawn, and the price-destroys-generalisation claim is back to three sightings (E28, E35, E39) until the
comparison is redone on windows where the cell actually changes. What survives is the mechanism: byte-identical
weights and 0.000 on an empty context, so the world model is in the context and not in the weights.
**The standing rule this produced:** no experiment in this line reports an aggregate cell accuracy without the
unchanged baseline beside it, and the metric of record is accuracy on CHANGED cells, where the constant scores
zero and a world model has to earn its keep (E25: the failures are the missed changes).

### 18.2 Two axes of recurrence as the answer to self-play's depth ceiling (the user's observation, 2026-09-30)

**The ceiling.** Self-play pretraining (§18.1, `refs/self_play_pretraining_zero_data_2609.30063.md`) has its
learner predict a program's OUTPUT BYTES — never its execution trace. So the learner must perform the program's
per-byte computation internally, and a fixed-depth autoregressive transformer cannot: log-precision transformers
sit in uniform TC⁰ (Merrill & Sabharwal 2023) and cannot do iterated composition or state tracking at fixed depth,
which chain-of-thought provably repairs. That predicts exactly the result the abstract reports — the curriculum
discovers "recognizable mathematical sequences", the arithmetic-progression end of program space, where per-byte
work is SHALLOW. A program with a genuine **nested loop** — an inner iteration per outer step, which is what
Perlin's octave sum is — has deep per-byte work, and no amount of curriculum lets a fixed-depth learner fit it.
The generator can propose it; the learner structurally cannot learn it.

**Why our substrate is the natural fix, and it is already designed.** §18 commits to TWO recurrences and its own
words map onto a nested loop exactly: *"Depth recurrence refines a fixed state; sequence recurrence ADDS state —
positions the loop can write to and read back, memory for a plan of unknown length."*

| axis | our mechanism | the loop it is |
|---|---|---|
| **depth** | the core applied K times to the same positions (`LoopedModel`, E26/E30) | the INNER loop — unbounded per-token computation; Saunshi et al. (`refs/looped_latent_thoughts_2502.17416.md`): T loops simulate T steps of chain-of-thought |
| **sequence** | Coconut's continuous thoughts, a position's final state fed forward | the OUTER loop — new state per step, memory of unknown length |

So a looped-plus-thoughts learner should reach program classes a fixed-depth learner cannot, and the gap should
appear **specifically on nested loops**. That is a sharp, falsifiable prediction about where their curve flattens.

**Two further consequences worth recording.**
*The curriculum gains a second dimension.* Their generator proposes programs at the frontier of the learner's
capability. If the learner's capability is itself elastic — more passes for a harder program, under the halting
rule §18 already commits to — then the frontier moves in DEPTH as well as in program length, and the curriculum
is two-dimensional where theirs is one.
*Depth is a speed prior.* Solomonoff weights a program by 2^(−length) and says nothing about runtime; Levin's Kt
adds log(time), which is Schmidhuber's speed prior. **A fixed-depth learner is a speed prior with a hard cutoff —
it can only represent programs whose runtime per output symbol fits in its layers. Depth recurrence turns that
cutoff into a cost.** (E29's retracted design was a Levin search; the measurement stands and the framing survives.)

**What this does NOT get for free.** Unbounded depth needs a halting rule — convergence testing (Geiping's KL
threshold) or a written halting head (ACT/PonderNet), both already in §18 — and Coconut's c = 3 instability is the
recorded hazard on the sequence axis. The full version of this is a TRAINING experiment and belongs after E32, not
before it.

**E45 — does depth buy what width cannot? (pre-registered 2026-09-30; CPU, gradient-free, testable now.)**
The expressivity half needs no training, because E43 handed us the case. Toggle's rule changes cells far from the
one acted on; E43 read that as needing a non-local PREDICATE (a content-addressed read). There is a second
candidate the two-axis view supplies and E43 never considered: **a distant effect is a signal that PROPAGATES, and
propagation is the same local rule applied K times** — depth, not width. One pass of a radius-1 rule moves
information one cell; k passes move it k cells, at k × 8 gather heads instead of a (2k+1)² window's.
Arms, on synthetic grid dynamics generated from short programs (so the generating rule and its description length
are known, and changed cells are unlimited — the fix for E42/E43's 21–34-item samples): cellular-automaton-style
rules with a propagation distance p ∈ {1, 2, 3, 4}, predicted by (a) one pass at radius p, (b) p passes at
radius 1, (c) one pass at radius 1 (the floor). Measured: accuracy on changed cells; parameters and gather heads
each arm costs; observations per distinct key (E44's factorization measure) for each.
**Pass:** (b) matches (a) on accuracy at every p while costing asymptotically fewer heads AND keeping observations
per key roughly flat where (a)'s collapses. **Refute:** (b) below (a) at any p ≥ 2 — depth does not substitute for
width, and E43's non-local-predicate reading is the only route.

### 18.3 E46 — a Brainfuck interpreter as a written looped block (pre-registered 2026-10-01)

Reading the body of 2609.30063 (§18.1's reference) removed the obstacle to building its result our way: **its
program space is Brainfuck** — eight instructions plus ten single-byte macros, byte output. Eight instructions
against our eight (§21.3), and every one of them maps:

| Brainfuck | written as |
|---|---|
| `>` `<` | the data pointer is a REGISTER; `Quantise` increments it |
| `+` `-` | the cell under the pointer is incremented — a `Row` on the pointed cell |
| `[` `]` | `Branch` on "the current cell is zero" — §21.3 item 7, which is how Giannou et al. get conditional branching |
| `.` `,` | `Readout` / an input register |
| the fetch-execute cycle | the DEPTH loop, one instruction per pass, the instruction pointer a register |

So this is E28's trick on a UNIVERSAL machine instead of a gridworld, and it is Giannou et al.'s "looped
transformer as a programmable computer" at a size we can actually compile and verify. The program sits in the
CONTEXT as tokens, which is §18's interpreter claim in its strongest form.

**Measured:** the written block executed on N sampled Brainfuck programs (drawn as the paper draws them), its
output bytes against a reference interpreter, under a step budget. Plus compile time, d, heads, and passes per
emitted byte. **Pass:** byte-exact on ≥ 0.99 of programs that halt within the budget, with every disagreement
traced to a named cause. **Refute:** any systematic disagreement, or a construction needing an instruction outside
§21.3 — which would be a precise statement of what our instruction set lacks for universality.

**Why it is worth doing before the learning half.** It makes the substrate for everything downstream: E45's
generated dynamics become Brainfuck programs; the ICL experiments get unlimited data with known generators and
known description lengths; and a verified written interpreter is the thing a self-play curriculum would then be
PRICED against rather than RL-trained against (§18.1). It needs no gradient and no training.

---

## 19. The library problem, and the continuous thesis (DESIGNED 2026-09-21)

**The library problem.** §5's library is a list I typed and §7 grows it by three steps that have all run. Every item
those steps ever minted is an INSTANCE of a kind already on the list: a particular permutation, a word in the
generators. They have never minted a KIND — a new form of predicate or feature. E17 met the wall exactly (a relation
is in no kind: 0.51), and E29's only real generalisation came from a kind typed for it. §7's own loss condition —
"the items a task needs are not inside any generic structure" — bites on any game the library was not written for:
counting, ordering, "the same colour as the key", a piece that rotates.

**The resolution, in the interpreter frame (§18).** The library is not a list of structures but the INSTRUCTION SET
of the written block — fixed, small, complete: what one attention head, the residual and a threshold can compute
(read by offset, read by content, read by direction, pool, write, compare, branch — §21.3). Everything else is a
program over it, data in the context; a new kind is a program, never a new instruction; and §7's third step becomes
right at the right level — a sub-program recurring across programs is compressed into a named macro. That is library
learning as DreamCoder does it (Ellis et al. 2021: wake = solve by search with the current library, sleep = abstract
recurring sub-programs into new primitives; Stitch and LILO are the successors), and it is the only known form that
grows KINDS by compression. Completeness comes from the instruction set (Giannou et al.: a one-instruction computer
suffices); the cost is the search, exponential in the length of what is new and tamed only by the ladder of macros.
E29's rule applies to the instruction set itself: it must be the substrate's own primitives, not features I like.

### 19.1 The three levels of naming — what the library problem shares with the missing word inventory (2026-09-27)

E36 found that the text learner has no word-level unit: the `Store` indexes contexts (the last R characters) and
never abstracts a chunk, so `amabat` exists in the model only as a set of values spread over many contexts, never as
an object. The first reaction was that this is the library problem one level down. **That was too strong**, and the
correction is the useful part — there are THREE levels of naming, and we sit at different heights in different
places:

| level | what is named | have we got it? | where it lives in compression |
|---|---|---|---|
| 1. **values in a fixed address space** | an outcome per pre-declared index | yes, everywhere | PPM, CTW, our chain, every count table |
| 2. **an exact recurrence** | a recurring composite gets a name, paid once, so its uses become cheap | yes ON MATRICES (§7 step 3: E3's 17 named permutations, E8's 3 generators, 161 → 121 bits), no on TEXT | the ZIP/LZ78/LZW dictionary; BPE; Sequitur |
| 3. **a pattern with holes** | a recurring SKELETON is named and the differences become arguments | **nowhere** | DreamCoder's abstraction; Goldsmith's signature; Morfessor's morph paradigm |

**The shared failure, stated exactly.** At level 1 the learner can only move within a pre-declared address space:
nothing in the price ever creates a new address. §7 step 3 is the operation that creates one — but it tests
EQUALITY, so it can only name things that recur *identically*, which means every name it mints is another INSTANCE.
That is precisely why the library has never minted a KIND (§19) and it is a different thing from why text has no
words. Text is missing level 2 **by omission**: we picked the context family of compressors when the dictionary
family is the one that mints units, and E3/E8 prove we can already do level 2 on matrices. The library is missing
level 3 **by absence**: no anti-unification exists anywhere in the codebase.

**What level 3 requires, concretely.** Anti-unification: given several observed objects, find their most specific
common generalisation, name the skeleton, and let the positions where they differ become argument slots. Our price
already knows how to charge for this — a name costs its definition once and its uses log₂(library size) — so the
missing piece is the operation, not the currency.

**Why this also explains three results we already have.** E17's relational goals failed at 0.51 because a relation
IS a pattern with holes ("the output is the reverse of the input" has an argument slot) and there was no object to
identify it into; identify-then-plan asked for a level-3 name from level-2 machinery. R_h is enormous where
structure recurs exactly (E12's abelian group, 134×) and nearly nothing where it recurs only up to arguments (E8,
1.33×). And §20's diffusion route works precisely because the SCHEDULE supplies the holes: the intermediate targets
are given, so no abstraction has to be found.

**The cheapest test of level 3 is morphology, and we now have the corpus for it.** A Goldsmith signature — a set of
stems crossed with a set of suffixes, chosen by description length (Goldsmith 2001, *Unsupervised Learning of the
Morphology of a Natural Language*; Creutz & Lagus's Morfessor) — is an anti-unification with two argument slots. It
is DreamCoder's abstraction step on a domain that is cheap, countable, has a 25-year literature of baselines, and
directly attacks the sparsity E36 measured (Latin's inflection is what explodes the type count to 69,629 at 2.7M
characters). So building it is not a detour from the library problem: **it is the library problem's missing
operation, tested where it is cheapest.** `corpora/latin_classical` (35.0M characters, 48 authors) exists for this.

**And the connection to §18's open question.** §19 lacks the OPERATION (abstraction with holes); §18 lacks the
DATA (successful traces to compress into programs — "what compressing behaviour writes into the block"). They are
one bottleneck seen from two ends, because anti-unification needs several instances of a solved thing before it can
find their skeleton. That is the honest argument for a puzzle curriculum: not that puzzles teach representations —
BrainBuilder writes those — but that they are a SOURCE OF SOLVED TRACES, which is the raw material level 3 consumes
and which we have none of for language. Searchformer and Stream of Search (§18) both show the second stage —
compress your own successful traces — beating the teacher.

**S4 RAN 2026-09-27 (`RESULTS.md`): the ladder needs a fourth rung.** Level 3 pays — signatures cost 0.625 of the
literal code against BPE's 0.711 at a matched vocabulary target, while paying a correction bitmap they never need —
and the suffixes MDL selects are Latin's real inflectional endings (-is, -um, -rum, -ibus, -tur, -ntur, -mus, -unt
…) plus the enclitic -que on 20,293 stems, with no labels or grammar anywhere. But the evidence multiple is 1.66×,
not the 3× pre-registered, and the reason is that the model makes ONE cut per word. Latin is stem + derivation +
inflection + clitic, so a single hole catches only the outermost layer: the biggest signatures are degenerate pairs
({NULL, -que}, {NULL, -s}) and `-sque` appears as an atomic suffix rather than -s + -que.

So the missing operation is not anti-unification but **recursive anti-unification — a pattern whose arguments are
themselves patterns.** That is the same gap OPEN-5 recorded at the other level ("Remaining: words of words") and it
is what DreamCoder's library is (macros built of macros). The two levels fail in the same way, which strengthens
this section rather than weakening it: one operator is missing, at both heights.

| level | what is named | have we got it? |
|---|---|---|
| 3. a pattern with ONE hole | a skeleton, its argument filled from a list | **yes, as of S4 — and it pays 12%** |
| 4. a pattern whose arguments are patterns | recursive composition of skeletons | no, at either height |

NOT DESIGNED: the recursive operator, its price, and where it sits (a sleep pass over the Store, or a `ziplearn`
method). The cheapest test is a SECOND cut per word (stem + derivation + inflection), which S4's code is three
lines from supporting.

**S4 — Level 3 on morphology** (CPU, ≤ 20 min, no network). On `corpora/latin_classical`: induce stems and suffixes
by two-part code (a lexicon of stems + a set of signatures + the cost of expressing each word type), against two
controls — no segmentation (the E36 baseline) and BPE at a matched vocabulary size (level 2, exact recurrences
only). Measured: total description length of the word types under each; tokens per *stem* against the 5.2 tokens
per surface form; and bits/char of a context model over the induced units against the character chain's 1.8225.
Pass: the signature model beats BPE on description length of the type inventory AND raises tokens per unit by ≥ 3×;
bits/char is reported, not required to win. Refute: no better than BPE — which would say the holes buy nothing and
level 2 is the whole story.

**The continuous thesis.** Transformers trained by gradient descent are the one design that has compressed a messy,
high-dimensional world, and they keep improving. So the substrate is a continuous transformer and ZipLearner is the
COMPRESSION THAT ACTS ON IT; the neat discrete world is the case where the cheapest description is exact (E28's block
is the M → ∞ limit of soft attention over sparse features) and is not special-cased. What is known about how gradient
descent uncovers continuous representations:

1. *Features are directions, more of them than dimensions* (Elhage et al. 2022, superposition; Bricken/Templeton et
   al. 2023–24, sparse dictionaries): a feature is a direction, sparse features pack into nearly-orthogonal
   directions, composition is roughly addition. Our one-hot code is this with orthogonality exact. **§22 turns this
   into a design choice rather than an accident.**
2. *Gradient descent extracts the data's statistical modes, strongest first, each in a burst* (Saxe, McClelland &
   Ganguli 2014, 2019): a deep linear net learns the singular modes of the input–output correlation in order of
   singular value, and in a hierarchical domain those modes are the hierarchy of concepts, coarse to fine. Rare
   compositional structure comes late or never — E0–E26's 0/8.
3. *The parameter-to-function map is biased toward simple functions* (Valle-Pérez, Camargo & Louis 2018; Mingard et
   al. 2021; Hochreiter & Schmidhuber 1997; Hinton & van Camp 1993): the volume of parameters mapping to a function
   falls off with its complexity, so init plus SGD samples from a Solomonoff-like prior. Gradient descent in an
   overparameterised net is already a soft approximate search biased toward short descriptions; ZipLearner's
   principle is what it approximates, made exact where exactness pays.
4. *In-context learning is learned inference over a task family* (Xie et al. 2021; von Oswald et al. 2023; Garg et
   al. 2022): the net's features are the family's parameters and the forward pass infers them — "fit first, name
   later" inside the forward pass.
5. *What it does not do*: exact algorithms and their length generalisation; compositions that are not statistical
   modes; learning without forgetting (E6b below chance vs E6's 100%); readability. Those are where description
   length must be enforced rather than approximated.

Hence the division of labour: (a) SLEEP = consolidate the learned representation into the cheapest description — a
sparse dictionary, low-rank factors, and where cheaper, exact tables and programs; §7's "look for a name" on
continuous activations, which is mechanically what a sparse autoencoder is; (b) THINKING = §18; (c) CONTINUAL
LEARNING = consolidation by price instead of replay. The library of §5 becomes the FORMS a compression can take —
dictionary, low-rank, convolution, program — priced.

**The reach** (asked 2026-09-21). *Mathematics*: in principle everything with a procedural or formal description, in
time exponential in the length of what is new under the current library; the ladder of macros is the mechanism, the
general instruction set is still to be written. *Abstract normative concepts (goodness, justice)*: the MECHANISM of
abstraction (a shared description with the varying parts as arguments, adopted when cheaper) and abstract THOUGHT
(programs over programs) are in scope; the concepts are not — they have no short program, are graded, relational and
normative, and need the continuous regime, language as the domain, and a source of value beyond one number.
*Vision*: the structural half is near — features at relative positions in an object-centred frame under a
transformation group (E28's frame in 2D, E12's group, SE(3) for 3D), and "a chair without the length of its legs" is
exactly what compression yields (the macro is what recurs, the arguments what varies). What is missing is the front
end from pixels to a code with parts, where §5's repeated diagonal (a convolution) is the unbuilt first step. The two
bounds on all three: the SEARCH and the CODE — a concept is short only in the right representation, which is the seam
§23 attacks.

**Where to start.** The claim everything depends on is that "look for a name" works on a continuous representation
(E32 — shelved with the gradient arm). The user's test case is a language model on the Latin corpus (E33), where the
discrete machinery is at home and the number of interest is data efficiency. Then a messy continuous version of what
we have (shapes with continuous variation, where "macro + arguments" can be checked against ground truth), and only
then pixels. Not to be done: throwing away the discrete machinery (it is the exact edge case), or starting from real
vision before the compression step is shown to work on something readable.

---

## 20. Credit assignment without backpropagation — the diffusion interpretation (DECIDED 2026-09-21)

**The objection this answers.** §19 left one thing to gradient descent: the *nonlinear composition of learned
features across layers*, because backpropagation is credit assignment through a differentiable composition and the
gradient-free alternatives had lost to it or were blind. The user's question: if the data is compressed into weights,
can the COMPOSITIONS be compressed too?

**The paper.** Shing, Koyama & Akiba (Sakana AI), *DiffusionBlocks: Block-wise Neural Network Training via Diffusion
Interpretation*, arXiv 2506.14202; `refs/diffusionblocks_2506.14202.md`. Its key sentence: "residual connections
naturally correspond to updates in a dynamical system. With minimal modifications to this system, we can convert the
updates to those of a denoising process, where each block can be learned independently by leveraging the score
matching objective." Trained one block at a time it matches end-to-end training on vision, diffusion, autoregressive,
masked-diffusion and RECURRENT-DEPTH transformers — the last being our looped block's family.

**What it means.** Credit assignment through depth is needed only when the intermediate representations are unknown.
The diffusion interpretation DEFINES them: block k's input is the data at noise level t_k, its target is the data at
t_{k−1}, and both are observable, so every block can be fitted LOCALLY by any method — including ZipLearner's — with
no gradient through the stack. Our loop was already this shape: Geiping et al. start the recurrence from noise, the
anchor is the conditioning, and E30's decaying reads of earlier passes are what multi-step ODE solvers do with
previous steps. Two corrections to §19 follow: (1) **nonlinearity is not the obstacle** — a denoising block may be as
nonlinear as we like because its target is known; what had no closed form was the composition, and the schedule
supplies it; (2) **size is not the obstacle** — the limit on end-to-end training was memory across the stack, which
blockwise fitting removes, so a written network can be as deep as its schedule.

**"Compress the compositions" — two levels.** *Composition fixed by a schedule*: each block's weights are the
cheapest description of its own (noisier in, cleaner out) pairs — DiffusionBlocks written rather than trained.
*Composition not given by a schedule* — programs whose intermediate states are not noisy versions of the answer: then
the composition itself is the description to compress (which block reads what, as flag-conditioned route tables; how
many passes, by convergence; which macro, by library growth). That is DreamCoder's abstraction step in our words, and
intermediate targets come from tools already here: §6/E2's inversion of the next layer, and Coconut's curriculum.
What the interpretation does not give: a schedule for arbitrary programs. Its class is refinement — generation,
repair, planning as iteration on the frame (VIN), and the masked-diffusion form of language.

**Where a gradient is needed — the measured answer (E34/E35).** *Not needed, shown:* exact structure over discrete
codes (E3, E8, E12, E18, E28); memory and continual learning (E6 vs E6b); choosing among experts (E34: the Bayesian
mixture equals the linear mixer); credit assignment through refinement depth (E35: eight blocks fitted locally);
linear representation learning (an SVD — and §23 argues this reaches further than it looks). *Needed, most likely, in
two local places:* (1) **combining evidence MULTIPLICATIVELY** — E34: eighteen contexts averaged gain nothing (1.820
vs 1.823), a geometric mean with Bayesian exponents gains 0.02, a raw product over-sharpens (7.7); the learned
exponents of a logistic mixer (PAQ) are a few thousand weights on a convex online loss with no backpropagation; round
2 sharpened the obstruction — a closed-form LDA solve on counted second moments MATCHES the grid posterior for
DIVERSE experts (−0.079 vs −0.077) and FAILS for NESTED ones (exact exponent ≈ 0), and text experts are maximally
nested (order-k ⊂ order-k+1), so the exponents are not recoverable by counting second moments; (2) **features inside a
block** when generalisation must come from shared structure rather than lookup — E35: a written radius-1 denoiser
either remembers windows (works, no compression) or collapses to the identity (the price), where DiffusionBlocks'
blocks generalise by learned features; per block, on the block's local target, never end to end. The alternative —
priced search over compositions (§19) — is unproven, not refuted. **§24 re-decomposes (1) and (2) against what a
neural language model actually gains, and finds (1) is the larger half.**

**The target, restated.** 1.0 bits/character is a number of enwik8's scale (10⁸ characters) and is unreachable on
2.7M characters of Latin by any method; §24.1 gives the corrected target and its derivation.

---

## 21. BrainBuilder: the three-way split, the blueprint, and the blueprint research programme (DESIGNED 2026-09-22; compiler BUILT 2026-09-22, PARTIAL)

**What this section decides.** ZipLearn splits into three: a LIBRARY both others import; ZIPLEARN, which gives a
built brain knowledge and skills by counting; and BRAINBUILDER, which runs BEFORE ZipLearn and writes the network as
a fully developed brain from a BLUEPRINT plus the architecture numbers. The genome analogy is exact in one respect
and is the rule: **the blueprint holds repeatable top-down instructions for building the brain — which circuits
exist, where they sit, how they connect — never a synaptic weight that encodes a fact.**

Every claim names a file, a function or a number. Where a piece is not designed it says NOT DESIGNED; §21.10 lists
them. Nothing here calls `backward` except §21.8, whose subject is what a gradient trainer must be ABLE to do to a
built brain.

### 21.0 Vocabulary

- *Substrate*: `experiments/transformers/h1_lid.py` — `Attn` (q/k/v/proj, RoPE by pairs, `causal`, the k/v `cache`),
  `AttnRes` (the flag-conditioned route), `Block` (attn + MLP `Linear(d,4d) → GELU → Linear(4d,d)`), `Model`,
  `LoopedModel` (a core applied K times with a boundary operator between passes). Five edits, §21.4.
- *Residual layout*: named SUBSPACES of the residual vector, each a slice of dims (E28: `self.T`, `self.N[j]`,
  `self.A`, `ROW`, `COL`, `OUT`, the flag dims, allocated by `base += C`).
- *Token class*: cell, entry, register, border, zero — one flag dim each; what a token is, readable by a head.
- *Instruction*: a weight TEMPLATE that one attention head, one MLP row pair, one route query or the boundary
  operator can execute exactly over one-hot codes (§21.3). The E29 rule (§19): the set is the substrate's
  primitives; no entry names a colour, a wall, a direction, an offset chosen because a game uses it, or a character.
- *Circuit*: one instruction INSTANCE placed in the brain (its parameters, the subspaces it reads and writes).
- *Blueprint*: the JSON listing subspaces, token classes, circuits, stores, registers, the loop, the routes and the
  tests. *Arch*: the size numbers (d, heads, head width, layers, position code, max length).
- *Brain*: the compiled artefact — an `h1_lid` model whose weights are written tensors, plus the resolved layout,
  the codec, the loop spec, the empty stores and registers. STRUCTURE and STATE are its two parameter classes (§21.8).
- *Store*: the one count table, with three tensor forms: memory tokens, an attended key/value BANK, MLP rows.
- *Pass*: one application of the core to every position. *Anchor*: what the boundary operator injects before a pass
  (E28: the action). *Boundary operator*: the written tensor op between passes (keep, clear, quantise, commit,
  anchor, select, halt). *Halt flag*: a flag dim the runner reads and nothing else.

### 21.1 The rules the blueprint must obey

1. **No typed content.** Offsets are not typed: the blueprint declares a receptive FIELD as an extent (every offset
   with |di| ≤ r, |dj| ≤ r, centre excluded — 8 at r = 1, 80 at r = 4), BrainBuilder writes one gather head per
   offset (overproduction), and ZipLearn's sleep pass zeroes the heads whose cells the masks dropped (pruning). The
   born neighbourhood is a number like d; the used neighbourhood is learned (E24: rules keep 2–4 of 9 cells). Query
   weights that depend on the number of cells are computed by the compiler from the layout, never typed.
2. **No Python dispatch at run time.** `Brain.run` is plain torch: encode, loop (pass, boundary operator, read the
   halt flag), coda. The Store's readers (`predict` by hash, `nearest` by matmul, product-key candidates) exist ONLY
   in ZipLearn's offline apparatus — sleep, the leave-one-out price, consolidation — never inside a pass (§18 rule 3).
3. **One block class, one substrate.** No `WrittenBlock` beside `Block`, no hidden substrate op. Every consolidated
   entry is written into the existing `Linear → GELU → Linear` MLP as a `Row` (key row, bias −threshold, value row)
   at a scale where GELU is a threshold; the Willshaw form is many `Row`s whose sparse keys share hidden units
   (§21.7), the same tensors. `norm="none"` is a flag, not a class.
4. **No noise source inside a head.** Noise is an INPUT: a register field `noise` filled before a pass by the runner
   from a seeded generator, exactly as the frame and anchor are filled (§21.8a).
5. **As many layers as the dependency order needs.** Heads in one layer run in parallel; gather → match → pool →
   compare needs three attention layers plus one MLP. The compiler orders layers from the circuits' declared
   `reads`/`writes`; the core is as many `Block`s as that order needs.
6. **The goal is never initialised by the blueprint.** The GOAL register is FILLED by the goal model (ZipLearn's
   counted win keys, E19) or by a harness that says so.
7. **Ties are reported, not forbidden.** E28 measured 1,434 and 1,475 ties on windows the table never saw; ties are
   between equally near entries and are independent of M. `verify` checks purity on the per-instruction unit tests
   (where every query has a unique target) and REPORTS the tie count on the blueprint tests.
8. **The product-key reader is Lample et al. 2019's, not an argmax over sub-key maxima:** sub-key scores select a
   CANDIDATE set (top-k per half), and the candidates that are stored keys are rescored exactly.

### 21.2 The module split — files and APIs

The library is `experiments/ziplearn/ziplib/`, one file per concept, each a MOVE of an existing component with the
old location deleted in the same commit (`feedback_decisive_full_cutover`). **The regression numbers that must not
move:** E18 1.000; E28 530/530 and 509/509; E24's masks [3, 3, 3, 2] on LockPath and 9,192 → 660 bits; E6 1.000
retention and 3 blocks; E33 1.873 frozen at 2.7M characters; E35 strict 0.946 / 0.880 / 0.800.

| file | holds | moved from |
|---|---|---|
| `ziplib/price.py` | `flag_bits`, `pay`, `precision_bits`; `entropy`, `table_price` (the two-part code of a merged table); `kt_bits` (the per-context KT code), `prequential_bits` (the blended-backoff code the predictor pays) | `ziplearner.py`, `arcgames.py` (`entropy`, `LocalRule._price`), `textlm.py` (`ContextLM._bits`, `_prequential`) |
| `ziplib/store.py` | `Store`: ONE count table over masked windows — `full`, `mask`, `table`, `majority`, `stats`, `observe`, `predict`, `sleep(strict)`, `chain()`, `price()`, `entries()`; `windows_grid`/`windows_seq`; the three tensor forms `as_tokens`/`as_bank`/`as_rows`; the offline readers `nearest`/`candidates`; `consolidate(n0, eps0, price)` | `arcgames.py` (`LocalRule`, `ActionModel`'s radius competition, `GoalModel`'s keys), `textlm.py`, `e35.py` (`NearestRule`) |
| `ziplib/layout.py` | `Layout.allocate(subspaces, d)` → slices; symbolic widths bound from `Arch.dims`; a free-list of spare dims; the head-channel allocator `channels(head, needs)`; error when the layout exceeds d | `e28.py` lines 75–87 |
| `ziplib/codec.py` | `Codec.encode(cls, **fields)` / `decode(vector)` generated from the token classes; coordinate codes `onehot` (exact, small grids) and `rope2d` (row phases in one half of a head's rotary pairs, column phases in the other; a sequence is the row-0 case = E18's RoPE); memory-token construction from `Store.entries()` with an all-zero subspace where the mask dropped a cell (a wildcard) | `e28.py` `encode`; `e18.py` |
| `ziplib/instructions.py` | the instruction set (§21.3): `Gather`, `Match`, `Pool`, `Broadcast`, `Row`, `Compare`, `Branch`, `Readout`, and the boundary parameters `Keep`, `Clear`, `Quantise`, `Commit`, `Anchor`, `Halt`; each with `needs`, `emit`, `test` | `e28.py`, `e18.py` `write`, `e5.py`'s gate templates |
| `ziplib/blueprint.py` | `Blueprint` and `Arch` dataclasses, `load`/`save`/`validate` | new |
| `ziplib/brain.py` | `Brain`: the model (`norm="none"`), the resolved `Layout` + `Codec`, the loop spec, `BoundaryOp`, the stores (empty), the registers, `verified=False`; `run`, `think`, `banks[name]`, `rows.write`, `param_groups()`, `save`/`load` | `e28.py` `rollout`; `h1_lid.py` `forward_embedded` |
| `brainbuilder.py` | `compile(blueprint, arch) → Brain`, `verify(brain) → Report`, `anatomy(brain)` (every nonzero traced to its circuit), `ablate(blueprint, circuit)`, the CLI, the experiments B0–B8 | new; generalises `WrittenSim.__init__` and `e18.write` |
| `ziplearn.py` | the learner over a VERIFIED brain: `observe`, `sleep`, `write`, `count_inverse` (GCML's W), `mix`, `lift` (the decompiler, §21.8), `ledger`, the harnesses `play(env, brain)` and `read(corpus, brain)`, the knowledge experiments Z1–Z4 | `ziplearner.py`, `arcgames.py` (`Player.observe`/`sleep`/`play`), `textlm.py`, `research/expA_exponent_grid.py` |
| `blueprints/*.json` | `gridworld.json` (§21.2.2), `induction.json` (E18) | new |

`arcgames.py` keeps only the environment adapter and the exploring `Player`; `textlm.py`, `e28.py`, `e34.py`,
`e35.py` become callers of `ziplearn.py` or are deleted.

```
ziplib.price:   flag_bits(is_exc, n_right, n_wrong) -> bits; pay(s, right); precision_bits(n, span, sigma) -> bits;
                table_price(merged) -> bits; kt_bits(ctx_keys, nxt, V) -> bits; prequential_bits(ctx, nxt, mask, V, limit) -> bits
ziplib.store:   Store(field|back, V).observe(keys, y) / predict(keys) -> (y | None) / sleep(strict=True) -> (before, after, cells)
                / chain() -> [mask] / price() -> bits / entries() -> [(key, counts, stats)]
                / as_tokens(layout, codec) -> Tensor[N, d] / as_bank(layout, codec) -> (keys[N, d], values[N, d])
                / as_rows(layout, code="onehot"|"sparse", k=None) -> [(key_row, threshold, value_row)]
                / nearest(keys, M) -> y / candidates(keys, k) -> [entry ids] / consolidate(n0, eps0, price) -> moved
ziplib.layout:  Layout.allocate(subspaces, d); Layout[name] -> slice; Layout.free(n) -> slice; Layout.channels(head, needs)
ziplib.codec:   Codec.encode(cls, **fields) -> Tensor[d]; Codec.decode(v) -> dict; Codec.coords(tokens) -> LongTensor[T, 2]
ziplib.instructions: Instruction.needs(params, layout, arch) -> Needs(heads, hd_min, content_dims, position_pairs, rows)
                     Instruction.emit(params, layout, arch, slot, M) -> [Write]; Instruction.test(params, layout, brain, n) -> Report
ziplib.blueprint: Blueprint.load(path) / save(path) / validate() -> [error]; Arch.load(path); Arch.auto(dims)
ziplib.brain:   Brain.run(tokens, anchors=None, max_passes=8) -> (tokens, passes); Brain.think(tokens, n=None) -> tokens
                Brain.banks[name]; Brain.rows.write(key, value); Brain.param_groups() -> {"structure": [...], "state": [...]}
                Brain.save(dir) / Brain.load(dir); Brain.to(device)
brainbuilder:   compile(blueprint, arch) -> Brain; verify(brain) -> Report; anatomy(brain) -> table; ablate(bp, name) -> Blueprint
ziplearn:       ZipLearn(brain).observe(before, action, after, outcome); .sleep() -> report; .write() -> bits
                .count_inverse(transitions); .mix(stream) -> table; .lift() -> {store: Store}; .ledger() -> {circuit: bits}
                .play(env, budget, max_levels, seed) -> results; .read(train_seqs, test_seq) -> (bpc_frozen, bpc_online)
```

### 21.2.1 The blueprint as data — the schema

Widths may be symbols bound from `Arch.dims`. Names are the only cross-references; the compiler turns them into
offsets, heads, layers and rows.

```
Blueprint = {
  "name": str,
  "field":     {"kind": "lattice", "r": int},                       # the born receptive field
  "subspaces": [{"name": str, "width": int|symbol, "kind": "onehot"|"flag"|"scalar"|"dist"|"code"}],
  "tokens":    [{"class": str, "flag": subspace, "fields": {subspace: "input"|"coord.row"|"coord.col"|"const:<v>"|"store"|"runner"}}],
  "circuits":  [{"name": str, "instr": key of INSTRUCTIONS, "params": {...}, "per": "offset"|null,
                 "place": {"layer": int|"auto", "head": int|"auto"}, "reads": [subspace], "writes": [subspace],
                 "prunable": bool}],
  "stores":    [{"name": str, "class": token class, "key": [subspace], "value": subspace, "form": "tokens"|"bank"|"rows",
                 "capacity": int, "nulls": [token class]}],
  "registers": [{"name": str, "slots": int, "holds": [subspace], "filled_by": "core"|"store"|"runner"}],
  "loop":      {"core": "auto"|[layer], "tied": true,
                "boundary": {"keep": [subspace], "clear": [subspace], "quantise": [{"read": [subspace], "write": subspace}],
                             "commit": [[from, to]], "anchor": {"sub": subspace, "from": register|"runner"}, "halt": flag},
                "sequence": {"feedback": [[from, to]]} | null},
  "routes":    [{"mixer": layer, "table": {flag: source}}],
  "codebook":  {"kind": "disjoint"|"colour"|"frame", "seed": int, "k": int|null} | null,   # §22
  "tests":     [{"name": str, "generator": str, "oracle": str, "criterion": {"exact": true} | {"min": float}}]
}
Arch = {"d_model": int|"auto", "n_head": int|"auto"|[int per layer], "head_dim": int|"auto", "n_layer": int|"auto",
        "pos": "onehot"|"rope2d", "max_len": int, "p_star": 0.99, "dims": {symbol: int}}
```

`validate` refuses: an `instr` not in `INSTRUCTIONS`; a `params` value that is a content symbol (a colour index, a
character, an offset list — an offset list is a FIELD, declared once under `field`); a register with an `init`; a
blueprint with no test. That is the mechanical E29 guard, plus the two rules a whitelist cannot enforce: offsets are
a field, goals are filled by the goal model.

### 21.2.2 The worked example — the E28 brain, `blueprints/gridworld.json`

Dims `H = 8, W = 11, C = 17` (16 colours + BORDER), `V = 16`, `nA = 4`; field `r = 1` for the regression, `r = 4`
for the generality test. Subspaces in allocation order: `colour(C)`, `nbr[o](C)` per field offset, `pred(V, dist)`,
`action(nA)`, `util(nA, scalar)`, `noise(nA, scalar)`, `goal(C)`, `task(8)`, `conf(1, scalar)`, `row(H)`, `col(W)`,
`out(V, dist)`, `flags(9)` = cell, entry, register, border, zero, surprise, halt, goal_met, bias. Token classes:
`cell` (colour from input, row/col from coords), `entry` (colour, nbr[*], action, out, conf from the store),
`register` (holds by name), `border` (colour = BORDER; the null for gathers), `zero` (empty; the null for entries).

```
gather[o]  Gather(offset=o, src=colour, dst=nbr[o], null=border, cls=cell)         reads row,col,colour  writes nbr[o]   prunable
lookup     Match(q=[colour, nbr[*], action], k=same, v=out -> pred, key_class=entry, null=zero, weight(action)="auto")
goal_read  Match(q=[colour, nbr[*]], k=same, v=flags.goal_met, key_class=entry(goal), null=zero)
bcast_act  Broadcast(reg=ACTION, sub=action)                                       reads register       writes action
bcast_goal Broadcast(reg=GOAL, sub=goal)                                           reads register       writes goal
surprise   Compare(a=pred, b=colour, flag=surprise)                                reads pred,colour    writes surprise   (MLP rows)
inverse    Pool(reg=ACTION, key_class=cell, src=goal - colour, W="inverse", dst=util)
nogo       Pool(reg=ACTION, key_class=cell, src=colour, W="nogo", dst=util, sign=-1)
task_write Pool(reg=TASK, key_class=cell, src=surprise, gate=surprise)             reads surprise       writes task
halt_row   Row(key={goal_met: 1}, threshold=0.5, value={halt: 1})                  reads goal_met       writes halt      (MLP row)
read_act   Readout(register ACTION.action -> the game's controls)
read_pred  Readout(pred -> vocab)
```

Stores: `rules` (class entry, key = colour + nbr[*] + action, value = out, form tokens, nulls border + zero),
`goals` (class entry, key = colour + nbr[*], value = goal_met). Registers: `ACTION` (action, util, noise; filled by
core), `GOAL` (goal, row, col; filled by store), `TASK` (task; filled by core), `STEP` (a one-hot pass counter;
filled by the boundary). Loop: core "auto", tied; keep = row, col, flags, task, goal; clear = nbr[*], util;
quantise = [{read: [pred], write: pred}, {read: [util, noise], write: action}]; commit = [[pred, colour]];
anchor = {sub: action, from: ACTION | runner}; halt = flags.halt. Tests: `rollout_vs_table` (E28's protocol, exact on
all-known plans ≥ 0.98) and `induction` for `induction.json` (criterion 1.0).

What the compiler makes of it at `Arch.auto`, r = 1: layer 0 = the 8 gathers + 2 broadcasts, layer 1 = lookup +
goal_read with the `surprise` rows in its MLP, layer 2 = the 3 pools with `halt_row` in its MLP; `d_layout` = 251;
layer 0 has 10 heads so `hd = max(ceil(251/10), H + W + 2, C) = 26` and `d = 260`; layers 1 and 2 are one-head layers
(`hd = d`). `M = ln((T − 1)·p*/(1 − p*))` with `T = 350`, `p* = 0.99` gives M = 10.4 (E28 typed 30). At r = 4:
80 gathers, `d_layout = 1475`; under `rope2d` the channel plan is the open problem of §22.3.

### 21.3 The instruction set

Each instruction is a class in `ziplib/instructions.py` with `needs` (heads, minimum head width, content dims,
position pairs, MLP rows), `emit` (writes addressed as `blocks.{L}.attn.qkv.weight[...]`,
`blocks.{L}.attn.proj.weight[...]`, `blocks.{L}.mlp.0.weight/bias[r]`, `blocks.{L}.mlp.2.weight[:, r]`, `emb.weight`,
`head.weight`) and `test`.

1. `Gather(offset, src, dst, null, cls)` — one head. Under `onehot`: query = own row/col code permuted by the offset,
   key = own code, value/proj copy `src` into `dst`; the `null` token's key scores 1.5·M against the class flag and
   catches an off-grid neighbour. Under `rope2d`: q and k from the `bias` flag in the position pairs, the key's
   phases `cos(di·θ_c), sin(di·θ_c)` on the row-axis pairs and `cos(dj·θ_c), sin(dj·θ_c)` on the column-axis pairs;
   the compiler computes the score profile over every (Δi, Δj) within `max_extent` and raises the pairs per axis
   until the gap between the target offset and every other is ≥ 1 (in units of M); a profile that cannot reach the
   gap is a compile error naming the head. `hd_min` = H + W + 2 (`onehot`) or 2·pairs·2 + 2 (`rope2d`).
2. `Match(q, k, v, dst, key_class, null, weights, mode)` — one head. Query and key = the named subspaces at weight M
   per dim; the attended token's `v` is written into `dst` (`mode = replace` writes v − dst, E28's `OUT − T`;
   `mode = add` adds). `weights` are per-subspace multipliers; `"auto"` means "one match on it outweighs a full match
   on all the others" and the compiler sets it to (number of other key dims) + 2 — E28's `g = nO + 2`, derived from
   the layout instead of typed. An unseen query lands on the NEAREST stored key (E28: better than "unchanged" on all
   four splits); attention has no "no entry". E18's induction head is `Match(q=tok, k=prev, v=tok, dst=out)` with
   content dims in the lowest rotary pairs (the compiler checks `max_len·θ_c < 0.1`).
3. `Pool(reg, key_class, src, W, dst, gate, sign)` — one head. The register token's query is the class flag (plus
   `M·gate` when a gate flag is named); the value is `W · src` of every token of the class; the head writes their
   mean into the register's `dst`. `W` is a value matrix in `Brain.banks` (identity by default; COUNTED by ZipLearn
   when named — GCML's inverse model). Without the gate the query lands on the `zero` token: a no-op.
4. `Broadcast(reg, sub, dst)` — one head. Every token attends to the register and copies its `sub` into `dst`.
5. `Row(key, threshold, value)` — one MLP hidden unit: `mlp.0.weight[r] = M·key`, `mlp.0.bias[r] = −M·threshold`,
   `mlp.2.weight[:, r] = value / (M·margin)`; at scale M, GELU is a threshold. The slot ZipLearn writes a
   consolidated entry into (Geva et al. 2021's key-value memory), and the halting row.
6. `Compare(a, b, flag)` — C rows: unit c = `GELU(M·(a_c + b_c) − 1.5·M)`, each summed with weight `−2/M` into
   `flag`, plus one bias row with value `+1`: the flag is 1 iff the two one-hot codes differ. E5's written gates.
7. `Branch(mixer, table)` — an `AttnRes` mixer whose query carries M on the named flag dims; the source whose flag is
   set takes the softmax mass (§15: a hard per-token route). Absent, the residual is the plain sum.
8. `Readout(sub → vocab, M_out)` — `head.weight[v, sub.start + v] = M_out`.
9. **`Relate(q, k, metric, v, dst, key_class, null)` — DESIGNED 2026-09-27, NOT BUILT (§23).** `Match` with equality
   replaced by a learned bilinear similarity: the query projection is `A` and the key projection `B` where
   `A ᵀB = Σ_qk`, the counted cross-covariance's truncated SVD (§23.2). Reduces to `Match` when `Σ_qk` is the
   identity on a one-hot subspace, which is why it is one instruction and not two. `needs`: one head, `hd_min` = the
   rank kept.
10. The boundary parameters — not heads; written tensors of `BoundaryOp`: `Keep` (a diagonal mask), `Clear` (its
    complement on the named subspaces), `Quantise(read, write)` (sum the `read` subspaces into one group, argmax,
    write a one-hot), `Commit(from, to)` (copy after quantise; E28's `pred → colour` advances the simulator),
    `Anchor(sub, from)`, `Halt(flag)` (the loop also stops when every kept register subspace is unchanged between
    two passes — exact, because registers are one-hot after `quantise`).

Sinks are not an instruction: `border` and `zero` are token classes that `Gather`, `Match` and `Pool` name as their
`null` (Xiao et al. 2023's attention sink; E28's two null tokens). **NOT in the set, and not designed:** a noise
generator (noise is an input, §21.8a); a learned-exponent mixer (E34's irreducible gradient, §20, §24); a new KIND
minted from data (§19: a new kind is a program). *Superposed subspaces moved OUT of this list on 2026-09-27: §22
designs them as a codebook, and `Layout` gains the allocation modes; the instructions are unchanged because a
codebook is a property of the LAYOUT, not of any head.*

### 21.4 The compiler (`brainbuilder.compile`) and the substrate edits

1. *Bind dims* from `Arch.dims`; expand `per: "offset"` circuits over the field; every width > 0.
2. *Allocate the layout* (`Layout.allocate`): sequential offsets in declaration order; spare dims on a free-list
   ZipLearn may claim. `d_model = "auto"` → the smallest `hd · n_head` that fits; explicit and too narrow → error
   naming the dims needed. **Allocation modes, §22.3:** `disjoint` (today's; one subspace per feature),
   `colour` (features that are never co-live share dims exactly, by graph colouring on the co-activation graph), and
   `frame` (co-live features share dims through a designed codebook at coherence μ, with the interference priced).
3. *Order the layers.* A circuit reading a subspace another writes sits in a later layer of the same pass (attention
   before the MLP within a `Block`), or the same layer of a later pass if the loop `keep`s that subspace.
   Topological sort → the minimal layer per circuit; `place.layer` pins override; a cycle no loop breaks is an error
   naming both circuits. The core is the resulting layer list, tied across passes.
4. *Allocate heads and channels.* First-fit per layer; a layer holding one `Match` whose query is the whole window is
   a one-head layer (`hd = d`); `n_head` is therefore a per-layer list. Within a head, `Layout.channels` puts content
   dims in the lowest rotary pairs and position dims in the highest; a conflict is an error naming the head. **This
   is where B0's `rope2d` cell fails today — "159 position-free pairs do not fit beside the position pairs of a
   465-wide head"; §22.3 says first-fit is the wrong algorithm and names the replacement.**
5. *Set the sharpness* `M = ln((max_len − 1)·p*/(1 − p*))`: with the intended key ahead of every rival by one match,
   the softmax mass on the target over T tokens is ≥ 1/(1 + (T − 1)e^{−M}); `p* = 0.99` gives M = 10.4 at T = 350 and
   13.0 at T = 4,400; `p* = 0.999` adds 2.3. The leak `1 − p*` is absorbed by `quantise`. This is the number
   Tracr-style compilation gets wrong by saturating (§21.8 obligation 1).
6. *Emit.* Collect every circuit's `Write` list; two writes to one address with different values → error naming both
   circuits; scatter with one `index_put_` per tensor into a zeroed state_dict (no per-element Python).
7. *Materialise.* `LoopedModel(core_layers=L, boundary=BoundaryOp, norm="none")` with the arch's `pos`; load; wrap as
   `Brain(..., verified=False)`. `Brain.save` writes `state.safetensors` (the tensor names `h1_lid.Block` already
   uses, so `.to("cuda")` and `torch.compile` need nothing), `blueprint.json`, `layout.json`, and one `<store>.npz`
   per store (the evidence the tensors are regenerated from).

*The scaling contract.* The NONZEROS written are the sum of the instructions' sizes (E28: ~550 nonzeros in two `Attn`
modules of 131,072 parameters), independent of d; the tensors are whatever the arch says; running them is ordinary
dense torch. An explicit `Arch(d_model=1024, n_head=16, n_layer=24)` holds the same circuits in its first layers, the
rest zero, and must give identical rollouts (B0's invariance check — CONFIRMED at `d_layout + 64`). Under `rope2d` a
gather head's parameters are 2 numbers per pair, independent of H, W and L — the scaling path; `onehot` is the exact
CPU check.

*Both recurrence axes (§18).* Depth: `Brain.run` applies the core, then `BoundaryOp`, once per anchor (a plan of n
actions = n passes; E28's rollout) or until the halt flag / register convergence, at most `max_passes`. Sequence:
`Brain.think` appends a thought position whose embedding is `BoundaryOp(final residual of the last position)` through
the loop's `feedback` map, using `Model.forward_embedded(h, caches, start)`; the thought count is the same halt rule.
ONE `BoundaryOp` serves both axes. Depth is WRITTEN (E28); sequence is DESIGNED and tested in B6.

*Verification* (`brainbuilder.verify`; ZipLearn refuses a brain whose `verified` is False): (1) per-instruction unit
tests — random tokens from the codec, the head's attention row read through a recording hook; pass iff the argmax is
the intended target on 100% of queries AND the smallest target-minus-runner-up logit gap is ≥ M AND the written
subspace equals the codec's expected value to 1e−3; (2) purity on those tests: every `onehot` subspace one-hot to
1e−3 after every layer; (3) the blueprint's own tests against an oracle that is not the brain, with the tie count
reported. Report → `runs/brainbuilder/<name>.json`.

*Substrate edits in `h1_lid.py`* — flags on the existing classes, defaults unchanged, no copies, **all five BUILT
2026-09-22 (E28 still passes 20/20 on the edited substrate)**: (1) `Block(norm="layer"|"none")`; (2)
`LoopedModel(core_layers=L, boundary=None|BoundaryOp)`; (3) `Model`/`LoopedModel` accept `n_head` as a per-layer
list; (4) `Attn.forward(x, cache, start, coords=None)` — with `coords` (T × 2 integers) the first half of the rotary
pairs rotates by `coords[:, 0]·θ` and the second by `coords[:, 1]·θ` (`rope2d`); `coords = None` is today's RoPE;
(5) `BoundaryOp(keep, quantise, commit, anchor, halt)`, a module whose tensors are written.

### 21.5 The executive — what trained models develop, and what BrainBuilder writes

Interpretability finds no centralised controller in trained transformers, but it finds parts: task/function vectors
(a few mid-layer heads whose summed output identifies the in-context task — Todd et al. 2023; Hendel et al. 2023);
attention sinks (Xiao et al. 2023); register tokens (Darcet et al. 2023); entropy neurons that scale output
confidence (Stolfo et al. 2024); copy-suppression heads (McDougall et al. 2023). BrainBuilder writes each as a
PLACEMENT of §21.3's instructions, and the PFC / basal ganglia / ACC / thalamus division of
`src/tbt/ARCHITECTURE.md` §3 becomes a placement too, not a module:

| part | brain region | how it is written | status |
|---|---|---|---|
| Register slots (working memory) | PFC persistent activity | tokens of class `register` at the front, one one-hot slot id each; their `holds` subspaces are `keep`-ed across passes; `Broadcast` reads them into every token | WRITTEN; what fills GOAL is the goal model's counted keys (COUNTED); the programs the loop runs beyond simulation NOT DESIGNED |
| Task vector | PFC | the `TASK` register's `Broadcast` puts the task id into every `Match` query, so one block with a different id reads a different store block — E6's `ContinualLayer.select` done by a head | WRITTEN; the ids minted by ZipLearn on refutation (COUNTED) |
| Flag-conditioned write gate | thalamic gating | a `Pool` head whose query carries `M·flag`; without the flag it lands on `zero` and the register keeps its content; a store accepts a new entry only when `surprise` is set (mint on refutation, §9 rule 2) | WRITTEN; the append is ZipLearn's `observe` reading the flag (COUNTED) |
| Surprise head | ACC | `Compare(pred, colour, surprise)` — prediction error computed by the block, not by Python | WRITTEN |
| Selection with noise | basal ganglia | `Pool` heads sum `W·(goal − colour)` (Go) and `−W_nogo·colour` (NoGo, counted from refuted actions) into `util`; `Quantise(read=[util, noise], write=action)` is winner-take-all over utility plus input noise | WRITTEN; W and W_nogo COUNTED (`count_inverse`); the noise gain and what fills `util` with no goal NOT DESIGNED (OPEN-10) |
| Halting | ACC / loop convergence | `halt_row` sets `flags.halt` from `goal_met`; `BoundaryOp` also halts when the kept registers are unchanged | WRITTEN |
| Routes | thalamus | `Branch` on the flags; `border`/`zero` sinks as the default-off channel | WRITTEN |
| Confidence calibration | entropy neurons / tonic dopamine | the coda read-out's scale and the per-context mixing (−0.041 bits/char) | COUNTED at the coda; inside the loop NOT DESIGNED |

The planner inside the block (GCML, `src/tbt/notes/gcml_neural_sampling_cognitive_maps.md` §6):
`Broadcast(GOAL)` → `Pool` utilities → `Quantise` with noise → `Broadcast(ACTION)` as the next pass's anchor →
`lookup` advances the imagined frame → `halt` on `goal_met` or convergence. WRITTEN for state goals; `W` COUNTED;
relational goals NOT DESIGNED (E17's 0.51). Not claimed: GCML's flatness ("one pass per step, calls flat in the
goal's distance") — winner-take-all over `W(s* − s)` is a one-step chooser and the imagined rollout still costs one
pass per imagined step. B3 measures passes per goal against distance.

### 21.6 The capacity answer — 2 bits per parameter, as measurable claims

Allen-Zhu & Li (arXiv 2404.05405) measure ~2 bits of extractable fact per parameter at ~1,000 exposures and ~1 at
100, unchanged at int8, lost at int4. Gardner (1988): a linear-threshold unit with n weights holds at most 2n random
±1 associations. So **2 bits per parameter is the CEILING for incompressible associations in a dense weight**, and
gradient descent at 1,000 exposures sits at it — both the operating point and, for that content, the limit. Four
axes, each a number:

1. *Per parameter, dense.* A written one-hot memory token holds one entry of ~18 bits in d = 128 floats: 0.14 bits
   per parameter — BELOW 2. Written storage does not beat gradient descent per dense parameter and the design does
   not claim it.
2. *Per storage bit.* The same token has ~6 nonzeros: ~3 bits per nonzero. Sparse associative writing (`Row`s with
   k-of-d keys, many entries superposed on shared hidden units — Willshaw 1969; Kanerva's SDM; Ramsauer et al. 2020)
   stores `P = ln2 · m·n/(k_in·k_out)` pairs in an m × n binary matrix: ~24k pairs at m = n = 2048, k = 11, i.e.
   ~0.47 bits per binary synapse at k ≈ log₂ n, falling to ~0.10 at k = 40. Against int8 gradient storage (0.25 per
   stored bit) that is a 2–4× factor, not a new regime. B4 measures; the cliff beyond P is the interference the
   consolidation price must see (B5).
3. *Per exposure.* A written store holds the whole fact after ONE observation: ~18 bits per exposure against
   18/1000. The only axis where the gap is 100–1000×; tabled beside the others so it is not mistaken for the answer
   to the per-parameter question.
4. *Hierarchy.* `R_h = K_flat / B_written`: the two-part price of the flat full-window table over the bits actually
   written after sleep, templates, words and consolidation. Already on disk: E24 LockPath 9,192 → 660 bits (13.9×);
   E3 31 → 8.7 per task (3.6×); E8 161 → 121 (1.33×); E12 1,190 → 8.9 (134×). Bits of flat knowledge per written
   parameter = `R_h × 0.14` (tokens) or `R_h × 0.47` (rows): **above 2 iff `R_h ≥ 14` (tokens) or `R_h ≥ 4.3`
   (rows)**. On Allen-Zhu & Li's random tuples `R_h = 1` by construction — the pre-registered control. Hierarchy
   stores shared structure ONCE; that, and only that, takes bits-of-knowledge per parameter past 2. *§22 adds a
   fifth axis: a designed codebook multiplies the dims a given number of features occupies, at a priced
   interference — capacity and legibility traded explicitly instead of implicitly.*

The cost side, also counted: memory tokens cost T² attention per pass (a 64×64 frame against 2M bank entries is
~2·10¹² MACs per pass: ~20 ms on an RTX 5090, ~200 s on this CPU — the bank's K/V is computed once through
`Attn.forward(cache=)`); rows cost d per row and interfere; consolidation is where the interference is paid.

### 21.7 The Store's three forms and the consolidation price

`Store` is the one table. Three tensor forms of one entry, all written by `ziplearn.write` from the evidence and
regenerated on demand:

- *tokens* — a row of the sequence (E28); read competitively by softmax; nearest-key default; costs d floats and a
  key per query;
- *bank* — the same row as a (key, value) pair in `Brain.banks[name]`, attended through the k/v cache rather than
  placed in the sequence: identical arithmetic, positions free, the GD-compatible form (§21.8);
- *rows* — a `Row` in the MLP: key row = the entry's one-hot key (or its k-sparse projection), bias = −(k_in − ½),
  value row = the entry's value; read by threshold; no nearest-key default; superposed when keys are sparse.

Consolidation (`Store.consolidate(n0, eps0, price)`, run by `ziplearn.sleep`): an entry moves from tokens/bank to
rows when its evidence is settled (n ≥ n0, exception rate ≤ eps0) AND the interference exceptions the row form
produces on the evidence — `full` replayed through the written rows, computable offline — cost fewer bits than the
token's storage saves. Young or contested entries stay tokens: complementary learning systems as a price. B5
measures. **What it does not price:** what the nearest-key default GAINS on unseen windows (E28's open point; E35's
collapse of every denoising block to the identity) — NOT DESIGNED, and B5 could therefore consolidate the neighbours
the kernel needs. The sleep pass on an overproduced field zeroes a `Gather` head when no entry's mask keeps its
offset.

### 21.8 Gradient-descent compatibility

A brain written by BrainBuilder must be pretrainable with gradient descent on new knowledge so the result is more or
less the same efficient, compressed, ACCESSIBLE knowledge ZipLearn would write (if less efficiently), as the natural
consequence of the structure; and a brain built by BrainBuilder + ZipLearn must be post-trainable normally (SFT, RL,
OPSD).

*Two parameter classes in every built brain* (`Brain.param_groups()`; a trainer sets the first's learning rate to 0
or small, and any adapter sits on top of it):

- **STRUCTURE** — the instruction set as written: every head's `qkv`/`proj` rows, the `Compare` and `halt` rows, the
  `Branch` queries, the `BoundaryOp` tensors, the codec's embedding rows, the `Readout` rows. Written once, frozen or
  slow.
- **STATE** — the stores' entries in bank or row form, the `Pool` value matrices `W`, the register tokens' initial
  contents (program tokens), the read-out's calibration table. ZipLearn writes these by counting; gradient descent
  writes them by backprop IN THE SAME FORMAT.

*The knowledge store is an explicit sparse addressable key-value memory.* The bank form is a product-key memory layer
(Lample et al. 2019) with our layout as the key space and a dense softmax reader at N ≤ 10⁵; Memory Layers at Scale
(Berges et al. 2024) train the same object by backprop with a top-k reader; ROME and MEMIT edit the same object
without gradients. The product-key candidate reader is the sparse reader at N > 10⁵ — its substrate hook (a top-k
inside `Attn`) is NOT DESIGNED; the dense reader is what runs today.

*The compiler's four obligations:*

1. **Temperature.** Write at the smallest M meeting the exactness criterion (`p* = 0.99`): softmax mass 0.99 leaves a
   gradient of order `1 − p*` on the logits, so gradients reach STATE through every head; E28's M = 30 saturates them
   to ~10⁻¹³. Tracr-compiled weights (Lindner et al. 2023) are the warning: written at saturation they are brittle
   and untrainable. `verify` reports, per head, the exactness AND the mean `|∂loss/∂logits|`; both must be nonzero.
2. **Headroom.** An explicit `Arch` wider or deeper than the layout leaves free residual dims, zero heads and zero
   MLP rows; the bank's `capacity` reserves rows. Synaptogenesis then pruning: gradient descent may claim them,
   ZipLearn's sleep zeroes what nothing reads. B0's invariance check says the free capacity does not change what the
   written circuits compute (CONFIRMED at `d_layout + 64`).
3. **Post-training hooks, written.** (a) `Readout(ACTION.action → controls)` is a softmax over the game's controls —
   a policy head, RL-ready; (b) the hard `halt_row` is the written form and a sigmoid over the same row's
   pre-activation is its PonderNet-differentiable form (one tensor, two read-outs); (c) the GOAL slot is a register
   token, so a goal is a prompt; (d) OPSD is the gradient form of the hindsight loop — the teacher is the same brain
   with the outcome register filled, the student the brain with it empty, and the loss is the KL between their
   read-outs on the student's own trajectories.
4. **A DECOMPILER.** `ziplearn.lift(brain) -> {store: Store}` reads GD-written STATE back into tables: for each bank
   or MLP row, decode by the layout (argmax per one-hot subspace → a window key; the value row → a count
   distribution), keep the row only if every key subspace is one-hot to a tolerance τ (reported), rebuild the `Store`
   with `stats` from the value's sharpness. Sleep then consolidates the lifted table like any other. Possible only
   because the store is explicit; a row that does not decode is reported as unliftable and the fraction is printed.
   *§24.5 states the general version of this — decompilation as an MDL problem — and why `lift` is the tractable
   special case.*

*Tests:* G1–G3 in §21.9.

### 21.8a Randomness — how much a built brain has, and where

The organising rule: **randomness is an input to the model, never something the model generates.** A deterministic
network cannot manufacture entropy; it must read entropy from a channel, as it reads a pixel. Three kinds:

1. **Structural — fixed at birth, seeded, then pruned.** (a) ADDRESSES: the keys of the Store — the codes for
   symbols, contexts, entries — are sparse random codes, k active of n (HTM's ~2%: 40 of 2048), neither designed nor
   learned: near-orthogonal, exponential capacity, a computable collision rate; the dentate gyrus and the fly's
   mushroom body do exactly this. A NEW thing gets a fresh random code — the TBT line's "a fresh random grid phase is
   the object's origin and identity" — which is how novelty is minted without collisions. The amount is the code's
   entropy, ~k·log₂(n/k) bits per address. *§22 makes this precise: a random sparse code is the cheapest member of
   the codebook family, and its coherence is the Welch bound plus slack.* (b) HEADROOM: the overproduced heads, rows
   and free dims are initialised SMALL AND RANDOM, as gradient descent initialises, because symmetry breaking is the
   one thing gradient descent cannot do without noise — identical weights receive identical gradients. The written
   structure needs none of this; the headroom needs all of it, or §21.8 fails at its first step. Build with ~2× the
   heads and rows the blueprint uses (biology overproduces roughly 1.5–2×) and let sleep prune. (c) SEEDS RECORDED: a
   blueprint plus a seed is a deterministic genome; different seeds are different individuals with one brain, and the
   variance across seeds measures what the blueprint does NOT determine — a research instrument.
2. **Computational — read at run time from a NOISE REGISTER.** One register is refilled from an external source each
   pass, and exactly three things read it: SELECTION (Thompson sampling where a posterior exists — the grid
   posteriors of the mixing research are posteriors; GCML's ε where none does; stochastic tie-breaking); the LOOP'S
   INITIAL STATE (Geiping et al. start from noise so the fixed point is path-independent; E35 generates from it); and
   EXPLORATION STEP LENGTHS (Lévy-distributed when nothing local is informative). How much: the posterior's own
   width — Thompson injects exactly the model's uncertainty and no more, so a sharp posterior makes the brain
   deterministic and a flat one makes it explore; where no posterior exists the scale is a temperature, PRICED in
   bits, never typed. Safety: GCML's homing term — even a bad noisy step is compensated because the next step again
   points at the goal — is why noise on selection is safe when a goal direction exists.
3. **Training — the trainer's, not the brain's.** Minibatch noise, dropout, Geiping's random iteration counts,
   DiffusionBlocks' noise levels, E35's corruption schedule. BrainBuilder only leaves room for them (the headroom).

*What has none:* the instruction set — heads, routes, boundary operator, halting threshold — is exact or it is a bug
(E28's 530/530 came from exactness). Blueprint entry: deterministic circuits; seeded random sparse addressing; seeded
small random headroom at ~2× use; one noise register wired to selection, loop initialisation and exploration, its
scale bound to uncertainty.

### 21.9 The blueprint research programme — experiments in order, each with a pass and a refute

**B0 — verification (run first; not an experiment). STATUS: PARTIAL, 2026-09-22** (`runs/brainbuilder/b0.json`).
Passing: compiled `induction.json` at `Arch(d=64, heads=2, layers=2, rope, max_len=128)` equals `e18.write`'s
state_dict tensor for tensor (29/29) and scores 1.000 with the derived M; compiled `gridworld.json` at r = 1,
`Arch.auto`, `onehot` reproduces `WrittenSim` bit-for-bit (all-known agreement [530, 530] seed 0 and [509, 509]
seed 1; ties 1,434/1,434 and 1,475/1,475; 900/900 and 867/867 plans identical) and identically at `d_layout + 64`;
compile 0.037 s. **The `rope2d` invariance cell: status UNRESOLVED, and the recorded failure is stale.** The
`b0.json` on disk records `CompileError: layer 1: 159 position-free pairs do not fit beside the position pairs of a
465-wide head (232 pairs)`, and v1.0 of §22.3 built a diagnosis on it. Checked 2026-09-27
(`research/expN_allocation.py`): the exact call that cell makes —
`compile(Blueprint.load("blueprints/gridworld.json"), gridworld_arch("rope2d"))` — **compiles**, at d = 320,
`n_head` [10, 1, 1, 4], `max_len` 350, as do r = 1 and r = 4 under both codecs. So the compile no longer fails; what
is NOT established is whether the compiled `rope2d` brain's rollouts match the `onehot` brain's, because the full
`b0_regressions` rerun did not finish — it ran 45 minutes against the 163.8 s the stale report records, which is
itself a datum: the variant that used to fail instantly now executes 900 rollouts through a much wider brain, and
that path is slow enough to belong in B8's cost budget rather than B0's. **B0's honest status is therefore: E18 and
E28 PASS as recorded; the `rope2d` cell is unrun, not failing.** Any other miss is a compiler bug, not a result.

**B1 — Generality: one brain, the games AND Latin, nothing changed** (CPU, ≤ 10 min). ONE blueprint
(`gridworld.json`, field r = 4, 80 gather heads, `rope2d`), ONE `Arch`, ONE compiled STRUCTURE state_dict —
byte-identical between the arms; the arms differ only in what the codec is given (a frame's cells with (row, col);
a text's characters with (0, position)) and which `Readout` the harness reads. Games arm: E28's protocol on LockPath
levels 0–1 with sleep. Text arm: the E33 corpus, 826,605 training characters, the first 20,000 of *De Bello Civili*
held out; ZipLearn writes the order-≤ 4 contexts as bank entries with wildcards for shorter orders; the bank size N
and the MAC count are printed before the run. Pass: written = planner ≥ 0.98 on all-known plans AND sleep zeroes 76
of the 80 heads, leaving E24's four neighbours, AND text ≤ 2.27 bits/char (xz online at that size 2.324; the blended
order-4 n-gram frozen 2.113 — the gap to it is reported as the softmax-vs-KT cost). Refute: games < 0.9, or text
> 2.324, or any head the masks kept zeroed. *Depends on B0's `rope2d` cell.*

**G1 — Gradient pretraining lands in the same store** (CPU, ≤ 5 min). N = 1,000 random (key → value) tuples (a
6-tuple over V = 16 → one of 16). Arm Z: ZipLearn writes them (one exposure each). Arm G: a fresh brain, empty bank
of 1,200 rows (keys small-random), STRUCTURE frozen, Adam on the `pred` read-out's cross-entropy for 100 epochs;
then `lift`. Measured: exact recall (Z, G); liftable fraction (G); bits per fact of the lifted table under
`table_price` against Z's; the gradient norm reaching STRUCTURE at `p* ∈ {0.9, 0.99, 0.999}`. Pass: G recall ≥ 0.9
and liftable ≥ 0.9 at `p* = 0.99`, lifted bits per fact within 2× of Z's. Refute: liftable < 0.5 (the store is not
the same object under GD) or G recall < 0.5 at every `p*` (the written structure is Tracr-brittle).

**G2 — Post-training does not break the built brain** (CPU, ≤ 10 min). ZipLearn builds on LockPath levels 0–1; then
(a) SFT on hindsight traces of level 2 and (b) REINFORCE on the action read-out with the level outcome as reward,
200 episodes, STRUCTURE frozen. Pass: written = planner on levels 0–1 stays ≥ 0.98 AND level 2 is solved in fewer
actions than the explorer alone. Refute: levels 0–1 below 0.93, or no gain on level 2.

**G3 — The round trip is a fixed point** (CPU, ≤ 3 min). Write → 20 Adam steps on the same transitions, STRUCTURE
frozen → `lift` → `write` again. Pass: ≥ 0.99 of entries identical and `table_price` within 1 bit. Refute: > 5% drift
or unliftable.

**B2 — The executive does E6 inside the block** (CPU, ≤ 2 min). E6's stream (shift 3, shift 5, affine 2x + 1, shift 3
again; V = 11; 40 pairs each; 20 streams) as (x, y) tokens: `surprise` gates the store append and the minting of a
TASK id; the TASK `Broadcast` selects the block every `Match` reads. `ContinualLayer.select` is deleted from the
path. Pass: retention 1.000 after every stretch, exactly 3 ids, the flag firing within 2 observations of each change.
Refute: retention < 0.95 or a 4th id on A's return.

**B3 — The planner is the loop** (CPU, ≤ 3 min). E9's environment (7 actions, 200 goals within 3 actions);
`count_inverse` fills W from k = 3 observations per action; the loop plans (GOAL broadcast → pool → quantise with
noise 0.1 → lookup advances → halt on `goal_met` or 8 passes), then the plan runs. Then E29's rooms (sides 12, 20,
40). Measured: goals reached; plan length vs the oracle; passes per real step against the room's side. Pass: ≥ 0.8 of
goals at k = 3 (E9's search: 1.000); refute: < 0.5. Passes-per-step growth is REPORTED, not claimed; relational goals
excluded by pre-registration (E17).

**B4 — Capacity on four axes** (CPU, ≤ 5 min). G1's tuples at N = 10³, 10⁴, 2·10⁴, 5·10⁴ written as (a) one-hot
tokens, (b) sparse `Row`s at d = 2048, k ∈ {11, 40}; measured: exact recall vs N; bits per parameter and per storage
bit at the largest N with recall ≥ 0.99; bits per exposure beside Allen-Zhu & Li's 2/1000. Then R_h recomputed on
E24, E3, E8, E12 and on the random set. Pass: (b) ≥ 0.3 bits per storage bit at k = 11 with recall ≥ 0.99 from one
exposure; R_h = 1.00 ± 0.02 on the random set; R_h > 4 on every structured set. Refute: (b) < 0.1, or recall < 0.9
below Willshaw's P, or R_h > 1.2 on the random set (the price leaks).

**B5 — Consolidation under the interference-aware price** (CPU, ≤ 5 min). E35's strict blocks 4 (28,811 entries) and
8 (30,629): `consolidate(n0=8, eps0=0.1)`; measured: entries moved, storage bits before/after, replayed false
positives, repair at t = 0.25/0.5/0.75 against E35 strict (0.946/0.880/0.800). Pass: ≥ 5× fewer storage bits with
repair within 0.01 at t = 0.5. Refute: repair drops > 0.05 (interference mispriced) or < 10% of entries qualify.

**B6 — The sequence axis** (CPU, ≤ 3 min). E28's rollout as Coconut thought positions — one appended position per
action through `BoundaryOp` + `forward_embedded` caches — must equal the depth-loop rollout on the same 300 plans;
the cost per step of both axes reported. Refute: any disagreement.

**B7 — The ablation ladder: which circuits general intelligence needs** (CPU, ≤ 15 min). `ablate` one circuit at a
time — the registers, `surprise` (writes ungated), noise = 0, `Branch` → plain sum, tokens → rows only, `nogo` — and
rerun B1, B2, B3. Pre-registered: removing the gate breaks B2 (ids multiply); noise = 0 lowers B3 on the rooms (no
escape from a utility tie); rows-only lowers B1's unknown-window agreement (E28: 246/300 toward the planner's
222/300); the route ablation changes nothing in B1 (one pass per action never reads earlier passes) and shows, if
anywhere, in B3's multi-pass imagination. Pass: the table exists with every cell filled; a failed prediction is
recorded as such.

**B8 — Compile cost at scale** (CPU, ≤ 5 min). `gridworld.json` at (H, W) = (8, 11), (32, 32), (64, 64) under
`onehot` and `rope2d`; banks of 121, 10⁴, 10⁶ entries; compile seconds and peak bytes. Pass: compile time linear in N
and ≤ 10 s at 10⁶; the `rope2d` gather's parameter count independent of H, W; `rope2d` = `onehot` on all-known plans
≥ 0.99. Refute: a per-element Python loop surviving (> 60 s at 10⁶).

### 21.10 Not designed, by name

The price of the nearest-key default's generalisation (E28's open point; E35's collapse; the leave-one-out term is
the candidate); what LEARNS the register contents — the programs the loop runs beyond simulation, repair and
one-step selection (§18's open question; OPEN-10's value when no plan exists); relational goals (E17); the sparse
top-k reader inside `Attn` for banks above 10⁵ entries; the KT escape as a written read-out; the pricing of the
blueprint's constants (`p*`, the noise gain, `max_passes`, `n0`, `eps0`) rather than typing them; the minting of a
new instruction (a new kind is a program, §19); a schedule for compositions that are not refinements (§20's second
level); confidence calibration inside the loop; whole-window `Match` under `rope2d` (B0's failing cell, §22.3).

*Removed from this list 2026-09-27:* superposition of subspaces — designed in §22 as a layout mode, not an
instruction; the similarity predicate — designed in §23 as `Relate`. Both are DESIGNED and NOT BUILT.

### 24.7 E38 — the crossover, on a corpus big enough to reach it (pre-registered 2026-09-27)

E36 measured the word-bigram's deficit against the character chain at word boundaries shrinking with data
(+0.644 / +0.574 / +0.425 / +0.259 at 100k / 289k / 769k / 2.73M characters) and the refit put the crossover at
**17–26M characters**. `corpora/latin_classical` is 35.0M characters, so the prediction is now testable rather than
extrapolated — which is the whole reason to have fetched it.

**Measured**, at training sizes 2.73M (the old corpus, as the tie-back), and as many larger sizes as memory allows
up to 35M: bits/char online and frozen; the chain's bits at word-initial positions; the counted word bigram's bits
at the same positions; word types, tokens per type, and boundary coverage. **Pass:** the bigram's deficit reaches
zero at or below 26M characters. **Refute:** still positive at 35M, i.e. the refit was optimistic too and the
crossover is genuinely out of reach for Latin. Either way the curve is the deliverable.

**RAN 2026-09-27 — REFUTED, and the apparatus rebuilt (`RESULTS.md`).** Deficits +0.646 / +0.559 / +0.470 / +0.390
at 2.73 / 5 / 10 / 20M characters; slope −0.296 per decade, which is E36's slope; extrapolated crossover 4 × 10⁸
characters. What moved is the INTERCEPT, and the cause is the split: E36 trained on 15 homogeneous books and held
out a 16th, E38 trains on 47 authors across a millennium and holds out Caesar. **The crossover is set by how alike
training and test are, not by volume** — E37's "stationarity is the discriminating variable", arriving again by a
different route. `ziplib/chain.py` now holds the vectorised chain (verified against `e34.Chain` to 3.6 × 10⁻¹⁵
bits; 35M characters in 449 s against the dict chain's 1012 s for 20M), and E39 adds the capacity budget, so text
is no longer memory-bound.

**The apparatus limit to watch, and it is real.** `e34.Chain` keeps nine Python dicts keyed by character tuples; at
2.73M characters the order-8 table already holds ~2.7M near-unique contexts. Extrapolating to 35M is ~30M contexts
at roughly 200 bytes each per order — several GB, past what this machine should be asked for. The run therefore
probes memory first and reports the largest size that fits, and a size that does not fit is reported as an
apparatus limit, never as a result. This is the first time the text line has been compute-bound rather than
data-bound, and the fix (packed integer keys, count cutoffs at the deep orders — standard PPM practice) is
NOT DESIGNED.

---

## 22. Legibility at scale — superposition with a codebook (DESIGNED 2026-09-27; the user's question)

**The problem this answers.** §0 says the safety case rests on auditability, and §21.6 says hierarchy is the only
way past 2 bits of knowledge per parameter. Those two pull against each other, and v0.9 stated the tension without
resolving it: what makes gradient-descent weights illegible is SUPERPOSITION — more features than dimensions, stored
in almost-orthogonal directions with interference (Elhage et al. 2022) — so a frontier-scale written brain looked
like a choice between keeping orthogonal channels and paying a large parameter multiple, or superposing and becoming
illegible on purpose. The user's question (2026-09-27) — *is there an interpretable way to do superposition, a clear
and unambiguous codebook?* — finds the third option, and it turns out to be the well-studied one.

**The answer in one sentence.** Superposition is illegible when the assignment of features to directions is CHOSEN
BY OPTIMISATION and recorded nowhere; it is legible when the assignment comes from a published codebook and is
written in the blueprint, because then the interference is a constant you compute once and the readout is exact
under a stated condition.

### 22.1 The ladder — four rungs, and we have two before legibility is spent

| rung | assignment | interference | legible? | capacity |
|---|---|---|---|---|
| 0. **disjoint** (today) | one subspace per feature, `Layout.allocate` | none | fully | dims = features; wasteful |
| 1. **colour** | features that are never CO-LIVE share dims exactly | **none, by a disjointness certificate** | fully | dims = the chromatic number of the co-activation graph |
| 2. **frame** | co-live features share dims through a designed codebook | bounded a priori by the coherence μ | fully — the codebook and the assignment are data | m ≫ d features, exact readout under a sparsity condition |
| 3. **learned** (gradient descent) | chosen by optimisation | data-dependent, unbounded | no | the most, and the reason nobody can read it |

Rungs 1 and 2 are both superposition in the sense that matters — the same dimensions carry more than one feature —
and both are fully auditable. **We have not yet used either.** That is the correction to v0.9's worry: legibility is
not spent until rung 3, and the design never needs rung 3.

### 22.2 Rung 2, the mathematics — this is compressed sensing, and the conditions are theorems

A codebook is m unit vectors (atoms) in d dimensions, m > d. Its **coherence** μ is the largest inner product
between two distinct atoms. Three facts make it a design object rather than a hope:

1. **The Welch bound.** μ ≥ √((m − d)/(d(m − 1))) for any m atoms in d dimensions. Equality is achieved by
   *equiangular tight frames* (Grassmannian frames) — every pair at the same, minimal angle. The floor on
   interference is known before anything is built.
2. **Uniqueness and exact recovery.** If at most k atoms are active at once and k < ½(1 + 1/μ), the sparse
   representation is UNIQUE and is recovered exactly by matching pursuit or ℓ1 (Donoho & Elad; Tropp). "Unambiguous
   codebook" is not a wish; it is a stated inequality relating the sparsity the data has to the coherence the
   codebook has.
3. **Explicit constructions exist** and are not exotic: maximum-length sequences, Gold codes, Kerdock codes, chirp
   (Alltop) sequences, mutually unbiased bases, difference-set (Steiner) ETFs. CDMA is the existence proof at
   industrial scale — many users' signals superposed in one band, each decoded unambiguously by its published
   spreading code, for decades. Engineered superposition is old, understood, and nothing like a black box.

**Measured (S2, 2026-09-27, `research/expM_codebook.py`; `RESULTS.md`).** All three hold, with the constants:
coherence comes out 4–6× the Welch bound for gaussian, k-sparse and Hadamard codebooks at d = 128 (random and
near-tight frames are far from optimal); the `k < ½(1 + 1/μ)` guarantee predicts 1.7–2.2 where the measured k at
≥ 99% exact readout is 3–6, so it is **conservative by about 2×** with the right shape; and the usable packing obeys
**m/d · k ≈ 32** (m/d = 16 at k = 2, 8 at k = 4, 4 at k = 8), collapsing sharply past it. That product is the number
a blueprint needs: it says how sparse the activity must be to buy a given packing.

The neighbouring literatures we already cite are the same object: vector-symbolic architectures / hyperdimensional
computing (Plate's HRRs, Kanerva, Gayler) superpose key-value bindings in one vector and unbind any one with bounded
noise, with published capacity formulas; Willshaw/SDM (§21.6 axis 2) is the binary case; Kanerva's sparse random
codes (§21.8a item 1a) are the cheapest member of the family — a random code sits near the Welch bound with slack,
which is exactly why "a fresh random grid phase is the object's identity" works.

### 22.3 Rung 1 — register allocation IS graph colouring (and what the `rope2d` cell actually was)

**The observation that makes rung 1 free.** Two features that are never active at the same time can occupy the SAME
dimensions with *zero* interference — not "almost orthogonal", literally disjoint in time. The number of dimensions
needed is the chromatic number of the **co-activation graph** (a vertex per feature, an edge when two are ever live
together), and the assignment is recorded in the blueprint as a colouring. It is exact, it is auditable, and the
co-activation graph is a COUNTED object — ZipLearn already counts what co-occurs.

**This is a compiler's register allocation, and it has been graph colouring since Chaitin (1981):** variables are
features, registers are channels, two variables interfere when simultaneously live, and when the graph will not
colour with the registers available you SPILL. The fix, in order of effort: (1) build the interference graph over
the channel needs of the circuits placed in a head, from their declared `reads`/`writes` and the loop's
`keep`/`clear` — which is exactly a liveness analysis; (2) colour it; (3) where it still will not colour, SPILL in
one of the two ways a transformer allows — split the match across two heads whose scores add in the residual (the
standard compiler spill, paid in heads), or move to rung 2 and pay interference in bits instead of dims.

**Correction, 2026-09-27 (`research/expN_allocation.py`).** v1.0 of this section claimed B0's failing `rope2d` cell
was this problem and that `Layout.channels` was "first-fit and first-fit is the wrong algorithm". Checked against
the compiler's own arithmetic, that was wrong on both counts and the example is withdrawn:

- `compile(Blueprint.load("blueprints/gridworld.json"), gridworld_arch("rope2d"))` — the exact call the invariance
  cell makes — **succeeds today**, at d = 320, `n_head` [10, 1, 1, 4], `max_len` 350. The failure recorded in
  `runs/brainbuilder/b0.json` is from an earlier state of the compiler; B0's status line (§21.9) is re-derived from
  a rerun, not from that file.
- The constraint that produced it is not first-fit but one line of `brainbuilder.py`:
  `room = P - (P // 2 + col_pairs) if col_pairs else P`. Under `rope2d` the first half of a head's rotary pairs
  rotates by the row coordinate and the second by the column, so when ANY head in the layer uses a coordinate the
  whole row half is reserved — a `Match` head needing 159 position-free content pairs then has at most `P/2` and
  fails however wide the head is. When the layer holds no `Gather` (`col_pairs == 0`) the branch gives it the whole
  head and it fits. So it IS an over-reservation and liveness IS the cure, but the live quantity is the coordinate
  AXIS per layer, not a general feature-interference graph, and the existing code already gets the common case
  right by accident.
- The 159 is real and worth keeping in view: a whole-window `Match` at r = 1 compares colour(17) + 8 × 17 + action(4)
  + two class flags = 159 content dims, and at r = 4 it is 1,381. Those are all live at the query simultaneously, so
  **rung 1 cannot help a `Match` query at all** — they genuinely interfere. Rung 1's saving is between circuits in
  different layers, not within one comparison.

What survives: rung 1 as a design (it is exact, auditable and counted), the Chaitin framing, and the spill options.
What does not: the claim that it fixes `rope2d`, and the claim that `rope2d` is broken.

### 22.4 What superposes and what does not — the design constraint

A bundle (a sum of atoms) is transformed correctly by any LINEAR map: the map applies to each superposed item at
once. So `Gather` (a coordinate permutation), `Broadcast` and the value/projection halves of `Match` and `Pool`
superpose for free. **The nonlinear instructions do not.** `Match`'s softmax over m superposed keys does not return
"the match for each of them"; `Compare`'s GELU threshold and `Quantise`'s argmax read a single winner. This gives a
hard rule for the blueprint:

> Features may share channels wherever only linear instructions read them. A subspace that a `Match` query, a
> `Compare` or a `Quantise` reads must be separated first — which is what `Quantise` is FOR, and why the boundary
> operator sits where it does.

That rule is also the reason rung 2 does not dissolve the block into mush: superposition lives in the STORAGE and
the linear transport, and the loop's boundary operator is the point at which the state is re-separated into one-hot
codes each pass. E28's `quantise` already does this; it was written for numerical hygiene and turns out to be the
structural precondition for superposition.

### 22.5 The price — the legibility tax becomes computable

Interference is bits. For a codebook at coherence μ with k active atoms, the expected extra codelength from misreads
is computable from the same two-part code everything else uses: the bits SAVED by packing m features into d channels
minus the bits LOST to the exceptions interference causes on the evidence (`Store.consolidate`'s replay, §21.7,
generalised from rows to any shared code). So **how superposed should a layer be** is not a hyperparameter — it is
`argmin over the allocation mode of (storage bits + interference bits)`, the rate–distortion trade of §8 applied to
the layout.

**Correction (S2).** The argmin is well defined but it is NOT self-contained: it depends on the
reads-per-stored-entry ratio of the workload, which is an application property, not a codebook property. Measured
with 2,000 stored entries against 10,000 reads the storage term dominates and the price picks maximum packing even
where readout is 0.000 — absurd. Until the workload's read/write ratio is supplied, the operational criterion is the
largest packing at ≥ 99% readout (m/d · k ≈ 32 above), and the price is a tie-breaker rather than the decision.

Two consequences worth stating plainly:

- The **legibility tax** (§0) stops being a vague worry and becomes a number per layer: the bits rung 0 or 1 costs
  over the rung-2 optimum, and the bits rung 2 costs over rung 3's (unmeasurable, but boundable by a trained model
  of the same width).
- §21.6 gains a fifth capacity axis. Rungs 1 and 2 multiply the features a given d holds, at a priced cost, without
  leaving the audit trail — so "is 2 bits per parameter gradient descent's limit or a real one?" splits into "2 bits
  is the dense incompressible ceiling" (unchanged) and "hierarchy (R_h) times codebook packing is what beats it".

### 22.6 Pre-registered

**S1 — Rung 1 fixes the allocator** (CPU, ≤ 5 min). Implement liveness + colouring in `Layout.channels`; recompile
`gridworld.json` at r = 1 and r = 4 under `onehot` and `rope2d`. Pass: B0's `rope2d` invariance cell compiles and
gives rollouts identical to `onehot` on all-known plans (≥ 0.99), and the colouring is printed as a table (feature →
channels, with the disjointness certificate for every shared channel). Refute: the graph does not colour at r = 1
even with liveness, i.e. the interference is real and rung 1 does not apply here.

**S2 — Rung 2 on the Store, priced** (CPU, ≤ 10 min). G1's 1,000 random tuples and E35's block-4 table (28,811
entries) stored as (a) disjoint one-hot keys, (b) a Gold-code or random-sparse codebook at m/d ∈ {2, 4, 8, 16} with k
active. Measured: exact recall vs m/d; the empirical coherence against the Welch bound; the measured misread rate
against the k < ½(1 + 1/μ) prediction; storage bits and interference bits under §22.5's price, and which mode the
price picks. Pass: recall ≥ 0.99 at m/d = 4 with the misread rate within 2× of the bound's prediction, and the price
picking the mode with the best measured total. Refute: recall < 0.9 at m/d = 2 (the theory does not transfer to this
key distribution), or the price picking a mode that measurably loses.

---

## 23. The semantic landscape without gradients (DESIGNED 2026-09-27; the user's question)

**The question.** Transformers know which words are related because their embedding space places related words near
each other. Is there a non-gradient way to give a model its semantic landscape? **Yes, and it is one of the
best-established gradient-free results in the field** — but the version we need is the asymmetric one, and that
version is what §21.3's new `Relate` instruction is.

### 23.1 What is known: a static embedding is a counted second-moment object

- **LSA/LSI** (Deerwester et al. 1990): the SVD of a term–document matrix. The original, and it works.
- **Levy & Goldberg (2014)**: word2vec's skip-gram-with-negative-sampling *implicitly factorises a shifted PMI
  matrix*. So `SVD(shifted PPMI)` ≈ word2vec: count co-occurrences in a window, form PMI, take a truncated SVD. No
  gradient, closed form.
- **Hellinger PCA** (Lebret & Collobert 2014): PCA on the square root of co-occurrence probabilities, presented
  explicitly as a gradient-free alternative and competitive on word similarity.
- **GloVe** (Pennington et al. 2014) is weighted least squares on log co-occurrence counts — it uses AdaGrad, but the
  objective is a matrix factorisation solvable by alternating least squares.
- **Random indexing** (Kanerva; Sahlgren): accumulate random vectors of context words. Pure counting, incremental
  and online, Johnson–Lindenstrauss guarantees — the cheapest member, and the one that matches §21.8a's addressing.
- **Brown clustering** (Brown et al. 1992): hierarchical hard clustering by bigram mutual information. Gradient-free,
  and it produces a HIERARCHY — which is the object §21.6 axis 4 prices as R_h.
- **Eigenwords / two-step CCA** (Dhillon, Foster & Ungar): provably consistent spectral embeddings, and — the
  important one for us — CCA produces TWO different projections, which is an asymmetric bilinear form.

None of this needs backpropagation, and all of it is a second moment of counts, which is the object `Store` already
accumulates.

### 23.2 The asymmetric version — and why it is one instruction

Word similarity is symmetric; attention is not. `softmax(qᵀk)` asks *which earlier token should THIS token attend
to*, which is not *which word is similar to this word*. The right object is the **cross-covariance** between the
features of the querying position and the features of the attended position, estimated by counting (which pairs
(i, j) the data makes relevant — for a language model, the empirical distribution of (token at i, token at j) over
attended pairs; for the games, (cell, neighbour) pairs at a transition). Write that counted matrix `Σ_qk` and take
its truncated SVD `Σ_qk ≈ U S Vᵀ`. Then

> `A = S^½ Uᵀ` and `B = S^½ Vᵀ` give `AᵀB = Σ_qk`: **the query and key projections of an attention head, written
> directly from counts.**

That is the `Relate(q, k, metric, v, dst, key_class, null)` instruction of §21.3 item 9. It is one instruction and
not two because it REDUCES to `Match` when `Σ_qk` is the identity on a one-hot subspace — equality is the special
case of similarity where the metric is diagonal. The rank kept is chosen by the same price as everything else:
`precision_bits` per retained singular value against the exceptions the truncation causes (§8 change 2).

Two things this buys beyond word similarity: it is the missing *soft match* of §24.3, and it is the mechanism by
which a written model could share statistical strength between contexts that are not prefix-related (§24.4) — the
Latin morphology problem.

### 23.3 Where it stops — stated, not glossed

1. **Static, not contextual.** These methods give one vector per type. A transformer's power at long range is
   CONTEXTUAL embeddings. Our position is better than it looks: contextuality can come from COMPOSITION (a written
   `Gather`/`Relate` over static codes) rather than from learning the contextual vectors themselves. Whether that is
   enough is exactly S3's question, and it is not settled by the static-embedding literature.
2. **Rare keys.** A counted second moment is bad where counts are thin. This is the one place our apparatus is
   ahead: the KT/prequential price (`ziplib.price`) says *how much to trust each entry in bits*, so the truncation
   rank and the shrinkage are priced rather than tuned.
3. **The nested-expert warning.** Round 2 found that closed forms on counted second moments match the grid posterior
   for DIVERSE experts and fail for NESTED ones (§20). A metric over nested contexts may hit the same wall; S3 must
   check it rather than assume the analogy fails to apply.
4. **Polysemy is superposition.** One vector per type cannot hold two senses; the honest answer is §22 — senses are
   separate features sharing channels, and the codebook is how they are told apart.

### 23.4 One counted matrix, two jobs

Worth recording because it is a coincidence with teeth: **PMI is the same object as the mixing exponent.** PMI is
`log p(x, y) / (p(x) p(y))`, and Allard et al.'s eq. 17 (round 1, `notes/gradient_free_mixing_and_features.md`)
makes a pooling exponent a conditional-to-marginal log-likelihood ratio — 1 under conditional independence, 0 when
one expert is a function of the others. The same counted matrix therefore serves as the semantic metric (§23.2) and
as the pooling exponents (§24.2). If a single counted second moment does both jobs, the "two irreducible gradients"
of §20 are one object seen twice; if it does not, S3 and D1 will say which half fails.

### 23.5 Pre-registered

**S3 — The counted metric on Latin — RUN 2026-09-27, REFUTED ON LATIN FOR A DATA REASON (E36, `RESULTS.md`).**
The metric is unusable on this corpus and the number says why: **5.2 tokens per word type** at the full 2.7M
characters (69,629 types), where distributional semantics needs 10²–10⁴. PPMI + rank-64 SVD gives neighbours that
share a stem 0.7% of the time against a 0.065% baseline — a 10× enrichment that is absolutely noise — and as a
predictor of a word's first character the low-rank Σ_qk costs 6.05 bits at rank 8, rising to 8.73 at rank 64, where
the character chain costs 4.52. A PPMI reconstruction is not a density estimator, and no rank fixes that.
**Re-aimed, not abandoned:** the construction needs a domain where tokens per type is large, and the games are that
domain — 17 colours, 4 actions and thousands of transitions per type against Latin's 5.2. S3 is re-registered on
frames, with E28's window keys as the query features and the transition's outcome as the key features. The original
text follows.

*Original pre-registration* (CPU, ≤ 15 min). On the E33 split: (a) build the character/word co-occurrence
counts over the training text; (b) form PPMI and take truncated SVDs at rank 8, 16, 32, 64, with the rank chosen by
`precision_bits` against exceptions; (c) use the resulting `Σ_qk` as a `Relate` metric for a soft-match context
expert — "the character that followed the most SIMILAR earlier context", against E34's match model, which uses
literal 6-gram equality and scored 2.88 solo. Measured: solo bits/char of the soft-match expert against the literal
one; the mixture's bits/char with it added to E34's eighteen; and, as the diagnostic, which context pairs the metric
rates as similar (are they morphological variants?). Pass: the soft-match expert beats the literal match model solo
by ≥ 0.1 bits/char AND the top-rated pairs are recognisably stem-sharing. Refute: no better than literal match
(similarity buys nothing here), or the top pairs are frequency artefacts.

---

## 24. The gap to a neural language model, decomposed (2026-09-27)

### 24.1 The corrected target

**And why text cannot escape it the way the games do (E44).** On the grid games the sleep pass keeps about three
cells out of forty-nine, and that subset selection removes the evidence collapse entirely — LockPath holds 185
observations per key at radius 3 where the unfactored window has 3.7. On text E33 found the same pass is a
**no-op**: the prequential price keeps all eight context positions. That is the difference between the two lines
in one number — **grid dynamics are SPARSE in relevance and factorize; language is DENSE and does not** — so the
collapse is a factorization problem in one case and a real information limit in the other.

**Why the floor is 1.8, measured (E40, 2026-09-27).** The chain is CONTEXT-bound, not data-bound: at 34M
characters it matches its full eight-character context at 91.7% of positions, with a median of 40 prior
observations, and still pays 1.78 bits there — converging (1.85 → 1.82 → 1.78 over 12.5× data) to
H(next character | previous 8) for Latin, just under 1.75. And no deeper window rescues it, because the evidence
disappears exactly where the information would start: coverage and median evidence fall 99.9%/4,775 at order 4 to
91.7%/40 at order 8, 46.5%/3 at order 12 and 12.3%/2 at order 16. **There is no depth at which a counting model
has both the context and the evidence** — the context space grows as V^R while the data grows linearly. So the
remaining 0.6–0.9 bits are not reachable by more Latin, a bigger table or a longer window; they need something
that GENERALISES across contexts instead of counting them.

1.0 bits/character is a fact about corpora of enwik8's scale (10⁸ characters). On 2.7M characters of Latin a
well-tuned neural language model would land nearer **1.3–1.5** — character LSTMs on Penn Treebank (~5M characters)
sit around 1.2–1.4, and Latin is smaller and less regular. So:

> **The legibility tax on this corpus is ≈ 0.4 bits/character, not ≈ 0.8.** (E33: 1.873 frozen, 1.820 online.)

**But the tax is a PROFILE, not a number (E37, 2026-09-27).** Measured or already on record, the constructive route
is AHEAD on four axes and behind on one:

| axis | tax | evidence |
|---|---|---|
| adaptation under distribution shift | **−6.04 bits/char** | E37: a Latin-trained chain on Middle High German, frozen 8.394 vs online 2.355; the gain still growing at the end of the stream |
| continual learning | −(below chance → 1.000) | E6 100% retention, 3 blocks, vs E6b forgetting to below chance |
| exact composition, held out | −(0/8 → 8/8) | E18 and E28 at 100% where the trained model solves 0/8 |
| storage density on structured content | −(R_h 13.9 to 134) | E24 9,192 → 660 bits; E12 1,190 → 8.9; 1.00 by construction on incompressible tuples |
| **bits/char on stationary bulk text** | **+0.4** | E33/E36, the number above |
| adaptation in-distribution | −0.05, halving per 3× data | E37/expO: +0.339 / +0.189 / +0.104 / +0.050 at 100k → 2.73M |

The discriminating variable is STATIONARITY, not the model class. A homogeneous held-out book drawn from the
training distribution is the single condition under which a frozen model is not penalised, and it is the condition
no deployed model enjoys. So the honest scoreboard names its axis: the number this project tracks is the tax **on
the stationary-text axis**, and the others are already won. The caveat that keeps this honest: every negative figure
above is against a FROZEN counted model or an earlier transformer of ours, not against a gradient-trained model run
under the same shift — that comparison needs the gradient arm, which is off.

And the compression literature's ladder on enwik8 says where that 0.4 is spent. Approximately: PPM ≈ 1.9,
PAQ-class context mixing ≈ 1.2, a large transformer ≈ 0.95. **The largest single step is PPM → context mixing, not
context mixing → neural.** We are on the PPM rung (E33 measured exactly that: "the discrete machinery on text is
PPM"). The mixer is the bigger half of the gap, and it is 20-year-old, fully documented technology — not a mystery
about learned representations.

### 24.2 A transformer's output layer IS a context mixer — two derivations, one answer

Expand a transformer by paths: the direct embed → unembed route is a bigram table; each attention head adds a
context-selected contribution; every component writes into the residual stream and the unembedding SUMS those
contributions into logits. **Summing log-probabilities is multiplying probabilities with learned exponents.** So a
transformer's output layer is structurally a geometric pool over a library of context experts: attention chooses the
context, the output matrix's scale is the exponent.

That is the same conclusion E34 reached by measurement — a linear (Bayesian) mixture can only CHOOSE among experts
(1.820 vs the table's 1.823), a geometric mean with Bayesian exponents gains 0.02, the raw product over-sharpens
(7.7), and what PAQ learns is the EXPONENTS. **We have the library and lack the pooling.** Two independent
derivations landing on one missing piece is the strongest evidence in the line about where to spend effort.

Round 2 already tested the gradient-free route to those exponents and it came back negative for our case: the
closed-form LDA solve on counted second moments ties the grid posterior for DIVERSE experts (−0.079 vs −0.077) and
fails for NESTED ones (exact exponent ≈ 0 → prune instead), and text experts are maximally nested. **§23.4 is the
one remaining gradient-free candidate** — if PMI and the pooling exponent are the same counted object, the nested
case may be an artefact of estimating the exponent from the wrong second moment.

### 24.3 Long context, decomposed

Not one capability, and the pieces have very different prospects:

1. **Retrieval by literal match — DONE.** E18 wrote an induction head exactly; E34 ran it as a context expert at
   2.88 bits/char solo. The compression literature's own answer to long range is the same thing (PAQ's match model,
   pure counting, one of its strongest components).
2. **Retrieval by soft match — the §23 route.** Our `Match` does equality on a written subspace; attention does
   learned bilinear similarity. That is the gap, and `Relate` is the proposed instruction. Unproven, not blocked —
   and, on Latin, not testable (§23.5: 5.2 tokens per type).

   **Measured 2026-09-27 (E36): the constraint is not the window.** Where the missing information sits is now a
   number. Word-initial characters are 13.4% of positions and 32.1% of the code length, at 4.518 bits against 1.482
   for word-interior characters — the whole gap is at the boundary, where character context has run out. And
   lengthening the context does nothing: R = 8 / 12 / 16 / 24 gives 1.8901 / 1.8902 / 1.8902 / 1.8903 bits/char,
   with the boundary cost getting slightly WORSE (4.518 → 4.634). So explanation (i) — "the model simply runs out of
   window" — is refuted, and what is needed at a boundary is information of a different kind, which is exactly the
   user's point that static-vs-contextual is the hurdle.
3. **Accumulated state that is not a copy of anything** — topic, register, what has been established — is the real
   wall, and the reason is precise: **there is no local target.** For induction the target is defined (copy what
   followed); for a summary nothing says what the intermediate should be, which is exactly the credit-assignment
   problem §20 escaped for diffusion by letting the noise schedule define the intermediates. MDL gives a well-posed
   objective (the state is whatever minimises the codelength of the rest) but optimising it is a search — §19's
   library problem again. So two-thirds of long context is writable or plausibly countable, and one-third reduces to
   the one open problem we already have.

### 24.4 The Latin morphology hypothesis — REFUTED 2026-09-27, and what replaced it

*The hypothesis was:* backoff shares statistical strength only along the SUFFIX chain, so on a heavily inflected
language our tables treat `amabat`, `amabant` and `amabamus` as unrelated contexts and pay for the stem every time;
the missing primitive would then be a factored (stem, suffix) context.

**It is wrong, and the error is elementary (E36, `research/expL_where_the_bits_go.py`).** The stem IS the left
context of the suffix, so a backoff chain over the last 8 characters already shares it. Measured on suffix
characters of words never seen in training: 1.786 bits when the 4-character stem was seen with other forms against
2.187 when it was not. The sharing the hypothesis said we lacked is worth 0.40 bits and we already have it.

**What replaced it.** The surplus is not inside words, it is AT THE BOUNDARY: word-initial characters are 13.4% of
positions and 32.1% of the code length (4.518 bits each, against 1.482 for word-interior). And the obvious counted
fix does not work either — a KT-smoothed p(first character | previous word) costs 5.043 bits, WORSE than the chain's
4.518, because 69,629 word types over 2.7M characters is too sparse to count. Its deficit does close with data
(+0.644 / +0.574 / +0.425 / +0.259 at 100k / 289k / 769k / 2.73M). The semantic landscape is not a different
mechanism from counting; it is the same counting at a data scale this corpus does not reach.

**Crossover, refitted 2026-09-27 (E36 addendum).** The figure of "near 10⁸ characters" was a bad extrapolation —
it averaged the deltas instead of fitting them, and the deficit is closing FASTER as data grows (−0.070, −0.149,
−0.166 per tripling). A least-squares fit of the deficit against log₁₀(characters) gives a slope of −0.28 to −0.32
per decade and a crossover at **10^7.2–10^7.4, i.e. 17–26 million characters**, centred near 20M. That is five times
sooner than stated and, unlike 10⁸, it is REACHABLE: the whole surviving classical Latin corpus is about 7.36M words
(PHI Latin Texts) ≈ 45–50M characters, and `cltk/lat_text_latin_library` is ~110 MB of Latin of all periods. So the
word-level context overtaking the character chain at boundaries is a testable prediction, not a thought experiment.
What a bigger corpus does NOT fix is the metric: Heaps' law fitted on our own four sizes gives types ∝ n^0.70, so
tokens per word type reaches only ~12 at 27M characters and ~18 at 108M, still an order of magnitude below what
distributional semantics needs. Latin's inflection is what multiplies the type count, which makes MDL morphology
induction (a two-part code over stems + suffixes — Goldsmith's Linguistica 2001, Creutz & Lagus's Morfessor; our own
principle applied to word structure, gradient-free and not a hand-written lexicon) the lever that a bigger download
is not. NOT DESIGNED.

### 24.5 Decompilation as an MDL problem

The general form of §21.8's `lift`. Reading a program out of weights is: find the blueprint B minimising
**`price(B) + price(W − compile(B))`**, with `ziplib/price.py` supplying both terms. Two things make this
better-posed than post-hoc interpretability:

- **A verification oracle.** Interpretability's core weakness is that "did I find the circuit?" has no crisp test.
  Here `compile(B)` either equals W up to a residual PAID FOR IN BITS, or it does not. The compiler's exactness
  turns a hermeneutic question into an arithmetic one.
- **Unlimited exactly-labelled (program, weights) pairs** at 0.037 s per compile — ground truth nobody else has in
  exact form. *The Tracr critique applies to us too:* compiled models are unrealistically clean, and a decompiler
  validated only on compiled weights has proved nothing about gradient-descent weights.

Two hard parts, both real. **The symmetry quotient:** `compile(B)` is defined only up to head permutation, channel
permutation, rotation within a subspace the rest of the network respects, and scale that RMSNorm absorbs — so
matching W to any B requires solving a subspace-matching problem first. That is the same binding failure as
`project_factorisation_not_application` (given the group element, 0.733; given demonstrations, 0.000), in new
clothes. **The search space** is §19's library problem. `lift` is the tractable special case precisely because it
skips both: the store is explicit and the layout is ours, so there is no matching to do and no search — which is why
it is the piece worth building first.

**The catch-22 to state before anyone builds a general decompiler:** a model small enough to analyse, trained on
2.7M characters, will not exhibit the gap you want to explain; a model large enough to exhibit it is too superposed
to decompile. The window where both hold is narrow, and the two experiments below are designed to need neither end
of it.

### 24.6 Pre-registered

**D1 — Attribute the gap per symbol.** *The gradient-free half RAN 2026-09-27 (E36): our own code length,
attributed. Pass criterion "the top 10% of positions carry ≥ 40% of the total" — measured 33.8%, just below, but the
structure is unambiguous and nameable, which was the substance of the test: 13.4% of positions carry 32.1% of the
bits and they are all the same thing (the first character of a word). The half that still needs a trained model is
the DIFFERENCE `bits_ours − bits_gd`; what ran is `bits_ours` alone against a structural partition.* Original
pre-registration (CPU, ≤ 10 min; needs one small trained model, so it waits on an explicit instruction to touch the
gradient arm). Bits are additive, unlike accuracy. For every character of the E33 held-out
book compute `bits_ours(c) − bits_gd(c)`, sort, and characterise the top few thousand by position-in-word,
preceding-context length, word frequency and whether the word's stem occurs elsewhere in training. Pass: the surplus
is CONCENTRATED (the top 10% of positions carry ≥ 40% of the total gap) and the concentration has a nameable
structure. Refute: the surplus is uniform — in which case the gap is a smoothing constant, not a missing mechanism,
and §24.4's hypothesis is dead.

**D2 — The projection test: does gradient descent stay inside the channel plan?** (CPU, ≤ 10 min; the cheap version
of "is a BrainBuilder brain decompilable after GD", and the one I would run first.) Compile a brain, train it with
gradient descent (G1's setup, STRUCTURE unfrozen for this test only), then decompose the weight change ΔW into the
part lying INSIDE the written channel plan (the subspaces `Layout` allocated, per head) and the part OUTSIDE it,
reporting both as a fraction of ‖ΔW‖ and the distance travelled ‖ΔW‖/‖W₀‖ beside them. Pass: the decomposition is
computed and reported for every head. Reading: if the improvement lives INSIDE the plan, the instruction set is
expressive enough and ZipLearn is failing to find good values — a search problem; if it lives OUTSIDE, **a primitive
is missing and the projection names which channels it wants**. Refute (of the method, not the design): ‖ΔW‖/‖W₀‖
large enough that the original plan is no longer the right basis — itself the informative outcome, and the trigger
for D3.

**D3 — Decompilation on our own output, as the honest baseline** (CPU, ≤ 5 min). Take a compiled brain, apply a
random head permutation and a random rotation within one subspace, then recover B by minimising
`price(B) + price(W − compile(B))` over the blueprint's own circuit set. Pass: the original blueprint is recovered
up to the symmetry, with residual price ≈ 0. Refute: not recovered — in which case the symmetry quotient, not the
search, is the binding constraint, and that is worth knowing before any general decompiler is attempted.

---

## 25. The order of work (set 2026-09-27)

Everything below is CPU, gradient-free, and runs against code and corpora that exist. The ordering is by what each
one would settle, not by effort.

**REVISED 2026-09-27 after the course correction of §18.1; the user chose (2) then (1).**
0. **E41 — in-context learning in a written block (§18.1).** The line's actual subject: can a written interpreter
   acquire a world model from CONTEXT alone, with the weights untouched? Nothing new needs building — E28's block
   and the counted rules exist. **First.**
0a0. **E46 — a Brainfuck interpreter as a written looped block (§18.3).** The substrate for everything after it,
   and the cheapest decisive test of whether §21.3's eight instructions are universal. No training.
0a. **E45 — does depth buy what width cannot? (§18.2).** CPU, gradient-free, and it tests the two-axis claim on
   the case E43 left open (Toggle) while fixing E42/E43's sample-size death with generated dynamics. Before E32.
0b. **E32 — "look for a name" on a continuous representation (§19).** The crux of the continuous thesis, shelved
   since 2026-09-21. Needs one small trained net AS THE SUBJECT, not as a comparison arm. After E41.

The text-line items below are kept but demoted; E40 is the reason (counting converges to H(c | 8) and cannot get
the partition, so none of them can reach the target by themselves).

1. **S4 — level-3 naming on morphology (§19.1).** The only experiment that tests the operation the whole library
   problem turns on: naming a pattern with holes. Cheap (it works on a word-type inventory, not a character
   stream), has controls (no segmentation; BPE = level 2 only), and a clean refutation (if signatures do not beat
   BPE, the holes buy nothing and level 2 is the whole story). **First.**
2. **E38 — the crossover on 35M characters (§24.7).** Tests a prediction this line made and can now reach, and
   produces the data-efficiency curve the project has wanted since E33. Compute-bound, so it goes second and
   reports its own apparatus limit.
3. **S3 on frames (§23.5).** The counted metric where tokens per type is thousands rather than 5.2. Needs the games
   harness rather than the text one; no new machinery.
4. **ziplearn.py + the cutover (§21.2).** The unbuilt second script and the deletions. Not research, but everything
   in §21.9 is blocked behind it.
5. **B-programme (§21.9)** in its own order, once (4) exists.

Not scheduled, and why: **S1** (the liveness allocator) lost its motivating example when `rope2d` turned out to
compile (§22.3) — it is a real design but nothing is blocked on it; **D1's second half, D2, D3, G1–G3** all need the
gradient arm, which is off by standing instruction; **the anti-unification operator itself** (§19.1) is NOT DESIGNED
and S4 is the experiment that would tell us what it must do.
