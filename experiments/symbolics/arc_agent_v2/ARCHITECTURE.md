# Arc agent v2: a brain of columns whose layers are the requirements for learning a general function

Started 2026-10-29 (the user). **This is the one agent under development.** **Progress, and what comes next: §7.**

- **v1** (`../arc_agent/`) is frozen. It is reference machinery: read it for pieces v2 needs and copy them across;
  never import it.
- **`../legacy/`** stays legacy.
- **Paths** in these docs are relative to `experiments/symbolics`.
- **Tests:** `../../venv312/Scripts/python.exe -m pytest arc_agent_v2 -q`.
- **Lab book:** `arc_agent_v2/docs/LAB.md`.

## 0. The hard rule: no meaning is built in (the user, 2026-10-29)

**No SDR has a built-in meaning, and no math is hardcoded.**
- **Symbols enter as random SDRs.** Inside the brain nothing may look up what a symbol is: no labels, no values, no
  roles. Labels exist only in the worlds, for building inputs and scoring.
- **Forbidden:**
  - integer or floating-point arithmetic on what symbols stand for;
  - slots assigned by input position;
  - an operation fixed to a symbol or to a layer;
  - **cargo-cult learning:** a learned quantity plugged into a hardcoded role or formula. For example, learning φ so
    that φ(c) = φ(a) + φ(b) hides addition in the learning target.
- **Why:** a brain with math hardcoded into it can't learn other domains. The meaning of an SDR in MNIST, in a game or
  in arithmetic must be learned the same way.
- **Innate, and only this:** the column's prior, which is domain-general and used identically for space, images and
  symbols.
  1. Inputs arrive as SDRs.
  2. There is a hidden place, and moves change it.
  3. Moves compose (one after another), and a move does the same thing wherever it is made, until refuted (path
     integration; place invariance).
  4. Content can be remembered at a place.
  5. Innate frames exist only where biology gives them (V1's retinotopy, M1's motor map: `config` "innate:<kind>"),
     never for an abstract domain.
- **How a symbol acquires its meaning (the user's question, the design target):** a symbol's meaning is what it does,
  the move it makes in the column's frame.
  - Its position on an axis is where its move takes you from the start ("argument 1 turns the hand to itself").
  - Its use as an argument is that it moves you from wherever you are ("argument 2 turns it by itself").
  - That is one learned fact, used in two ways. The axis itself is the orbit of the moves (13 moves of "1" mod 13
    bring you back): found, not given.
- **The test for any mechanism:** would it learn MNIST, or a game, unchanged? If it needs to know that its inputs are
  numbers, it breaks the rule.

**The general-learner standard** (the user, 2026-10-29): "The moment you start thinking about modifying what is
supposed to be a general-purpose learner so that 'I can get this many coordinates or this specific outcome' you have
failed and started cargo-culting... You would not do this to a transformer, hold this model by the same standard."
- **Two levers only:**
  1. **The data:** feed it what a human learning the task would get, and ask "does this learn as well or better than a
     human with the same information?"
     - **Not more:** no hints a human wouldn't get.
     - **Not less** (the user's clarification): what a human would need must not be omitted. "It is impossible for a
       human to have a complete understanding of the number line if nobody has shown them all of the digits."
  2. **Sound architectural choices** that make it a better general learner, justified across diverse worlds, never by
     one milestone's target.
- **Milestones measure the general learner.** They are not specifications for new parts: no "place-value module", no
  "carry register", no change of notation to suit the learner.

## 1. The theory

From `arc_agent/docs/DESIGN.md` §1a (the user's derivation, 2026-10-29):
- **Learning a general function means discarding the particular examples,** keeping only the trait they share, which
  predicts unseen cases.
- **You can only discard them if you already know how to recognize places, content and operations.** Without massive
  scale that knowledge has to come first: the no-free-lunch theorem says no learner generalizes without a prior.
- **Networks that grok modular addition build a coordinate system first,** a circle, and the operation becomes a
  rotation on it.
- **So the cortical column is that prior.** It assumes every problem splits into four parts, which its layers provide
  (an analogy with Hawkins' roles, not a claim about anatomy):

| layer | what it is | the requirement it meets |
|---|---|---|
| L6 | the coordinate system | places: an address for everything the column models |
| L4 | place: the argument slots | where the operation's inputs sit (the user marked this one "?") |
| L5 | the operation | the transformation between input and output (a displacement in L6's frame) |
| L2/3 | content | what fills the slots: what is at each place |
| hippocampus (not a layer) | memorizing | the column is very bad at it, so it is delegated |

The rest of the v1 thesis carries over (DESIGN.md §1–§4):
- columns are models over frames, innate or learned;
- objects are derived, not primitive;
- a router (the PFC) picks the right column for a task.

## 2. The datatype: SDRs, never tokens (`HDClibrary.py`, BUILT)

Every representation is an **SDR**, a sparse binary vector:
- **dimension** d = 2048 (`config.SDRConfig.d`, the user's default);
- **free SDRs** have w = 40 active bits (2%);
- **similarity** is overlap, the number of shared bits. Two random SDRs share 0.8 bits on average, so a few shared
  bits already mean "related";
- **symbols:** a symbol enters as a random SDR (`CategoryEncoder`), so any structure among symbols must be learned.

**Block codes, the coordinate systems (`Blocks`):**
- **Shape:** an SDR whose 2048 bits form 32 blocks of 64, with exactly one active bit per block. Each block is a
  module, a ring or torus of 64 cells.
- **Binding** adds the blocks' offsets and **unbinding** subtracts them. The binding is invertible, commutative,
  associative and keeps the code sparse.
- **Powers:** `power(g, x)` multiplies g's offsets by x. So `bind(power(g, a), power(g, b)) == power(g, a + b)`:
  argument 1 turns the hand to itself, argument 2 turns it by itself (tested in `test_power_turns_the_hand`). A block
  code is a point on a product of rings, so a coordinate system is something an SDR can be.

**The trade-off that L6 must settle (§5, question 1):**
- **Metric codes:** near values share bits (`ScalarEncoder`; `GridEncoder`, grid-cell modules of 8 × 8 cells whose
  periods grow by 1.4). They are not exactly algebraic: binding two places agrees with their sum in only about a
  quarter of the blocks.
- **Algebraic codes:** powers of a block code are an exact group code, but distinct values share almost nothing.

**The rest of the library:**
- **set algebra:** & | - ^;
- **kwta:** the k most excited bits;
- **thin:** a random subset;
- **bundle:** a sparse superposition that keeps what many inputs share;
- **cdt:** context-dependent thinning, a binding of free SDRs that keeps similarity;
- **permutations:** roles and positions;
- **ItemMemory:** cleanup, by an inverted index;
- **Synapses:** HTM permanences with Hebbian learning;
- **SpatialPooler:** HTM's learned encoder.

All of it is tested in `tests/test_hdc.py`.

## 3. Configuration (`config/config.py`, BUILT)

`BrainConfig` holds the SDR's dimensions and the **regions of the neocortex**. Each region has:
- a number of columns;
- a frame: innate ("innate:retinotopic-2d", "innate:motor-map", ...) or "learned";
- feedforward and feedback links to other regions or sensors.

`validate()` checks the links (sources exist, feedforward acyclic) and returns the regions in feedforward order.

**Presets:**
- `ARITHMETIC`: one learned region;
- `MNIST`: a V1 sheet with an innate 2-D frame under a learned IT;
- `HUMAN`: the target layout. V1 → V2 → V4 → IT (what), V2 → MT → PPC (where), entorhinal cortex, PFC, PMC → M1. It is
  data only and not wired yet.

Hyperparameters are added to the config as the parts that use them are built.

## 4. The files

**The folders** (the user, 2026-10-29):
- `architecture/` holds the brain.
- `config/` holds the config.
- `worlds/` holds the environments. The brain never imports them (§0; `tests/test_rule.py` checks every module in
  `architecture/` and the config).
- `tests/` holds the tests and the pre-registered harnesses.
- `docs/` holds the lab book and the research notes.
- `runs/` holds the run outputs (not committed).

The brain's files are all in `architecture/` except the config.

| file | what it owns | status |
|---|---|---|
| `HDClibrary.py` | the SDR and every operation on it | **built** |
| `config/config.py` | hyperparameters; the regions; efference copies | **built** |
| `brain.py` | builds the regions, columns, bus and hippocampus from a config; awake steps; sleep (consolidation) | **built** (parts 3–6) |
| `bus.py` | the thalamocortical bus: stamped, exactly-once delivery of every input by receptive field; efference copies | **built** (part 4) |
| `thalamus.py` | the active gate on the bus: holding, gating and routing values into argument slots (`docs/CHAINS.md` §6) | not designed yet |
| `column.py` | the coordinator: owns one column's four layers and passes SDRs between them in a fixed order each step | **built** |
| `l6.py` | the coordinate system: the column's place, and its (phase, coordinate) | **built** |
| `l4.py` | the argument slots: the place(s) the current input sits at; feedforward input arrives here | **built** (sensing only) |
| `l5.py` | the operation: the moves and shifts that take a place to the next | **built** |
| `l23.py` | content: what is at each place | **built** |
| `hippocampus.py` | episodes: fast, one-shot memory of what happened (DG and CA3; CA1 not built), and episode ends | **built** |
| `frameLearning.py` | consolidation: P3, P4 (per context), coordinates, factoring, shift classes and their relation | **built** (parts 3–6) |

**The layers on milestone A** (a direction, to be pre-registered in detail with the code, `docs/LAB.md` part 3). The
input is the written equation, "a + b = c", one random SDR per symbol, one equation per episode. The brain's only task
is to predict the next SDR.
- **L6:** the hidden place, an SDR. Each episode starts at the same place.
- **L5:** each input's move, learned as a mapping from places to places. A move is assumed to do the same thing at
  every place, until refuted.
- **L2/3:** the content remembered at a place, which is the prediction of the next input there.
- **L4:** the input, at the current place.
- **How it generalizes:** when two sequences of moves end at the same place from the start, they are taken to be
  equal everywhere (path integration). From the training equations this closes the moves into a group, the way coset
  enumeration (Todd–Coxeter) builds a group from relations, and fills in the held-out pairs.
- **What is not given:** which symbols move the place and which don't, how many places there are, the cycle, or
  commutativity. All of it is learned.

(A first proposal here hardcoded roles, the operation and integer math, and the user rejected it, 2026-10-29: see §0.)

**frameLearning.py's starting material** (surveyed 2026-10-29). The sources are `../legacy/docs/COORDINATES.md`
parts 1–13 (code in `../legacy/structure/claude/`), RESIDUAL_TRICK.md and REAL_ADELIC.md, and v1's
`docs/HIDDEN_STATE.md` and `unapplied/`. They answer COORDINATES.md's own question, "can the right embedding be derived
from the data instead of handed over?": yes. What each offers for the hard cases:

| hard case | what was found | for v2 |
|---|---|---|
| the structure is not the one hypothesized | parts 2–4: axiom sets (abelian group, group, quasigroup, semilattice, chain, none) compete by prequential evidence. An axiom costs nothing and pays when true; a random table is rejected (1.6 million contradictions) | P4 is one hypothesis (a regular group action) among several. A2 needs this: subtraction is a quasigroup, max a chain, × mod p has an absorbing 0 |
| little data | part 7: associativity in all four forms, with the cancellation patterns. Table known after 2–3% of the entries; code 1.35× the information minimum. The remaining gap is tracking which labels are placed, which is coordinates | stronger completion. Coordinates also feed inference, not only output |
| aliased observations | part 8: the structure found from 2–3 repeated colours almost as fast as from unique ones, places given by each geometry hypothesis's path integration. A true geometry outside the family yields a convincing partial model; the fix (all lattices, Hermite normal form) was never run | under aliasing P3 breaks (futures repeat). Places must come from path integration under geometry hypotheses |
| uncertain moves | part 9: one map per position hypothesis (FastSLAM). A single blurred map reinforces its own errors | several place hypotheses, each with its own content; loop closure |
| non-commutative structure | parts 5–6: decomposing the regular representation gives characters (abelian) or irreducible matrices (A₅: 3 × 3 rotations); "abelian" is read off the decomposition | coordinates from this decomposition of the consolidated move table (part 5's transition table). In SDR form: block codes whose block periods are the characters' orders (learned module sizes, as grid modules differ in scale); permutation representations for non-abelian frames |
| an encoding the learner can't use | part 1, K5: the right coordinates as circles didn't help a small transformer; as residue lookup tables they did | codes must suit the column's operations (overlap, binding) |
| place value and carry | REAL_ADELIC: 3-adic digits (81 = 3⁴) speed learning ≥ 6.6×, a prime gains nothing. Part 1 found 105 = 3 · 5 · 7 from the table; part 10 found the subgroup lattice | A3: place value is a chain of quotients (Z/81 → Z/27 → Z/9 → Z/3), found from the group's subgroups, with no base given |
| geometry vs content | part 10: within one episode a striped colouring is a smaller torus; structure is what persists and constrains actions. Part 11: geometry pays at planning (the shortcut test) | test coordinates on plans and shortcuts, not only on next-symbol prediction |
| hidden states | v1 HIDDEN_STATE §7–8: an event-conditioned graph learned by Baum–Welch recovered a lock and a mode machine exactly | P3 under ambiguity, made probabilistic |
| learned permutations under repeated labels | v1 part 30 draft: each operator's allowed sources narrowed by every observation | constraint propagation, as in part 7's Sudoku rules |

**Not in the folder:**
- the lattice family for geometries outside the hypothesis space;
- whether part 9's remaining gap is the data's or the beam's;
- the commutator as a measure of how much joint search a composition needs (RESIDUAL_TRICK H1, untested).

## 5. Questions: open and settled

1. **L6's code (open; for innate frames only).** For a learned frame, places are learned SDRs and there is no numeric
   substrate (§0). For V1's innate retinotopic frame, the candidates are:
   - the metric grid code (`GridEncoder`: coarse modules, with long periods, make near places overlap; fine modules
     tell near places apart);
   - an exact algebraic code.

   This is decided at B1.
2. **L4 (open):** the user marked "place / arguments" with "?". The working reading is the input at the current
   place. Feedforward sensory input arrives there, as in TBT.
3. **The hippocampus (settled, the user, 2026-10-29: accepted):** DG, CA3 and CA1 built on SDRs, to be researched,
   then built.
   - **DG (pattern separation):** a random expansion followed by kwta.
   - **CA3:** a binary associative memory written in one shot by Hebbian learning, both autoassociative (completion)
     and heteroassociative (sequence, "what followed"). For example Willshaw's memory, or Kanerva's sparse
     distributed memory, itself a model of CA3.
   - **CA1:** maps the recalled pattern back into the cortex's SDRs.
4. **A1's input (settled, the user, 2026-10-29):** the written equation, every symbol a random SDR (operator and "="
   included), one equation per episode. Triples without an operation symbol were rejected: a format that leaves the
   operation out hardcodes what it means.

## 6. Milestones (success criteria fixed now; each build step gets its own pre-registration in `docs/LAB.md`)

**A. Arithmetic with no operation given.** Numbers are random SDRs; the brain must find the coordinate system and the
operation.

| # | test | success | control |
|---|---|---|---|
| A1 | addition mod p (p = 13 and 97), the written equation "a + b = c", every symbol a random SDR, one equation per episode; the only task is to predict the next SDR; train on half the p² equations at random, test the prediction after "=" on the other half (decoded by the world to the nearest symbol) | held-out accuracy ≥ 0.99 | the same brain with frame learning off (memorizing only): chance, 1/p |
| A2 | three operations mixed (+, −, × mod p), each named by a random SDR | ≥ 0.95 on each (× lives on another coordinate system: the multiplicative group) | as A1 |
| A3 | base-10 addition of multi-digit numbers (digits as random SDRs), tested 2 digits longer than trained | ≥ 0.95 exact | as A1 |

**B. MNIST, unsupervised.**
- **Data:** `C:/Users/julia/GitHub/data/MNIST/raw/`, outside the repository and never committed: 60,000 training
  images and 10,000 test images, idx format, read with numpy.
- **Labels** are used only to score.
- **Scores:**
  1. accuracy with 10 clusters, matched to the labels one-to-one (Hungarian);
  2. accuracy with at most 100 clusters, each labelled by its training majority, measured on the test set;
  3. 1-nearest-neighbour accuracy on the learned SDRs, a check that the representation keeps what matters.
- **Reference points from the literature (not run here):** k-means on raw pixels, about 0.53–0.59 with 10 clusters;
  1-nearest-neighbour on raw pixels, 0.969.

| # | test | success |
|---|---|---|
| B1 | V1 columns sense patches at innate 2-D places; IT learns the digits as content at places | 100 clusters ≥ 0.90; 10 clusters ≥ 0.60 |
| B2 | the digit in its own learned frame: shift, scale and slant become L5 transformations between the retinal frame and the digit's ("the transform is the difference") | 10 clusters ≥ 0.80 |

## 7. Progress and build order (the one place to track where v2 is)

**The goal (the user, 2026-10-29):** one model that, by unsupervised and self-supervised learning only:
- predicts MNIST digits;
- does all arithmetic.

Then back to the ARC-AGI-3 public games.

**The next experiment does both at once** (the user, 2026-10-29): "To force you to make a true general learner, your
next experiment will have to do both math and MNIST digit prediction with the same model. Mind you, it will be
unlabeled, unsupervised learning." One brain and one code: no domain flag anywhere in `architecture/`. The audit
(`docs/AUDIT.md`) lists what that experiment must remove (§3) and which mechanisms it must re-justify (§2).

**Where things are written:**
- **This section:** what is done and what comes next.
- **§6:** the milestones' success criteria.
- **`docs/LAB.md`:** the lab book. One "part" per experiment: its pre-registration (before the code), then its results.
- **`runs/`:** the run outputs (not committed).

Vertical slices: nothing counts until it is wired into the brain and exercised by a test. Each step is pre-registered
in `docs/LAB.md`, built, run, and recorded.

**Performance is among the highest commitments** (the user, 2026-10-29).
- **The limit:** a part's run, its regression included, takes under 2 minutes.
- **The regression is tiered** (`tests/battery.py`):
  - **tier 1** goes with every part;
  - **tier 2** (p = 97, and the full fingerprints of `tests/test_fingerprints.py`) runs when consolidation, the column
    or the thalamus changes.
- **A speedup must leave results identical:** the fingerprints and the battery check it.

**Milestones** (§6):

| milestone | what | status |
|---|---|---|
| A1 | addition mod p, from written equations | **done**: held-out accuracy 1.0 for p = 13 and 97 (part 3) |
| A2 | +, − and × mod p, mixed | **done**: held-out accuracy 1.0 on each, p = 13 and 97, 3 seeds; subtraction by digits only ever added 1.0 (k = −1 found), 0.0 without the relation (part 6) |
| A3 | base-10 numbers of several digits, with carry | not started |
| B1 | MNIST: V1 patches at innate places; IT learns the digits | not started |
| B2 | MNIST in the digit's own learned frame | not started |

**Steps:**

| step | what | lab part | status |
|---|---|---|---|
| 0 | the SDR library, the config; the hard rule; A1's world | parts 1–2 | done |
| 1 | A1: the hippocampus, the column's layers, consolidation by P3 and P4 | part 3 | **done** |
| 2 | coordinates learned from the frame; the bus and efference copies | part 4 | **done** |
| 3 | one fact, two uses: one coordinate for every place, factored into (phase, coordinate) | part 5 | **done** |
| A2 | three operations mixed; subtraction from addition facts (the reversal) | part 6 | **done** |
| chains A | chained expressions of one precedence level ("a + b − c ="), zero-shot | part 7 | **done**: chains of 2–4 operations 1.000, never trained; 0.000 with the phase merge off |
| hippocampus | keys by the cortex's location (TEM); a day's memory first; relabelled walk worlds | part 8 | **done**: revisits by new paths recalled 100% where the column knows its place; 0% without location keys or without recency |
| union column | the column's state as a weighted union of (phase, coordinate) hypotheses: Bayesian filtering, bursting on surprise; replaces part 7's merge (`docs/NEURAL_COLUMN.md`) | part 9 | **done**: chains 1.000 in {+, −} and in {+} only (part 7 couldn't); k − 1 bursts per chain; control 0.000; answers ~1–1.6 bits (the Laplace prior is too broad) |
| A3, step 1 | two-digit numerals mod 100, measured (`docs/A3.md`) | part 11 | **done**: held-out 1.000; the chain found is CRT (2, 4, 5, 25), not decimal; tens shift = 10 × units, unrelated; no naming; withheld answers 0.000 (no coordinate: one position's moves are paths, the wrap is the carry) |
| audit | every mechanism against the general-learner standard: where each fires across 10 worlds, and a verdict (`docs/AUDIT.md`) | — | **done**: parts 5–10 fire only in arithmetic, except the context split, fixed points and shift classes. Replace or remove: the relation (x ↦ kx), the thalamus's gate contexts and operations, recency as an override. Limits for the joint experiment: identity-only comparison, cycles-only coordinates, one frame per brain |
| joint, next | arithmetic and MNIST in one brain, unlabeled and unsupervised (the user); scored by §6's A and B measures. The design (`docs/JOINT.md`): a region per sensor (S learned, V innate retinotopic); content by overlap; the hippocampus conjoins what was sensed; objects as hidden causes (L2/3), which also replace recency; one frame per receptive field. Then the association region (13), the thalamus's replacements (14), the relation's general form (15) | 12 | design done; the pre-registration next |
| A3 | the general learner on the data a human gets for multi-digit addition, all the digits and the number line shown, against that human (`docs/A3.md` §4) | — | waits for the joint experiment |
| chains B | precedence ("a + b × c ="): working memory in a `thalamus.py`; the gate's operations as hypotheses, taught by prediction error | part 10 | **done**: 2-operation expressions 1.000 (precedence learned from 300 examples; gate off 0.000 on the sensitive ones); 3 operations 0.85–0.88 (releases before + and − unlearnable from 2-operation data); the run 3.4 min, over its 2-minute target |

**Next, proposed in the order the goal needs (for the user to confirm, 2026-10-29).** Analogy and the reversal curse
are folded in where the goal uses them:
1. **A2 (done, part 6):**
   - P4 split by context, and a constant move for the absorbing 0;
   - shift classes, related by k = −1: the reversal curse, as subtraction learned from addition facts.

   **Left open:**
   - × gets its own coordinates (the discrete logarithm) but generalizes only through P4's table;
   - the MDL search would not survive a fourth class (LAB part 6, limits).
2. **Chained expressions** ("a + b + c ="). The phase automaton must generalize to "+" after a result, which the data
   never show. The user (2026-10-29) expects this to need, very likely:
   1. **Working short-term memory.**
   2. **The thalamus as an active gate:**
      - how and why it gates the brain;
      - where the cortical columns' modules plug into it;
      - how the PFC does its job and influences the rest of the brain mechanically: what wiring, to where, lets it do
        something; what it selects for; what commands it issues, and how.
   3. **How brains solve chained expressions.**
   4. **How these interact** to solve chained expressions without exponential blowups anywhere.

   **Researched** (`docs/CHAINS.md`, 2026-10-29). Two proposals:
   - **A:** loops in the phase automaton, by P3 lifted to phases, with the coordinate as the running result.
   - **B:** a held second value for precedence, through a `thalamus.py` that holds, gates and routes values into
     argument slots: the complement to `bus.py`, as the user suggested.
3. **A3:** place value as the group's chain of quotients (part 4, finding 3).
4. **B1:** a V1 sheet of columns, a frame per receptive field, the spatial pooler, IT.
5. **B2:** the digit in its own frame. Analogical mapping as aligning a digit's frame to a stored one.
6. **The hippocampus's CA1 and attractor recall,** when something needs them.
7. **Back to the ARC-AGI-3 public games.**

**Recorded direction (the user, 2026-10-29):**
- **Analogical mapping as a native ability:** an analogy is a map between two frames that carries moves to moves.
  With coordinates it is one change of coordinates, found from one aligned pair (Kanerva's "What's the Dollar of
  Mexico?"; Gayler & Levy 2009).
- **The reversal curse should fall to the same logic:**
  - a fact is a move between places, moves are one-to-one (P4), and the frame holds each map in both directions, so
    the converse is the same fact read backwards;
  - for relations that aren't one-to-one, the converse is the transpose, not the inverse
    (`experiments/intelligent_constraints/`).
- **A goal cortex** (the user, 2026-10-29; not designed yet).
  - **What it is:** a part of the neocortex, made of columns that predict reward and punishment as generally as
    possible.
  - **Active inference:** goals are ideal states the model tries to stay in.
  - **The drives:**
    - **The first goal:** "be correct".
    - **Later, in agentic experiments:** "achieve the WIN" and "don't waste actions", as two separate drives.
  - **Why it matters for chains:** it could drive multi-step chained actions to be correct in the first place.
  - **It meets B's gate:** "be correct" as a drive is a learning signal of the kind the gate needs (`docs/CHAINS.md`
    §10).
