# Arc agent v2: a brain of columns whose layers are the requirements for learning a general function

Started 2026-10-29 (the user). **This is the one agent under development.**

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

All of it is tested in `test_hdc.py`.

## 3. Configuration (`config.py`, BUILT)

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

| file | what it owns | status |
|---|---|---|
| `HDClibrary.py` | the SDR and every operation on it | **built** (16 tests) |
| `config.py` | hyperparameters; the regions | **built** (2 tests) |
| `brain.py` | builds the regions, columns, bus and hippocampus from a config; runs one step (sense → bus → columns → bus → act) | not designed yet |
| `bus.py` | the thalamocortical bus: carries SDRs between regions (feedforward drivers, feedback); later the PFC's gating | not designed yet |
| `column.py` | the coordinator: owns one column's four layers and passes SDRs between them in a fixed order each step | not designed yet |
| `l6.py` | the coordinate system: the column's frame and its place in it (a block code); moving the place by a displacement (path integration) | proposal below |
| `l4.py` | the argument slots: the place(s) the current input sits at; feedforward input arrives here | proposal below |
| `l5.py` | the operation: which displacement or transformation of L6's frame turns the input into the output; the column's output to the bus | proposal below |
| `l23.py` | content: what is at each place (place ⊗ content, recalled by overlap); lateral voting between columns | proposal below |
| `hippocampus.py` | episodes: fast, one-shot memory of what happened, for what the columns can't generalize | **not designed yet** (§5, question 3) |
| `frameLearning.py` | learning coordinate systems, transformations and functions from examples | proposal below |

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

**frameLearning.py's starting material:**
- **v1 and legacy research:**
  - the cyclic group's characters as coordinates (`../legacy/docs/COORDINATES.md` parts 6 and 9);
  - quotients as content regularities (DESIGN.md §3, part 10);
  - learned permutations narrowed by every observation (v1 part 30 draft, `../arc_agent/unapplied/permute_frame.py`);
  - action models lent across a frame's symmetries (part 31b draft, `conjugate.py`).
- **The math for milestone A, the first research item:**
  - **Learning the moves:** learning, from episodes alone, which inputs move the place and what each move does.
  - **Closing them under composition:** a learned automaton, with relations between move sequences inferred from
    shared endpoints, and coset enumeration to complete it. Nothing may assume the group is cyclic, abelian or
    numeric.
  - **Coordinates as a compression of the closed moves, if wanted later:** the group's characters, as the grokking
    networks find.

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

## 7. Build order

Vertical slices. Nothing counts until it is wired into the brain and exercised by a test.
1. **A1:**
   - frameLearning for a group from partial triples;
   - a column with minimal L6, L4, L5 and L2/3;
   - a one-region brain;
   - the memorizing control.
2. **A2, then A3.**
3. **B1:** a V1 sheet of columns, the bus, the spatial pooler, IT.
4. **The hippocampus,** after the design question is settled.
5. **B2.**

Each step: pre-register in `docs/LAB.md`, build, run (about 10 minutes of compute at most), record.
