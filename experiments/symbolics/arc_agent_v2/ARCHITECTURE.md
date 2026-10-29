# Arc agent v2: a brain of columns whose layers are the requirements for learning a general function

Started 2026-10-29 (the user). **This is the one agent under development.**

- **v1** (`../arc_agent/`) is frozen. It is reference machinery: read it for pieces v2 needs and copy them across;
  never import it.
- **`../legacy/`** stays legacy.
- **Paths** in these docs are relative to `experiments/symbolics`.
- **Tests:** `../../venv312/Scripts/python.exe -m pytest arc_agent_v2 -q`.
- **Lab book:** `arc_agent_v2/docs/LAB.md`.

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

**The layers on milestone A** (a proposal, to be pre-registered with the code). The input is a triple of random SDRs
(a, b, c), seen as a ∘ b = c, with ∘ never named:
- **L6** holds a learned map from each symbol to a place, a block code power(g, φ(symbol)). frameLearning finds φ, so
  that φ(c) = φ(a) + φ(b) on the training pairs.
- **L4** has two slots: argument 1 sets the place (turn the hand to a), and argument 2's place is the displacement.
- **L5** applies the operation: bind(place(a), place(b)), a rotation.
- **L2/3** reads the content at the resulting place: the symbol whose place it is (cleanup by overlap).

**frameLearning.py's starting material:**
- **v1 and legacy research:**
  - the cyclic group's characters as coordinates (`../legacy/docs/COORDINATES.md` parts 6 and 9);
  - quotients as content regularities (DESIGN.md §3, part 10);
  - learned permutations narrowed by every observation (v1 part 30 draft, `../arc_agent/unapplied/permute_frame.py`);
  - action models lent across a frame's symmetries (part 31b draft, `conjugate.py`).
- **The math for milestone A:**
  - The training triples define, for each b, a partial permutation P_b : a ↦ c.
  - These permutations commute and compose (P_b P_b' = P_{b+b'}): a partial regular representation of the group.
  - Its characters, the joint eigenvectors (a discrete Fourier transform), give the circle coordinates. This is what
    the grokking networks find by gradient descent; here it is to be found by the group structure itself.
  - How to do this from partial data, without assuming the group is cyclic, is the first research item.

## 5. Open questions (for the user)

1. **L6's code:** a metric code (grid modules: near places overlap), an exact algebraic code (powers of a block code),
   or both, coarse modules metric and fine ones algebraic?
2. **L4:** the user marked "place / arguments" with "?". The working reading is the argument slots: the places the
   operation's inputs occupy. Feedforward sensory input arrives there, as in TBT.
3. **The hippocampus:** the v1 memory (Spotlight cells keyed by exact joint states) is to be replaced. A proposal to
   research first: DG, CA3 and CA1 built on SDRs.
   - **DG (pattern separation):** a random expansion followed by kwta.
   - **CA3:** a binary associative memory written in one shot by Hebbian learning, both autoassociative (completion)
     and heteroassociative (sequence, "what followed"). For example Willshaw's memory, or Kanerva's sparse
     distributed memory, itself a model of CA3.
   - **CA1:** maps the recalled pattern back into the cortex's SDRs.
4. **A1's input:** the triples alone (no operation symbol at all), as proposed above.

## 6. Milestones (success criteria fixed now; each build step gets its own pre-registration in `docs/LAB.md`)

**A. Arithmetic with no operation given.** Numbers are random SDRs; the brain must find the coordinate system and the
operation.

| # | test | success | control |
|---|---|---|---|
| A1 | addition mod p (p = 13 and 97) from triples (a, b, c); train on half the p² pairs at random, test on the other half (the answer: the nearest symbol) | held-out accuracy ≥ 0.99 | the same brain with frame learning off (memorizing only): chance, 1/p |
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
