# TBT-LLM Maze Experiments — Design Plan

## Motivation

Before building a full TBT-LLM for sequence modelling, we need to validate the
core claim of Thousand Brains Theory: that a cortical column — given only a local
sensor, a motor output, and a reference frame — can build a map of its environment
and self-localize within it. The maze is the simplest environment where this claim
is testable and falsifiable.

Two experiments are planned:

- **Experiment 1** (single column) — validates the sensorimotor loop in isolation:
  path integration, place cell binding, and predictive navigation.
- **Experiment 2** (multi-column + thalamus) — validates multi-column coordination
  via thalamic gating. This is the prerequisite architecture for the PFC attention
  column and ultimately the TBT-LLM.

---

## Theoretical Background

This section records the TBT theory that informs every design decision below.
It is intended as a self-contained reference — future readers should not need to
hunt for the original papers to understand why the code is structured as it is.

### Sources

Primary papers (all read for this design):

- **Hawkins & Ahmad 2016** — "Why Neurons Have Thousands of Synapses, a Theory
  of Sequence Memory in Neocortex." *Frontiers in Neural Circuits.*
  HTM foundation: distal dendrites, minicolumn bursting, sequence prediction.

- **Hawkins & Ahmad 2017** — "A Theory of How Columns in the Neocortex Enable
  Learning the Structure of the World." *Frontiers in Neural Circuits.*
  Introduces the sensorimotor column with motor efference copy + L4/L6 roles.

- **Lewis, Purdy, Ahmad & Hawkins 2019** — "Locations in the Neocortex: A Theory
  of Sensorimotor Object Recognition Using Cortical Grid Cells." *Frontiers in
  Neural Circuits.*
  L6 grid cells as reference frame; L5 displacement cells; L4 place cells.

- **Clay, Leadholm & Hawkins 2024** — "The Thousand Brains Project: A New Paradigm
  for Sensorimotor Intelligence." *arXiv:2412.18354.*
  Model-free/model-based policies; goal states; voting; Monty architecture.

Supporting neuroscience:

- **Zikopoulos & Barbas 2013** — PFC L5 → TRN → relay nucleus pathway.
  *Journal of Neuroscience.*
- **eLife 2024** — Convergence of basal ganglia with L5 motor cortex in motor
  thalamus; basal ganglia as gate, not voter.


### The One-Algorithm Hypothesis

TBT's central claim: every cortical column, regardless of location or modality,
runs the same algorithm. V1 columns processing edges, PFC columns processing
abstract goals, motor columns generating movements — all have the same laminar
structure and the same computational primitive. The column learns a model of its
input space, self-localises within that model, and generates motor outputs to
improve its predictions.

This is why the maze experiment generalises: if the algorithm works for spatial
navigation (maze cells), it works for sequence navigation (token positions), object
recognition (shape space), and attention (representational space). Each domain
differs only in what the "reference frame", "sensor", and "motor output" mean.


### Layer Structure and Roles

Every cortical column has six layers. The roles most relevant to TBT:

```
L2/3  —  Output to other cortical columns (lateral voting, long-range).
          Carries the column's current best hypothesis to neighbouring columns
          for consensus-building. Also receives top-down context from L1.

L4    —  Place cells. Input layer. Receives thalamic relay from sensory organs
          (or from other cortical areas in higher regions). L4 cells represent
          "what feature is present at the current reference-frame location."
          Crucially, L4 is GATED by the L6 location signal: the same sensory
          feature produces different L4 responses depending on WHERE the sensor
          is currently located in the reference frame. This gating is what
          makes L4 cells "place cells" rather than pure feature detectors.

L5    —  Motor output / goal-state layer. Thick-tufted L5 neurons project
          subcortically to thalamus, basal ganglia, brainstem, spinal cord.
          In TBT (Lewis et al. 2019), L5 neurons are "displacement cells":
          they represent DESIRED NEXT POSITIONS in object-centric coordinates.
          L5 output is a GOAL STATE, not a raw motor command. The motor system
          downstream converts goal states to joint/muscle commands.

L6    —  Grid cells / reference frame. Receives efference copies of motor
          commands (what the body just did). Maintains the column's position
          within its reference frame via PATH INTEGRATION: each motor command
          shifts the position estimate without requiring sensory confirmation.
          L6 output modulates L4 via feedback projections — this is the
          mechanism by which "where I am" gates "what I expect to see."
```

The sensorimotor loop per time step:

```
1. Motor command issued (efference copy → L6)
2. L6 updates reference frame position (path integration)
3. L4 predicts expected sensory features at new position (L6 gates L4)
4. Actual sensory input arrives at L4
5. Prediction error = mismatch between predicted and actual L4 activity
6. If error > 0: L4 updates model; minicolumn bursts (see HTM below)
7. L5 generates goal state (desired next position to reduce uncertainty)
8. Goal state passed to basal ganglia for action gating (see below)
```


### HTM Sequence Memory (the Foundation)

Before TBT, Hawkins developed Hierarchical Temporal Memory (HTM). TBT inherits
HTM's sequence mechanism and extends it to sensorimotor sequences.

**The minicolumn / cell structure:**

In HTM, each cortical column contains many minicolumns, and each minicolumn
contains ~4-6 cells. Cells within the same minicolumn receive the same feedforward
input (same feature at same location). They differ in their CONTEXT sensitivity:
each cell has thousands of distal dendrites that detect patterns in nearby cells
(same layer, same column, nearby columns). These distal dendrites make the cell
"predictive" before the feedforward input arrives.

**Distal dendrites and sequence prediction:**

When a cell's distal dendrites detect a previously learned context pattern, the
cell enters a slightly depolarised "predictive state." If the feedforward input
then arrives, that cell fires strongly while its minicolumn-mates (which weren't
predicted) remain silent. This is how the column represents "feature X in context
C" distinctly from "feature X in context D" — same feedforward input, different
cell within the minicolumn.

**Bursting — the "I didn't predict this" signal:**

When feedforward input arrives at a minicolumn but NO cell in that minicolumn was
in the predictive state, ALL cells in the minicolumn fire simultaneously. This
"burst" has two effects:
1. It signals upstream: "I received input but had no prediction — this is novel."
2. It triggers learning: the currently active context patterns are associated with
   the bursting cells via Hebbian synaptic growth on distal dendrites.

Bursting is how HTM learns new sequences: unexpected transitions cause bursts,
which cause synaptic growth, which creates predictions for the next time this
transition is encountered.

**The sensorimotor extension (2017):**

Pure HTM cannot distinguish "moved left then saw wall" from "moved right then saw
wall" if both produce the same sensory pattern. The 2017 paper adds motor efference
copies: copies of motor commands are fed to L6 BEFORE the sensory consequences
arrive. L6 updates the reference frame, which gates L4, so the same sensory
feature at two different reference-frame positions produces distinct L4 responses.
Motor commands become part of the context that disambiguates otherwise identical
sensory observations.

This is path integration at the cortical level: the column tracks where it is
relative to the object/environment using a running sum of efference copies, not
by waiting for sensory confirmation of each position.

**Apical vs. basal dendrites:**

- Basal (distal) dendrites: receive lateral input from nearby cells in same layer.
  Detect sequential context within the same processing level. Used for bottom-up
  sequence prediction (HTM mechanism above).
- Apical dendrites: receive top-down feedback from higher cortical areas (L1 axons
  from distant columns). Represent the column's hypothesis about what the WHOLE
  OBJECT is, not just the current feature. Apical input biases the column toward
  the current object hypothesis without overriding bottom-up input — it is a "soft
  prior" from the larger context.

In TBT: basal dendrites predict the next sensory feature in the sequence; apical
dendrites carry the current voted-upon object hypothesis from L2/3 consensus. The
interaction: if both agree, the column fires strongly and confidently. If they
disagree, the column either defers to feedforward (apical too weak) or updates its
hypothesis (apical strong, bottom-up prediction error).


### Voting — For Object Identity, Not for Actions

TBT includes a voting mechanism in which multiple columns communicate via L2/3
long-range horizontal connections. The vote is specifically about **object
identity and pose** — what object the column is currently observing and where the
sensor is positioned on that object.

The vote does NOT extend to action selection. Once consensus is reached on object
identity, each column independently generates a goal state (L5 output). There is
no published TBT mechanism for resolving conflicting goal states across columns.
This is an acknowledged open problem.

In our maze experiment: there is only one column in Experiment 1, so no voting
occurs. In Experiment 2, the three columns do not vote on maze cell identity —
each has its own specialised role (sensory, spatial, PFC) and they interact only
through the thalamus.


### Action Selection: Two Policies

TBT (Clay et al. 2024) describes two policies that a column switches between:

**Model-free policy** — fast, habitual movement following learned sequence
structure. The column has learned a sequence "at position A, action N leads to
position B" and executes that sequence directly without deliberation. This is
the basal ganglia direct pathway (D1 neurons facilitating the learned action).
Used when: object/position is confidently recognised and the route to goal is
known.

**Model-based policy** — deliberate selection of the action that will most reduce
uncertainty about the current hypothesis. Direct quote from Clay et al. 2024:
*"moving a sensor to a location that will minimize uncertainty about the currently
observed object."* This requires computing, for each candidate action, what the
expected hypothesis uncertainty would be after taking that action. Used when:
position is ambiguous, or a new environment is being explored.

The two policies correspond to the brain's dual-process architecture:
- Model-free ≈ direct basal ganglia pathway (fast, low cognitive load)
- Model-based ≈ prefrontal + hippocampal deliberation (slow, high cognitive load)

Policy switching criterion in our implementation: switch to model-based when the
belief distribution entropy exceeds a threshold (position uncertain); switch to
model-free when entropy is low and a route to goal exists in the learned map.

**The active inference / free energy connection:**

The model-based policy is exactly active inference (Friston et al.). In free energy
terms: the column maintains a generative model (the PlaceMap), a belief distribution
over hidden states (current position), and selects actions to minimise expected
free energy = expected prediction error + expected uncertainty. "Minimise expected
free energy" and "maximise expected information gain" are equivalent when the
generative model is accurate. We use the information-gain formulation because it is
more tractable to compute for a discrete maze.


### L5 as Goal-State Generator

L5 thick-tufted neurons are "displacement cells" (Lewis et al. 2019). They do not
output raw motor commands (muscle activations or joint angles). They output DESIRED
NEXT POSITIONS in the column's reference frame: "I want to be at position X+Δ."

This is critical for hierarchy: a high-level column (PFC) outputs a goal in
abstract space ("move toward goal cell"). A lower-level column converts that
abstract goal into a more concrete sub-goal ("turn north"). The motor cortex
converts that into joint commands. Each level translates goal-state language into
its own reference frame.

For the maze experiment: the PFC column's L5 output is "move in the direction of
the goal cell." The spatial column interprets this as a displacement in allocentric
maze coordinates. There is no further level below — the displacement IS the motor
command in the discrete maze.


### Basal Ganglia — Gating, Not Voting

The basal ganglia gate which L5 goal state gets executed. This is confirmed by
neuroscience (eLife 2024): inhibitory afferents from globus pallidus / substantia
nigra converge on motor thalamus cells that also receive excitatory L5 input. The
basal ganglia can suppress or release specific L5-driven action channels.

Mechanism (standard neuroscience, not TBT-specific):
- Direct pathway (D1): facilitates the selected action (disinhibits thalamus)
- Indirect pathway (D2): suppresses competing actions (further inhibits thalamus)
- Net result: winner-take-all over competing goal states

TBT does not specify the algorithm by which the basal ganglia select among
competing columns' goal states. For our Experiment 2, we implement a soft
approximation: the PFC column sets a gate vector on the thalamus that attenuates
spatially-inconsistent action proposals. This is inspired by the biology but is
not claimed to be the exact mechanism.


### The Thalamus — Routing and Attention

The thalamus routes sensory information to cortex and is the physical substrate of
the attentional searchlight. The PFC → TRN → relay nucleus pathway (Zikopoulos &
Barbas 2013):

```
PFC L5  →  TRN (inhibitory)  →  relay nucleus  →  target cortical column L4
```

PFC L5 excites TRN neurons; TRN inhibits relay nuclei; therefore high PFC L5
output → more TRN inhibition → less relay transmission → target column sees
attenuated signal. This is the mechanism for top-down attentional suppression.

In our model: the thalamus has one relay nucleus per sensor channel (N/S/E/W/goal).
PFC L5 outputs a gate vector that the relay nuclei apply to each channel. Gate
1.0 = full transmission, 0.0 = complete suppression. The goal_flag channel (bit 4)
is never gated — the agent must always detect arrival at the goal.

This model is intentionally simplified. Biological TRN is inhibitory (reduces
firing rate), not a binary gate. We approximate it as a multiplicative scalar on
each binary sensor bit.


### What TBT Leaves Open

These questions are acknowledged gaps in the theory as of 2024:

1. **Conflicting goal states across columns.** No published resolution mechanism.
2. **Formal optimization objective for action.** Information gain is inferred from
   Monty's behavior; TBT never states an equation.
3. **PFC reference frame.** Hawkins claims PFC columns use the same algorithm in
   abstract space; what the abstract space's dimensions are is unspecified.
4. **L5 distal dendrites.** HTM's sequence mechanism via distal dendrites is
   established for L2/3/4. Whether L5 uses the same mechanism for goal-state
   sequence learning is stated but not elaborated.
5. **Abstract concept columns.** Grid cells firing in abstract conceptual space
   is confirmed by Constantinescu et al. 2016 (Science) and two 2023/2024
   Nature papers, but not integrated into TBT's column algorithm.


### Neuroscience of Reference Frames: Exp, Log, and Hyperbolic Geometry

Research surveyed 2026-04-20 in response to Odrzywolek (2026) "All Elementary
Functions from a Single Operator" (arXiv:2603.21852). Key question: does the
brain use exp/log as computational primitives, particularly in its reference
frame and path integration systems?

**What the literature establishes:**

**1. Logarithmic encoding of quantities is ubiquitous (well established).**

- *Nieder & Miller (2003, Neuron)*: Number-selective neurons in macaque PFC and
  parietal cortex have tuning curves that are Gaussian on a log scale, not a
  linear one. The internal number line is logarithmic.
- *Dehaene (2003, Trends in Cognitive Sciences)*: "The logarithmic mental number
  line" — behavioral and neural review. Weber-Fechner law has a direct neural
  correlate.
- Auditory cortex: tonotopically organised on a log(frequency) axis (octaves).
  Built into anatomy across all species studied.
- *eLife (2022)*: Hippocampal time cells are log-compressed — fields widen with
  time, overrepresenting early moments. Weber-Fechner law for time itself.

Implication: if a quantity Q is encoded as log(Q) in firing rate, then
subtraction of two neural signals computes log(Q1) - log(Q2) = log(Q1/Q2),
i.e., ratio computation is free. This is structurally identical to what makes
EML computationally complete.

**2. Hippocampal spatial geometry is hyperbolic, not Euclidean (strong, recent).**

- *Peer, Sapir, et al. (2021/2022, Nature Neuroscience)*: "Hippocampal spatial
  representations exhibit a hyperbolic geometry that expands with experience."
  CA1 place cells in rats represent space with non-Euclidean, hyperbolic geometry.
  Representation expands proportional to the logarithm of exploration time.

Hyperbolic geometry has exponential distance growth built in: nearby cells are
represented with fine resolution; distant cells with coarse resolution. The
reference frame the brain uses for space is the geometry of exp/log, not flat
integers. Path integration in hyperbolic space requires exp/log operations to
stay accurate. Our current integer-grid AllocentricFrame is a flat-Euclidean
approximation that will accumulate systematic errors at scale.

**3. Grid cell module spacing is geometric (well established).**

- *Stensola et al. (2012, Nature)*: "The entorhinal grid map is discretized."
  Grid cell scales cluster in discrete modules with a constant ratio between
  successive modules. Mean ratio ~1.42 (~sqrt(2)); some animals show 1.65-1.74
  (close to e^0.5 ≈ 1.65).

The scale hierarchy is logarithmically organised: module index predicts
log(scale). This is a multi-resolution coordinate system, analogous to a
wavelet decomposition or a hierarchical SDR over different spatial scales.
Module 1 tracks coarse position; module N tracks fine position. Together they
provide a unique code for any position in the navigated space.

**4. Multiplication is implemented as exp(log(a) - log(b)) in identified neurons.**

- *Gabbiani, Krapp, Koch & Laurent (2002, Nature)*: The LGMD neuron in the locust
  computes angular velocity × (1 / angular size) for looming detection. Mechanism:
  subtraction of two log-encoded inputs in the dendritic tree, followed by
  exponentiation via voltage-gated membrane conductances. Formula stated in
  the paper: exp(log(a) - log(b)). This is eml(log(a), b) — one substitution
  from the EML operator.
- *Gabbiani et al. (2004, Journal of Neuroscience)*: Confirmed the log compression
  happens in the dendritic tree itself, not in upstream circuits.

Caveat: this is a single identified neuron in an invertebrate. The mechanism
has not been directly confirmed in mammalian cortex. It establishes that
dendritic log-subtract-exponentiate is a biologically viable primitive, not that
it is universal.

**5. Divisive normalization is the canonical cortical computation (well established).**

- *Carandini & Heeger (2012, Nature Reviews Neuroscience)*: Divisive normalization
  R = E^n / (sigma^n + sum(E_i^n)) is documented throughout visual cortex,
  auditory cortex, olfactory bulb, hippocampus, attention, and value systems.

In log space: log(R) = n*log(E) - log(sigma^n + sum(E_i^n)). The denominator
involves log of a sum, not log of a single input — so divisive normalization is
structurally related to log-subtraction but is not identical to EML. It is a
more complex operation (pooled denominator, saturation constant). The brain's
most canonical computation is a generalization that subsumes EML but does not
reduce to it.

**What is NOT supported:**

- The specific base e is not confirmed. Auditory cortex uses base-2 (octaves);
  grid cell spacing is closer to sqrt(2) than e^0.5 in most reported data.
  The natural logarithm is not privileged over log2 in any known neural system.
- EML as a whole (exp(x) - ln(y) as a single circuit gate) has not been
  identified in any mammalian cortical neuron.
- There is no theoretical neuroscience literature on EML specifically — the
  operator was defined in 2026. No prior theoretical framework to anchor to.

**The architectural conclusion:**

Exp and log are plausible biological primitives; EML as an atomic gate is not.
The relevant biological operations are:
  - Log encoding of incoming signals (population-level, ubiquitous)
  - Subtraction of log-encoded signals (equivalent to ratio computation)
  - Exponentiation via membrane conductances (dendritic output nonlinearity)
  - Divisive normalization (pooled version of log-subtraction, canonical)

For TBT maze experiments, the most important implication is the hyperbolic
geometry finding and the geometric grid cell module spacing. Our current
flat-integer reference frame is a valid first approximation (verified working)
but does not scale. The biologically faithful upgrade is a multi-scale
hierarchical reference frame with log-spaced resolution levels — matching the
Stensola et al. grid cell module structure.

**Testable prediction from the hyperbolic geometry finding:**

If the biological reference frame is hyperbolic, then path integration errors in
large environments should be non-uniform: errors should be larger in peripheral
(rarely visited) regions and smaller near frequently visited landmarks. A flat-
grid path integrator would show uniform random drift; a hyperbolic integrator
would show the characteristic hyperbolic error pattern. A large-maze experiment
(50x50+) could test this prediction by comparing the two reference frame types.


### Implications for Architecture

The current architecture (v1, verified working on 5x5 maze) uses:
  - Flat integer grid AllocentricFrame (L6 approximation)
  - Binary 5-bit sensor: [wall_N, wall_S, wall_E, wall_W, is_goal]
  - Union SDR PlaceMap with linear Bayesian belief update
  - Single-scale representation

For larger and harder mazes, three upgrades are motivated by the neuroscience:

**Upgrade 1: Multi-scale reference frame (motivated by Stensola et al.)**
Instead of one frame at integer resolution, maintain K frames at scales
s_1 < s_2 < ... < s_K (e.g., 1, 2, 4 cells per unit). Each frame has its own
PlaceMap and belief. The joint belief is the product. Coarse frames localise
quickly; fine frames disambiguate. This is the grid cell module hierarchy.

**Upgrade 2: Extended sensor with wall distances (motivated by log encoding)**
Instead of binary wall-or-not, encode the distance to the nearest wall in each
direction (0=adjacent wall, 1, 2, 3+ cells). Log-compress: floor(log2(d+1)).
This gives 2 bits per direction = 9-bit SDR with much greater discrimination
power in large mazes where many cells share the same binary wall pattern.

**Upgrade 3: Log-domain belief (numerical stability for large mazes)**
With 1000+ cells, belief probabilities underflow to 0. Represent belief as
log-probabilities: log_belief[p] += log(match(sdr, p) + epsilon). Equivalent
to the current linear Bayesian update but numerically stable. Motivated by the
log-encoding ubiquity — the brain does not operate on raw probabilities.

These upgrades maintain the same algorithmic structure (path integration +
Bayesian belief + information-gain selection) while extending to the scale and
geometry that biological systems actually use.

---

## What We Are Not Doing

The existing `src/` (column.py, cortex.py, reference_frames.py) is a **vision
system** built around the fixation-over-images paradigm. We are **not refactoring
it**. The maze code lives entirely in `tbt_llm/src/` and `tbt_llm/experiments/`.
It imports `ReferenceFrame` and `AllocentricFrame` from `src/reference_frames.py`
(which are already correct for maze navigation) but does not use `Cortex`,
`MacroColumn`, or any vision-specific code.

The maze column is a **clean reimplementation** of the sensorimotor loop, written
to exactly match the maze domain rather than bending the vision API into an
unintended shape.

---

## Directory Structure

```
tbt_llm/
├── PLAN.md                     (this file)
├── README.md
├── src/
│   ├── __init__.py
│   ├── maze_env.py             Simple grid maze: walls, goal, step dynamics
│   ├── sensor.py               Local sensor SDR: 5 bits [N,S,E,W,goal]
│   ├── place_map.py            PlaceMap: position→sensor model + belief update
│   ├── thalamus.py             Thalamus: relay nuclei + TRN gating
│   └── brain.py               SingleColumnBrain, MultiColumnBrain
├── experiments/
│   ├── exp1_single_column.py   Experiment 1: single column learns maze
│   └── exp2_multi_column.py    Experiment 2: multi-column + thalamus
└── tests/
    ├── test_maze_env.py
    ├── test_sensor.py
    ├── test_place_map.py
    └── test_thalamus.py
```

---

## The Maze Environment (`maze_env.py`)

A simple discrete 2D grid with walls, a start cell, and a goal cell.

```
MazeEnv
  grid:    np.ndarray (H, W) of uint8 — 0=open, 1=wall
  start:   (row, col)
  goal:    (row, col)
  pos:     (row, col) — current agent position
  prev_pos: (row, col) — position before last step (for collision detection)
  actions: N=0, S=1, E=2, W=3

  reset()              → (row, col)         start new episode at self.start
  reset_at(pos)        → (row, col)         start at arbitrary open cell
  step(action)         → (row, col), bool   take one step; returns (new_pos, hit_wall)
  valid_actions()      → list[int]          actions that don't hit a wall
  reached_goal()       → bool
```

The maze is specified as an ASCII string or a numpy array. A 5×5 default maze with
a clear path but some dead-ends is hard-coded for reproducibility. Walls block
movement (the agent stays in place if it tries to move into a wall). `step()`
returns `hit_wall=True` when this happens so the brain can correct path integration
without needing to compare positions directly.

**Wall collision and path integration:**

When the agent tries to move into a wall, the environment keeps the agent in place
but the brain issued a motor command. The L6 grid cells integrated that command and
now the frame position is wrong. Two correction strategies:

1. **Environment-assisted:** `step()` returns `hit_wall=True`. The brain
   immediately reverses the frame update. This is used in Experiment 1.
2. **Sensor-assisted:** The brain detects that the sensor reading did not change
   after a step, infers a collision, and reverses the frame. This is biologically
   closer (proprioception + vestibular, not an oracle).

We use strategy 1 for Experiment 1 to isolate the path integration question from
the collision detection question.

---

## Local Sensor (`sensor.py`)

The sensor at each cell produces a **5-bit binary SDR**:

```
[wall_N, wall_S, wall_E, wall_W, is_goal]
```

Each bit is 0 or 1. The full sensor space has 2^5 = 32 possible values, of which
only a small subset appear in any given maze. The SDR is intentionally minimal —
it gives the column just enough signal to distinguish cell types without encoding
position. The column must learn position itself via path integration.

Two cells with the same sensor reading (e.g. two open cells in the interior) are
indistinguishable from the sensor alone. The column must use its accumulated path
integration history to tell them apart. This is the core test: does the reference
frame + Bayesian belief update correctly disambiguate cells with identical sensors?

```
LocalSensor
  encode(env: MazeEnv) → np.ndarray shape (5,) dtype int8
```

---

## Place Map (`place_map.py`)

A reimplementation of the minicolumn model tuned for the sequential sensorimotor
domain. In TBT terms: this is the column's stored model of its environment —
the map that L4 place cells collectively represent, indexed by the L6 reference
frame position.

### What the PlaceMap represents

The PlaceMap stores:

```
position_key → expected sensor SDR at that position
```

This is directly the TBT prediction: at any reference-frame position, the column
can predict what sensory features should be present there. Prediction error =
mismatch between stored SDR and observed SDR. Consistent prediction = column is
correctly localised and its model is accurate.

### Learning rule: union model

When the column visits a position, it writes the observed sensor SDR into the
stored model via bitwise OR (union). The union grows monotonically: once a bit is
set at a position, it stays. This matches MiniColumn.learn_one() in the vision
system and represents the same biological principle: the stored model is the union
of everything ever observed at that location. For the maze, sensor readings are
deterministic (no noise), so the union stabilises after the first visit.

```
PlaceMap
  n_cells:   int                         total grid cells (for coverage denominator)
  _model:    dict[tuple, np.ndarray]     position_key → sensor SDR (union)
  _visits:   dict[tuple, int]            visit count per position

  observe(sdr: np.ndarray, pos: tuple) → None
      Write sensor SDR at position (union model).

  predict(pos: tuple) → np.ndarray | None
      Return stored SDR at pos, or None if unseen.

  match(sdr: np.ndarray, pos: tuple) → float
      Fraction of active sensor bits in sdr also set in stored union at pos.
      Returns 0.0 if pos unseen. Always in [0.0, 1.0].
      Used for Bayesian belief update: belief[p] *= match(observed_sdr, p).

  prediction_error(sdr: np.ndarray, pos: tuple) → float
      1.0 - match(sdr, pos).

  coverage() → float
      len(_model) / n_cells.

  localize(sdr_history: list[tuple[np.ndarray, tuple]]) → tuple | None
      Given a short history of (sdr, displacement) pairs, return the most
      likely current position by replaying displacements through the map and
      scoring total match. Used for diagnostics.
```

---

## Reference Frame for Maze Navigation

We reuse `AllocentricFrame` from `src/reference_frames.py` unchanged:

```python
frame = AllocentricFrame(position=(0.0, 0.0), resolution=1.0)
```

- `position_key()` returns `(row, col)` as integers
- `update((dr, dc))` path-integrates one step (L6 efference copy integration)
- `set_position((r, c))` resets to a known position (episode start)

The frame starts at `(0, 0)` at episode start. If the agent knows its true start
position, the frame is initialised to match it. If not (localisation experiment),
the frame starts at `(0, 0)` and accumulates displacement — the belief update
must then match frame-relative positions against the stored allocentric map.

---

## Action Selection: The Information-Gain Algorithm

### L5 as goal-state generator

In biological TBT, L5 does not output which muscle to contract. It outputs a
desired-next-position in the reference frame — a goal state. For the maze,
the goal state is simply the target cell the column wants to move to next.
The "motor system" converting goal state to action is trivial in a discrete
grid: goal_state (r+dr, c+dc) → action (N/S/E/W matching (dr, dc)).

L5 generates its goal state by the model-based policy (see below). In the
multi-column experiment, PFC L5 additionally sets the thalamic gate vector.

### Model-free policy (habitual)

Once the column has confidently localised (high belief mass on one position) and
the PlaceMap contains a known path from current position to goal, execute that
path without deliberation. Implemented as: follow the stored sequence of actions
that was previously successful. Switch to model-based if the path fails (unexpected
sensor reading = prediction error → burst → re-localise).

For Experiment 1 Phase 3 (navigation), we do NOT implement model-free path
following. The column always uses the model-based policy. Model-free is included
in Experiment 2 as a comparison.

### Model-based policy (deliberate, information-gain)

The model-based policy maintains a **belief distribution** over possible current
positions and selects actions to maximally reduce that uncertainty.

**Belief state:**

```
belief: dict[tuple, float]   — position → probability, sums to 1.0

Initialisation:
  If true start known: belief = {start_pos: 1.0}
  If unknown: belief = uniform over all positions in PlaceMap

Bayesian update after observing sensor SDR at current position:
  for each p in belief:
      belief[p] *= place_map.match(observed_sdr, p)
  Renormalise.
  If all weights → 0 (sensor reading inconsistent with all stored positions):
      Reset to uniform (column is lost; must re-explore).
```

**Information-gain action selection:**

The column asks: "if I take action A, what sensor reading will I observe at the
next position, and how much will that reduce my uncertainty about where I am?"

```
For each valid action a (displacement Δ):
    expected_entropy[a] = 0.0
    For each candidate position p where belief[p] > ε:
        predicted_sdr = place_map.predict(p + Δ)
        if predicted_sdr is None: continue   # unseen cell — skip

        # Hypothetical belief after observing predicted_sdr at p+Δ
        hyp_belief = {q: belief[q] * place_map.match(predicted_sdr, q)
                      for q in belief}
        hyp_belief = normalise(hyp_belief)

        expected_entropy[a] += belief[p] * entropy(hyp_belief)

Select: action = argmin_a expected_entropy[a]
```

This selects the action that, in expectation over current position uncertainty,
produces the most informative next observation. It is equivalent to maximising
expected information gain (KL divergence between prior and posterior belief).

**Why not prediction-error minimisation?**

Prediction-error minimisation sends the agent to the most familiar cell (lowest
mismatch between predicted and observed SDR). This is wrong when two positions
have the same sensor reading: visiting either tells you nothing. Information-gain
minimisation instead seeks the cell that discriminates between the two candidate
positions — even if that cell is less familiar.

**Policy switching:**

```
if place_map.coverage() < MIN_COVERAGE:          # map too sparse
    → random walk (pure exploration)
elif entropy(belief) < CONFIDENCE_THRESHOLD:     # well-localised
    → model-free path following (if path exists)
    → otherwise model-based (navigate toward goal via information gain)
else:                                             # uncertain position
    → model-based (reduce position uncertainty first, then navigate)
```

Constants (tunable): `MIN_COVERAGE = 0.3`, `CONFIDENCE_THRESHOLD = 1.0 nat`
(≈30% of cells seen; belief entropy below 1.0 nat = position fairly certain).

**Connection to active inference / free energy:**

This is exactly Friston's active inference with a discrete state-space generative
model. The belief update is variational Bayesian inference (exact here because the
model is discrete and the maze is small). The action selection minimises expected
free energy, which in this setting equals expected entropy of the posterior — the
same quantity computed above. TBT does not use the free energy vocabulary but the
mathematics are identical.

### Basal ganglia in Experiment 2

In the multi-column experiment, the basal ganglia role is played by the thalamus
gating mechanism: the PFC column's goal-direction output sets the thalamic gate
vector, attenuating L5 outputs from columns whose proposed actions are
inconsistent with the current goal. This is a soft approximation to the
biological D1/D2 winner-take-all and is explicitly marked as such.

---

## Thalamus (`thalamus.py`)

The thalamus is the routing hub that makes Experiment 2 architecturally distinct
from Experiment 1. Without it, adding more columns is just adding more independent
learners. With it, the PFC column can modulate what information reaches other
columns — implementing attention.

### Biology (simplified)

```
Cortical column L5  →  TRN (inhibitory)  →  Relay nucleus  →  target column L4

PFC L5 excites TRN; TRN inhibits relay nucleus; therefore:
  high PFC L5 activity → strong TRN inhibition → weak relay → target sees less
  low  PFC L5 activity → weak  TRN inhibition → strong relay → target sees more
```

We model this as a multiplicative gate on each relay nucleus:

```
Thalamus
  n_nuclei:  int                           one per sensor channel (N/S/E/W/goal)
  gate:      np.ndarray (n_nuclei,) ∈ [0,1]   initialised to 1.0 (all open)

  route(sdr: np.ndarray, nucleus_id: int) → np.ndarray
      gated_sdr = (sdr.astype(float) * gate[nucleus_id] >= 0.5).astype(int8)
      Full pass-through when gate=1.0; complete suppression when gate=0.0.

  set_gate(nucleus_id: int, value: float) → None
  set_gate_vector(values: np.ndarray) → None
  reset() → None     # restore all gates to 1.0
```

### What PFC L5 computes

```
goal_direction = goal_position - spatial_col.best_estimate()
gate[N] = sigmoid( goal_direction[0])    # favour N if goal is north
gate[S] = sigmoid(-goal_direction[0])
gate[E] = sigmoid( goal_direction[1])    # favour E if goal is east
gate[W] = sigmoid(-goal_direction[1])
gate[goal_flag] = 1.0                    # never suppress goal detection
```

This suppresses directional channels pointing away from the goal, focusing the
spatial column's attention on goal-relevant sensor information. Interpretable
and deliberately simple — the point is to test the wiring, not to build a
sophisticated PFC.

---

## Brain Wrappers (`brain.py`)

### `SingleColumnBrain`

```
SingleColumnBrain
  place_map:             PlaceMap
  frame:                 AllocentricFrame
  belief:                dict[tuple, float]
  epsilon:               float = 0.3     # annealed over episodes
  min_coverage:          float = 0.3
  confidence_threshold:  float = 1.0     # nats

  reset(start_pos, known_start: bool = True)
      frame.set_position(start_pos)
      if known_start: belief = {start_pos: 1.0}
      else: belief = uniform over place_map._model keys
      # place_map is NOT reset — the map persists across episodes

  update_belief(sdr)
      belief[p] *= place_map.match(sdr, p)  for all p
      Renormalise. On collapse: reset to uniform.

  observe(sdr)
      place_map.observe(sdr, frame.position_key())
      update_belief(sdr)

  select_action(valid_actions) → int
      if coverage < min_coverage or random() < epsilon:
          return random.choice(valid_actions)
      return argmin_a expected_entropy_after_action(a, valid_actions)

  step(action, env) → sdr
      (new_pos, hit_wall), sdr = env.step(action)
      frame.update(DELTA[action])
      if hit_wall:
          frame.update((-DELTA[action][0], -DELTA[action][1]))  # undo
      self.observe(sdr)
      return sdr
```

### `MultiColumnBrain`

Three columns + one thalamus:

```
MultiColumnBrain
  sensory_col:   PlaceMap        sensor SDR at current allocentric position
  spatial_col:   PlaceMap        same reference frame; receives gated sensor
  pfc_col:       PlaceMap        goal-relative reference frame
  thalamus:      Thalamus        5 relay nuclei (N, S, E, W, goal_flag)
  frame:         AllocentricFrame

  reset(start_pos, goal_pos)
  step(action, env) → sdr
      raw_sdr = sensor.encode(env)
      for each nucleus: gated_sdr[nucleus] = thalamus.route(raw_sdr[nucleus], nucleus)
      sensory_col.observe(raw_sdr, frame.position_key())
      spatial_col.observe(gated_sdr, frame.position_key())
      goal_direction = goal_pos - spatial_col.best_estimate()
      thalamus.set_gate_vector(pfc_l5_output(goal_direction))
      frame.update(DELTA[action])

  select_action(valid_actions) → int
      Use spatial_col belief + information-gain policy.
```

---

## Experiment 1: Single Column Maze (`exp1_single_column.py`)

### Hypothesis

A single cortical column with an allocentric reference frame can:
1. Build a complete map of a 5×5 maze from random exploration
2. Self-localize using Bayesian belief update over the stored map
3. Navigate to the goal using information-gain action selection

### Protocol

**Phase 1 — Exploration (build the map, 200 episodes, random walk):**

Track `place_map.coverage()` per episode. Expected: saturates near 1.0 as all
open cells are visited.

**Phase 2 — Localization test (50 episodes, random walk):**

At each step, compute the column's best-estimate position (argmax belief) and
compare to env.pos. Report fraction of steps where they match.

**Phase 3 — Navigation (100 episodes, information-gain policy):**

Random start each episode. Track steps-to-goal. Compare to random-walk baseline
(same maze, same starts, no PlaceMap). Expected: column reaches goal in fewer
steps than random walk.

### Metrics and pass criteria

| Metric | What it tests | Pass criterion |
|---|---|---|
| Coverage at 200 episodes | Map completeness | ≥ 0.90 |
| Localization accuracy | Bayesian belief + path integration | ≥ 0.70 |
| Steps-to-goal vs. random | Information-gain navigation | Statistically < random |
| Mean prediction error | Map accuracy | Monotonically decreasing |

### Output

- `results/exp1_coverage.png`
- `results/exp1_localization.png`
- `results/exp1_steps_to_goal.png`
- `results/exp1_prediction_error.png`
- `results/exp1_map.txt` — ASCII render of learned map

---

## Experiment 2: Multi-Column Brain + Thalamus (`exp2_multi_column.py`)

### Hypothesis

Thalamic gating controlled by a PFC column (goal-direction L5 output) improves
navigation efficiency relative to the single column of equivalent total capacity.

### Architecture

```
Sensory column   ← raw sensor SDR                   AllocentricFrame
Spatial column   ← thalamus-gated sensor SDR         AllocentricFrame (shared)
PFC column       ← spatial column position estimate  goal-relative frame
Thalamus         ← PFC L5 gate vector                5 relay nuclei
```

Goal position is told to PFC at episode start (testing the routing mechanism, not
goal discovery).

### Comparison

| Condition | Mean steps to goal |
|---|---|
| Random walk | baseline |
| Single column (Exp 1) | after 200 explore episodes |
| Multi-column + thalamus | same 200 explore episodes |

**Additional diagnostic:** gate vector trace over one episode — shows how PFC
attention shifts direction as the agent approaches the goal.

---

## Implementation Order

### Phase 0 — Environment
1. `maze_env.py` — grid, walls, step + hit_wall, ASCII render
2. `sensor.py` — 5-bit local sensor
3. `tests/test_maze_env.py` + `tests/test_sensor.py`
4. Smoke test: random agent, verify coverage saturates

### Phase 1 — Single column
5. `place_map.py` — observe, predict, match, coverage, localize
6. `brain.py` — SingleColumnBrain
7. `tests/test_place_map.py`
8. `experiments/exp1_single_column.py`

### Phase 2 — Thalamus + multi-column
9. `thalamus.py` — route, set_gate, set_gate_vector
10. `brain.py` — MultiColumnBrain (extend)
11. `tests/test_thalamus.py`
12. `experiments/exp2_multi_column.py`

### Phase 3 — Analysis
13. Localisation via history matching
14. Gate trace visualisation
15. Write up findings → update `SEQUENCE_MODELING.md`

---

## What These Experiments Tell Us About the TBT-LLM

**If Experiment 1 passes:** The core sensorimotor loop is validated in the spatial
domain. The same algorithm applies to sequence modelling with a temporal reference
frame: each "position" is a point in sequence history rather than a maze cell, and
the "sensor" is the next token rather than a wall pattern. The information-gain
policy translates directly: "which next token to predict" becomes "which context to
attend to in order to reduce uncertainty about the next token."

**If Experiment 2 passes:** The thalamic gating mechanism is validated. A PFC-
analog column can route task-relevant information to other columns and suppress
irrelevant signals. In the TBT-LLM this becomes the attention mechanism: a
goal-tracking column (current prediction target) gates which parts of the sequence
context are routed to the prediction columns. The sequence modelling problem is
then not "attend to all tokens then predict" but "determine what you're trying to
predict, then gate the relevant context through the thalamus."

**If either fails:** The failure mode will specify the fix. Most likely failure in
Experiment 1: path integration drift (frame position diverges from true position
over long episodes). Most likely failure in Experiment 2: gating too coarse
(suppressing 4 of 5 sensor channels loses too much information for accurate maps).

---

## Key Design Decisions and Rationale

**Why not use the existing `MacroColumn`?**
`MacroColumn` is a batch fixation API: `begin_image() → observe() × N → commit()`.
The maze requires a continuous step-by-step API where learning happens after every
action. Bending `MacroColumn` into this shape would produce misleading code. The
`PlaceMap` reimplementation is ~80 lines and directly expresses maze semantics.

**Why not N minicolumns for the maze column?**
In the vision system, N minicolumns compete over N object hypotheses per retinal
patch. In the maze, each cell has one ground truth (no object hypothesis
competition). The Bayesian belief distribution plays the role of the evidence
competition — it is mathematically equivalent but cleaner to represent explicitly
as a dict than as N parallel minicolumn evidence accumulators.

**Why union model?**
Same as `MiniColumn`: sensor readings at a cell may vary slightly with approach
direction if we later add noise. Union over visits is robust to this. In the
noiseless case it stabilises after the first visit.

**Why is the thalamus a multiplicative gate?**
The biological TRN is inhibitory (reduces firing rate). In binary SDR terms:
inhibition = suppressing active bits. A multiplicative gate approximates this
at the cost of exact biological fidelity. This is a deliberate simplification.

**Why is the goal known to PFC?**
We are testing the routing mechanism, not goal discovery. Knowing the goal
isolates the variable of interest. Goal discovery from reward signals is a
separate, harder problem that builds on this foundation.

**Why information gain and not reward maximisation?**
TBT is not a reinforcement learning theory. Columns learn to build accurate models
of their environment and act to improve those models. Reward (reaching the goal)
is used only to define what "success" means for evaluation. The action-selection
objective is model improvement, not reward accumulation.

---

## HTM Sequence Memory — Detailed Mechanism

Sources: Hawkins & Ahmad 2016 (Frontiers), Hawkins, Ahmad & Cui 2017 (Frontiers),
Lewis, Purdy, Ahmad & Hawkins 2019 (Frontiers, PMC6491744).

### Three dendritic zones

Each HTM pyramidal neuron has three functionally distinct input zones:

**Proximal dendrites** (~20 synapses per NMDA spike, near soma): Define the
classical receptive field. When 8–20 co-located synapses activate, they trigger
a local dendritic spike that causes an action potential — the cell becomes
*active*. This is the feedforward sensory input path.

**Basal dendrites** (distal from soma, ~100 synapses per segment, ~25 segments):
Detect patterns of activity in surrounding cells — temporal context from the same
processing level. When enough synapses on one segment are co-active, the cell
enters a *predictive state*: sub-threshold depolarisation without an immediate
spike. If the feedforward input then arrives, this cell fires strongly while
unpredicted minicolumn-mates remain silent. This is the sequence prediction
mechanism.

**Apical dendrites** (tuft, electrotonically isolated from soma): Receive
top-down feedback from higher cortical areas via L1 axons, and motor efference
copies in TBT. Apical input biases the cell toward a "soft prior" — it amplifies
predictions but cannot override strong feedforward disagreement. When both basal
and apical are active simultaneously, the cell fires with high confidence.

### The minicolumn burst — "I didn't predict this"

When feedforward input arrives at a minicolumn but NO cell in that minicolumn was
in the predictive state, ALL cells burst simultaneously. This has two effects:

1. **Upstream signal:** "This context is novel/unexpected." Downstream modules
   see a high-entropy representation rather than a sharp point.
2. **Learning trigger:** The currently active context patterns on basal dendrites
   of the bursting cells are associated via Hebbian growth. Next time this same
   context precedes this sensory input, those cells will predict it and suppress
   the burst.

The burst corresponds biologically to the column failing to predict the next
element of a sequence. It is the learning signal, not an error signal sent
anywhere in particular — the learning happens locally at the synapses.

### How motor efference copies disambiguate sequences (2017)

Pure sensory HTM cannot distinguish "moved left then saw wall" from "moved right
then saw wall" if both produce the same sensory pattern. The 2017 paper resolves
this: motor commands are issued as efference copies to L6 **before** sensory
consequences arrive. L6 updates the reference frame (path integration), which
gates L4, so the same sensory feature at two different frame positions produces
distinct L4 responses.

Concrete example from the paper: a robot touches a cylinder at heights A, B, C, D
from bottom to top, then D, C, B, A from top to bottom. Without motor signals, HTM
must learn both sequences separately. With motor efference copies, the system
learns "feature A is at height 0" regardless of exploration order, because the
height (L6 frame position) is always part of the context for L4.

The disambiguation is therefore not "which sensory token came before this one" but
"where am I in the object's reference frame" — a fundamentally different and more
powerful representation.

### Grid cells in L6 — path integration at cortical scale (2019)

Lewis et al. 2019 propose that L6 contains grid cell modules — populations tiling
their reference space in the same multi-scale pattern seen in entorhinal cortex.
Unlike hippocampal grid cells (physical space), cortical grid cells encode location
of the sensor on the **object** in object-centric coordinates.

Path integration in L6:
```
location_{t+1} = location_t + motor_displacement_t
```
The active population pattern shifts as a "bump" across the tiled grid cells.
Sensory input (L4 feedback to L6) can correct drift: if the observed feature
doesn't match the prediction at the current estimated position, the bump is
nudged toward the consistent location.

Capacity result from Lewis et al.: k grid cell modules enable reliable recognition
when each feature appears at least k times in the object. More modules → more
discriminative power without linear growth in cell count.

### What this means for our PlaceMap

Our `PlaceMap` is the L4 abstraction: it stores `position_key → sensor SDR`,
indexed by the L6 frame position. Our `AllocentricFrame` is the L6 abstraction:
it path-integrates motor commands (step displacements) to maintain a position key.
The Bayesian belief distribution over position plays the role of the bursting
dynamics: high entropy belief = many cells bursting (many minicolumns in novel
state); low entropy = sharp single-cell prediction (one minicolumn dominant).

**The union model is correct for our maze**, because sensor readings in a
noiseless maze are deterministic — the union stabilises after one visit per cell.
If we later add sensor noise, the union correctly represents "everything ever
observed at this cell" as a coverage region rather than requiring exact match.

**The PlaceMap does NOT implement cell-level disambiguation** (multiple cells per
minicolumn predicting different successors). This is a deliberate simplification:
in a discrete maze with exact position tracking, there is no sensory ambiguity
that requires cell-level disambiguation — the frame position alone disambiguates.
If we later move to a continuous environment or add partial observability, we would
need to add N competing PlaceMap instances (one per hypothesis) or implement
explicit HTM-style distal dendrite prediction within each map cell.

### The HTM → TBT evolution summarised

| Aspect | HTM 2016 | HTM 2017 | TBT 2019+ |
|---|---|---|---|
| Location | Implicit (dendrite graph) | Allocentric L6 signal | Explicit grid cell tiling |
| Motor integration | Absent | Efference copy modulates predictions | Path integration via grid cells |
| Sequence storage | Cell-to-cell transitions | Feature-location associations | Grid location ↔ sensory features |
| Disambiguation | Temporal context only | Motor + temporal | Motor-driven location frame |
| Generalisation | Must see exact sequence order | Better (feature at location) | Excellent (grid encoding scales) |
| L5 role | Not emphasised | Receives motor signals (proposed) | Displacement cells / goal states |
| L6 role | Not emphasised | Computes/receives location | Grid cell modules + path integrator |
