# Cipher's TBT Agent

A from-scratch agent based on the Thousand Brains Theory of the neocortex, aimed at the ARC-AGI-3 interactive
reasoning benchmark (ARC Prize 2026).

It is not a large neural network trained by gradient descent on a big dataset. One small, reusable cortical
column learns the structure of whatever it is given (a space, a number line, a game's rules, a language's words)
as a geometry it can navigate. The same column, composed into a small neocortex, plays games, recognizes objects,
does arithmetic, and models language, with no task-specific rules.

## Status

This is a research prototype. It is not a finished system and it is not a claim of general intelligence. The
results below come from an offline ARC-AGI-3-style replica built for development and from small demos, with 78
automated tests covering them. The agent has not yet been run on the official ARC-AGI-3 games. A bridge to the
official agent SDK is in place (`src/arc_sdk.py`), and connecting it to the live games is the current work.

## How it works

### One mechanism: learn a map, predict by navigating it

A cortical column does one thing. It learns a map of some structure, then predicts by moving through that map. It
has four parts, loosely matching cortical layers (`src/tbt/column.py`):

- Where (L6): a location code. The column places what it sees into a coordinate frame computed from how things
  connect. That frame is the eigenbasis of the successor representation, the same code grid cells are thought to
  use in the brain (Stachenfeld et al., 2017). It comes out grid-like for open space, ring-like for a cycle, and
  correct for a branching tree, all from one rule.
- How (L5): displacement operators. Movement is a first-class object. "Add 5" is stepping the successor operator
  five times.
- What (L4): a content codebook. What sits at each location (a digit, a colour, a tile).
- Memory (L23): an object store of bound "what is where" facts.

### Memory through a gated recurrence

The location code can run as a recurrent state, a selective gated integrator in the style of a modern
state-space model (`src/tbt/recurrence.py`). This lets the agent path-integrate: it tracks where it is from its
own moves, without seeing its absolute position, and it remembers what it has seen. The same recurrence drives
both the location frame and a small language model.

### Composition: a small neocortex

Harder tasks use more than one column. A task column and a space column are combined through a thalamus, which
routes a goal from one column into another, and a basal-ganglia gate, which selects which column handles which
structure (`src/tbt/thalamus.py`, `src/tbt/basal_ganglia.py`). The roles emerge from competition and
reinforcement rather than being assigned. Keeping the columns separate and additive avoids the combinatorial
blow-up of one giant state.

### Perception and planning

Perception turns a frame into objects by connected-component segmentation, then identifies each object
pose-invariantly so a rotated object is still read as the same one (`src/perception/perceive.py`,
`src/tbt/recognize.py`). This follows the evidence-based recognition of Numenta's Monty (Clay et al., 2024): an
object is stored once in its own reference frame, and its pose is solved for rather than memorized per
orientation. The planner rolls the learned model forward and scores reached states by a signed value
(`src/tbt/neocortex.py`, `src/tbt/reward.py`). Reaching a goal, clearing a blocker, covering a target, and
avoiding a hazard all come out of that one loop, with no typed sub-goals and no per-game branches.

## What it does so far

These run with one agent and no per-game rules. The agent learns each game's rules by playing it.

- Plays the replica games. The same agent solves the levels of several small ARC-AGI-3-style games in
  `src/tasks/games/`: a key-and-door game, Sokoban (including multi-cell rigid blocks pushed as one body), a
  collect-everything game, a reversible-switch game, and Tetris (it controls a multi-cell piece and rotates it
  through the recognized object's own rotation operator). It learns which colour does what and what counts as a
  win from the sparse score, then plans. The sub-goal order is learned, not coded.
- Recognizes objects under rotation. It learns an object set online by watching, with no labels, and recognizes
  a known object at an orientation it never saw, by solving for the pose (`src/tbt/recognize.py`).
- Tracks itself when it cannot see everything. Given a small window around the body, it path-integrates its
  position and remembers the map (`src/demos/recurrent_location.py`).
- Does exact arithmetic by navigation. It learns a number line and adds by walking it, including place value and
  carry, and stays correct on sums with more digits than it was shown (`src/demos/arithmetic.py`,
  `src/demos/carry.py`).
- Models language at small scale. From small Latin and Middle and Old High German corpora, the same column
  mechanism gives a word geometry where related words land near each other, and a next-token model that improves
  on n-gram baselines in the sparse-data regime (`src/demos/language_recurrent.py`).

## Install

Python 3.11 or newer for the core and the tests. The live ARC-AGI-3 SDK needs Python 3.12 or newer (see below).

```bash
python -m venv venv
. venv/Scripts/activate          # Linux or macOS: source venv/bin/activate
pip install -e .                 # core deps: numpy, torch. add ".[demos]" for the language demo's scipy
```

Run the tests:

```bash
python -m pytest src/tests
```

Run a demo (each is a module under `src/demos/`):

```bash
python -m demos.arithmetic
python -m demos.language_recurrent
```

## Repository layout

```
src/
  tbt/             the cortical column and the shared machinery
    column.py        the column (L6 location frame, L5 operators, L4 codebook, L23 memory)
    recurrence.py    the selective gated recurrence (path integration and the language model)
    thalamus.py      cross-column goal routing
    basal_ganglia.py emergent column allocation
    reward.py        the value critic and the replay-based planner
    recognize.py     pose-invariant object recognition
    neocortex.py     the forward-model rollout planner
    agent.py         the thin agent that drives an environment
  perception/      frames to objects and scenes, plus the planner adapter
  tasks/           the offline ARC-AGI-3-style replica (games, harness, a BFS oracle)
  wm/              an earlier symbolic agent, kept as a reference
  demos/           runnable validations (number line, arithmetic, language, control loop, and more)
  arc_sdk.py       the bridge to the official ARC-AGI-3-Agents SDK
  tests/           the automated tests
corpora/           the language data (Latin, Middle and Old High German)
```

## The ARC-AGI-3 target

ARC-AGI-3 is an interactive reasoning benchmark. An agent plays short games it has never seen, with no
instructions, and works out the goal from a sparse score (ARC Prize 2026). Per step the agent gets a frame, a
score, a game state, and the set of available actions, and nothing else. This matches the design here, where the
goal is learned rather than given.

`src/arc_sdk.py` bridges this agent to the official ARC-AGI-3-Agents framework by mapping it onto the framework's
`choose_action` and `is_done` methods. Running against the live games needs an API key from three.arcprize.org
and Python 3.12 or newer with the `arc-agi` toolkit installed.

## References

Thousand Brains Theory and Monty:

- Hawkins, Lewis, Klukas, Purdy, Ahmad (2019). A Framework for Intelligence and Cortical Function Based on Grid
  Cells in the Neocortex. Frontiers in Neural Circuits.
- Lewis, Purdy, Ahmad, Hawkins (2019). Locations in the Neocortex: A Theory of Sensorimotor Object Recognition
  Using Cortical Grid Cells. Frontiers in Neural Circuits.
- Clay, Leadholm, Hawkins (2024). The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence.
  arXiv:2412.18354.
- Leadholm et al. (2026). Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and
  Inference. Neural Computation. arXiv:2507.04494.
- Numenta and the Thousand Brains Project: https://thousandbrains.org ,
  https://github.com/thousandbrainsproject/tbp.monty

Grid cells, place codes, and path integration:

- Stachenfeld, Botvinick, Gershman (2017). The hippocampus as a predictive map. Nature Neuroscience.
- Burak, Fiete (2009). Accurate Path Integration in Continuous Attractor Network Models of Grid Cells. PLoS
  Computational Biology.

Sparse representations:

- Ahmad, Hawkins (2016). How do neurons operate on sparse distributed representations? A mathematical theory of
  sparsity, neurons and active dendrites.

Planning, replay, and value:

- Mattar, Daw (2018). Prioritized memory access explains planning and hippocampal replay. Nature Neuroscience.
- Schrittwieser et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).
  Nature.
- Wang, Liu, Ye, You, Gao (2024). EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data.
  ICML. arXiv:2403.00564.

Disentangled structure:

- Higgins et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.
- Locatello et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
  Representations. ICML.

Sequence models and the benchmark:

- Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
- Chollet (2019). On the Measure of Intelligence. arXiv:1911.01547.
- Sutton (2019). The Bitter Lesson.
- ARC Prize 2026 and ARC-AGI-3: https://arcprize.org/arc-agi/3

## License

MIT-0 (MIT No Attribution). See `LICENSE`.
