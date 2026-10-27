# Columns as models over coordinate systems: a design note for the ARC agent

**Status:** a thesis and a plan, 2026-10-27. Nothing in `src/tbt/` has changed on its basis. The evidence so far is in
`../../docs/COORDINATES.md` (parts 1–9), and the next experiments are listed at the end.

## 1. The thesis (the user's, 2026-10-27)

The earlier Thousand Brains (TBT) agents failed, and we never understood why. Much of the trouble was one question:
how should an object be defined, learned and stored? The proposal is that the question itself is the mistake:
- **A cortical column is a reusable model acting over its own coordinate system:** its own equivalent of positional
  embeddings, with its own geometry.
- **The brain represents a task's structure and its content separately.** The prefrontal cortex (PFC) recruits
  circuits across the brain to solve a task.
- **"Object" is not needed as a primitive.** Models are learned over coordinate systems.
- **Some coordinate systems are present at birth,** like the retinotopic map of visual cortex. Others are learned.
  This says concretely how evolution puts priors into a brain: as geometries.

## 2. Why: what failed, and what worked

**What failed:** the two agents made opposite mistakes about objects.
- **The symbolic agent (`../gemini/`, Gemini's) dodged the question by making colour the object.**
  - `brain.py` builds 16 columns, one per colour.
  - Each column learns, for every colour its own colour touches, whether moving into it is free, blocked or shifted.
  - All of this runs over a hand-written motor map (ACTION1 = up, and so on).

  Its columns are keyed by content, and its geometry is given, not learned: the reverse of the thesis. A column for
  "red" can't transfer to a game where the same role has another colour.
- **The main-line TBT agent (`src/tbt/`, and its archived predecessor) took objects as primitive, and spent most of
  its effort on them:**
  - segmentation;
  - recognizing a known object at a new pose;
  - occlusion;
  - duplicate objects;
  - when to mint a new object.

  Each answer raised the next question.

**What worked, in this folder and earlier:**

| result | what it shows |
|---|---|
| grokking mod 105 takes 12 epochs with the residue embedding handed over, 88 without (`CONTROLS.md`) | the right coordinates make learning nearly trivial |
| the hypothesis "commutative group" completes Z/105 from half its table, and its shift operators' eigenvectors are the exact characters (COORDINATES part 1) | the coordinates can be computed from data instead of handed over |
| seven structural hypotheses compete by Bayesian evidence and the right one wins on 6 of 6 tables, from a quarter of the entries (parts 2–4) | the data can choose the structure |
| a model over computed coordinates transfers when every label is reshuffled; A₅ gets a 3 × 3 rotation address (parts 5–6) | structure and content separate: structure transfers, content is rebound |
| the torus is identified from 3 colours about as fast as from unique labels (part 8) | the geometry can be learned from ambiguous observations |
| place invariance needed a factored state and one shared operator over reference frames (the `src/tbt/` rebuild) | the same lesson, met earlier |
| in a transformer's weights, × and + are the same circuit in two coordinate systems | an operation is a choice of basis |

## 3. The architecture it implies

Three parts, plus two consequences.

**A. Geometries** (coordinate systems). Each geometry supplies three things:
1. **Coordinates:** an address for every place it covers.
2. **Path integration:** how each action moves an address. For a group, an action is a group element; the address
   moves by multiplication, a phase step for commuting actions and a rotation matrix for non-commuting ones (parts
   5–6).
3. **A hypothesis space for content:** its quotients. "Content is the same wherever the address is the same modulo
   H", with H a subgroup. A content regularity is a quotient of the geometry. Stripes on a torus are content constant
   on the cosets of a subgroup.

Geometries are either innate or learned:
- **Innate:** for ARC, the 64 × 64 frame grid with its translations, and the rotations and reflections (the dihedral
  group). These are given, the way retinotopy is given.
- **Learned:** found by hypothesis tests, with the data scored entry by entry (prequential evidence: parts 2–4, 8).
  The candidates must be composable (products, quotients, shears), not a hand-written list: part 8's control showed
  that a missing geometry yields only a partial structure.

**B. Content models.** One memory keyed by coordinates, reused by every geometry, together with the content
hypotheses the geometry supplies (its quotients). When content is a function of a quotient, the model predicts places
it has never visited. This is the generalization that came free when coordinates were handed over.

**C. A router** (the PFC's role). It chooses which geometry, or which product of geometries, a task uses, by
evidence. It is a selection among models, not a model of the task.

**Consequence 1: objects are derived, not primitive.** An object is a frame that moves as a unit: a set of features
whose coordinates all change by the same transformation.
- **Its pose** is the group element relating its frame to the scene's frame.
- **Finding objects (segmentation)** is a hypothesis test like part 8's: these features keep the same coordinates in
  one moving frame.
- **The old questions get short answers:**
  - define = a frame;
  - learn = discover what moves together;
  - store = content keyed by frame coordinates.
- TBT's object models become the special case of a rigid frame attached to a thing.

**Consequence 2: uncertain position needs one map per hypothesis.** When the agent can't trust its path (part 9:
slips with repeated colours), the content memory must keep a separate map for each hypothesis about its position. A
single blurred map reinforces its own errors.

## 4. How it differs from TBT

| | TBT (Hawkins) | this thesis |
|---|---|---|
| unit of knowledge | an object | a model over a coordinate system |
| reference frame | location on an object, from sensorimotor movement | any geometry: innate (topographic maps) or learned (groups, lattices, graphs) |
| combining columns | voting among columns | routing: the PFC selects and composes geometries |
| objects | primitive | derived: frames that move as a unit |
| where priors live | not specified | in the innate geometries |

What it keeps from TBT: a repeated, general-purpose circuit, and models built over reference frames.

**Brain evidence:**
- **Structure and content are stored separately:**
  - the Tolman–Eichenbaum machine (Whittington et al. 2020);
  - entorhinal and ventromedial prefrontal cortex abstract a task's structure across new stimuli (Baram et al.
    2021);
  - medial PFC keeps a problem's structure while the hippocampus remaps (Samborska et al. 2022);
  - grid-like codes appear in abstract spaces (Constantinescu et al. 2016).
- **The PFC as a router:** it biases which pathways carry a task (Miller & Cohen 2001); the "multiple-demand"
  network (Duncan 2010).
- **Innate geometry:**
  - topographic maps are laid out by molecular gradients before experience (Sperry's chemoaffinity; ephrin
    gradients), and spontaneous retinal waves refine them before the eyes open;
  - a genome can't specify weights, but it can specify a gradient and a local wiring rule (Zador 2019).
- **A caution:** learning novel patterns transfers only partly across retinal positions (Dill & Fahle 1998). Innate
  geometries are weaker than a CNN's built-in translation symmetry.

## 5. Predictions that would falsify it

| # | prediction | what would refute it |
|---|---|---|
| F1 | transfer follows shared geometry, not shared content: with the geometry carried over, a new episode costs one visit per place; relearning costs the geometry's discovery on top | episodes with a shared geometry gain nothing from carrying it over |
| F2 | content generalizes to unvisited places exactly when it is a function of one of the geometry's quotients | content that is a function of a quotient isn't predicted at unvisited places, or arbitrary content is |
| F3 | objects emerge as frames that move together, with no object primitive: segmenting by common transformation recovers the objects of ARC-like scenes | the objects that matter in ARC games are defined by role (a key opens a door), not by common motion, so co-moving frames miss them |
| F4 | when the true geometry is outside the hypothesis family, the learner finds a partial structure; composing geometries (products, quotients, shears) recovers the true one | composition doesn't recover it, or the family has to be hand-written per game |
| F5 | in ARC, the innate frame geometry plus a few learned ones suffices: what an agent must learn per game is mostly geometry and its quotients | ARC games need structure no small family of geometries expresses |

## 6. Open problems

1. **Per-model geometry or a shared one?** Sensory areas have their own innate maps. Abstract tasks seem to borrow
   one shared structural code from the entorhinal system, which is the Tolman–Eichenbaum machine's picture. Which
   holds decides whether each model builds its own geometry or borrows one.
2. **A grammar of geometries.** Products, quotients, shears, and the structural forms of Kemp & Tenenbaum (2008)
   (chain, ring, tree, grid, cylinder). How is the space searched in polynomial time?
3. **Beyond groups.** Groups give exact coordinates. General graphs give only approximate eigen-coordinates (the
   successor representation's eigenvectors, which are grid cells). Relational state spaces (Sokoban: a linear value
   can't hold its value function) have no obvious coordinates.
4. **Binding under uncertainty.** One map per hypothesis is exponential in the worst case (part 9's hardest case).
5. **Role-defined objects.** A key matters because of what it does, not how it moves. Can "what it does" be a
   geometry: a coordinate in a task-space map, as in the hierarchy notes?

## 7. Experiments

**Done** (`../../docs/COORDINATES.md`): parts 1–9, mapped to the thesis in §2.

**Next:**
1. **Part 10, the thesis in miniature:** learn the geometry from repeated colours, compute its coordinates, then
   reuse one content model over new episodes with new colourings, including content that is a function of a
   quotient. This tests F1 and F2.
2. **The moving piece:** a board plus a piece that moves as a unit. Discover that some features share a moving frame,
   and read off the piece's pose. This tests F3, and is the ARC-shaped case.
3. **Composable geometries:** products, quotients and sheared lattices, so part 8's out-of-family cycle is recovered.
   This tests F4.
4. **The regression line, restarted under the thesis:** search the coordinates in which the target becomes a sum
   first, then fit the content. Finder v3's outer-function inversion was a first step.

**If the thesis holds, the ARC agent becomes:**
- a geometry library (the innate frame grid with its groups, plus learned geometries);
- one content memory keyed by coordinates;
- a router choosing geometries by evidence;
- objects as co-moving frames.

The current `src/tbt/` object machinery (segmentation, recognition, object minting) would be replaced, not extended.

## 8. Updates

**2026-10-27, part 10** (`../../docs/COORDINATES.md`):
- **F2 is supported.** Content that is a function of a quotient is predicted at unvisited cells (0.87 against
  chance 0.17), and arbitrary content is not.
- **Two corrections to the thesis as written:**
  1. **Geometry and content regularity are the same kind of object, a quotient.** Within one episode, a learner
     searching geometries finds a striped colouring as a smaller torus, and does as well as the carried content model.
     So the structure/content split isn't visible in the data of one episode. Structure is defined operationally: the
     part of the model that persists across episodes and constrains what actions do. §3 should say that.
  2. **F1 isn't supported on next-colour prediction.** On a torus, rediscovering the geometry inside an episode
     costs about 0.02 accuracy, because early revisits are short loops every candidate agrees on. Geometry should pay
     before any revisit: planning a shortcut (Tolman's test). F1 needs that test, or episodes too short to rediscover
     the geometry, or a geometry family too large to search within one episode.
