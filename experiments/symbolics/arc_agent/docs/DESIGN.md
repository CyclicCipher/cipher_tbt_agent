# Columns as models over coordinate systems: a design note for the ARC agent

**Status:** a thesis and a plan, 2026-10-27. Nothing in `src/tbt/` has changed on its basis. The evidence so far is in
`../../legacy/docs/COORDINATES.md` (parts 1–9), and the next experiments are listed at the end.

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

## 1a. What each layer of the column does: the requirements for learning a general function (the user, 2026-10-29)

**The missing piece:** the thesis said why TBT was only partly right, but not what each layer of the column does in
the new theory. It can be deduced:
1. From first principles and evidence, work out what learning a general function requires.
2. Map those requirements onto Hawkins' proposed roles for the layers. This is an analogy, not a claim about the
   anatomy.

**The worked example: modular arithmetic.**
- Draw the number line in polar coordinates: a dial, as on a clock.
- Modular addition adds one angle to another: it is a rotation.
- Argument 1 turns the hand to itself; argument 2 then turns the hand by itself.
- If you know that base-10 digits are a polar coordinate system, one per digit, the rule is trivial (4 + 3 is a
  rotation). But that assumes the answer. Pretend you don't know it: from a few memorized sums, how would you derive
  the rule that solves 4 + 3?

**What generalizing is:**
- **The general rule is the trait the examples share.**
  - That trait predicts examples outside the training distribution (OOD).
  - Finding it reduces complexity and discards information.
  - The question is therefore: how do we know what information to discard?
- **An analogy: morality.** Many people were taught, or derived by mistake, an incoherent definition of morality,
  memorized from examples. One evil act the definition doesn't cover refutes it. The general rule (the NAP, the
  non-aggression principle) is what the examples share. What do modular addition and the NAP have in common?
- **How to tell generalizing from memorizing every example:** use inputs where memorizing would need almost
  infinitely many examples. For instance, use floating-point values (a rotation by any angle) rather than integers.
- **The observation that set this off:** after the softmax fix (arXiv 2610.31642), generalization proceeds smoothly,
  governed by a ratio between weight decay and an optimizer quantity (the user didn't recall which). What that means
  semantically is unclear, because it is continuous computation that can't be put into words. Hence the analogy.

**The derivation:**
1. **To discard the specific examples, you must already know how to recognize place, content and operation.** You
   cannot discard them before you know what those are.
2. **Without massive scale that is impossible.** You need the abstraction of places and content before you can learn
   the operation. This is what the no-free-lunch theorem requires: no learner generalizes without a prior.
3. **The evidence:** networks that grok modular addition build a coordinate system first (the circle / Fourier
   representation). The operation is then a rotation in it.
   - The same in this project: grokking mod 105 takes 12 epochs with the residue coordinates handed over, and 88
     without (§2).
4. **So the column's structure is the prior that the no-free-lunch theorem requires for learning a general function.**
   It assumes every problem splits into four parts:
   - a coordinate grid;
   - a set of arguments (slots);
   - content that goes in the slots;
   - transformations (functions) between input and output.
   It works because it leaks the task's structure to the model.
5. **It also explains why the neocortex is not the home of episodic or recent memory.** It is very poorly suited to
   memorizing, so it delegates that to the hippocampus.

**The layers, by analogy with Hawkins' roles.** This is also Gary Marcus's algebraic mind (*The Algebraic Mind*,
2001): variables, operations over variables, and instances bound to variables. The coordinate system the variables
live in is added.

| layer | Hawkins (TBT) | this theory: the requirement it meets | in the arc agent (`ARCHITECTURE.md` §3) |
|---|---|---|---|
| L6 | location on the object | **the coordinate system** | the frame (`frames.py`): its addresses and its group of operators |
| L4 | sensory feature at a location | **place: the argument slots** (the user marked this one "?") | the column's state (its address in the frame) and the context the operator reads (contents at positions relative to the column, other columns' states) |
| L5 | displacement, motor output | **the operation: the displacement between input and output** | the operator model per action (`Column.keys`, one `ConditionalOps` per action key; `frame.apply(state, op)`) |
| L2/3 | the object, settled by voting | **content: what fills the slots** | the content memory (`Column.content`, a Spotlight lattice: a translation column's template colours) |
| hippocampus | — | **memorization, which the column is bad at** | `Brain.hc` (part 28): (level, joint state) × action → next joint state |

**How the agent answers "what information to discard?" today:**
- **The screen position is discarded by construction.** An action's operator is keyed by (action, column), not by
  where the column is, so what an action does carries to every place (part 16).
- **Context features are discarded by evidence.** Each situation feature has its own head with a learned sharpness
  θ_f, weighted by its prequential evidence (part 27b). A feature that doesn't predict loses weight and is in effect
  discarded.
- **Open:** discarding examples before the frame exists, which is the no-free-lunch step. The agent is given a menu
  of frame types and doesn't yet learn new ones from scratch (part 30's Permute frame is the first learned one).

**A note (Claude):**
- **The no-free-lunch theorem says:** averaged over every possible target function, no learner beats another. So any
  learner that generalizes has a prior.
- **So the claim is not that this prior is the only one possible.** It is that the world's tasks (and ARC's) do
  factor into place × content × operation, so this prior is the one that matches them.
- **What would refute it:** a task family that doesn't factor this way, on which the column's prior loses to a
  structure-free learner of the same size.

## 2. Why: what failed, and what worked

**What failed:** the two agents made opposite mistakes about objects.
- **The symbolic agent (`legacy/gemini_tbt_agent/`, Gemini's) dodged the question by making colour the object.**
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
| the layers (§1a) | L6 location, L4 feature at location, L5 displacement and motor output, L2/3 object | L6 coordinate system, L4 place (argument slots), L5 operation, L2/3 content; memorizing is the hippocampus's job |

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

**Done** (`../../legacy/docs/COORDINATES.md`): parts 1–9, mapped to the thesis in §2.

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

**2026-10-27, part 10** (`../../legacy/docs/COORDINATES.md`):
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

**2026-10-27, parts 11–12:**
- **F1 is supported where it should be: planning.** On the shortcut test, carried geometry gets home by the
  shortest way in every trial, wrap-arounds through unexplored territory included. Without it:
  - a planner that gambles fails 5–27% of the time;
  - a planner that plays safe takes the shortcut in only 2–21% of shortcut trials, with routes 3–5 times the
    shortest;
  - remembered transitions get home in 13–15%.
- **F3 is supported, with a condition.** Objects come out as frames that move together, with no object primitive,
  only if segmentation is revisable. One frame pair can't reveal a fixed point of the motion (a rotation's pivot) or
  the inside of a uniform run (the aperture problem). With re-explanation on refutation, every trial ended with the
  exact object.
- **The learner finds which coordinate system each action lives in:** translations world-frame, rotation
  body-frame. That map is what transfers to a new piece: errors drop from 5.2 to 2.1 per 10 steps.
- **Colour as the object (the symbolic agent's assumption) predicted 0.3% of frames.**

**2026-10-27, part 13: the push_block rebuild** (`push_block_rebuild.py`). The first version of the architecture of
§3, on the symbolic agent's own world:
- **Columns:** frames found when cells move together, and split when they stop.
- **Drivers:** each frame's motion is attributed by evidence to static, action, time (step count mod n) or contact
  with another frame.
- **Router:** attention over drivers by track record.

Results:
- **Prediction:** exact next-frame accuracy 0.93–0.98, against 0.36–0.87 for colour as the object.
- **Time:** a patroller that moves without the agent acting is attributed to time with period 6 in every run.
- **Play:** after one WIN, every attempt is optimal.
- **What it still lacks:**
  1. **Goal discovery:** the goal is copied from one WIN; it doesn't yet generalize across layouts.
  2. **Directed exploration:** random play costs 500–1,000 steps to the first WIN, and leaves the box's contact role
     unresolved in 3 of 9 runs.
