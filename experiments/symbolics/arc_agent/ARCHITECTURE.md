# The arc agent: columns as models over reference frames, built from a brain blueprint

Written 2026-10-27, before the code. **This is the only agent in `experiments/symbolics`, and the only one we work on**
(the user, 2026-10-28). Everything else in the folder is in `legacy/`: kept for reference, never imported, never
worked on. That includes Gemini's TBT agent (`legacy/gemini_tbt_agent/`), which this agent replaced, and the
experiments that led to it (`legacy/claude_predecessor_experiments/`; `legacy/docs/COORDINATES.md`; parts 1–19 of
`arc_agent/docs/AGENT.md`), whose mechanisms that worked are rebuilt here. The agent's code, docs (`arc_agent/docs/`),
run outputs (`arc_agent/runs/`) and game harness (`arc_agent/kaggle/harness.py`) all live in `arc_agent/`. Paths in
these docs are relative to `experiments/symbolics`.

## 1. The theory it implements

From `arc_agent/docs/DESIGN.md`, in one paragraph: TBT's main problem is taking objects as a primitive. A cortical column is
instead a reusable model over a reference frame (a coordinate system), either innate or learned, and the frames
together represent the task's structure. Structure (the frame and how actions move addresses in it) is stored apart
from content (what is at an address). Objects are derived: a set of cells that one transformation moves together.
A prefrontal router, later, chooses and composes frames for a task.

## 2. The blueprint (`blueprint.py`)

A blueprint is data: a list of **areas**. Each area says how many columns it has, which reference frame they use
(innate: given; learned: chosen by evidence), and which areas feed it. `Brain(blueprint)` builds the columns. The
default blueprint:

| area | columns | reference frame | what it learns |
|---|---|---|---|
| **V1** (visual) | 1 | innate: the retina, 64 × 64 addresses, with translations | the scene: what is at each address now, and (through its Spotlight memory) what lies under things that moved off it |
| **M1** (motor) | 1 | innate: the motor map, the key actions plus click addresses aligned with V1's | which actions exist, and where clicks are worth trying (proposals ranked by what has responded) |
| **assoc** (association) | n = 32 slots | learned, per column: translation (Z², a set of cells moving together), recolour (a permutation of colours on fixed cells), or clock (Z, a counter that advances on every action) | its frame type by evidence; its content (template); for each action, which operator the action applies to its frame (the generalized efference copy) |
| **HC** (hippocampus) | 1 | the product of the assoc columns' frames (a joint state is one point in it) | episodes: Spotlight memory addressed by joint state; which states were visited, what each action did there, where WINs happened |
| **PFC** | 1 (`pfc.py`, part 24) | a choice frame: which kind of model is responsible | which assoc column to trust where several explain the same change: Bayesian model selection whose prior is the PFC column's own prediction, learned like any column's, so the right model is picked faster with experience (learning to learn, emergent). Goals, plans and drives are still in `executive.py` |
| **hidden state** | 5 (`hidden.py`, part 27) | choice frames whose state is not read from the screen: 4 delay columns (the event 1-4 steps ago) and 1 graph column (6 states) | the delay columns learn their own dynamics like any column (lag k copies lag k - 1); the graph column's state is inferred: an HMM with event-conditioned transitions, fitted by Baum-Welch on the visible columns' own memories (each planned column's operator, given its most trusted key's value). Both reach the other columns as ordinary "other column" features |

The blueprint is where innate structure lives (DESIGN.md: priors are geometries). V1 and M1 get their frames from the
blueprint; assoc columns get only a menu of frame *types* and must find their own frames from the data. Adding a
column type, an area, or a PFC later is an edit to the blueprint, not to the brain.

## 3. A column (`column.py`)

Every column has the same parts, whatever its frame:
- **A frame:** the coordinate system, with its group of operators (translations; colour permutations; clock ticks).
- **A state:** the column's address in its frame now (an offset; a phase; a tick count).
- **Content memory:** a Spotlight lattice over the frame's addresses (section 4). For an assoc column, its template:
  the colour at each of its cells' offsets. Structure (the offsets) and content (the colours) are kept apart, so a
  column whose content is relabelled keeps its structure.
- **An operator model per action, the generalized efference copy (part 16):** for each action, a posterior over which
  operator of its frame the action applies to this column. Counts with a prior from the overhypothesis (part 19) and a
  Dirichlet predictive. This is the piece that stops relearning: it is keyed by (action, column), not by the screen,
  so what an action does to a column carries to every screen and every level.
- **Evidence for its frame type:** a running log-likelihood of its predictions, compared between frame types for the
  same cells (attention by track record, part 13). A column whose frame keeps failing is released and its slot reused.

**Actions, seen from a column:** a key is its own action. A click is described by where it lands relative to the
column: on the column's own cells ("on me"), or on another region, named by that region's content signature (colour
and size). So "clicking a button changes a column elsewhere" is learnable: a remote binding (cue competition, the
Rescorla–Wagner rule in its count form).

**Conditions (part 20, replacing a contact rule; in-context attention since part 26, one head per situation feature
since part 27b):** what an action does to a column is predicted by attention over keys. One key is unconditional;
every other key reads one context feature in the column's own frame (the content at one position relative to the
column, with "edge" off the screen and the remembered content beneath it; another column's state, hidden-state columns'
included; a quotient of its position; the numeric expert). Every observation is kept as an episode (the situation,
every key's value, the operator seen, and what each key had predicted).

Heads compete to say which episodes are like now. Head 0 counts every episode (part 20's track record). Head f, for
each situation feature (another column's state, what lies beneath, the column's own state), counts episodes that match
now on f as 1 and the others as exp(-theta_f). In each head, the same weights say:
- which key is read: its prior plus its weighted record of predictions;
- what the key's value means: the weighted counts of the operators seen with it.

theta_f is learned by the gradient of the head's log probability of each new operator; a large theta_f is part 21's
conjunction cases on f (a door that blocks only while the switch is off). Heads are weighted by a prior times their
prequential evidence, so a feature that does not matter loses weight instead of shrinking every episode's (part 26's
product over features did; HIDDEN_STATE.md section 8). Nothing says what blocks or what changes a move: walls, one-way
tiles, ice, switches and hidden locks are whichever key predicts, in the situations where it predicts.

## 4. Spotlight memory (`spotlight.py`)

What we take from Spotlight (Percepta): memory is a lattice of cells at addresses, separate from the model, and grows
without changing any model; a cell is read and written at its address, or through a neighbourhood when the address is
uncertain. What we change (COORDINATES part 5): addresses are computed and path-integrated by the column's frame, not
learned by projecting a key, so exact reads are the normal case.
- **A cell** is a small associative memory updated by the delta rule with one-hot keys: w[key] ← w[key] + β·(onehot(v)
  − w[key]). With β = 1 it stores the last value; with β < 1 it keeps a distribution. This is DeltaNet's update in its
  simplest form.
- **Soft reads:** given a belief over addresses (a distribution), the read is the belief-weighted mixture of the cells.
  When the address itself is uncertain, part 9's lesson applies: one lattice per address hypothesis, not one blurred
  lattice.
- **Its three uses here:**
  1. V1: what lies under each address (object permanence for things that moved, were covered, or are hidden);
  2. each assoc column: its content template;
  3. HC: episodes, addressed by joint state; key = the action, value = what followed (next joint state, WIN).

## 5. One step (`brain.py`)

1. **Predict:** every assoc column predicts its next state for the action taken; V1 renders the predicted scene.
2. **Compare:** the cells the prediction got right are explained. Each column's operator model is updated with what
   the action actually did to it.
3. **Explain the rest (recruitment):** the unexplained change is split into sites (connected sets of changed cells).
   For each site, frame types compete:
   - translation: the displacement d with the most votes from changed old cells to same-colour changed new cells (a
     Hough vote: common fate measured, not assumed); the column's cells are those that moved by d;
   - recolour: same cells, a consistent colour map;
   - clock: a site that changed on every action, whichever action it was (needs a few steps of evidence).

   The type that explains the most changed cells wins (description length); a free assoc slot takes it.
4. **Remember:** V1 stores what is now visible, and what was uncovered, in its Spotlight lattice; HC stores the
   transition.

The state used for planning is factored: the tuple of the assoc columns' states, with clock columns left out (a clock
is predictable from the step count, so it is not part of what the agent controls). The clock is kept as a variable:
it is the remaining budget.

## 6. The executive (`executive.py`, a stand-in for the PFC)

Part 15b's lexicographic satisficing agent with part 17's commitment, over the brain's model:
- **Goal hypotheses** (part 15, about kinds, refuted when they hold without a WIN):
  - `on(kind, c)`: some column of this kind lies wholly on scene colour c;
  - `at(kind, state)`: some column of this kind is in this state.

  Prior: half per family, split evenly. A kind is a column's content signature (its template), so evidence carries
  across levels.
- **Search:** breadth-first over the model's predicted joint states, to depth 12, over keys plus the clicks whose
  effect is known or uncertain. An action whose outcome is uncertain here is a leaf worth its outcome's entropy.
- **Choice:** (1) the shortest plan with P(WIN) ≥ 0.5; else (2) the plan with the most information per step, if
  ≥ 0.05 bits; else (3) the plan most likely to win; else (4) RESET if the agent has acted in this level, or an
  untried action.
- **Commitment:** a plan runs until it ends, a WIN, a surprise (a column not where predicted), or a drop in its value.
- **RESET** is never sent before acting in a level: the engine would restart the whole game.

## 7. Control arms

Until part 26 a separate explorer (a table of whole screen and action to next screen, per level) was run as a control
arm. The user deleted it on 2026-10-28; its recorded results stay in `arc_agent/docs/AGENT.md`. A control arm is now
this agent with one mechanism removed (as `test_conditions.py --no-hidden`), or this agent at an earlier commit. All
games run through `arc_agent/kaggle/harness.py` (level-weighted (human/agent)² score, human baselines from the
toolkit).

## 8. Predictions (pre-registered, before the code)

The control's results are already recorded (`legacy/runs/explorer_v0_fixed.json`: 5 of 183 levels, mean score 0.0012).

| # | prediction |
|---|---|
| B1 | the counter is a clock column: in ≥ 20 of 25 games a clock column is recruited during level 1, and no clock column spans more than 2 rows and more than 2 columns of the screen (a bar, not a board region) |
| B2 | no relearning: in the 7 avatar games (tu93, wa30, ls20, g50t, dc22, sc25, bp35), at least one key's operator on the avatar's column is certain (≥ 0.98) within 15 actions of level 1. In ar25, cn04 and dc22, no single key takes more than 50% of level-1 actions |
| B3 | the brain solves ≥ 10 of the 183 levels (control: 5) |
| B4 | the brain's mean score is ≥ 2 × the control's (≥ 0.0024) |
| B5 | transfer inside a game: the brain completes level 2 in ≥ 3 games (control: 0) |
| B6 | cost: ≤ 50 ms per step on average; all 25 games in ≤ 10 minutes |

**What would show the architecture is wrong rather than unfinished:** B1 and B2 failing would mean columns over
learned frames don't capture what actions do in these games. B3–B5 failing with B1–B2 holding would point to the
missing goal model (most games show their goal on screen, and `on`/`at` don't cover "match this template"), which is
the next part, not a refutation.

    venv312/Scripts/python.exe experiments/symbolics/arc_agent/kaggle/harness.py --arm brain       (and --arm control)
