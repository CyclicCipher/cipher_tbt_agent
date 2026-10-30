# Audit: which of v2's mechanisms were built for an outcome (2026-10-29)

**The standard** (ARCHITECTURE.md §0, the user): a general learner is changed only in two ways:
- by its data;
- by an architectural choice that makes it a better learner across worlds.

It is never changed to get one milestone's result. The smell test: would I do this to a transformer?

**The next experiment** (the user, 2026-10-29): "your next experiment will have to do both math and MNIST digit
prediction with the same model. Mind you, it will be unlabeled, unsupervised learning." This audit is read against
that. MNIST is the second world family that parts 5–10 never had.

## 1. Where each mechanism fires

- **The apparatus:** `tests/audit_fire.py` wraps v2's functions in counters from outside, with no change to
  `architecture/`.
- **The worlds:** ten.
  - **Arithmetic** (p = 13, seed 7):
    - A1 {+};
    - A2 {+, −, ×};
    - the reversal {+, −};
    - A1 with answers withheld;
    - two-digit numerals mod 100 (part 11).
  - **Walks** (part 4, seed 0), each with a relabelled day 2 (part 8):
    - E1, the torus;
    - E2, the cycle;
    - O1, A₅;
    - O2, the box.
  - **Precedence** (part 10, p = 13, seed 7). Its day 1 is A2's world, so its consolidation rows repeat A2's (marked
    =). Only its column and thalamus rows are its own.
- **The output:** `runs/audit_fire.json`.

| mechanism (part) | A1 | A2 | rev. | withheld | numerals | E1 | E2 | O1 | O2 | prec. |
|---|---|---|---|---|---|---|---|---|---|---|
| P4: poolings made (3) | 2 | 18 | 20 | 32 | 286 | 0 | 0 | 0 | 2 | = |
| P4: poolings refused, conflict (3) | 0 | 324 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | = |
| pairs split by context (6) | 0 | 162 | 158 | 0 | 90 | 0 | 0 | 0 | **12** | = |
| poolings refused, under 2 agreements across contexts (6, am. 4) | 0 | 4 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | = |
| fixed-point instances, not evidence (6, am. 5) | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | **6** | = |
| constant moves filled (6) | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | = |
| orbits with coordinates (4) | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | = |
| shift classes (6) | 1 | 2 | 2 | 1 | **3** | 1 | 1 | – | – | = |
| phases left out of the coordinate (6) | 0 | 1 | 0 | 0 | 0 | 0 | 0 | – | – | = |
| naming accepted (5) | 1 | 1 | 1 | 1 | 0 | 0 | 0 | – | – | = |
| relations found / shifts they gave (6) | 0 | 1 / 4 | 1 / 7 | 0 | 0 | 0 | 0 | – | – | = |
| union: hypotheses dropped, transition without shift (9, am. 3) | 0 | 79 | 0 | 0 | 0 | 0 | 0 | – | – | 7,399 |
| union: bursts (9) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – | – | 74,022 |
| union: re-seeded from the place (10, am. 1) | 0 | 79 | 0 | 0 | 0 | 0 | 0 | – | – | 3,513 |
| day 2: today's memory overrides the cortex (8) | – | – | – | – | – | 129 | 130 | 143 | 0 | – |
| thalamus: hold / release contexts learned above the prior (10) | 0 | 0 | 0 | 0 | 0 | – | – | – | – | 30 / 121 |

(– : not applicable. O1 and O2 have no coordinates, so nothing is factored. Only the walks have a day 2 of single
inputs. Only precedence has a day after a sleep, which is when the thalamus learns.)

**What the table says:**
- **Parts 3, 4 and 8 fire in both families:** P3, P4, the coordinates and the day memory.
- **Three of parts 5–10's mechanisms fire outside the world they were built for:**
  - the context split, built for A2, also fires in the box O2;
  - fixed points: A2, and O2;
  - shift classes, built for A2, give three classes in the numerals.
- **Everything else from parts 5–10 fires only in arithmetic:**
  - naming;
  - constant moves;
  - the two-agreement rule;
  - the relation (only where − is);
  - the union's drops and re-seeding (only where × is);
  - bursting, and the thalamus (only in precedence).
- **So the walks never tested parts 5–10's generality.** They only checked that those parts stay out of the way.
  The joint experiment is the first real test.

## 2. The verdicts

**Three kinds:**
- **Keep:** a general rule of inference or learning, or one that fires in more than one world family.
- **Re-justify:** plausibly general, but so far exercised only by arithmetic, or with a constant tuned on a smoke
  case. It needs a derivation from a general principle, or a world outside arithmetic where it should fire, and does.
- **Replace or remove:** shaped for one outcome. Its general form replaces it, or it goes.

### 2.1 Keep

| mechanism | part | why it is general |
|---|---|---|
| P1, P2: one start place, content at places | 1 | the column's prior (§0), approved |
| P3 strict and full; agreement only on a content that identifies places | 3; 4 am. 2 | Myhill–Nerode. Evidence for a merge must discriminate places. Fires in every world |
| the episode's end, observed | 6 am. 1 | a transformer's end-of-sequence token. A human sees an equation end |
| P4: pairs as displacements, pooled; one-to-one refutation | 3 | place invariance, the column's prior 3. Fires in arithmetic and O2 |
| generalize, then split by context on refutation | 6 | generalize, then correct (the thesis). Fires in O2, which it wasn't built for |
| a fixed point is not evidence of identity | 6 am. 5 | every map that fixes the place predicts that instance, so it can't tell them apart. Fires in O2 |
| coordinates from commuting moves (characters, `_cyclic`) | 4, 5 | the grid-module substrate. Found in A and in E1, E2. It declines honestly on O1 and O2 (C6) |
| one coordinate per place, a shift per input, MDL (the fewest inputs that shift) | 5 | P4's regularity across the frame. MDL is general |
| a place with nothing ahead constrains no coordinate | 5 am. 1 | nothing to predict |
| shift classes, generalize then split by phase | 6 | fires in the numerals (3 classes), which it wasn't built for |
| a phase left out of the coordinate when no class fits | 6 | the honest fallback |
| naming, as a hypothesis accepted only by consistency | 5 | **a transformer's tied embeddings:** a symbol's code as output is its code as input (its move). Refused on relabelled answers (F4), as a general learner should. Its limit (one symbol names one place) is in §3 |
| the union of hypotheses: Bayes, Laplace, cells, bursting only when no hypothesis has the transition | 9 | Bayesian filtering, and HTM's minicolumn: a cell is (shared code, context), and a burst is an input nothing predicted |
| the gate learned by EM from prediction error | 10 | learning from prediction error, no reward |
| the input weighed before the decision it prompts | 10 am. 6 | an operation can't change the likelihood of what prompted it |
| the day memory keyed by the cortex's location | 8 | TEM. Fires in the walks |
| `information.py` (the fixed cross-entropy) | 9 | measurement |
| PRUNE, MAX_HYPS, MAX_THOUGHTS, `_choose`'s limit | 9, 10, 6 | resource bounds, like a context length |

### 2.2 Re-justify

| mechanism | part | the problem | what would justify it |
|---|---|---|---|
| across contexts, pooling needs 2 agreements | 6 am. 4 | a constant, 2, set against one coincidence in A2 | derive it as evidence: how likely two unrelated maps are to share k place pairs by chance |
| constant moves (an input that led to one place from 2 places of a context leads there from all) | 6 | fires only for × 0 | a world outside arithmetic with a constant move. ARC-AGI-3's RESET is one, as is a teleport. And a constant move as the degenerate displacement, not a separate rule |
| re-seeding the union from the place | 10 am. 1 | relocalization in name, but it patches the column's two location states (the place path, and the factored union) against each other | one location state. The column has two because part 5 added the factored one beside part 4's |
| USE: at test, branch only where learning raised P(yes) above the prior mean | 10 am. 2 | a compute cut. 0.01 was refuted on a smoke case, then the prior mean was chosen | one pruning rule shared by the thoughts and the union |
| decision points are read symbols only | 10 am. 3 | in a world of actions (efference copies) the thalamus never acts. ARC is such a world | working memory that can hold across actions (subgoals) |
| the module chain in increasing order of period | 4 | chosen so that Z/81's overlap would be 3-adic closeness (C3), a target representation. The content never chooses the basis: the numerals got the CRT chain, not the decimal one | the basis chosen by what predicts the content (MDL). That is suggested by A3, so it needs a justification outside A3 first |
| shift classes taken greedily, largest phase first | 6 am. 8 | an order, not a criterion | MDL: the fewest classes |

### 2.3 Replace or remove

| mechanism | part | why it is shaped for an outcome | its general form |
|---|---|---|---|
| **the relation between shift classes:** x ↦ k·x in each cyclic module, k a unit | 6 | the special case subtraction needs. It fires only where − is. It can't relate a 2-D frame's controls rotated or mirrored (a map that mixes modules), the simplest case in a game | a group map between two classes' shifts, over the whole group, found from the shared inputs and chosen by MDL. This is the user's direction note: "analogical mapping as a native ability", a map between frames that carries moves to moves. So replace, rather than remove. If it is removed, the reversal's subtraction by never-subtracted digits goes back to 0.0 (part 6's G5 control); A2 doesn't use it |
| **the thalamus's gate contexts:** (the input before the last, the current input) for hold; (the held input, the current input) for release | 10 | features picked because precedence depends on the two operators | the thought's own column state (its phase or cell) and the input |
| **the thalamus's operations as built:** hold = restart at the frame's start and re-feed the last input; release = feed the held state the current coordinate's name | 10 | the expression parser. The capacity under it (suspend a state, pursue another, resume with the result) is general: nested sentences, subgoals | a subroutine call, justified by a world outside arithmetic. Re-feeding and name-feeding as built are kept only if that world needs them too |
| **recency as a hard override:** today's memory at a place beats the cortex | 8 | a rule, where the rest of the column weighs evidence. Amendment 5's best place (part 10) exists only because the override reads one place | weigh the day's memory against the column's prediction, with how much to trust recency learned from how often the world changes (a hazard rate). Each thought then reads the memory at its own place, and the best-place rule goes |

## 3. The structural limits the joint experiment hits

These are not cargo cult. They are single-domain assumptions, each justified by more than one world, A3 not among
the justifications.

1. **Identity only.**
   - **The limit:** every comparison in v2 is exact identity of SDRs:
     - P3's futures;
     - P4's pairs;
     - the hippocampus's places (DG separates every situation);
     - L2/3's counts.
   - **Why it fails on images:** MNIST's patches never repeat exactly, and neither do ARC's frames.
   - **The transformer comparison:** its embeddings are graded, so similar inputs give similar states.
   - **The requirement:** content compared by overlap, which reduces to identity on random SDRs (two random SDRs share
     about 0.8 of 40 bits). Then arithmetic is unchanged, and images become possible.
2. **Coordinates need complete cycles of a finite abelian group.**
   - **Which frames get none:**
     - O2's box;
     - a learned frame for the eye over a 28 × 28 image;
     - ARC's 64 × 64 grids.
   - **They are all bounded.** One digit position's 0–9 is the same case. That is A3's, but A3 isn't the
     justification.
3. **One frame per brain, one coordinate per frame, the thalamus on one column.**
   - **What exists now:**
     - `brain.sleep` consolidates one frame for every column ("Not built yet: one frame per receptive field");
     - `factor` keeps only the system covering the most places;
     - the thalamus serves `columns()[0]`.
   - **What breaks:** a brain learning arithmetic and images at once would put both into one frame and factor only the
     larger. ARC needs space and the task as two frames at once.

## 4. What I propose

1. **The joint experiment comes next, as the user asked.**
   - **The setup:**
     - one `Brain`;
     - one config;
     - one stream of episodes, arithmetic and MNIST interleaved;
     - no domain flag anywhere in `architecture/`.
   - **Scoring:** both are scored by §6 (A: held-out answers; B: clusters matched to labels, labels used only to
     score).
   - **The order:** a design note (`docs/JOINT.md`), then part 12's pre-registration, then the code.
2. **§3's limits are what it must remove,** in that order: overlap first, then bounded frames, then one frame per
   receptive field. Each is justified across worlds, not by a score.
3. **§2.3 goes in the same move as its replacement** (the full cutover), with the before and after recorded on A2,
   the reversal and precedence. What has no general form by then is removed, and its results go with it.
4. **§2.2's items are checked when the joint experiment exercises them.** An item that still fires only in arithmetic
   after MNIST is in has failed its re-justification.
5. **A3 waits.** It is a measurement of the general learner on the data a human gets, all the digits shown. That
   includes the number line, which part 11's world never showed.

## 5. For the user to decide

- **§2.3:** remove now, or replace inside the joint experiment's design? Removing now costs the reversal (1.0 → 0.0),
  and probably precedence, until the general forms exist.
- **"The same model":** I take it as one brain instance and one code. Regions differ only by wiring, as cortex does:
  the eye's movements move the place and the retina is sensed, as in the walks; read symbols move the place. Is a
  region per sensor acceptable, or must one region receive both?
