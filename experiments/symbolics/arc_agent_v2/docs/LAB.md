# Arc agent v2: lab book

Each part is pre-registered (the change, the tests, predictions) before its code, then its results are added.
v1's lab book is `../../arc_agent/docs/AGENT.md`. The design is `../ARCHITECTURE.md`.

## Part 1: the foundation (2026-10-29)

**The user's decisions:**
- **A new agent:** v1 is frozen as reference.
- **Every representation is an SDR of d bits** (default 2048), never a token.
- **The files:** brain, config, HDClibrary, bus, column, l23, l4, l5, l6, hippocampus, frameLearning.
- **First goals:** arithmetic with no operation given, and MNIST unsupervised.

**Built:**
- **`HDClibrary.py`:**
  - the SDR datatype and its set algebra;
  - similarity;
  - kwta, thin, bundle;
  - cdt binding;
  - permutations;
  - block codes, whose binding is offset addition, so powers are an exact group code;
  - encoders: category, scalar, and 2-D grid modules;
  - item memory with an inverted index;
  - HTM synapses and spatial pooler.
- **`config.py`:** the SDR's dimensions and the regions, with three presets (arithmetic, MNIST, a human-like layout
  that is data only).

**Tests:** 18 of 18 pass (`test_hdc.py`, `test_config.py`).

**Three wrong guesses, corrected in the tests (not the code):**
- a scalar run of 40 bits spans only 2 units of a 0–100 range;
- the grid code's overlap at a distance of 1 is 10 of 32 blocks, not more than 16;
- 359.5° sits 2.8 bits before 0° on the ring, so a 37-bit overlap is correct.

**Found while building:** a quantized code can't be both metric and exactly algebraic.
- Binding two grid-module codes agrees with the code of their sum in only about a quarter of the blocks.
- Powers of a block code are exact, but distinct values share almost nothing.
- Which one L6 uses is open question 1 in ARCHITECTURE.md §5.

**MNIST:** found at `C:/Users/julia/GitHub/data/MNIST/raw/` (60,000 + 10,000 images, idx files; outside the
repository).

## Part 2: the hard rule, and A1's world (2026-10-29)

**The user's corrections:**
1. **No SDR has a built-in meaning, and no math is hardcoded.** Cargo-cult learning is forbidden too: a learned object
   slotted into a hardcoded role or formula.
2. **My first A1 proposal broke the rule three times:**
   - roles fixed by position in a triple;
   - L5 hardwired to `bind` (addition);
   - φ fitted so that φ(c) = φ(a) + φ(b), then `power(g, φ)`.
3. **A1's input is the written equation,** every symbol a random SDR, "+" and "=" included.

The rule is written into ARCHITECTURE.md §0. The user's question, "how does a digit's SDR acquire its meaning, as an
argument or as a position on an axis?", is the design target. The answer to build to: **a symbol's meaning is the move
it makes.**

**The hippocampus design** (DG, CA3 and CA1 on SDRs) was accepted.

**Built: `arithmetic_world.py`:**
- each equation "a op b = c" is an episode of SDRs;
- the world splits the equations at random into training and held-out;
- it decodes a prediction to the nearest symbol (scoring only).

**Tests:** 23 of 23 pass. The scorer is verified both ways: an oracle scores 1.0, and a constant guess scores below
0.2 (chance is 1/13).

## Part 3: A1, learning what each symbol does (pre-registered 2026-10-29, before the code)

**The world:** addition mod p, the written equation "a + b = c", every symbol a random SDR, one equation per episode.
The brain's only task is to predict the next SDR. Training: an epoch of the training equations, shown once.

**The innate prior, and nothing else (ARCHITECTURE.md §0):**
- **P1:** every input moves the column's place: a hidden SDR, the same at the start of every episode.
- **P2:** the content at a place predicts the next input there.
- **P3, same future, same place:** two places whose observed futures agree (and never disagree) are one place. This
  is the Myhill–Nerode test, recognizing a place by what follows it.
- **P4, regularity:** a move is determined by where it takes a single place. Moving from where symbol x takes you to
  where symbol y takes you is one displacement, the same at every place, until refuted. It is space's property ("3
  right" is 3 right everywhere), here assumed of any frame.

**Not given:** which symbols move the place and which don't, how many places there are, cycles, commutativity, order,
or any value.

**The mechanism, by file:**
- **`hippocampus.py`, memorizing:**
  - DG mints a fresh place SDR (random expansion + kwta) for every situation not met before.
  - CA3 stores each transition in one shot: (place, input) → next place.
  - CA1 reads a recalled place back out.
  - Alone, this is a trie of the episodes.
- **`frameLearning.py`, consolidation (replay of the hippocampus):**
  - merges places by P3;
  - collects every observed instance of each relative move (P4);
  - merges the instances that share a place pair;
  - fills in the transitions never observed.
- **The column:**
  - **L6:** the current place;
  - **L5:** each input's move, from the consolidated transitions;
  - **L2/3:** the content at each place;
  - **L4:** the input at the place.
- **`brain.py`:** builds one region from `config.ARITHMETIC`, resets the place at an episode's start, and steps one
  input at a time, returning its prediction of the next.

**Arms:**
- **full;**
- **memorize-only:** consolidation off. This is the control the success criterion names.

**A guard against cargo-cult hardcoding:** a test checks that no module of the brain imports the world or uses a label.

**Predictions** (3 seeds each):

| # | prediction |
|---|---|
| M1 | full, train fraction 0.5: held-out accuracy ≥ 0.99 for p = 13 and p = 97 (the criterion A1 states) |
| M2 | memorize-only, train fraction 0.5: held-out ≤ 1/p + 0.05; training accuracy ≥ 0.99 for both arms |
| M3 | data efficiency: full reaches held-out ≥ 0.99 at train fraction 0.3 for p = 13, and 0.1 for p = 97 (larger groups give more instances of each move) |
| M4 | the learned frame, after consolidation: (a) the places reached after "a + b =" number exactly p; (b) the relative moves on them form a group in which every move but staying returns to its starting place after exactly p steps (p prime): a ring, found not given; (c) the move of "=" commutes with every relative move (it carries the ring onto itself without reordering it) |
| M5 | cost: one run under 60 s for p = 97 |

**What would refute the design:**
- **M1 failing with M4 holding:** the frame is found but not used for prediction (a wiring fault).
- **M4 failing:** P3 and P4 are not enough to find the ring from half the table.

## Part 3 results (2026-10-29)

**Build check: the first run was killed after minutes.** The cost was my implementation, not the architecture.
- **The memories were quadratic:** every store scanned all items.
- **`Conjunction` used dense 2048 × 2048 projections.**
- **A wasted loop in P4's fill:** it scanned every pair at a place, not only the new move's.
- **`SDR !=` ran a numpy array comparison.**
- **A 14-process pool.**

All fixed (commit `11039a7`), and the outputs are identical to before on every consistent configuration. A run at
p = 97 now takes about 4 s with scoring; p = 13 takes 54 ms end to end.

**Results** (`runs/a1.txt`, `runs/a1.json`; 3 seeds each; 83 s for all 24 runs):

| p | train fraction | arm | held-out accuracy | training accuracy | seconds per run |
|---|---|---|---|---|---|
| 13 | 0.5 | full | 1.0, 1.0, 1.0 | 1.0 ×3 | 0.1 |
| 13 | 0.5 | memorize-only | 0.0, 0.0, 0.0 | 1.0 ×3 | 0.0 |
| 97 | 0.5 | full | 1.0, 1.0, 1.0 | 1.0 ×3 | 3.9–4.1 |
| 97 | 0.5 | memorize-only | 0.0, 0.0, 0.0 | 1.0 ×3 | 1.9–2.0 |
| 13 | 0.3 | full | 1.0, 1.0, 1.0 | 1.0 ×3 | 0.1 |
| 97 | 0.1 | full | 1.0, 1.0, 1.0 | 1.0 ×3 | 4.1–4.2 |
| 13 | 0.5 | full, "?" control | 0.012, 0.024, 0.024 (chance 0.077) | 1.0 ×3 | 0.1 |
| 97 | 0.5 | full, "?" control | 0.0002, 0.0, 0.0 (chance 0.010) | 1.0 ×3 | 6.8–6.9 |

**The learned frame (M4), identical in every seed:**

| p | fraction | moves inferred | held-out equations | places after "a + b =" | displacements | full permutations of the places after "a + b" | their order |
|---|---|---|---|---|---|---|---|
| 13 | 0.5 | 85 | 85 | 13 | 12 | 12 | 13 |
| 13 | 0.3 | 118 | 118 | 13 | 12 | 12 | 13 |
| 97 | 0.5 | 4705 | 4705 | 97 | 96 | 96 | 97 |
| 97 | 0.1 | 8468 | 8468 | 97 | 96 | 96 | 97 |

**The prediction at every position of the held-out equations** (seed 0; what comes next after k symbols):

| p | arm | a | "+" | b | "=" | c |
|---|---|---|---|---|---|---|
| 13 | full | 0.035 | 1.0 | 0.0 | **1.0** | **1.0** |
| 13 | memorize-only | 0.035 | 1.0 | 0.0 | 0.0 | 0.0 |
| 97 | full | 0.008 | 1.0 | 0.0 | **1.0** | **1.0** |
| 97 | memorize-only | 0.008 | 1.0 | 0.0 | 0.0 | 0.0 |

- **Positions a and b are the world's free choices.** b scores 0 by construction: the brain predicts a b seen after
  "a +" in training, and a held-out b never was.

**Verdicts:**
- **M1 holds:** held-out accuracy is 1.0 for p = 13 and p = 97.
- **M2 holds:** memorize-only scores 0.0 on held-out equations and 1.0 on training ones.
- **M3 holds:** 1.0 at train fractions 0.3 (p = 13) and 0.1 (p = 97).
- **M4 (a) holds:** exactly p places after "a + b =".
- **M4 (b) holds, with a deviation:** it was measured on the places after "a + b", where the displacements act; no
  place offers two inputs leading to the places after "=". There, every displacement but staying is a full
  permutation of order exactly p: the ring was found, not given.
- **M4 (c) is not measurable as pre-registered,** for the same reason. In its place, "=" carries the ring one-to-one
  onto the places after "=" in every run.
- **M5 holds:** about 4 s per run at p = 97.
- **Apparatus check:** on the random table there is no full permutation, and held-out accuracy is at or below
  chance. There is no leak.

**What this shows:**
- **The meaning of each digit was learned as the move it makes.** No symbol, value, role or operation was given:
  "=" included, which the brain now anticipates on equations it never saw.
- **The generalization is P4.** Moves are pooled by the place pairs they share, until each digit's relative move is
  a full permutation of the ring. Exactly the held-out moves were inferred, with no spurious ones.
- **Memorizing alone generalizes nothing.**

**Limits:**
1. **One operation, one fixed format, numbers of a single symbol.**
2. **P4 holds exactly for a group,** so 1.0 is what an exact learner should reach. A2 (several operations) and A3
   (multi-digit numbers with carry) are the real tests.
3. **On the random table P4 still infers some wrong moves** (19–112), and nothing retracts them: "generalize, then
   correct" has no correction yet.
4. **The ring after "a" and the ring after "a + b" are different places.** The same displacements act on both, but
   the brain does not yet know that "the place 7 takes you to from the start" is "the place a sum of 7 reaches".
   That is the other half of "one fact, two uses".

## Part 4: coordinates learned from the frame (step 2; pre-registered 2026-10-29, before the code)

**Why (the user):** coordinate learning is the natural next step. Pre-registered with two tests from the survey
(ARCHITECTURE.md §4):
- a structure outside the hypothesis space, the case where COORDINATES part 8's partial models fooled the scoring;
- a shortcut test (part 11), where carried structure pays before any revisit.

**The goal:** place codes in which each learned move is one fixed operation on the code, the same at every place,
instead of a row in a table.

### The change

1. **Coordinates from the frame** (`frameLearning.py`, the method of COORDINATES parts 1, 5 and 6):
   - **The generators:** the frame's maps that are full permutations of a set of places, i.e. P4's pooled
     displacements, and any input whose move permutes a set of places. The places they act on split into orbits;
     each orbit gets coordinates.
   - **The read-out "abelian":** do the generators commute on the orbit? If not, the step declines and the orbit keeps
     its table. (Codes for non-commuting frames, permutation representations, are a later step.)
   - **Characters:** the common eigenvectors of the generators' permutation matrices (a random complex combination),
     each normalized at the orbit's base place, the first of its places met. Each character has an order n, and a
     phase index (0 … n−1) at every place.
   - **Modules, finding 3:** go through the characters in increasing order of n, and keep one only if it splits some
     class of places that the ones kept so far leave together. This keeps the group's own chain of quotients:
     - Z/105 gives 3, 5, 7 (the CRT factors);
     - Z/81 gives 3, 9, 27, 81 (the 3-adic digits);
     - Z/6 × Z/10 gives 2, 2, 3, 5.
   - **The code** (a new library class, `Modules`, block codes whose blocks have their own periods):
     - **Blocks:** each kept character of order n becomes R blocks of n cells, each block with a random unit u
       (coprime to n). R = max(1, min(⌊d / Σn⌋, ⌈w / number of modules⌉)), so about w active bits within d.
     - **A place's code:** offset u · (its phase index) mod n in each block.
     - **A move:** the shift δ it makes in every block. It is applied by binding (offset addition), the same at every
       place, and is found from a single instance (s → t) as δ = unbind(code(t), code(s)).
2. **Inputs that move, and inputs that are sensed** (`config`, `column`, `hippocampus`):
   - **Each sensor's inputs either move the place** (P1; an efference copy, or A1's read symbols) **or are sensed at
     it** (content, P2; they don't move it).
   - **A1:** its symbols move, as before.
   - **The walk worlds:** actions move; observations are sensed.
   - The hippocampus stores transitions for moving inputs, and content for both.
3. **P3 in its full form,** as part 3 worded it: "two places whose observed futures agree (and never disagree) are one
   place". Part 3 built only the strict form (identical futures), which merges nothing in a walk, whose futures never
   repeat.
   - **After the strict merges:** places are merged if, over every input sequence both have seen, their content never
     differs where both are determined (seen with one next input), and they agree on at least one such content.
   - **Merging folds their children together.** A merge that would make some place's content conflict is undone.
   - **Candidates** are compared only within groups of equal determined content.
4. **The route (L5's inverse):** the moves that lead from here to a goal place.
   - **With coordinates:** the displacement δ = unbind(goal, here), looked up in a table of shortest words built once
     by a breadth-first walk over the codes.
   - **With a table:** a breadth-first search over the known moves.
   - It is used only by the shortcut test.

### The line between innate and learned (for the user to check)

- **Innate:**
  - P1–P4 (approved);
  - modules: rings whose offsets add when moves compose. This is the grid-module substrate; the library supplies
    it, as it supplies binding;
  - that a sensor's inputs either move or are sensed. This is wiring: efference copy against sensory afferent, as
    TBT's location is path-integrated from movement while sensation lands in L4. It is set per sensor in `config`.
- **Learned from the data:**
  - the places;
  - every move;
  - whether the moves commute;
  - how many modules there are, and their periods;
  - each place's phase;
  - each move's shift.
- **Never used:** a symbol's value, label or position in the input. The base place is the first met, and the units
  are random. The eigen-decomposition is consolidation's computation over the learned moves, as the integer row
  addresses are.

### Worlds

- **A1's frames:** addition mod p, train fraction 0.5, 3 seeds; p = 13, 97, 105 and 81.
- **`walk_world.py` (new):**
  - **Episode 1:** a walk from a start node, actions and nodes' labels as random SDRs, with a unique label per node.
  - **Inside the hypothesis space:**
    - **E1:** the 6 × 10 torus, actions N, S, E, W.
    - **E2:** the cycle of 60, actions ±1 and ±7. This is part 8's E2: outside part 8's family of rectangular tori,
      where a 15 × 15 torus won as a partial model. It is inside v2's (any finite abelian group).
  - **Outside it:**
    - **O1:** A₅'s Cayley graph, actions a, b and b⁻¹ (a group, but not commutative);
    - **O2:** a bounded 6 × 10 box, moves blocked at the walls (not a group: a blocked move is not one-to-one).
- **Episode 1 is 300 steps,** part 11's length, with 3 seeds per world.

### The shortcut test (part 11, in v2)

- **Trials:** 20 per seed, on E1 and E2. Each trial has a new start node and new unique labels; the start is the
  brain's base place, the worlds being homogeneous. Then:
  1. a 40-step random walk;
  2. an excursion: k steps of action 0, then m of action 2;
  3. the query: the brain's route home, executed in the world.
- **Arms:**
  - **(a)** episode 1's frame carried, with coordinates;
  - **(b)** episode 1's frame carried, as its table, without coordinates;
  - **(c)** relearning: a frame consolidated from this trial's steps only;
  - **(d)** the hippocampus only: a search over this trial's remembered transitions.
- **Measures:**
  - success (back at the start node);
  - length over the true shortest length;
  - on shortcut trials (where the shortest way home wraps around), how often the shortest way is taken.
- **What it separates:** carried structure from relearning, as part 11 did. If P4 completes the carried table, (a)
  and (b) should plan identically. That tests that the codes lose nothing, not that they add anything. What codes add
  beyond a complete table is measured by C2 (a move from one instance) and C3 (quotients).

### Predictions

| # | prediction |
|---|---|
| C1 | A1's frames, on the ring of places after "a + b": module periods [13], [97], [3, 5, 7] and [3, 9, 27, 81] for p = 13, 97, 105 and 81, in every seed; every place's code distinct |
| C2 | on those rings, every pooled displacement is one fixed shift: code(t) = bind(code(s), δ) for 100% of its (s, t), and δ found from one (s, t) reproduces the whole map (100%) |
| C3 | finding 3: on Z/81's ring, two places' codes overlap in exactly R × #{k ≤ 4 : 3^k divides their difference} bits, for every pair. SDR overlap is 3-adic closeness. (The difference is read from the world's labels in the harness only) |
| C4 | no regression from P3's full form: part 3's A1 results (M1–M4) reproduce exactly for p = 13 and 97 |
| C5 | E1 and E2, in every seed: consolidation finds exactly 60 places; P4 completes the move table; the read-out says abelian; the modules are [2, 2, 3, 5] (Z/6 × Z/10) and [2, 3, 4, 5] (Z/60). On E2, v2 is not fooled where part 8 was |
| C6 | O1 and O2, in every seed: the step declines (O1: not abelian; O2: the moves are not permutations) and installs no codes. P4's inferred moves are 100% correct on O1 (a group, where P4 is exact) and ≥ 95% correct on O2 |
| C7 | the shortcut test, arm (a): success in every trial on E1 and E2, always by the shortest way (length ratio 1.00), wrap-arounds included |
| C8 | arm (b) plans exactly as (a) in every trial. Arm (c): success ≤ 0.9, or the shortcut taken in ≤ 30% of shortcut trials. Arm (d): success ≤ 0.3 |
| C9 | design and cost: ≥ 30% of trials are shortcut trials; all of part 4 runs in under 60 s |

**What would refute the step:**
- **C5 failing on E2:** v2 fooled where part 8 was.
- **C6 failing:** codes installed on a structure they can't represent, which is a convincing wrong model.
- **C2 failing with C1 holding:** the code doesn't carry the moves.
- **C3 failing:** the module chain isn't the quotient structure.

**Not in this step:**
- an aliased episode 1 (geometry hypotheses scored by path integration, part 8);
- slips (one map per hypothesis, part 9);
- codes for non-commuting frames;
- competing structure hypotheses (needed for A2).

**The guard (`test_rule.py`)** extends to every new brain module. `walk_world.py` is a world; the brain never imports
it.

### Part 4 amendments (2026-10-29, during the build, before any run)

1. **Episode 1 is 1,000 steps, not 300.**
   - **The problem:** a random walk of 300 steps visits all 60 nodes in only 16% of walks (torus 6 × 10: 0.159;
     cycle ±1 ±7: 0.172; 2,000 simulated walks each). The coordinate step needs every node, so C5 would fail by
     design.
   - **The fix:** at 1,000 steps the chance is 0.998 and 0.9995.
   - **Why part 11 could use 300:** its learner searched a family of tori, which doesn't need full coverage.
2. **"Agree" in P3's full form is agreement on a content that identifies places.**
   - **The rule:** a determined content counts as evidence for a merge only if it has never been the determined
     content of two places shown to differ (a pairwise check within each group of equal content, before any merge).
   - **Why:** read literally, two places after different first digits in A1 both have the determined content "+".
     When their observed continuations happen not to overlap, nothing ever contradicts merging them. At p = 13 and
     half the table that is about 2 expected false merges per run.
   - **The consequences:**
     - "+" fails the test, because places after different digits conflict further down;
     - a walk's unique node labels pass it;
     - with repeated labels nothing passes, so no false merges are made. The hard case stays open, as declared.
3. **Arm (d) searches this trial's remembered transitions between places identified by P3** (graph memory, as in
   part 11), without P4 or coordinates.
   - **Why:** the hippocampus's raw episode tree has no repeated places, so no way back can be searched in it at all.
   - **Arm (c)** stays the whole mechanism (P3, P4, coordinates), on this trial's steps alone.


## Part 4 results (2026-10-29; `runs/a4.txt`, `runs/a4.json`; 54 s for everything)

**Built:**
- **`bus.py`:** the one router. It stamps inputs, delivers them exactly once in step order to each column's receptive
  field, and marks a column that missed a message lost instead of letting it drift. Tested in `test_bus.py` with three
  columns sharing a field and a region that sees only the retina.
- **Efference copies:** `config.efference`.
- **Moving and sensed sources:** `Region.moves`.
- **`Modules`:** block codes with a period per block.
- **P3's full form,** with the identifying-content reading.
- **The coordinate step** (generators, orbits, the commutativity read-out, characters, the module chain, codes and
  shifts).
- **L5's route.**
- **`walk_world.py`.**

**Build check: efference copies had to become their own kind of input.**
- **The failure:** the first walk episode merged nothing (1,001 places). Every action had been stored as content of
  the place it was taken from, so two visits to one node always "disagreed": the agent took different next actions.
- **The reason:** an action is what the agent does at a place, not something the place predicts. A place's observed
  future is what the world shows for each action taken (the actions are the input alphabet, as in a Moore machine).
- **The fix:** `config.efference` names the sensors that carry the agent's own actions. They move the place, and they
  are never content: the bus marks them so for every subscriber.
- **The same distinction answers the user's warning about efference copies:** a copy reaches every column whose
  receptive field includes it, as the same SDR at the same stamp, and each column's L5 learns what it does in that
  column's own frame.

**C1–C3: A1's frames** (3 seeds each; the ring of places after "a + b"):

| p | module periods (every seed) | codes distinct | maps rebuilt from one instance | instances matched | overlap = R × 3-adic closeness |
|---|---|---|---|---|---|
| 13 | [13] | yes | 12/12 | 12/12 | — |
| 97 | [97] | yes | 96/96 | 96/96 | — |
| 105 | [3, 5, 7] | yes | 104/104 | 104/104 | — |
| 81 | [3, 9, 27, 81] | yes | 80/80 | 80/80 | 3,240/3,240 pairs |

**C4:** part 3's A1 results reproduce exactly, in all six runs (p = 13 and 97, 3 seeds).

**C5–C6: walk worlds** (1,000-step episode, unique labels, 3 seeds):

| world | places | table complete | moves P4 inferred | wrong moves | coordinates | installed shifts right | bus faults |
|---|---|---|---|---|---|---|---|
| E1 torus 6 × 10 | 60 | yes | 6, 4, 2 | 0 | one orbit of 60: [2, 2, 3, 5] | 240/240 | 0 |
| E2 cycle ±1 ±7 | 60 | yes | 5, 9, 7 | 0 | one orbit of 60: [2, 3, 4, 5] | 240/240 | 0 |
| O1 A₅ | 60 | no (in every seed) | 0 | 0 | seed 1: declined, "not abelian"; seeds 0 and 2: 30 orbits of 2, period 2 | 60/60 | 0 |
| O2 box | 60 | no | 0 | 0 | declined, "no move permutes a set of places" | — | 0 |

**C7–C9: the shortcut test** (20 trials × 3 seeds per world):

| world | shortcut trials | arm | success | length ÷ shortest | shortest way taken on shortcut trials |
|---|---|---|---|---|---|
| E1 | 0.77 | (a) carried coordinates | 1.000 | 1.000 | 1.000 |
| | | (b) carried table | 1.000 | 1.000 | 1.000 |
| | | (c) relearning | 0.133 | 1.683 | 0.043 |
| | | (d) graph memory | 0.133 | 1.683 | 0.043 |
| E2 | 0.95 | (a) carried coordinates | 1.000 | 1.000 | 1.000 |
| | | (b) carried table | 1.000 | 1.000 | 1.000 |
| | | (c) relearning | 0.217 | 3.833 | 0.070 |
| | | (d) graph memory | 0.217 | 4.194 | 0.070 |

(a) and (b) planned identically in all 120 trials.

**Verdicts:**
- **C1, C2, C3, C4: confirmed** in every run.
- **C5: confirmed.** On E2, v2 found Z/60 exactly, where part 8's torus family settled on a 15 × 15 partial model.
- **C6: confirmed on O2; refuted on O1 in 2 of 3 seeds.**
  - P4 inferred nothing on A₅, so its table stayed incomplete (178–179 of 180 moves).
  - Only the involution a came out as a full permutation in seeds 0 and 2. Its 2-cycles became 30 small orbits, each
    with exact period-2 coordinates.
  - In seed 1 a different gap left a and b both complete, and the step declined: "not abelian".
  - C6 predicted a decline assuming a complete table.
  - **What was installed is correct, not a convincing wrong model:** 0 wrong moves, and every installed shift right
    (60/60). But it is a fragment: the cosets of one generator.
- **C7, C8, C9: confirmed.**
  - Carried coordinates got home in every trial by the shortest way, wrap-arounds included.
  - The carried table planned identically.
  - Relearning and graph memory got home in 13–22% of trials (≤ 0.9 and ≤ 0.3).
  - Shortcut trials were 77% and 95%.
  - 54 s in all.

**What this shows:**
1. **Coordinates come from the moves alone.**
   - Each group's own chain of quotients comes out: the CRT factors of 105, the 3-adic digits of 81.
   - Every move becomes one fixed shift, found from a single instance.
   - In Z/81, two codes' overlap is exactly their 3-adic closeness: the p-adic metric appears in SDR overlap without
     being given (finding 3).
2. **The case that fooled part 8 doesn't fool v2,** because v2's hypothesis is any commutative group read off the
   moves, not a family of shapes.
3. **No wrong structure was installed anywhere:** 0 wrong moves in 12 walk episodes, every installed shift right, and
   no bus faults.
4. **Carried structure pays at planning, as in part 11. The codes lose nothing against the complete table.**
   Relearning equals graph memory: within a 50-step trial, P4 infers nothing at the excursion's unvisited end.
   - Part 11's relearner won 73–95% because it searched an innate family of tori. v2 has no innate geometry for an
     abstract frame, so without carried structure it can only use what it walked.
5. **What codes add beyond a complete table is C2 and C3:** a move from one instance, and quotient structure in the
   overlap. The shortcut test can't show it, by design.

**Limits:**
1. **P4 is weak on non-commutative groups:** 0 inferences on A₅. Few input pairs there share a displacement. Partial
   tables then give partial orbits (cosets of one generator) instead of a decline or the whole group.
2. **One frame for every region:** consolidation doesn't yet build a frame per receptive field.
3. **Aliased labels:** P3's full form makes no false merges with repeated labels, and so it finds no places either.
   Part 8's geometry hypotheses are the next step there.
4. **Codes for non-commuting frames** (permutation representations) are not built.

## Part 5: one fact, two uses (step 3; pre-registered 2026-10-29, before the code)

**Why (the user):** a symbol's meaning is the move it makes.
- **Its position on the axis** is where that move takes the hand from the start ("argument 1 turns the hand to
  itself").
- **Its use as an argument** is moving the hand from wherever it is ("argument 2 turns it by itself").
- **These are one fact.**

**Where A1's frame stands:** it holds them apart. The ring after "a" and the ring after "a + b" are two orbits with
separate codes, and every answer is a content memorized at its own place.

**What "one" cannot mean:** the two rings can't become one place. The place after "3" is followed by "+", the place
after "1 + 2" by "=", so P3 rightly keeps them apart. They are the same number in different phases of the equation.
So step 3 is a factorization: each place becomes (phase, coordinate). Then:
- one coordinate system covers every place;
- each moving input is one shift of it wherever it is made;
- the phases form a small automaton.

### The change (`frameLearning.py`, the column)

1. **One coordinate for every place** (a potential over the frame's moves):
   - **Shared codes:** coordinated orbits that the same displacements act on share one code (one `Modules`), each
     with an unknown offset.
   - **Each moving input is one shift δ,** the same wherever it is made (P4's regularity, extended across the frame).
     So every move (u, x, v) says c(v) = c(u) ⊕ δ_x.
   - **Naming, a hypothesis tested per input y:**
     - **The hypothesis:** "the symbol that comes next, and is determined, at the places y leads to names where you
       are". That is, its own position (where its move leads from the start) has the place's coordinate.
     - **Accepted** only if it is consistent at every place where it applies, and it pins the offsets.
     - **Rejected** otherwise, for example when the answers are relabelled.
   - **Solving:**
     - propagate from the first orbit (offset 0) through moves and accepted namings;
     - where something is still free, the fewest inputs that shift: try each undetermined input as the identity
       shift, and keep the consistent solution with the most identity shifts (MDL: a move that changes nothing costs
       nothing);
     - every move is then checked;
     - any contradiction, and the step declines (the frame keeps part 4's form).
   - **Generators, corrected:** a P4 displacement map that permutes one set of places is a generator there even if
     it is partial elsewhere. This matters when some sums are never seen.
2. **Phases:** the coordinate group's action extends to every place with a coordinate (the place reached by the same
   moves from a shifted place). Its orbits are the phases: in A1, the start, "after a digit", "after +", "after the
   second digit", "after =", "after the answer".
3. **Phase-level moves and content:**
   - **Moves:** each input's phase transition, (phase, x) → phase. It holds at every coordinate of the phase (P4 at
     the phase level).
   - **Content:** a phase's content is either one symbol for the whole phase ("+", "=") or, where naming was
     accepted, the symbol whose own position is the coordinate.
4. **The column's state becomes (phase, coordinate)** wherever the frame is factored:
   - **A move:** phase' = the transition; coordinate' = coordinate ⊕ δ_x.
   - **A prediction:** the phase's content.
   - **Places never visited are reached this way:** (after "+", 3) moved by "4" is (after the second digit, 7), even
     if no equation ever summed to 7.
   - **Where a phase transition is unknown,** the column falls back to part 4's places.

### The line between innate and learned (for the user to check)

- **New innate assumptions:**
  - **MDL:** of the solutions the data allow, the one with the fewest inputs that shift.
  - **Naming,** as a hypothesis: the determined symbol that comes next names the place's position. It is accepted only
    by consistency over every place it covers.
- **Learned:**
  - every shift;
  - which inputs shift at all;
  - the phases, and their transitions;
  - which phase's content is named;
  - where the hand starts. The start's coordinate comes out as the identity digit's position, derived, not set.
- **Never used:** a symbol's value or label. "+", "=" and "0" are found to be identity shifts, not told.

### Worlds

- **A1:** as part 3, train fraction 0.5. p = 13, 97 and 105, 3 seeds.
- **A1, answers withheld (new):** a random set H of ⌊p/3⌋ values never occurs as an answer. Every equation whose sum
  is in H is removed from training. Half of the rest is the training set; the test is every equation whose sum is in
  H. Every value still occurs as an operand. p = 13 and 97, 3 seeds.
- **A1, answers relabelled (control, new):** each answer is replaced by π(a + b), with π a fixed random permutation of
  the digits that is not a translation. Naming should be refused. Same splits as above.
- **The walk worlds:** labels are not moves, so naming never applies, and nothing should change.

### Predictions

| # | prediction |
|---|---|
| F1 | A1 (p = 13, 97, 105; every seed): every place gets a coordinate. The place after "a + b" has the same coordinate as the place after the digit z = a + b, for all p² pairs (100%). The phases number exactly 6 |
| F2 | the digit's two uses: each digit's shift from the start equals its shift from the places after "+" (every move consistent, 100%). Exactly one digit's shift is the identity: the world's "0", checked with labels in the harness only. "+" and "=" come out as identity shifts. The start's coordinate equals the identity digit's position |
| F3 | answers withheld (p = 13 and 97): accuracy on equations whose answer never occurred ≥ 0.99. Part 4's brain on the same split: ≤ 0.05 |
| F4 | answers relabelled: naming refused in every seed; the zero-shot test of F3 ≤ 0.05; ordinary held-out accuracy (answers seen, relabelled) still ≥ 0.99, since P3 and P4 don't need naming |
| F5 | no regression: part 3's M1–M4 and part 4's C1–C8 reproduce; the walk worlds' results are unchanged |
| F6 | cost: all of part 5 under 60 s |

**What would refute the step:**
- **F1 or F2 failing:** the moves aren't one shift each across phases, or MDL picks the wrong gauge.
- **F3 failing with F1 and F2 holding:** the factored column doesn't use its coordinates.
- **F4 failing:** naming is assumed, not earned.

**Not in this step:**
- chains ("a + b + c ="): they need the phase automaton to generalize, "+" after a result, which P3 and the data don't
  give;
- several operations (A2);
- numbers of several digits (A3).

### Part 5 amendments (2026-10-29, during the build, before the run)

1. **Places with no observed future are left out of the coordinate.**
   - **The problem:** P3 rightly merges every place after an answer (the episode's end) into one place. Then "answer
     z leads from the place after '=' to the end" can't be one shift per digit: the end would have to sit at every 2z
     at once. The first build therefore declined every A1 frame.
   - **The rule:** a place with no content and no move out of it has nothing for a coordinate to predict, so the moves
     into it don't constrain the coordinate.
2. **MDL is computed exactly:**
   - the constraints first eliminate every free parameter they pin down (those with coefficient ±1);
   - each group of parameters that occur together is then searched exhaustively for the most identity shifts.

   This replaces "try each undetermined input as the identity", which is order-dependent: picking digit 5 first gives
   a consistent solution with fewer identities.

## Part 5 results (2026-10-29; `runs/a5.txt`, `runs/a5.json`)

**Built:**
- **`frameLearning.factor`:**
  - one coordinate for every place, solved as affine forms over free parameters;
  - elimination of what the constraints pin down;
  - an exact MDL search over each group of free parameters;
  - naming tested per input;
  - phases (the coordinate group's orbits, with blocks of inputs);
  - phase-level moves and content.
- **`coordinate` corrected:**
  - a map that permutes some cycles counts as a generator there even if partial elsewhere;
  - orbits of one group share one code.
- **The column tracks (phase, coordinate) beside its place,** and predicts from it where the place has no content.
- **Two world variants:** answers withheld, and answers relabelled.

**F1–F2: one fact, two uses** (A1, every seed):

| p | factored | phases | after "a + b" = after the digit a + b | each digit's shift as argument = its shift from the start | identity shifts found | naming accepted | the start is "0"'s position |
|---|---|---|---|---|---|---|---|
| 13 | yes | 6 | 169/169 | 169/169 | "+", "0", "=" | "=" | yes |
| 97 | yes | 6 | 9,409/9,409 | 9,409/9,409 | "+", "0", "=" | "=" | yes |
| 105 | yes | 6 | 11,025/11,025 | 11,025/11,025 | "+", "0", "=" | "=" | yes |

Every place with an observed future got a coordinate.

**F3–F4: answers never seen as answers, and the relabelling control** (3 seeds each, identical in every seed):

| world | p | step 3 | part 4's brain | namings accepted |
|---|---|---|---|---|
| answers withheld | 13 | 1.000 | 0.000 | "=" |
| answers withheld | 97 | 1.000 | 0.000 | "=" |
| answers relabelled | 13, 97 | 1.000 | 1.000 | none |
| relabelled and withheld | 13, 97 | 0.000 | 0.000 | none |

**F5: no regression.**
- **Part 3:** reproduces exactly (p = 13 and 97, 3 seeds).
- **Part 4:** E1, E2 and the shortcut test (all 120 trials) reproduce exactly. Three differences, explained below.

**Verdicts:**
- **F1, F2, F3, F4: confirmed** in every seed.
- **F5: confirmed,** with the differences below.
- **F6: refuted.**
  - Part 5's runs took 102 s, against 60 s, plus 63 s for the regression.
  - The checks dominate: F1–F2 walk all p² prefixes twice per seed, about 20,000 walks at p = 97.
  - The brain's own training and sleep at p = 97 take about 3 s.

**Part 4's differences, found by this run:**
1. **A bug in part 4's C2 checker.**
   - **What it did:** the checker took the first orbit whose code layout fit, which was the ring after "a", not the ring
     after "a + b". So it checked one coincidental instance per map: "12/12" at p = 13 was 12 maps, one instance each.
   - **Now:** both rings share one code, and the same check covers every instance: 156/156, 9,312/9,312,
     10,920/10,920 and 6,480/6,480, all matched.
   - **So:** C2's conclusion holds, but part 4 had verified it on far less than it claimed.
2. **O1 (A₅) now declines in every seed:** "its moves don't connect its places". The pre-registered generator
   correction made the involution's 2-cycles part of one connected set, which the only complete generator can't
   connect. Part 4's C6 prediction (decline) now holds in all seeds, where it held in one.
3. **O2 (the box) declined without a reason.**
   - **Why:** a blocked move maps a place to itself, a cycle of length 1, and that counted as a generator connecting
     nothing.
   - **The fix (after the run, recorded here):** fixed points are excluded, since they carry no structure.
   - **Checked:** the box declines again with "no move permutes a set of places" in all 3 seeds. E1 is unchanged and
     the tests pass. No code was installed before or after.

**What this shows:**
1. **A digit's position and its use as an argument are one learned fact.**
   - Each digit is one shift of a single coordinate, the same from the start and from after "+".
   - The places after "a + b" sit at the coordinate of the digit a + b.
   - "+", "=" and "0" come out as moves that change nothing. The hand moves only by digits, as the user's dial has
     it.
   - The start's position comes out as "0"'s. Nothing about any of this was given.
2. **The answer names the result's position.** Naming is accepted only where it is consistent everywhere, and refused
   when the answers are relabelled.
3. **The payoff:** a learner that has seen a value only as an operand answers equations whose answer is that value,
   1.0 against 0.0. That comes from the factored state (phase, coordinate), which reaches places never visited.

**Limits:**
1. **Chains** ("a + b + c =") still need "+" after a result, a phase transition the data never show. See ARCHITECTURE.md
   §7 for what that needs.
2. **One coordinate system per frame:** the first group's.
3. **The MDL search is exhaustive per group of free parameters,** which is fine for 2 parameters (11,025 assignments
   at p = 105) and would not scale to many.

## Part 6: A2, three operations mixed, and subtraction from addition facts (pre-registered 2026-10-29, before the code)

**Why:**
- **Milestone A2:** +, − and × mod p, mixed in one stream, ≥ 0.95 on each.
- **The agreed order puts the reversal curse here,** as subtraction learned from addition facts.

**What breaks today:**
- **P4** names a displacement by a pair of inputs. Under "+", the pair (3, 5) means "2 further"; under "×", it means
  "5/3 times". Mixed in one stream, the pair's instances contradict each other, and P4 would refute the pair
  everywhere.
- **Step 3's one shift per input fails too:** under "−" a digit shifts backwards, and under "×" it doesn't shift at
  all.

### The change: generalize, then split where refuted

1. **P4 per context, on refutation.**
   - **Unchanged first:** P4 runs as before.
   - **On refutation:** a pair refuted as one displacement is tried again as one displacement per context, where a
     place's context is the input that led to it (the operator just read).
   - **Only refuted pairs are split,** so A1 and the walks are unchanged.
2. **Constant moves (absorbing elements).** An input that, from every place of one context where it was seen (at
   least 2), led to the same place, leads there from every place of that context, until refuted. This is "× 0" always
   giving 0, and it is the case where the move is determined by where it takes one place because it takes every
   place there.
3. **Coordinates per family.** The displacements of a split context are their own family, and each family's orbits
   get their own code. A place can lie in orbits of several families.
4. **Shifts per phase class, generalize then split.**
   - **First:** step 3's one shift per input everywhere is tried.
   - **Where it contradicts:** step 3's phases are grouped greedily, largest first. A phase joins the first class whose
     shifts it shares. Otherwise it starts a class of its own. If its moves contradict any shifts, it is left out of
     the coordinate (its moves are still known from P4).
5. **Relating classes: the reversal, and analogy as a change of coordinates.**
   - **The relation:** two classes in which the same inputs shift are related by one automorphism of the coordinate
     group: x ↦ k·x in each module, k a unit.
   - **Found** from the inputs seen in both classes, and **applied** to the inputs seen in only one.
   - **Accepted** only if one k holds for every input seen in both.

### The line between innate and learned (for the user to check)

- **New innate assumptions:**
  - generalize, then split on refutation, by the input that led to the place;
  - constant moves;
  - classes related by an automorphism of their coordinate group.
- **Learned:**
  - which pairs split;
  - which contexts differ;
  - the families and their codes;
  - the classes;
  - the automorphism. For subtraction it should come out as k = −1, the converse, found, not given.

### Worlds

- **A2:**
  - **The equations:** "a op b = c", op ∈ {+, −, ×}, mod p, every symbol a random SDR, the operators included.
  - **The stream:** one equation per episode, the three operations mixed.
  - **The split:** train on half of the 3p² equations, test on the other half.
  - p = 13 and 97, 3 seeds.
- **Reversal:**
  - **The equations:** {+, −}, with S a random third of the digits.
  - **Training:** every subtraction "a − b" with b ∈ S is withheld. The rest are split in half for training, as usual.
  - **The test:** every subtraction by b ∈ S. Those digits are seen as the second argument only of additions.
  - p = 13 and 97, 3 seeds.

### Predictions

| # | prediction |
|---|---|
| G1 | A2, held-out accuracy in every seed: + ≥ 0.99, − ≥ 0.99, × ≥ 0.95 |
| G2 | P4 splits the pairs shared across operators; every move it infers is correct (checked against the world's tables) |
| G3 | coordinates per family: an additive orbit of the results with the chain [p]; and under ×, an orbit of the p − 1 nonzero results with the chain of the multiplicative group, [2, 3, 4] for p = 13 and [2, 3, 4, 8, 16, 32] for p = 97 (discrete-log coordinates, found, not given; 0 is outside it, absorbing) |
| G4 | factoring with classes: the digits shift alike from the start and after "+"; after "−" they shift by the negated amounts, related by k = −1; "×" is left out of the coordinate; naming for "=" |
| G5 | the reversal: subtractions by b ∈ S ≥ 0.99. With the class relation off: ≤ 0.05 |
| G6 | no regression: part 3 (A1), part 5 (F1–F3) and the walk worlds reproduce |
| G7 | cost: training and sleep for A2 at p = 97 under 30 s per run |

**What would refute the step:**
- **G1's × below 0.95:** splitting and constant moves don't recover the absorbing 0 and the ratios.
- **G5 failing:** the relation isn't found, or isn't used.
- **G6 failing:** splitting changed the single-operation worlds.

**Not in this step:**
- chains;
- numbers of several digits;
- division;
- relations between classes that are not automorphisms (for × against +, the relation is the logarithm).

### Part 6 amendments (2026-10-29, during the build, before the run)

The build was developed on seed 0. Amendment 6 was found on seed 1 at p = 13, a pre-registered case, during the build.
No other pre-registered case was looked at before the run.

1. **The end of an episode is observed** (`hippocampus.finish`; a new innate assumption).
   - **The rule:** on each content sensor that gave nothing at an episode's last place, what came next was the end,
     one fixed SDR of the hippocampus (as every episode's start is one fixed place).
   - **The problem it fixes:** at p = 13, training held one subtraction from 0, "0 − 12". So the place after "0 −"
     had seen only "12" come next, and so had the place after "6 + 6 =", where the episode then ended. With the end
     unrecorded, the two never disagreed, and P3's full form merged them. Every "0 − b" was then answered wrong
     (− at 0.871).
   - **Unchanged:** the walk worlds (their episodes end on a retina reading, and an efference copy is never content).
   - **Part 5's leaf rule becomes:** a place whose only observed future is the end is left out of the coordinate.
2. **Every pair is split once any is refuted** (replacing item 1's "only refuted pairs are split").
   - Displacements that agree across contexts still pool through the place pairs they share.
   - A1 and the walks have no refutation, so they are unchanged.
   - **Why:** a pair not yet refuted can still mix contexts where it has few instances in one of them.
3. **Refutation is detected first, cheaply.**
   - A pair whose own instances are not one-to-one is refuted whatever else is pooled. When there is one, P4 goes
     straight to the split pass.
   - Otherwise P4 runs as before.
   - **Why:** the unsplit pass ran instance by instance through components mixing +, − and ×, to be thrown away. P4 at
     p = 97 took 39 s, about 10 s now.
4. **Across contexts, two classes pool only where they agree on at least two places.**
   - Within a context one shared place pair still suffices (P4).
   - **Why:** a translation and a multiplication can meet at one place by coincidence.
5. **A fixed point (s → s) belongs to a map, but is not evidence that two pairs are one displacement.**
   - **Why:** under ×, every pair at the place after "0 ×" maps the result 0 to itself. That joined every × class into
     one component.
6. **Constant inputs name no displacement** (in the split pass).
   - **The rule:** an input whose move is constant in a context (item 2's rule, on the moves observed) is left out of
     the pairs at that context's places.
   - **Why it follows:** where it leads is one place, whatever place it was made from. So a pair with it maps one place
     to many, or many to one, which P4 forbids.
   - **Found on seed 1, p = 13:**
     - the pair (0, 9) in the × context was seen once (after "2 ×": result 0 to result 5);
     - it was applied after "8 ×", where it filled "8 × 9" with 5, not 7;
     - in all, 14 wrong moves, and × at 0.122.
7. **The coordinate for factoring is the system whose orbits cover the most places.**
   - **Why:** at p = 13, the multiplicative orbit (12 places) was listed first, ahead of the additive one (13).
8. **Shift classes, made precise.**
   - **Order:**
     - step 3's one class is tried first;
     - otherwise phases are taken in order of the moves made from them (seen and inferred), most first, ties by the
       first met.
   - **A phase "shares a class's shifts"** if its moves and the class's solve together, with the coordinates.
   - **Naming** is tested on the moves keyed by class.
9. **The relation, made precise.**
   - **Found** for each pair of classes, as k per module among the units.
   - **If the shared inputs leave several k,** a shift is given only where every remaining k gives the same image. k is
     recorded only where one k per module remains.
   - **Applied** after the MDL solve.
   - **Side effect:** it also gives the other class shifts for "+", "−" and "=". They are never used from those phases.
10. **The solver** (`_solve`, `_choose`).
    - **Propagation is event-driven:** a move is looked at again when one of its unknowns becomes known.
    - **MDL scores** each form on the axes of its own parameters.
    - **The limit rises** from 200,000 to 2,000,000 assignments. A2 has 3 free parameters in one group: 97³ = 912,673.
    - **Checked:** part 5's recorded one_fact (p = 97 seed 0, p = 105 seed 1, p = 13 seed 2) and withheld (p = 97
      seed 0) come out identical.
11. **P4's bookkeeping, sped up.**
    - Pooled classes are resolved lazily.
    - Refused poolings are remembered: a live class's map only grows, so a conflict is permanent.
12. **G2's apparatus** (`test_a6.py`).
    - **Labels:** every frame place is labelled with the world's state, by walking training prefixes over seen moves
      only.
    - **The check:** every move of the frame, seen or inferred, is checked against the world's table.
    - **A place given two states** counts as a conflict (a wrong merge).

## Part 6 results (2026-10-29; `runs/a6.txt`, `runs/a6.json`)

**Built:**
- **The hippocampus records an episode's end** (amendment 1).
- **P4 per context** (`frameLearning.regular`):
  - refutation detected first;
  - every pair split once any is refuted;
  - two agreeing places needed to pool across contexts;
  - fixed points not taken as identity evidence;
  - constant inputs name no displacement.
- **Constant moves** (`constant`): the absorbing "× 0".
- **Coordinates per family** of displacement classes.
- **Shift classes** (`_classes`): generalize, then split by phase; a phase that contradicts every class is left out.
- **The relation between classes** (`_relate`): an automorphism x ↦ k·x per module, found from the shared inputs and
  applied to the rest.
- **The column** moves its coordinate by the shift of its phase's class.
- **`Brain(relation=False)`** turns the relation off (G5's control).

**The first attempt crashed in G6:** an IndexError in `Pool.live`, which looked up a class when none existed yet (the
box's split pass). Part 6's own runs had finished but weren't saved. The bug was fixed and the whole run repeated; the
harness now saves part 6's results before the regression. The compute was about 3 min for the failed attempt and 4.4
min for the run.

**G1: held-out accuracy** (every seed):

| p | + | − | × |
|---|---|---|---|
| 13 | 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000 |
| 97 | 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000 |

**G2: P4 splits, and every move is the world's** (identical in every seed):

| p | pairs split | moves (inferred) | wrong | places with two states |
|---|---|---|---|---|
| 13 | 162 | 585 (253) | 0 | 0 |
| 97 | 9,318 | 28,809 (14,113) | 0 | 0 |

- **The table is complete:** 585 = 13 + 39 + 507 + 13 + 13. That is the start's moves, the operators after each digit,
  every "a op b" (3p²), "=" and the answers.
- **The split:** every pair seen, 13 · 12 digit pairs plus 3 · 2 operator pairs.

**G3: coordinates per family** (every seed):
- an additive orbit of all p results, chain [p];
- a multiplicative orbit of the p − 1 nonzero results: chain (2, 3, 4) at p = 13, (2, 3, 4, 8, 16, 32) at p = 97.

The coordinate on the multiplicative orbit is the discrete logarithm, found, not given. Result 0 lies outside it.

**G4: factoring with classes** (every seed):

| p | classes | left out | k | digits alike from the start and after "+" | negated after "−" | naming | identity shifts |
|---|---|---|---|---|---|---|---|
| 13 | {S, A, after "+", C}, {after "−"} | after "×" | 12 = −1 | 13/13 | 13/13 | "=" | "+", "−", "×", "=", "0" |
| 97 | the same partition | after "×" | 96 = −1 | 97/97 | 97/97 | "=" | the same |

- **Notation:** S is the start; A is after a digit; C is after "a op b".
- **Seed 0 at p = 97:** the place after "−" came first in the greedy order, so the classes are numbered the other way:
  {after "−", A, C}, {S, after "+"}. A and C make no digit moves, so they fit either class.

**G5: the reversal** (subtractions by the digits never subtracted, seen only as added):

| p | with the relation (k found) | without |
|---|---|---|
| 13 | 1.000, 1.000, 1.000 (k = 12) | 0.000, 0.000, 0.000 |
| 97 | 1.000, 1.000, 1.000 (k = 96) | 0.000, 0.000, 0.000 |

**G6: no regression.**
- **Part 3:** reproduces exactly (p = 13 and 97, 3 seeds).
- **Part 5:**
  - F1–F2 reproduce exactly (p = 13, 97 and 105, 3 seeds);
  - F3 reproduces exactly (p = 13 and 97, 3 seeds).
- **Part 4's recorded file differs** only where part 5 already found differences, and identically:
  - **A1, C2 instances** (12 → 156 at p = 13, and so on): part 4's checker bug;
  - **A₅ in seeds 0 and 2:** now declines (part 5's generator correction);
  - E1, E2 and both shortcut tests reproduce exactly.
- **The box:**
  - **Matches part 4 exactly.** Part 5's run recorded a reason-less decline, which part 5's post-run fix (fixed points
    excluded) removed.
  - **It is the one walk with a refutation:** a blocked move maps a place to itself. It splits 12 pairs in every seed,
    against amendment 2's claim of no refutation in the walks.
  - **The split changes nothing recorded.** The box infers no move either way.

**G7: cost** (training and sleep, A2 at p = 97): 14.7 s, 16.0 s, 15.6 s (p = 13: 0.2 s).

**Verdicts:** G1–G7 all confirmed in every seed. G6's box: the frame and checks are unchanged, but amendment 2's
reasoning was wrong.

**What this shows:**
1. **One brain, three operations, one stream.**
   - **Splitting by context:** P4 splits its pairs by the input that led to the place, so the operator just read
     becomes the context of the next digit's move.
   - **The absorbing 0:** a constant move. Once 0 is excluded from displacements, every inferred move is right.
2. **The operators shift nothing.** "+", "−", "×" and "=" have identity shifts. An operator's meaning is the shift
   class it leads into, that is, how the next digit moves, not a move of its own.
3. **The converse is found.**
   - **What comes out:** a digit's shift after "−" is k times its shift after "+", with k = −1, the same in every seed
     and at both p.
   - **The payoff:** subtracting a digit that was only ever added is answered at 1.0, against 0.0 with the relation
     off.
   - **What carries it:** the relation is found from the digits seen in both classes, and carries each addition fact
     to its subtraction.
4. **× has its own coordinates, the discrete logarithm, found from the moves.** They are not used yet. × is left out
   of the factored coordinate, and its answers come from P4's table.

**Limits:**
1. **× generalizes only through P4.** Its answers come from the move table, not from a coordinate. With half the
   equations the table is complete; with fewer it may not be. Relating × to + needs the logarithm, a relation between
   classes that is not an automorphism (not in this step).
2. **The MDL search is exhaustive per group of parameters.**
   - **Today:** A2 puts 3 in one group (97³ = 912,673 assignments).
   - **Next:** a fourth class would put 97⁴ ≈ 88 million in one group, above the limit, and the factoring would decline.
   - **So:** A3 needs a better search.
3. **The cost is seconds, not milliseconds:** about 15 s per run at p = 97, about 3 s of it training. Under the
   profiler, which inflates them:
   - P4's fill loop: about 10 s;
   - the class trials: about 4 s;
   - installing 28,809 moves into L5: about 4 s.
4. **Chains are still out:** "+" after a result is a phase transition the data never show (§7).

## Part 7: chained expressions of one precedence level (pre-registered 2026-10-29, before the code)

**Why:**
- **The next step** in ARCHITECTURE.md §7, proposal A of `docs/CHAINS.md`: "a + b − c =".
- **The user (2026-10-29):** compare one chain in training with no chains in training. B (precedence) follows.

**What breaks today** (CHAINS.md §1): after "a + b", the phase automaton has no move by an operator, so the column is
lost, and there is no answer at any length above one operation. The running result is already right: it is the
coordinate.

### The change

1. **Phases merge as places do** (P3, lifted to phases).
   - **The rule:** two phases are one phase if both hold:
     - **positive evidence:** they share an input whose shift is the same from both;
     - **they never disagree:** for every input both make moves by, the shifts are equal and the phases they lead to
       are again one phase (a fold, as P3's full form folds places); for each sensor, the merged phase still predicts
       one symbol, or still names its coordinate, wherever both did.
   - **The expected effect:**
     - the start and "after +" share every digit's shift (part 6, G4), so they merge;
     - the phases their digits lead to must then be one phase: "after a digit" and "after a op b";
     - the automaton becomes a loop: digit, operator, digit, …, "=". No chain need have been seen.
2. **A collision is a merge before it is a refutation** (P4).
   - **The rule:** when a displacement would map two places to one (or one to two), the two places are first tried as
     one place by P3's fold. Only if the fold disagrees is the displacement refuted.
   - **Why:** a chain gives one result place a future its siblings lack ("a + b" followed by "−"). P3 then keeps it
     apart from the other places of the same result, and every translation through it collides. Without this, one
     chain would refute about half of the displacements of "+" (worked out in CHAINS.md).
3. **Shift classes are tried with naming.**
   - **The rule:** generalize, then correct. The classes are chosen with the naming pairs in the trials. If the final
     solve then fails, they are chosen again without them.
   - **Why:** a chain's last places are pinned only by naming (its answer names its result). Without that, the greedy
     can put them in the wrong class, and then naming fails everywhere.

### The line between innate and learned (for the user to check)

- **New innate assumptions:**
  - phases that share an input's shift, and never disagree, are one phase;
  - a collision is first tested as one place.
- **Learned:**
  - which phases merge;
  - that a result continues like a digit;
  - which operators may follow it.

### Worlds

- **Training** (both arms): A2's world restricted to {+, −}: half of the 2p² equations "a op b = c".
- **Arm "no chains":** that only.
- **Arm "one chain":** that, plus one chain "a op b op′ c = d", drawn at random once per seed.
- **Test:**
  - 1,000 random chains each of 2, 3 and 4 operations (3–5 operands), the operators random from {+, −};
  - the held-out single equations.
- p = 13 and 97, 3 seeds.

### Predictions

| # | prediction |
|---|---|
| H1 | single equations: held-out + and − ≥ 0.99 in both arms (A2's level) |
| H2 | no chains in training: chains of 2, 3 and 4 operations ≥ 0.99 (zero-shot) |
| H3 | one chain in training: chains of 2, 3 and 4 operations ≥ 0.99 |
| H4 | the mechanism: {start, after "+"} and {after a digit, after "a op b"} merge in both arms; in the one-chain arm, the chain's own phases merge into these; every move of the frame is the world's (part 6's check) |
| H5 | controls: with the phase merge off, chains ≤ 0.05 in both arms; with the collision merge off, the one-chain arm's held-out + falls below 0.99 |
| H6 | no regression in the answers of parts 3, 5 and 6 and the walks. Phase counts fall as change 1 predicts: A1 from 6 to 4, A2 from 8 to 6 |
| H7 | cost: training and sleep at p = 97 under 30 s per run |

**What would refute the step:**
- **H2 failing:** shared shifts don't identify phases, or the loop doesn't form without seeing a chain.
- **H3 failing while H2 holds:** one chain hurts. The collision merge isn't enough.
- **H6 failing in the answers:** the merges changed single-operation worlds.

**Not in this step:** precedence ("a + b × c", part B), × in chains, parentheses.

### Part 7 amendments (2026-10-29, before the code)

1. **No chain in training: the one-chain arm is dropped** (the user: "We will no longer run a single chain in the
   training").
   - **Removed with it:** changes 2 (a collision as a merge first) and 3 (classes tried with naming). Both existed
     only for that arm, and untested machinery is not built.
   - **What remains:** part 7 is change 1, the phase merge.
   - **Predictions removed:** H3, and H5's collision control.
   - **H2–H5 now read:**
     - **H2:** chains of 2, 3 and 4 operations ≥ 0.99, zero-shot;
     - **H4:** {start, after "+"} and {after a digit, after "a op b"} merge, and every move of the frame is the
       world's;
     - **H5:** with the phase merge off, chains ≤ 0.05.
2. **The control** is `Brain(phase_merge=False)`.
3. **The regression (H6)** covers:
   - part 3 (`runs/a1.json`);
   - part 5's F1–F3 (`runs/a5.json`), where only the phase count may change;
   - part 6's A2 and reversal (`runs/a6.json`);
   - part 4's walks and shortcut tests (`runs/a4.json`).

   Part 4's A1 entries are left out: their known difference is part 4's own checker bug (part 5).
4. **Part 6's harness names phases by the states they contain.** A merged phase holds several states (S and "after
   +"), so a phase can no longer be named by its first state. For unmerged frames this gives the same values.

## Part 7 results (2026-10-29; `runs/a7.txt`, `runs/a7.json`, `runs/a7_a2_phases.txt`)

**Built:**
- **`frameLearning._merge_phases`:** P3 lifted to phases. The seeds are phases of one shift class sharing an input;
  the fold is as P3's full form folds places. Contents are compared with `_kind`, now shared with `factor`.
- **`Brain(phase_merge=False)`:** the control.
- **`ArithmeticWorld.score_chains`:** chains, only ever tested.
- **`tests/test_a7.py`:** the harness.

**Found during the build, after the pre-registration:** H6's prediction that A1 goes from 6 phases to 4 was wrong.
- **What the build showed:** part 5's fast test still found 6 phases.
- **Why:** in A1 a digit is always followed by "+", and a result always by "=". Both contents are determined and they
  differ, so the fold refuses.
- **What I did:** the prediction was left as registered and is reported refuted below.

**H1, H2, H5: accuracy** (every seed, p = 13 and 97):

| | the brain as built | phase merge off |
|---|---|---|
| single equations, + and − (held out) | 1.000 | 1.000 |
| chains of 2 operations (1,000) | 1.000 | 0.000 |
| chains of 3 operations | 1.000 | 0.000 |
| chains of 4 operations | 1.000 | 0.000 |

No chain was ever trained on.

**H4: the mechanism** (every seed): the {+, −} world's 7 phases become 5.
- **What merged:**
  - the start with "after +";
  - "after a digit" with "after a op b".
- **What stayed apart:** "after −".
- **The frame:** every move is the world's, and no place holds two states.

**H6: no regression in the answers.**
- **Part 3:** reproduces exactly.
- **Part 4:**
  - the box and both shortcut tests reproduce exactly;
  - A₅ differs in seeds 0 and 2 as in parts 5 and 6 (part 5's generator correction);
  - E1 and E2 show `shifts_right` 240 against "240". This is the harness's JSON normalization turning a numpy integer
    into a string (fixed after the run), not a change: the values are equal.
- **Part 5:** F1–F3 reproduce exactly.
- **Part 6:** A2 and the reversal reproduce exactly (every recorded field).
- **Phase counts:**
  - A2 goes from 8 to 6 in every seed, as predicted (measured after the run, since part 6's records hold no phase
    count: `runs/a7_a2_phases.txt`);
  - A1 stays at 6, against the prediction of 4.

**H7: cost** (training and sleep at p = 97): 9.5 s, 9.9 s, 9.5 s.

**Verdicts:**
- H1, H2, H4, H5 and H7 confirmed in every seed.
- H6 confirmed for the answers and A2's phases; refuted for A1's phases.

**What this shows:**
1. **Chains of any length, never seen, are answered exactly.**
   - **The running result needs no store of its own:** it is the coordinate, as a grid code is the running position.
   - **The phase automaton becomes a loop, and the evidence is part 6's shift classes:**
     - the start and "after +" shift every digit alike, so they are one phase;
     - so the phases their digits lead to are one: a result continues like a digit.
2. **The generalization is only as bold as the data allow.**
   - **In A1** (one operator), what follows a digit is always "+", and what follows a result is always "=". The two
     places disagree, so P3 keeps them apart, and A1's brain would not chain.
   - **With + and −,** what follows a digit varies, so nothing contradicts the merge.
   - **This is P3's rule working as written:** a determined content is evidence. Whether a learner should override
     it ("a result is a value like any other") is a question for the user.
3. **The merge has a cost.** After "a op b" the merged phase predicts nothing, since it now expects "=" or an
   operator. The hippocampus still recalls "=" after prefixes it has lived through.

**Limits:**
1. **One precedence level only:** "+" and "−", left to right. Precedence ("a + b × c") is part B.
2. **Chains work only where what follows a digit varies** (point 2).
3. **× is not in chains:** × is left out of the factored coordinate (part 6), so "a × b + c" has no coordinate after
   "a × b".
