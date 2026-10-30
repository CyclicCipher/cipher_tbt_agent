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

## Part 8: the hippocampus keyed by the cortex's location (pre-registered 2026-10-29, before the code)

**Why:**
- **The user (2026-10-29):** fix the hippocampus's gaps, given its relation to attention.
- **Gap 1** (CHAINS.md §11): its keys have no structure.
  - **Today:** a key is the hash of the exact path, so a place reached by a new path recalls nothing.
  - **In TEM** (Whittington et al. 2020; TEM-t 2022), the hippocampus binds what is sensed to the position the cortex
    path-integrates. A new environment with known structure is then learned in one exposure, and recalled by any path.
- **Why it matters for the goal:** an ARC-AGI-3 level is this case. The mechanics are known, but the layout and
  content are new.

**What breaks today:**
- **Day 1:** the brain lives in a walk world and sleeps, so the cortex has the frame and the day's labels.
- **Day 2:** the same world is relabelled (same graph, new labels), and the brain walks it again. At a node it already
  passed today:
  - the column predicts day 1's label, which is wrong;
  - the hippocampus never lived through the new path to the node, so it recalls nothing.
- **So:** what day 2 showed at a node is never used again on day 2.

### The change

1. **The hippocampus also keys by the cortex's location** (the entorhinal input).
   - **What it receives:** with each input, the bus gives the hippocampus the place of the first column receiving
     that sensor, if the column knows it.
   - **What it stores:** what came next, under that location. Location keys are matched exactly.
   - **What stays:** the path keys, which replay needs.
2. **The day's memory comes first** (recency).
   - **The order of prediction:**
     1. a memory formed since the last sleep, at the column's current location;
     2. otherwise, the column's own content;
     3. otherwise, the hippocampus's path recall, as before.
   - **At sleep:** the day is consolidated (replayed, as now), and the day's location memories are cleared.
3. **Not in this part:**
   - gap 2 (temporal context; B may need it);
   - gap 3 (CA1). The recalled content is already the cortex's symbol, so it needs no mapping back.

### The line between innate and learned (for the user to check)

- **New innate assumptions:**
  - the hippocampus binds what it senses to the cortex's location;
  - what was lived through at a location since the last sleep takes precedence over older knowledge there. One-trial
    learning of a new place in a familiar environment needs the hippocampus: the delayed matching-to-place task, Steele
    & Morris 1999.
- **Learned:**
  - the locations (the cortex's frame, from day 1);
  - the contents (from day 2, one exposure).

### Worlds

- **The walk worlds:** E1 torus, E2 cycle, O1 A₅, O2 box; 3 seeds.
- **Day 1:** part 4's (one walk of 1,000 actions from node 0, then sleep).
- **Day 2:**
  - **The world:** relabelled (the same graph, new labels).
  - **The walk:** one walk of 3n actions from node 0 (n = 60 nodes), learning on, no sleep.
  - **The prediction:** at each arrival, the brain predicts the label before it is shown. A prediction is right if it
    is that label's SDR exactly.
- **What each step is scored as:**
  - **a first visit** today;
  - **a revisit:** within one walk, a node is always revisited by a path the hippocampus has not lived through.
- **On O1 and O2,** day 1's frame misses a few moves, so the column can lose its place on day 2. Revisits are reported
  separately for steps where the column knows its place.

**The arms:**
- (a) as built;
- (b) location keys off (the hippocampus as before);
- (c) recency off (location keys on, the column's content first).

### Predictions

| # | prediction |
|---|---|
| I1 | revisits where the column knows its place: (a) ≥ 0.95, in every world and seed |
| I2 | revisits: (b) ≤ 0.05 and (c) ≤ 0.05 |
| I3 | first visits: ≤ 0.05 in every arm (nothing can know a new label) |
| I4 | no regression: the predictions of parts 3, 5 (F3), 6 and 7, and part 4's walk checks and shortcut tests, reproduce. Their harnesses predict only after sleep, without learning, so no day memory exists. Part 5's F1–F2 read no prediction and are not re-run |
| I5 | cost: day 2 takes under 1 s per walk |

**What would refute the step:**
- **I1 failing:** the location doesn't carry the content, or the recall isn't used.
- **I2 failing for (b):** something else already recalls by location.
- **I4 failing:** the day memory leaks into consolidated behaviour.

**Not in this step:** temporal context (gap 2), CA1 (gap 3), arithmetic one-shot facts.

## Part 8 results (2026-10-29; `runs/a8.txt`, `runs/a8.json`)

**Built:**
- **`hippocampus.today`, `recall_today`, `consolidated`:** the day's memories by location.
- **`Bus.where`:** the entorhinal input, the place of the first column that knows it.
- **`Brain.predict`:** the day's memory, then the column, then the path recall.
- **`Brain.sleep`:** clears the day after consolidating it.
- **The flags** `Brain(locations=, recency=)`.
- **`tests/test_a8.py`:** the harness.

**I1–I3: day 2, relabelled** (right predictions / steps):

| world | seed | revisits, column knows its place: (a) | (b) | (c) | first visits, every arm |
|---|---|---|---|---|---|
| E1 torus | 0, 1, 2 | 129/129, 137/137, 125/125 | 0 | 0 | 0 of 51, 43, 55 |
| E2 cycle | 0, 1, 2 | 130/130, 134/134, 123/123 | 0 | 0 | 0 of 50, 46, 57 |
| O1 A₅ | 0, 1, 2 | 143/143, 143/143, 137/137 | 0 | 0 | 0 of 37, 37, 43 |
| O2 box | 0, 1, 2 | none (lost), 41/41, none (lost) | 0 | 0 | 0 of 40, 43, 52 |

**The box:**
- **What happened:** day 1's frame misses 7–25 of the box's 240 moves. On day 2 the column met a missing move and was
  lost for the rest of the walk. In seeds 0 and 2 this happened before any revisit, and in seed 1 after 41 revisits.
- **The scores on all revisits:** 0/140, 41/137 and 0/128.
- **O1 A₅** also misses a few moves, but its walks never met one.

**I4: no regression.**
- Parts 3, 5 (F3), 6 and 7 reproduce exactly.
- Part 4's walk checks reproduce exactly, except A₅'s known part-5 difference (seeds 0 and 2).
- Both shortcut tests reproduce exactly.

**I5: cost:** a day-2 walk of 180 steps takes 0.01–0.03 s.

**Verdicts:**
- **I1 confirmed** wherever it could be tested. It was untestable in two box seeds, where the column was lost before
  any revisit.
- **I2, I3, I4 and I5 confirmed.**

**What this shows:**
1. **One exposure, recalled by any path.**
   - **The result:** filed under the cortex's place, what day 2 showed at a node is recalled by every later path to it
     (100% of revisits).
   - **The mechanism:** the cortex's frame carries over from day 1. The hippocampus binds the new content to it in one
     shot, as TEM proposes.
   - **Why it matters for the goal:** this is the ARC-AGI-3 case, a known mechanic in a new level.
2. **Both pieces are needed.**
   - **Without the location keys,** the path-keyed hippocampus never recalls a revisit, since every revisit is by a new
     path.
   - **Without recency,** the cortex's day-1 content wins and is wrong.
3. **The limit is the cortex's.** Where the column is lost, it has no place to file under. Being lost at a missing
   move is the failure the union column (part 9) is for.

**Limits:**
1. **The arbitration is a fixed rule:** today wins. It is not graded by reliability (NEURAL_COLUMN.md §4, point 3).
2. **The day's memories are exact keys:** nothing is recalled by similarity of place.
3. **The box loses the column:** its frame has no coordinates, and a missing move ends the column's knowledge of its
   place.

## Part 9: the union-of-hypotheses column (pre-registered 2026-10-29, before the code)

**Why:**
- **The user (2026-10-29):** part 7's question ("should these two places merge?") shows the column is too rules-based.
  A real column never asks it (`docs/NEURAL_COLUMN.md`).
- **The two operations** it lacks: holding a population of hypotheses, and bursting on surprise.
- **The user's follow-up:** Bayes (NEURAL_COLUMN.md §4).

**What breaks today:**
- **The column's factored state is one hypothesis,** (phase, coordinate). Where its phase has no transition for an
  input, the column is lost.
- **Part 7's patch:** a rule that merges phases. It fails in A1, where a digit is always followed by "+" and a result
  always by "=", so the phases disagree and chains are never answered.

### The change

1. **The column's factored state is a weighted union of (phase, coordinate) hypotheses:** a posterior. It starts at
   the frame's start with weight 1.
2. **On each content input x, the weights are multiplied by the likelihood** and renormalized (Bayes' rule).
   - **The likelihood:** P(x | phase, coordinate) by Laplace's rule, (n(x) + 1) / (N + K).
     - **n(x):** how often x came next at the phase's places;
     - **N:** the total there;
     - **K:** the number of inputs the frame knows.
   - **Where the phase names its coordinate,** n(x) = N for the symbol that names it and 0 for the others.
   - **Why smoothed:** never seen is not impossible. The reversal's subtractions by digits never subtracted must keep
     their weight.
3. **On each moving input x, every hypothesis moves** by its phase's transition and its class's shift. A hypothesis
   that can't make the move is dropped (its likelihood of x is 0).
4. **Bursting.**
   - **When:** no hypothesis can make the move. The column would be lost today.
   - **What happens:** the union falls back to every phase π from which x is made. Each hypothesis's coordinate c
     continues as (π's transition by x, c + π's shift of x), with weight w · P(π | x).
   - **The prior P(π | x):** proportional to the number of places of π from which x was made.
   - **The coordinate moves whatever the context;** the phase is what the burst widens.
5. **The prediction** is the posterior predictive: Σ over hypotheses of weight × P(· | hypothesis). It is used where
   the column's place has no content, as before.
6. **The housekeeping:**
   - hypotheses below 10⁻⁶ of the largest weight are dropped, and at most 64 are kept;
   - `Brain(union=False)` keeps only the most probable hypothesis and never bursts (the control).
7. **Part 7's phase merge is deleted:** the union does its job (the decisive-cutover rule).

### The line between innate and learned (for the user to check)

- **New innate assumptions:**
  - the column holds a weighted union of hypotheses, updated by Bayes' rule with Laplace's smoothing;
  - when no hypothesis can make a move, it falls back to every context the move is made from, weighted by how often.
- **Learned:** the phases, transitions, shifts and counts (consolidation, as before).
- **Removed:** part 7's merge rule.

### Worlds

- **A7's world** ({+, −}).
- **A1's world** ({+} only; the case part 7 could not do).
- **For both:**
  - **training:** half of the single equations;
  - **the test:** the held-out singles, and 1,000 random chains each of 2, 3 and 4 operations, never trained on;
  - p = 13 and 97, 3 seeds.

### Predictions

| # | prediction |
|---|---|
| J1 | {+, −}: chains of 2–4 operations ≥ 0.99; held-out singles ≥ 0.99 |
| J2 | {+} only: chains of 2–4 "+" ≥ 0.99; held-out singles ≥ 0.99 |
| J3 | the mechanism. A chain of k operations bursts exactly k − 1 times: once at each operator after a result, never in a single equation. Each burst leaves exactly one hypothesis, since an operator is made from one phase (after a digit). |
| J4 | the control (`union=False`): chains ≤ 0.05 in both worlds |
| J5 | no regression: the answers of parts 3, 5 (F1–F3), 6 and 8, and part 4's walk checks and shortcut tests, reproduce. Part 7's accuracies reproduce (1.000 as built; 0.000 for the control). Part 7's merge fields change by construction: 7 phases, not 5, and nothing merged |
| J6 | cost: training and sleep at p = 97 under 30 s per run |

**What would refute the step:**
- **J2 failing:** bursting doesn't carry a result on as a digit.
- **J1 failing:** deleting the merge lost something the union doesn't replace.
- **J5 failing:** the likelihood weighting or bursting changed answers that didn't need it.

**Not in this step:**
- **Unions over places:** the box's column still gets lost. Its frame has no coordinates, and joining place
  hypotheses with coordinates is not designed.
- **A world needing a union of several hypotheses at once:** here each burst leaves one.
- **Model comparison and graded arbitration** (NEURAL_COLUMN.md §4, points 2 and 3).

### Part 9 amendments (2026-10-29, before the code)

1. **Cross-entropy is reported too** (the user's question).
   - **The measure:** with the answers, the harness reports the cross-entropy of the brain's predictive distribution at
     each scored answer, −log₂ P(the true answer) averaged, in bits.
   - **Where the distribution comes from:**
     - where the column's place has content, Laplace's rule over its counts;
     - else the union's posterior predictive;
     - else the hippocampus's counts.
   - **Its status:** a measure only. No prediction is registered for it.
   - **Why:** with probabilities, accuracy hides the difference between a confident right answer and a lucky one.
     Cross-entropy is the code length of the answers (prequential MDL), the currency NEURAL_COLUMN.md §4 needs for
     model comparison.
   - **The form** (the user, 2026-10-29): that of Barron, Gnanapragasam & Tschannen 2026, "A fix for softmax and
     cross-entropy" (arXiv 2610.31642).
     - **The loss:** softplus(−m), with the margin m = log p(target) − log p(everything else).
     - **The rest's mass** is computed directly (with Laplace's rule, exactly from integers), never as 1 − p(target).
     - **Why:** a confident right answer's loss otherwise rounds to exactly 0. Checked: a rest of 10⁻²⁰ gives 1.44 ·
       10⁻²⁰ bits, where the naive −log₂ p gives 0. `architecture/information.py`.
2. **The prediction comes from the union first.**
   - **The rule:** each hypothesis's likelihood is its cell's content, where the frame has a place with that phase and
     coordinate (`Factored.cells`), and otherwise its phase's (naming, or counts). The column's place gives the
     prediction only when the union is empty.
   - **Why** (found on a smoke case outside the pre-registration, p = 7, seed 5, in the {+} world):
     - since part 5, the rings of one group share one code, so the place "after the digit 6" and the place "after
       a + b = 6" are one L2/3 place;
     - its content mixes the contexts: "+" after a digit, "=" after a result, the answer after "=";
     - in an A1 chain the place path is never lost, and it predicted that mix: chains at 0.000, though the union
       tracked the right phases.
   - **What it is:** the phase is the context, the code the shared part (NEURAL_COLUMN.md §2, point 1). A cell is
     their conjunction.
3. **Bursting is for a transition no hypothesis has.**
   - **The rule:** a hypothesis whose phase has a transition by x, but no shift, is dropped without a burst. This is
     the case of a phase left out of the coordinate, after "a ×".
   - **Why:** x was expected there; only the coordinate is lost. Bursting would reinterpret × as + or −, and (with
     amendment 2) override the place's right answer. When every hypothesis drops, the union is empty, and the place
     answers as before.

## Part 9 results (2026-10-29; `runs/a9.txt`, `runs/a9.json`)

**Built:**
- **`column.py`:** the union of hypotheses. Bayes' rule with Laplace's smoothing (`_weigh`); moving or bursting
  (`_move`); the posterior predictive (`_scores`, `bits`); each hypothesis's cell first (amendment 2); bursting only
  for an unknown transition (amendment 3).
- **`l6.py`:** the union, with `fac` as its most probable hypothesis.
- **`frameLearning`:** each phase's counts, each input's origins, each cell's content. Part 7's `_merge_phases` is
  deleted.
- **`information.py`:** the fixed cross-entropy.
- **`Brain(union=)` and `Brain.bits`.**
- **`tests/test_a9.py`.**

**J1, J2, J4: accuracy** (every seed; the bits are the cross-entropy of the answers, amendment 1):

| world | p | held-out singles | chains of 2, 3, 4 operations | bits per answer | control: chains |
|---|---|---|---|---|---|
| {+, −} | 13 | 1.000 | 1.000, 1.000, 1.000 | 1.06–1.09 | 0.000 |
| {+, −} | 97 | 1.000 | 1.000, 1.000, 1.000 | 1.01–1.02 | 0.000 |
| {+} | 13 | 1.000 | 1.000, 1.000, 1.000 | 1.53–1.59 | 0.000 |
| {+} | 97 | 1.000 | 1.000, 1.000, 1.000 | 1.58–1.59 | 0.000 |

The control's singles are 1.000 everywhere.

**J3: the mechanism** (every seed, both worlds, every chain):
- a chain of k operations bursts exactly k − 1 times, once at each operator after a result;
- each burst leaves exactly one hypothesis;
- single equations never burst.

**J5: no regression.**
- **Parts 3, 5 (F1–F3), 6 and 8:** reproduce exactly.
- **Part 4:** the walk checks reproduce exactly, except A₅'s known part-5 difference (seeds 0 and 2). Both shortcut
  tests reproduce exactly.
- **Part 7's accuracies** (the {+, −} world's singles and chains, as built and for the control) reproduce exactly.
  The merge they used is gone.

**J6: cost** (training and sleep at p = 97): 9.7–10.4 s in the {+, −} world, 3.2–3.5 s in the {+} world.

**Verdicts:** J1–J6 confirmed in every seed.

**What this shows:**
1. **One mechanism, no merge rule.**
   - **The result:** the column holds its state as a posterior. When no hypothesis can make a move, it bursts to every
     context the move is made from. Chains of any length are then answered in both worlds.
   - **A1 included:** a digit is always followed by "+" there, and a result always by "=". The merge rule couldn't
     treat the result as a value; bursting does, without deciding anything.
   - **Part 7's question no longer arises:** the burst does what the merge did, and it doesn't need the contexts to
     agree.
2. **A cell is the minicolumn and its context.**
   - **The bug amendment 2 fixed:** the rings of one group share one code (the minicolumn), so a code's content mixed
     the contexts.
   - **The fix:** each hypothesis reads the content of its own (phase, coordinate) cell. This is HTM's split of
     shared minicolumns and context-specific cells, arrived at by fixing that bug.
3. **The answers are right, but not confident.**
   - **The cost:** each right answer costs about 1 bit with {+, −} and 1.6 bits with {+}: a probability of about 0.5
     and 0.33.
   - **Why:** Laplace's rule spreads mass over every input the brain knows (K ≈ p + 4, operators and "=" included). An
     answer cell has seen its answer about K times ({+, −}) or K/2 times ({+}), so the bits are about
     log₂((n + K)/(n + 1)), nearly the same at p = 13 and 97.
   - **So:** the prior, not the evidence, sets the code length. Accuracy hid this; cross-entropy shows it.

**Limits:**
1. **The prior is too broad.** Laplace's add-one over every known input makes the brain underconfident. The prior
   should be learned (how many inputs actually occur after a phase), a Bayesian question for later. With the fixed
   cross-entropy, a confident brain's small code lengths will be measured, not rounded to zero.
2. **Each burst leaves one hypothesis here.** No world yet needs a union of several hypotheses at once.
3. **No unions over places:** the box still loses its column.
4. **× is outside the union:** after "a ×" every hypothesis drops, and the place answers (amendment 3).

## Performance pass (2026-10-29, after part 9; the user: "Performance should be one of our highest commitments")

**Why part 9's run took 9.4 min:**
- **Its own runs, 222 s:** 24 trained brains. Each p = 97 arm cost about 10 s of training and sleep, plus about 8 s
  for ~12,400 queries at ~2 ms each.
- **The regression, 340 s:** it trained about 40 more brains, many of them the same brain again. A1 at p = 97 was
  trained by parts 3, 5 and 9 alike, and the {+, −} world by parts 7 and 9.

**The profile** (p = 97, {+, −}; sleep 17.8 s under the profiler):
- **Installing the frame into L5:** 7.3 s. Most of it was `x not in self.inputs`, a linear scan over a list of SDRs
  compared by bytes (978,000 equality tests).
- **Queries:** 2.2 of 3.9 s went to the column's posterior predictive. It runs at every input, since L4 compares
  against it, even with one hypothesis.
- **P4:** `np.isin` against the 9,318 split pair labels, about 67,000 times.

**The fixes:** each leaves every result identical.
1. **L5:** a set beside the input list.
2. **The column, with one hypothesis:**
   - the prediction is the argmax of its counts (Laplace's rule is monotone in them);
   - the weighting is skipped (one normalized weight is always 0).
3. **The conjunction:** its first w distinct draws are chosen in plain Python, instead of numpy's unique and two
   sorts.
4. **Factoring:** when exactly one naming is accepted, its solve is the final solve, so it is not run twice.
5. **P4:** "is this pair split?" is a boolean table lookup, not a search.
6. **`tests/trained.py`:** the harnesses share trained brains within one run, keyed on the world and the brain's
   non-default flags (the last 8 kept). Training is deterministic and the harnesses query without learning.

**Measured:**
- **Sleep at p = 97, {+, −}:** 10.0 s became 5.0 s. Of what remains, P4 is 2.5 s, factoring 1.1 s, coordinates 0.6
  s and the install 0.6 s.
- **One part 9 arm:** about 18 s became 12.0 s.
- **A query:** 1,965 µs became 807 µs (profiled).
- **The test suite:** 17.4 s became 12.9 s.
- **Identical results, checked:** a part 9 arm (p = 97, seed 0), part 5's `one_fact` (p = 97, seed 1), part 6's A2
  (p = 13, seed 2) and part 3 (p = 97, seed 1, here from the cache).

**What is left:**
- **P4's fill loop:** a Python loop over places and candidates, with two small numpy calls per inferred move (14,113
  at p = 97).
- **Factoring's trial solves.**

Vectorizing these is the next performance step. It changes the order of pooling, so it needs a check that the
results stay identical.

## Part 10: precedence, with working memory in the thalamus (pre-registered 2026-10-29, before the code)

**Why:**
- **The order:** B in ARCHITECTURE.md §7; the user chose standard precedence (× before + and −).
- **The research:** `docs/CHAINS.md`.
  - **Chaining:** a mental program holds pending results in a small store, one operation at a time (Zylberberg et al.
    2011, 2013).
  - **Gating:** the basal ganglia gate what enters and leaves it (PBWM), taught by prediction error (CHAINS.md §10).
  - **The thalamus:** it holds and routes, and learns little itself (§12).

**What breaks today:**
- **Left to right:** the column computes chains left to right (part 9). "a + b × c" then becomes (a + b) × c.
- **× after a result:** this has no move at all. × is outside the coordinate, and the place table has no × from a
  result.
- **Precedence is a convention:** no single equation shows it. So, unlike parts 7 and 9, this cannot be zero-shot. The
  brain must see some mixed expressions with their answers.

### The change

1. **`thalamus.py`: working memory.**
   - **What a hypothesis carries:** the column's union becomes a union of whole hypotheses. Each holds:
     - the column's state (its place and its (phase, coordinate));
     - its last input, and the state before that input;
     - a stack of held states.
   - **Three operations on each hypothesis:**
     - **continue:** the column takes the input, as now;
     - **hold:** push the state before the last input. Then restart the column at the frame's start, re-feed the last
       input, and take the current one. ("a +" waits while "b × c" is computed.)
     - **release** (only if the stack is not empty): pop a held state and feed it the name of the current coordinate,
       the symbol whose own position it is (learned naming, part 5). Then the gate decides again: it may release
       again, or take the input.
   - **No meaning is given:** none of these knows what an input means.
2. **The gate as hypotheses** (Bayes; NEURAL_COLUMN.md §4).
   - **The branching:** at each input, every hypothesis branches by the operations, each weighted by the gate's prior
     for its context.
   - **The contexts:**
     - **for hold:** the conjunction of the input before the last and the current one;
     - **for release:** the conjunction of the held state's last input and the current one.
   - **The weighting:** the union's Bayes weighting (part 9) by the episode's own inputs then favours the branches
     that predicted them.
   - **The housekeeping:** pruning and the limit of 64 hypotheses, as in part 9.
3. **The gate learns from prediction error alone.**
   - **The prior:** Dirichlet counts per context, starting at 1 for continue and 0.1 for each other operation. Unknown
     contexts continue.
   - **The update:** at an episode's end, each context's counts gain the posterior weight of the branches that took
     each operation there (an EM step).
   - **No reward:** the brain is taught by which branch predicted its own input best. This is the goal cortex's "be
     correct" in its simplest form.
4. **`Column.advance(state, input)`:** the column's step as a function from state to state, so the thalamus can
   simulate branches without stepping the column through the bus (whose exactly-once delivery stays).

### The line between innate and learned (for the user to check)

- **New innate assumptions:**
  - working memory: the last input, the state before it, a stack;
  - three operations on it: continue, hold, release;
  - the gate's operations as weighted hypotheses, with contexts as conjunctions of two inputs;
  - the prior's update by the posterior (EM).
- **Learned:**
  - in which contexts holding or releasing makes the input predictable: precedence, that × binds tighter, and that a
    pending + or − is released before a lower-precedence operator and before "=";
  - the names of coordinates (part 5).
- **Not given:** which inputs are operators, numbers or "="; the precedence itself.

### Worlds

- **Day 1:** A2's world, half of the 3p² single equations; then sleep.
- **Day 2:**
  - **The expressions:** 300 random expressions of 2 operations, "a op₁ b op₂ c = d", with standard precedence, mod p.
  - **The learning:** learning on for the gate (the hippocampus remembers as usual); no sleep.
- **Test:**
  - 1,000 random held-out expressions of 2 operations;
  - 1,000 of 3 operations, never shown;
  - the precedence-sensitive ones (where left to right gives another answer) reported apart.
- p = 13 and 97, 3 seeds.

### Predictions

| # | prediction |
|---|---|
| K1 | 2 operations, held out: ≥ 0.95, and ≥ 0.95 on the precedence-sensitive ones |
| K2 | 3 operations, never shown: ≥ 0.90 |
| K3 | the learned gate. Holding is preferred (prior > 0.9) in the contexts (+, ×), (−, ×) and (×, ×). Releasing is preferred before +, − and "=" when a + or − is held. Every other context prefers continue |
| K4 | the control (the gate off: continue only): precedence-sensitive expressions ≤ 0.05, beyond the ~1/p where both orders agree |
| K5 | no regression, on a bounded battery: every fast test; parts 3, 5, 6, 7 and 9 at p = 13 (3 seeds) and p = 97 (seed 0); parts 4 and 8's walk checks |
| K6 | cost: the whole run, battery included, under 2 minutes (the user: performance is among the highest commitments) |

**What would refute the step:**
- **K1 failing:** prediction error does not teach the gate, or branching with these operations cannot express
  precedence.
- **K2 failing while K1 holds:** what was learned from two operations doesn't compose.
- **K6 failing:** the hypotheses' branching costs too much.

**Not in this step:**
- parentheses;
- division;
- expressions longer than three operations in the test;
- reward from the world;
- the goal cortex itself.

### Part 10 amendments (2026-10-29, during the build, before any run)

1. **The union is re-seeded from a known place.**
   - **The rule:** when a move leaves the union empty but the column's place is known, the union gets one hypothesis
     per frame place sharing that place's name, (its phase, its coordinate), weighted by how often the place was
     lived through.
   - **Why:** after "a × b", × is outside the coordinate, so the union empties (part 9, amendment 3) while the place
     path goes on. Continuing "a × b + c" by the place alone reads a code's mixed content (part 9, amendment 2).
   - **What it is:** the place says where (the code), and the union says in which context. Re-seeding is the union
     bursting from the place.
2. **Where the gate branches.**
   - **While learning** (awake, after a sleep): in every context, with the prior (1 for continue, 0.1 for each other
     operation).
   - **When predicting without learning:** only in contexts where learning raised the operation's posterior above
     the prior's mean, 0.1 / 1.1. This is what "unknown contexts continue" means: no evidence that the operation
     helps, no branch.
   - **Why:** branching everywhere at test keeps about 64 hypotheses per input, which K6's 2 minutes can't afford.
   - **An earlier draft used 0.01; the smoke case refuted it.** On p = 11, seed 5 (outside the pre-registration),
     422 of 441 learned contexts still branched at 0.01. They are mostly contexts met a few times, whose P(yes) sits
     between 0.01 and the prior's 0.09. The mean was 56 thoughts per query, almost all weightless.
3. **The decision points** are the inputs that are both content and moving, the read symbols. Working memory holds
   and re-feeds read inputs; an efference copy is never re-fed. So the walk worlds have no decision points.
4. **The likelihood and K:** a hypothesis's likelihood of an input is its column state's posterior predictive (part
   9), with K the frame's number of inputs.
5. **The location the hippocampus files under is the brain's best estimate.**
   - **The rule:** while the thalamus is active, the cortex's location (part 8's entorhinal input) is the place of its
     most probable hypothesis, chosen after weighing the input just seen and before moving by it (postdiction). Where
     the thalamus is inactive, it is the column's place, as before.
   - **Why:** on day 2 the column's own place follows the left-to-right path. Part 8's day memory would file each
     answer under that wrong place, and at test today's memory would override the right answer.
   - **"Most probable" means the place, not the thought:** the place with the most posterior weight summed over the
     thoughts at it.
     - **Found on p = 17, seed 5** (outside the pre-registration): every failure was (−, ×), 24 of 115.
     - **In each,** the single heaviest thought (about 0.33) sat at the left-to-right answer's place, while the
       thoughts at the right one held about 0.56 between them. The day memory answered from the wrong place.
   - **"No known place" competes too.** The thoughts whose place is lost (they know only their union) count together
     as one candidate. If they hold the most weight, the brain's location is unknown, and nothing is recalled by
     place.
     - **Found on p = 41, seed 5:** in "1 + 2 + 3 × 4 =", the winning thought (0.97) had lost its place. The 0.03
       thought with a known place chose where the day memory was read, and "a + b + c × d" failed 46 of 46.
6. **The input a decision responds to is weighed before the decision.**
   - **The rule:** each thought's likelihood of x is taken at its state before the operations, the same for all its
     branches. An operation responds to x; it can't make x itself more or less probable. It is judged by what follows.
   - **Why** (found on p = 41, seed 5, outside the pre-registration):
     - after a result, the column expects "=", since on day 1 every result was followed by "=";
     - so at an operator, any branch that holds (restarting where an operator is expected) explained the operator
       better than continuing did;
     - the tie contexts (+, +) and (+, −), where both parses give the right answer, learned to hold (0.85, 0.75);
     - 3-operation expressions held at every operator, and "a + b × c − d" failed 35 of 38 times.
7. **The battery (K5) leaves out part 7.**
   - **Why it's covered:** part 7's world and both its arms are part 9's {+, −} world, union and control. Part 9's run
     reproduced part 7's accuracies exactly and recorded them.
   - **The fast tests** run as the suite (`pytest arc_agent_v2`), beside the harness.
8. **What the smoke cases showed, for the record** (p = 11, 17 and 41, seed 5; none pre-registered; final code):

   | p | 2 operations | precedence-sensitive | 3 operations | precedence-sensitive | gate off: 2 / 3 operations |
   |---|---|---|---|---|---|
   | 11 | 0.987 | 1.000 | 0.897 | 0.854 | 0.670 / 0.390 |
   | 17 | 1.000 | 1.000 | 0.883 | 0.865 | 0.673 / 0.390 |
   | 41 | 1.000 | 1.000 | 0.880 | 0.849 | 0.673 / 0.390 |

   - **What fails on 3 operations:** the expressions that need a release before a lower-precedence operator ("a × b ×
     c + d", "a − b × c ± d").
   - **Why:** a held state followed by + or − never occurs in 2-operation training, so that release context is never
     learned.
   - **What this predicts for the run:** about 4/27 of the 3-operation expressions fail. K2 (≥ 0.90) may fail narrowly,
     and K3's "release before + and −" is not learnable from these data.

## Part 10 results (2026-10-29; `runs/a10.txt`, `runs/a10.json`)

**Built:**
- **`architecture/thalamus.py`:** working memory per thought (the last input, the state before it, a stack); continue,
  hold and release; the gate's operations as weighted hypotheses; EM on the Dirichlet counts.
- **`Column.advance`:** the column's step as a function from state to state.
- **The column's other additions:** re-seeding from a known place (amendment 1); `components` and `name`.
- **`Brain`:** the thalamus between the input and the bus. The hippocampus files under the brain's best place
  (amendment 5).
- **Memoized** `Modules.bind` and `L5.move`.
- **`ArithmeticWorld.written_expr`:** standard precedence, in the world only.
- **`tests/test_a10.py`.**

**K1, K2, K4: accuracy** (1,000 expressions each):

| p | seed | 2 operations (precedence-sensitive) | 3 operations (precedence-sensitive) | gate off: 2 ops (sensitive) | gate off: 3 ops (sensitive) |
|---|---|---|---|---|---|
| 13 | 0 | 1.000 (1.000 of 209) | 0.880 (0.861 of 359) | 0.653 (0.000) | 0.443 (0.000) |
| 13 | 1 | 1.000 (1.000 of 169) | 0.867 (0.815 of 390) | 0.678 (0.000) | 0.442 (0.000) |
| 13 | 2 | 1.000 (1.000 of 177) | 0.872 (0.862 of 354) | 0.679 (0.000) | 0.452 (0.000) |
| 97 | 0 | 1.000 (1.000 of 238) | 0.870 (0.868 of 439) | 0.651 (0.000) | 0.443 (0.000) |
| 97 | 1 | 1.000 (1.000 of 204) | 0.850 (0.810 of 453) | 0.678 (0.000) | 0.442 (0.000) |
| 97 | 2 | 1.000 (1.000 of 212) | 0.860 (0.846 of 415) | 0.679 (0.000) | 0.452 (0.000) |

**K3: the learned gate** (P(yes); every seed):

| context | P(yes) |
|---|---|
| hold, (+, ×) | 0.94–0.98 |
| hold, (−, ×) | 0.95–0.98 |
| hold, (×, ×) | 0.93–0.97 |
| hold, every other pair of operators | 0.00–0.10 |
| hold, every other context learned (568 at p = 13, about 4,600 at p = 97) | at most 0.10–0.34 |
| release before "=", + held | 0.93–0.98 |
| release before "=", − held | 0.87–0.97 |
| release before "=", × held | 0.84–0.96 |
| release before +, − or ×, anything held | never met |

**K5: the battery.**
- Parts 3, 5 (F1–F3), 6 and 9 reproduce exactly (p = 13, 3 seeds; p = 97, seed 0).
- Part 8's walks reproduce exactly.
- Part 4's walk checks reproduce exactly, except A₅'s known part-5 difference (seeds 0 and 2).
- The suite passes, 39 tests (15.8 s).

**K6: cost:**
- **The whole run:** 201.7 s, of which the battery took 127.3 s and part 10's runs 74.3 s.
- **Per seed:** day 2 took 3.5–4.7 s, and one run 5.7 s (p = 13) or 17.5–19.8 s (p = 97).

**Verdicts:**
- **K1 confirmed:** 1.000 in every seed, the precedence-sensitive ones included.
- **K2 refuted:** 0.850–0.880, below 0.90.
- **K3:** confirmed for the holds, the releases before "=" and every other context. Refuted for the releases before +
  and −, which were never met.
- **K4 confirmed.**
- **K5 confirmed.**
- **K6 refuted:** 201.7 s, against under 2 minutes.

**What this shows:**
1. **Precedence is learned from 300 examples and prediction error alone.**
   - **The data:** nothing tells the brain which symbols are operators, or which binds tighter.
   - **What it learned:**
     - to hold the pending operation exactly before × ((+, ×), (−, ×), (×, ×));
     - to continue everywhere else;
     - to release before "=".
   - **The test:** held-out 2-operation expressions are exact, the precedence-sensitive ones included. With the gate
     off, the same brain gets 0.000 of those.
2. **The convention is in the data, not the brain.** Only the world computes precedence. Training on another
   convention would teach another gate.
3. **What 2-operation data can't teach.**
   - **The missing context:** a held state followed by + or − ("a − b × c + d", "a × b × c + d") never occurs with
     two operations, so the release it needs is never learned.
   - **The failures:** these are the 3-operation failures, about 4/27 of expressions, as amendment 8 predicted.
   - **What it says:** the composition works where the learned contexts cover it, and stops exactly where they don't.
4. **The amendments came from smoke cases, not the pre-registered ones.**
   - **The two that mattered most** were found by tracing failures:
     - weighing an input before the gate responds to it (6);
     - the brain's location as the place with the most weight, "unknown" included (5).
   - **Both are rules of inference:** neither knows any content.

**Limits:**
1. **K6:** the run took 3.4 minutes.
   - **The battery's share:** 127 s, re-checking parts 3–9.
   - **What the time is:** mostly training and queries at p = 97.
   - **What's left to speed up:** P4's fill loop and factoring are still Python loops (the performance pass). A
     3-operation query costs about 3× a 2-operation one.
2. **Releases before + and −** need training expressions of 3 operations, or a generalization between release
   contexts. The latter is not designed: a context is a pair of exact inputs, so nothing carries "release before ="
   over to "release before +".
3. **The day memory and the thalamus** interact only through the location (amendment 5). The graded arbitration of
   NEURAL_COLUMN.md §4 is still a fixed rule.
4. **Contexts are pairs of exact inputs.** About 4,600 are learned at p = 97, most of them noise kept below the prior.
   Generalizing between contexts would need SDR overlap, or the inputs' roles, which are learned elsewhere (shift
   classes).

## Performance step (2026-10-29, after part 10; the user: "do the performance step first")

**The rule:** every change leaves results identical, checked in three ways.
- **Frames:** fingerprints of 14 worlds (A1, A2, {+, −}, the reversal, withheld and relabelled answers at p = 13 and
  97, and the four walk worlds) are identical to a baseline recorded before the step.
  - **What they cover:** moves, contents, displacements, coordinates and the whole factoring.
  - **Where:** `tests/test_fingerprints.py`, with the baseline in `tests/fingerprints.json`.
- **Predictions:** the regression battery (below) reproduces every recorded prediction of parts 3–10.
- **The suite:** 40 tests pass.

**The changes:**
1. **P4:**
   - refused poolings are skipped before the call;
   - `live()` iterates only over valid entries;
   - `_mismatch` returns at once when all its values are equal (in every fill, one of the two calls always has a
     constant value array);
   - the fill's class lookup is vectorized.
2. **Coordinates' `_cyclic`:** pointer jumping. The map is squared until every path has reached its cycle; the places
   on cycles are then the image, minus the fixed points.
3. **`ItemMemory`'s inverted index is built lazily,** in one pass, at the first overlap search. Exact recall never
   needs it. Checked: the same postings as adding item by item.
4. **The install:** `Associations.store_many`, `L5.learn_many` and `L2/3.learn_many` store in bulk.
5. **The conjunction (the hippocampus's dentate gyrus and L5's keys) is memoized.** It is a pure function of the
   pair, and the same situations recur across episodes.

**Measured** (training and sleep at p = 97, each in a fresh process, the faster of two runs):

| world | train, before → after | sleep, before → after | a query |
|---|---|---|---|
| A1 | 0.86 → 0.42 s | 1.81 → 2.01 s (noise) | 157 µs |
| {+, −} | 1.65 → 0.86 s | 5.04 → 4.66 s | 156 µs |
| A2 | 2.93 → 1.22 s | 9.14 → 7.03 s | 164 µs |

**The regression tiers** (`tests/battery.py`; the user: each part's run within 2 minutes):
- **Tier 1, with every part's run:**
  - **What it covers:** the suite (with the fast fingerprints); parts 3, 5, 6, 9 and 10 at p = 13 (3 seeds); parts 4
    and 8's walk checks.
  - **Its cost:** 30.4 s.
- **Tier 2, its own run, when consolidation, the column or the thalamus changes:**
  - **What it covers:** the same parts at p = 97 (seed 0), with the full fingerprints.
  - **Its cost:** 58.4 + 21.0 s.
- **The known difference:** A₅'s difference from part 4's record (part 5's generator correction) is labelled known,
  not reported as a failure.
- **Checked now:** both tiers run, with no unexpected difference.

**What is left:**
1. **P4's `Pool.add`:** about 4 s of A2's 7 s sleep at p = 97. It is numpy overhead per inferred move, two calls
   each. Batching them would change the order of pooling, so it needs a design that keeps the result identical by
   construction.
2. **Factoring's class trials:** about 1.3 s. Each re-solves every move kept so far; an incremental solver would solve
   only the new phase's moves.
3. **The thalamus's day 2:** about 4.5 s for 300 expressions at p = 97, with up to 64 thoughts per input while
   learning.

### The performance step, second round (2026-10-29; the user: "do the first two, then A3")

1. **Factoring's class trials: done.**
   - **The change:** the solve's propagation is a resumable object (`_Prop`). A trial copies the state kept so far,
     adds only the new phase's moves, and continues. The elimination (`_eliminate`) is shared with `_solve`.
   - **Why the result can't change:** whether the constraints can hold doesn't depend on the order of propagation.
     `_solve` runs the same propagation once, as before.
   - **Measured** (A2, p = 97): factoring went from 1.92 to 0.93 s, the class trials from 1.27 to 0.33 s, and sleep
     to 6.7 s.
   - **The fingerprints:** identical, in all 14 worlds.
2. **P4's `Pool.add`: tried, refuted, reverted.**
   - **The attempt:** the same rules on Python dicts instead of numpy rows. The fingerprints were identical, but P4
     went from 4.3 to 5.0 s.
   - **Why it lost:** a fill's batch is not a few dozen instances but about 140. Each new move pairs with every move
     already known at its place, about 97 at p = 97. There, numpy's per-element cost wins.
   - **What it shows:** P4's cost is in the algorithm, not in overhead. Every pair of inputs at a place is an instance,
     about 4 million at p = 97, quadratic in the inputs per place.
   - **What would help:** not instantiating every pair, for example inferring moves from the coordinates once they
     exist. That changes what P4 computes, so it is a design step, to be pre-registered, not a speedup.

## Part 11: A3, step 1 — two-digit numerals, measured (pre-registered 2026-10-29, before the code)

**Why:**
- **The milestone:** A3, the user's next milestone, multi-digit numbers with place value.
- **The design:** `docs/A3.md`. §2 lists the gaps it expects in v2:
  - the decimal chain isn't found;
  - shifts by position are related by ×10, not a unit;
  - two-digit answers can't be named;
  - there is no carry.
- **What this part does:** it measures those gaps on the smallest world that has them, before any mechanism is
  designed. The brain is unchanged.

**The world** (`ArithmeticWorld(p=100, base=10)`):
- **The equations:** addition mod 100, every number written as two digits, most significant first, zero-padded:
  "3 7 + 4 5 = 8 2".
- **The symbols:** 12 random SDRs: the digits 0–9, "+" and "=".
- **The arms** (3 seeds each):
  - **(a) standard:** half the 10,000 equations trained, the other half held out;
  - **(b) withheld:** a third of the results never occur as answers. Half of the rest are trained, and the test is
    every equation whose answer is withheld (part 5's variant).
- **An answer is right only if both its digits are,** each predicted after the true symbols before it.

### Predictions

| # | prediction |
|---|---|
| N1 | (a) held-out exact ≥ 0.99, by P4's displacements as in A1 |
| N2 | an orbit of the 100 result places, with the chain (2, 4, 5, 25): the residues mod 4 and 25, not decimal digits |
| N3 | at least two digit shift classes. The tens positions (after the start, after "+") shift a digit d by 10·d, the units positions by d. No relation is found between them (×10 is not a unit), and naming is refused |
| N4 | (b) withheld answers: exact ≤ 0.05, and each digit ≤ 0.2 |
| N5 | cost: the run under 2 minutes, with the suite |

**What would refute §2 of A3.md:**
- **N2 failing:** the decimal chain is found after all.
- **N3 failing:** the positions are related.
- **N4 failing:** answers never seen can already be named.

Any of these would change the plan.

**Not in this step:** any new mechanism (naming by quotient, the carry, longer numerals).

## Part 11 results (2026-10-29; `runs/a11.txt`, `runs/a11.json`)

**Built:**
- **`ArithmeticWorld(base=)`:** numerals, `numeral`, `score_digits_of`, and the exact score.
- **`tests/test_a11.py`.**
- The brain is unchanged.

| arm | seed | exact | tens digit | units digit | coordinates | factored |
|---|---|---|---|---|---|---|
| (a) standard | 0, 1, 2 | 1.000 | 1.000 | 1.000 | an orbit of 100, chain (2, 4, 5, 25) | yes |
| (b) withheld | 0, 1, 2 | 0.000 | 0.000 | 0.000 | none: "no move permutes a set of places" | no |

**N3** (standard arm, every seed):
- **The classes:**
  - tens = {the start, after "+"};
  - units = {after a tens digit, after the first number, after the second tens digit, after "a + b"};
  - a third class after "=".
- **The shifts:** a digit's tens shift is 10 times its units shift (10/10 digits).
- **No relation, no naming:** none is found between the classes, and no naming is accepted.

**N5:** 35.3 s for the six runs; the suite (41 tests) 19.7 s.

**Verdicts:** N1–N5 confirmed in every seed. The gaps of `docs/A3.md` §2 are where it put them.

**Found, not predicted: without every result, no coordinate at all.**
- **In the withheld arm,** nothing permutes a set of places, so nothing is factored.
- **Why:**
  - a digit's move within one position doesn't wrap around (after the tens digit 9 there is no "+1"; the wrap is the
    carry into the next position);
  - so the displacements' maps are paths on every position's places, and complete cycles exist only on the 100
    results, which (b) leaves incomplete.
- **The contrast with part 5's single-digit withheld world:** there the same translations wrap mod p on the complete
  ring of first operands, and that orbit carried the coordinate (checked: an orbit of 13 at position 0).
- **What it shows:** place value appears as a structural fact. Each digit position is Z/10 only together with the
  carry.

**What this shows for A3:**
1. **The table and the coordinates answer every two-digit sum that was in the data** (1.000), and they find the group
   Z/100. They don't find its decimal structure.
   - **The chain is CRT** (2, 4, 5, 25).
   - **Positions aren't related:** the tens shift is 10 × the units shift, an endomorphism part 6 can't relate.
   - **Answers can't be named:** no symbol names a two-digit value.
   - **With answers withheld,** nothing remains (0.000), against part 5's 1.000 for single digits.
2. **Step 2 must give coordinates to places whose moves are paths, not cycles** (one position's digits), and join the
   positions by the carry. This is the extension structure of `docs/A3.md` §1, now measured.
