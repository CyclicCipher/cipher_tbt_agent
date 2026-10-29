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
