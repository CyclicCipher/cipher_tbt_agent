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
