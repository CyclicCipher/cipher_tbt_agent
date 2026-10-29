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
