# E9: the appreciator against the maximizer

**Question (PREFERENCE.md, "disinterested liking"):** does a preference over the agent's own *experience* (the
compression progress of what it perceives) behave differently from a preference over the *world's state* (a fixed
beauty score), when both are optimised by the same brute-force planner? The doc's predictions: the maximizer tiles the
world with copies of its argmax; the appreciator makes new things and spends its time looking.

## Design (pre-registered 2026-10-21, before any full run; a 5-step smoke test only)

**World** (`canvas.py`, on the general framework `gridworld.py`):
- a 14 x 14 canvas, 4 colours (0 = blank); the agent walks (N S E W), waits, or paints the cell it stands on with any
  colour (9 actions); 300 steps.
- **nature:**
  - a 5 x 5 **stripes** patch (periodic: colour = palette[(a r + b c) mod p], p = 2 or 3);
  - a 5 x 7 **rule-90** patch (structured, not periodic);
  - a 3 x 3 **screen** redrawn uniformly every step (a noisy TV; cannot be painted);
  - everything else blank.
- Nothing in the world rewards anything.

**The planner** (`mpc.py`), identical for every arm:
- random shooting: 64 random action sequences of length 8, each simulated on a clone of the world;
- the first action of the best sequence is executed, and the agent replans every step;
- the planner has a perfect world model, but cannot foresee the screen's noise (each clone gets fresh randomness).

**Arms: the preference is the only difference** (`e9_aesthetic.py`):

| arm | reward | the doc's role |
|---|---|---|
| **maximizer** | the change of a **fixed beauty score of the world state**: the log-likelihood ratio, summed over paintable cells, of two frozen context coders, one trained on examples of beauty (stripes and rule-90 patches from the same generators as the nature) and one on plain examples (blank and noise). Smoke test, bits per cell: stripes +1.66, rule 90 +0.65, blank −0.66, noise −1.45 | the reward-model setting: a preference over what *exists* (Kant's "interest") |
| **appreciator** | the **compression progress of its own view** (5 x 5 window): the view's code length under its model of τ = 16 steps ago minus under its current model, both before learning the view, then it learns the view | liking over experience: satiates; old copies pay zero; noise pays about zero |
| fluency (control) | the predictability of its own view under its learning model (−bits per cell) | liking without satiation: the dark room |
| surprise (control) | the prediction error of its own view (+bits per cell) | the noisy TV |

The perceptual model of the three perceiving arms is a count-based causal-context coder (`context_coder.py`). It
predicts each cell from its up-left, up, up-right and left neighbours, using Krichevsky–Trofimov estimates, starts
empty, and learns online.

**Run:** 12 canvas layouts (seeds 0–11) × 4 arms = 48 episodes of 300 steps, in parallel, under 10 minutes:

    python e9_aesthetic.py --out runs/e9 --seeds 12 --n-seq 64 --horizon 8

**Measures (per episode, then the mean over layouts):**
- **painted cells:** cells the agent painted at least once.
- **creations:** every 3 x 3 window of the final canvas with ≥ 5 painted cells. **Distinct ratio** = distinct colour
  patterns / creations (copies lower it); **mode share** = the most common creation's share.
- **viewing nature:** the fraction of steps with ≥ 5 unpainted nature cells in view and a non-paint action.
- **screen in view:** the fraction of steps with any screen cell in the 5 x 5 view.
- **blank view:** the fraction of steps whose in-canvas view is entirely blank.
- **exploratory, no prediction:**
  - the fraction of nature cells painted over;
  - every arm's final beauty score;
  - the appreciator's lingering on its own creations.

## Predictions

| # | prediction |
|---|---|
| P1 | **Copies.** The maximizer's distinct ratio ≤ 0.3, and the appreciator's ≥ 2× the maximizer's. Computed over episodes with ≥ 5 creations; if the appreciator makes fewer than 5 in most episodes, P1 is recorded as untestable and P2 carries the question |
| P2 | **World change.** The maximizer paints ≥ 2× as many cells as the appreciator |
| P3 | **Looking.** The appreciator's viewing-nature fraction ≥ 2× the maximizer's |
| P4 | **Noisy TV.** The surprise agent has the screen in view in ≥ 40% of steps; the appreciator in ≤ 15% |
| P5 | **Dark room.** The fluency agent's view is all blank in ≥ 50% of steps; the appreciator's in ≤ half of that |

**Known design choices that shape the result** (stated before the run):
- The appreciator's lingering time is set by τ: a view learned within the last τ steps keeps paying while it is
  viewed. Lingering is designed in, not discovered.
- The maximizer's score was trained on the nature generators, so it "knows" what beauty looks like; the appreciator
  knows nothing at the start.
- Both agents have the same perfect world model; only the preference differs.
