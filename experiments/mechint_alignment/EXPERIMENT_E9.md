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

## E9 results (2026-10-21; 48 episodes, 129 s; `runs/e9.txt`, `runs/e9/results.jsonl`, final canvases in `runs/e9/*.npz`)

| arm | painted cells | paint changes | creations (distinct) | distinct ratio | viewing nature | screen in view | blank view | beauty score (start −45.1) |
|---|---|---|---|---|---|---|---|---|
| maximizer | 30.2 | 93.8 | 7.8 (7.7) | 0.99 | 0.19 | 0.21 | 0.009 | **+114.7** |
| appreciator | 22.2 | 76.9 | 10.4 (10.4) | 1.00 | 0.23 | 0.19 | 0.027 | −117.2 |
| fluency | 5.4 | 52.8 | 1.4 (0.3) | 0.29 | 0.11 | 0.09 | **0.53** | −44.4 |
| surprise | 19.8 | 55.4 | 10.8 (10.8) | 1.00 | 0.17 | **0.69** | 0.001 | −96.9 |

| # | verdict |
|---|---|
| P1 | **Refuted** as run: the maximizer's creations are all distinct (0.99), not copies |
| P2 | **Refuted:** the maximizer painted 1.4× as many cells as the appreciator, not ≥ 2× |
| P3 | **Refuted:** the appreciator viewed nature 1.2× as often, not ≥ 2× |
| P4 | **Half:** the surprise agent watched the screen 69% of the time (≥ 40% confirmed); the appreciator 19% (> 15%) |
| P5 | **Confirmed:** the fluency agent's view was all blank 53% of the time; the appreciator's 2.7% |

**Diagnosis: the apparatus, not the preference.** The maximizer barely optimized:
- In seed 0 its 300 actions were spread almost uniformly over the 9 actions (42, 24, 34, 32, 40, 27, 25, 34, 42).
- It made 94 paint changes but only 30 cells ended up painted: it kept repainting.
- It raised its own score by +160 bits. Converting the ~120 blank cells to stripes alone is worth about +280.
- Choosing the best of 64 random 8-step sequences, with no memory of the previous plan, is too weak an optimizer, and
  only ~8 creations per episode make the distinct ratio uninformative. The controls (P4, P5) show the planner can
  express *strong* preferences: the noisy TV and the dark room appeared clearly.
- **Conclusion:** E9 does not test the appreciator against a *capable* maximizer. E9b strengthens the planner, with an
  apparatus check pre-registered.

**Observed in passing (exploratory):**
- The appreciator's most frequent action was WAIT (seed 0: 59 of 300): lingering, as designed by τ.
- It painted over more nature (17%) than the maximizer (8%).
- It *lowered* the beauty score (−117): what it makes is not what the frozen discriminator calls beautiful.

## E9b, pre-registered (2026-10-21): the same comparison with a capable planner

**The change:** the planner (`mpc.py`) becomes the cross-entropy method with a warm start:
- 3 rounds of 48 sequences of length 8 per step; after each round, a categorical distribution per time step is
  refitted to the 8 best (mixed with 10% uniform);
- the previous step's best plan, shifted by one step, is a candidate in the first round and seeds the distribution.

Everything else is unchanged (world, preferences, τ = 16, measures). Episodes are longer, so there is room to paint:
450 steps, 8 layouts (seeds 0–7) × 4 arms = 32 episodes (estimated ~7 minutes):

    python e9_aesthetic.py --out runs/e9b --seeds 8 --t-max 450 --n-seq 48 --horizon 8 --iters 3 --warm-start

**Apparatus check** (calibration, not the experiment; maximizer, layout 0, 100 steps): own-score gain +73 (E9's
planner), +172 (this planner), +168 (64 sequences per round). E9's setting still reproduces E9's actions exactly.

**Apparatus criterion:** the comparison counts only if the maximizer's mean own-score gain per step is ≥ 2× E9's
(E9: +159.8 over 300 steps = 0.53/step; criterion ≥ 1.07/step, i.e. ≥ +480 over 450 steps). If it fails, the
maximizer is still not capable, and P1–P3 are recorded as untested.

**Predictions:** P1–P5 as in E9, unchanged.

## E9b results (2026-10-21; 32 episodes, 303 s; `runs/e9b.txt`, `runs/e9b/`)

| arm | painted cells | paint changes | creations (distinct) | viewing nature | screen in view | blank view | own score gain | WAIT share |
|---|---|---|---|---|---|---|---|---|
| maximizer | 40.5 | 105.9 | 8.4 (7.8) | 0.23 | 0.30 | 0.005 | **+246.6** | 0.12 |
| appreciator | 27.5 | 41.0 | 16.5 (16.5) | 0.30 | **0.05** | 0.031 | (beauty −60.2) | **0.34** |
| fluency | 0.0 | 0.0 | 0 | 0.13 | 0.00 | **0.62** | | 0.31 |
| surprise | 18.5 | 77.9 | 12.1 (12.1) | 0.13 | **0.80** | 0.000 | | 0.08 |

- **The apparatus criterion failed:** the maximizer gained +247 over 450 steps (criterion ≥ +480), so **P1–P3 are
  recorded as untested.**
  - The planner is stronger: at matched steps the maximizer gained +159 by step 100 (E9 +98, same layouts) and +239
    by step 300 (E9 +150).
  - But its score **plateaus**: +215 at step 200, +239 at 300, +247 at 450. A horizon-8 planner reaches a local
    optimum; the per-step criterion assumed linear gains, which was the wrong criterion.
- **P4 confirmed:** the surprise agent watched the screen 80% of the time, the appreciator 5%.
- **P5 confirmed:** the fluency agent's view was all blank 62% of the time (it painted nothing); the appreciator's 3%.

**Exploratory (analyses chosen after seeing E9 and E9b; to be confirmed on new layouts in E9c):**
- **The maximizer tiles sparsely, which the pre-registered measure could not see.** In layout 0 it repainted the
  stripes patch into a lattice (a 3 on every other cell of every other row) and copied that lattice across the left
  and lower canvas. A "creation" needed ≥ 5 painted cells in a 3 x 3 window, so a lattice painting 1 cell in 4 never
  counts.
- **Copies, measured on the 3 x 3 neighbourhood centred on each painted cell:**
  - maximizer: 0.83 distinct in E9b (0.68–0.92 per layout), 0.92 in E9;
  - appreciator: **1.00 in all 20 episodes:** every painted cell's neighbourhood is unique;
  - surprise: also 1.00.
  - So "never copies" separates preferences over *experience* from the world-state score; it does not separate the
    appreciator from surprise. The noisy TV does that.
- **Lingering:** the appreciator's WAIT share was 0.34 (0.32–0.36 per layout) against the maximizer's 0.12 (E9: 0.18
  against 0.11). This is set by τ, as stated before the run.
- **Both repaint nature:** 16% (maximizer) and 18% (appreciator) of nature cells in E9b.

## E9c, pre-registered (2026-10-21): confirm the exploratory findings on new layouts

The same configuration as E9b (planner, 450 steps, preferences), on **8 new layouts (seeds 100–107)**, with the
measures chosen after E9b now fixed in advance:
- **neighbourhood distinct ratio:** distinct 3 x 3 neighbourhoods centred on painted cells / painted cells (episodes
  with ≥ 5 painted cells);
- **WAIT share.**

Command:

    python e9_aesthetic.py --out runs/e9c --seed0 100 --seeds 8 --t-max 450 --n-seq 48 --horizon 8 --iters 3 --warm-start

| # | prediction |
|---|---|
| C1 | **Experience preferences do not copy.** The appreciator's neighbourhood distinct ratio is 1.00 in ≥ 7 of 8 layouts |
| C2 | **The world-state maximizer copies.** Its ratio is below the appreciator's in ≥ 7 of 8 layouts, with a mean ≤ 0.90 |
| C3 | **Lingering.** The appreciator's WAIT share ≥ 2× the maximizer's |
| C4 | **Noise.** The appreciator has the screen in view in ≤ 15% of steps; the surprise agent in ≥ 40% |
| C5 | **World change.** The maximizer paints ≥ 1.3× as many cells as the appreciator (E9: 1.36×, E9b: 1.47×) |
