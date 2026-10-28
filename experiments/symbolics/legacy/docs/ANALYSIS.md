# The symbolics folder: what holds up (2026-10-26)

This folder came out of a Gemini session (2026-10-25/26; `cipher's written summary.txt`). This analysis checks each
claim against a control: the same experiment with one component removed. The controls and their pre-registered
predictions are in `CONTROLS.md`; the raw outputs are in `runs/`. Every control script imports the original files
and changes none of them.

## The bottom line

- **The real finding is the grokking result, read properly.** The same small transformer generalizes addition mod
  105 in 88 epochs on its own. It takes 12 epochs when its inputs are given as residues mod 3, 5 and 7 (the
  Chinese-remainder coordinates), and 29 epochs when it is only asked to *predict* those residues next to the
  answer. That is a clean, cheap, measured instance of "an operation is a basis": in the right coordinates the task
  becomes three tiny tables, and generalization comes fast. The weakness is that the coordinates come from the
  experimenter.
- **The Boolean-logic result is real, and it comes from the node targets.** Supervised on every subtree's value, the
  transformer evaluates unseen expressions at 99.2% even without the tree structure; with it, 100%. Supervised on the
  answer alone it memorizes: 66.5% held-out, 72.2% with the tree structure. The "2-adic" structure adds speed, not
  the result.
- **Both successful transformers share one mechanism: auxiliary targets that expose the computation's intermediate
  structure.** These are the answer's residues in one case and each subtree's value in the other. That is the
  finding worth carrying forward.
- **Most of the "adelic" math is not adelic** (the user's suspicion, confirmed for three of four pieces). The
  embedding is genuine; I first said otherwise, which was wrong (see `CONTROLS.md`, "The math"). For Z/N, the adelic
  coordinates are the Chinese-remainder factors plus a circle, so handing them over hands over the factorization by
  construction. The "Tate character" term is wrong, and it has no gradient. The "2-adic
  ultrametric" is the tree path length. That is the genuine metric on the tree of 2-adic balls, but it is not an
  ultrametric, as it is labelled. The "adele distance" is a
  smoothness discount; the real adelic absolute value is 1 for every nonzero rational.
- **The other headline effects are smaller than reported, or come from the setup.** Adelic search: +6 points on
  random instances, not the clean wins of the four hand-built cases. Continual learning without replay: it works
  only because each task reads a different input, which makes the gate a task label. Attention residuals beating
  additive residuals: a gated model beats an ungated one on a piecewise target.

## The real versions (`REAL_ADELIC.md`)

The user asked for Gemini's proxies to be replaced with the genuine constructions (originals kept in each
subject's `gemini/` folder). Results:
- **Search:** local-global counting. p-adic local densities lift greedy subset sum from 30% to 95% on small
  instances, but fade as the numbers grow (25% for numbers up to 1,000, 0% up to 10^6). The real place hurts.
- **Grokking:** the real adelic loss gives only 1.1–1.5×. The real adelic coordinates give about 6.7× wherever N
  factors (N = 105, and N = 81 through its 3-adic digits) and nothing for the prime 97.
- **Logic:** the 2-adic ultrametric bias doesn't rescue answer-only training (72.5%). On top of node targets it is
  the fastest arm (95% by epoch 14, one seed).
- Two corrections to this analysis came out of the pass. The grokking embedding is genuinely adelic. Gemini's tree
  bias is a genuine 2-adic object (the path metric on the tree of 2-adic balls), mislabelled as an ultrametric.

## Claim by claim

| claim (from the summary's "Lessons") | verdict | evidence |
|---|---|---|
| Additive residual fitting turns O(B^d) search into O(d·B) | true as a count, and a known method: forward stagewise regression, matching pursuit, boosting. It holds when each greedy pick is right and fails when terms interact | subset sum, random planted instances: greedy success 30%; exhaustive search 100% (`search/claude/controls_adelic_search.py`) |
| Attention residuals cut the error and land closer to the truth | the comparison is between a softmax-gated mixture of terms (gate on [x, 1]) and a plain sum, on f = x² for x < 0, sin x otherwise. A plain sum cannot represent a piecewise function at all, so any gate wins. It says little about attention residuals as such | `regression/gemini/test_attention_residuals_symbolic.py`, read |
| Adelic distance is strangely powerful for search | small and real: +5.8 points on random instances (30.0 → 35.8%), +11.3 on instances built like the test cases. A shuffled discount gives nothing, so divisibility matters. Mechanism: it prefers next residuals whose primes are all in the base, a cheap proxy for one step of lookahead | `CONTROLS.md` S1 and S2 (both refuted as predicted, in opposite directions) |
| A circular/adelic loss groks modular arithmetic almost at once, with practically no train/test gap | the speed is real (8 epochs against the baseline's 88). The embedding gives about 10× of it; the residue targets alone give about 3×. "No gap" holds for every arm, the plain baseline included, by epoch 120. The "Tate" term does nothing | `CONTROLS.md` G1–G4 |
| Promising results in Boolean logic too | true, and the node targets carry it: 99.2% with them and no structure, 72.2% with the structure and no targets, 66.5% with neither (majority class 51.7%). The "2-adic" bias is a tree-distance positional bias; it speeds learning (95% at epoch 22 vs 36) | `CONTROLS.md` L1–L3 (L3 refuted) |
| Thousand Brains columns do not need replay | parameter isolation with a label read off the input; it fails as soon as two tasks share an input | `CONTROLS.md` T1 |

## The ARC part (read, not run)

- **`arc_agent/gemini/`: `column.py`, `brain.py`, `bus.py`.** The column's motor map is hand-written (ACTION1 = up, etc.). The update
  constants (+3, +4, −1, −2) are hand-set. `brain.py` assigns colour roles: the tool is colour 4 if present, the target
  colour 2 or 3. The action cost is a hand-written push heuristic: box-to-target distance plus player-to-push-spot
  distance. This is the colour-as-role mistake the bitter-lesson rule warns about.
- **`arc_agent/gemini/test_suite.py`.** Its docstring says "no test harness cheating", but tests 2 and 3 set the routing logits and
  the box physics by hand before checking that the column uses them.
- **`arc_agent/gemini/hippocampus.py`.** It is not Spotlight. Percepta's Spotlight memory is a lattice of recurrent DeltaNet cells
  with learned addresses, read and written through a sparse 3×3 neighbourhood. This file is a visit counter on world
  coordinates with a Gaussian kernel: a familiarity map. That is useful for novelty, but it stores no content.
- **`arc_agent/gemini/sleep upgrade plan.txt`.** A plan; nothing of it is built.

## Bugs and measurement problems found on the way

- `regression/gemini/test_symbolic_resnet.py:132`: the "generalization on unseen inputs" check prints the target formula, not the
  learned model's prediction.
- `neural/gemini/test_transformer_grokking.py`: the Tate term uses argmax, so it never gets a gradient. Its phase identity is also
  wrong: it is zero for only 35 of the 105 correct answers. The right CRT weights are (2, 1, −6), because
  1 = 35·2 + 21·1 + 15·(−6).
- `neural/gemini/test_transformer_discrete_logic.py`: there is no padding mask. Padding positions sit at tree distance 0 from every
  token (the distance matrix is zero there), so they get the strongest attention bias of all. They carry no answer
  information, so this is not a leak, but they act as unplanned global registers.

## What is worth keeping

1. **The coordinates result** (grokking): a 20-second experiment that measures how much a basis is worth. As
   input it gives about 10×, as a target about 3×. A natural next experiment asks whether a learner can *find* the
   factorization itself, but it is not designed yet. Even asking for the answer mod every divisor would still hand
   it the structure.
2. **Process targets** (logic): supervising every intermediate value turns a 66% memorizer into a 99% evaluator,
   on the same data and the same network. The grokking residue heads are the same lever on a smaller scale (3×). Both
   point at one question, relevant to the mechint line's data-efficiency thesis: where can such intermediate targets
   come from when nobody writes them down?
3. **The "wind tunnel" habit**: tiny symbolic testbeds that run in seconds. It is a good habit, as long as each
   comes with its baseline arm. Every headline effect changed meaning once a control was added. Adelic search
   shrank. Continual learning and attention residuals came from the setup. The two that held came from a different
   component than the one credited: coordinates and targets, not "adelic" geometry.
