# Learnings about objective functions

**What this file is:** every finding about *objective functions* (what an agent or a model is rewarded or trained
for, how the objective is combined, and what that does to its behaviour or its outputs), with the evidence and the
theoretical implication. Created 2026-10-21 at the user's request. Add to it whenever an experiment or a source
teaches something about objectives.

**Entry format:**
- **finding:** one sentence;
- **evidence:** the experiment or source, with its status: **confirmed** (pre-registered and held), **exploratory**
  (found after the fact; to confirm), or **source** (literature, checked or from memory);
- **implication:** what it means for objective design;
- **open / caveats.**

The lab records are `EXPERIMENT_E*.md`; the theory is in `BEAUTY_MATH.md`, `PREFERENCE.md` and
`PREFERENCE_GRAPH.md`; the beauty experiments in one place are in `BEAUTY_RESULTS.md`.

## A. Fixed objectives, optimized hard

### L1. A fixed learned score is exploited, not fulfilled
- **Finding:** an agent maximizing a frozen, learned beauty score covered the canvas with isolated dots: a pattern
  no example looked like, which the score rates highest.
- **Evidence:** E9b–E9c, **exploratory** (seen in the canvases). The frozen score gives an isolated dot 15.0 bits,
  a dot lattice 8.5, stripes like its examples 4.1. 51–66% of the maximizer's painted cells are isolated dots,
  against 3% for the appreciator.
- **Implication:** this is peak shift / reward hacking. The optimizer finds the score's supernormal stimulus: the
  input on which the learned discriminant extrapolates most, here a lone coloured cell in a blank context, which the
  plain examples never contain. A learned reward model is only trustworthy near its training distribution, and a
  strong optimizer leaves that distribution.
- **Open:** does a typicality bound (score + λ·log prior, RLHF's KL term) remove the hack, or trade it for
  blandness (L3)?

### L2. A fixed, translation-invariant local score is maximized by a tiling
- **Finding:** a score that is a sum of the same local term at every position is maximized by a ground state: for
  generic scores, a periodic tiling of the locally best motif; for special rule sets, a rigid aperiodic order (Wang
  tiles, Berger 1966; Penrose matching rules).
- **Evidence:** **theory** (statistical physics of ground states); consistent with E9b–E9c, where the maximizer
  repeats itself (distinct neighbourhoods 0.76 against 1.00) and tiled a dot lattice in one layout.
- **Implication:** adding principles to a fixed score changes *which* motif is tiled, not *that* it tiles. Open-ended
  novelty needs an objective with no fixed argmax (L5).
- **Caveat:** the maximizer's planner plateaued at a local optimum; full tiling of the canvas is untested.

### L3. Maximizers sharpen a distribution; samplers preserve it
- **Finding:** optimizing a reward that contains a typicality term turns the reference distribution into a sharper
  power of itself. Iterated learning with maximizing learners amplifies biases; with sampling learners it converges
  to the learners' prior.
- **Evidence:** **source:**
  - Zhang et al. 2025 (Verbalized Sampling): reward = r_true + α log π_ref gives π* ∝ π_ref^γ exp(r_true/β), with
    γ = 1 + α/β; α̂ ≈ 0.6 measured in human preference data;
  - Griffiths & Kalish 2007 (sampling → the prior);
  - Kirby, Dowman & Griffiths 2007 (maximizing amplifies; from memory);
  - Kirk et al. 2024 (RLHF reduces output diversity).
- **Implication:**
  - Mode collapse in fine-tuned language models is the γ > 1 sharpening of a human preference for the familiar.
  - Wherever the system generates (speech, art, exploration), it should sample, not maximize.
  - A "KL to the reference" term bounds peak shift only as far as the reference itself is unbiased.

### L4. The output distribution is evidence about the objective's structure
- **Finding:** with the optimizer held identical, a fixed world-state score and a reward over the agent's own
  changing knowledge produce very different output distributions: one motif repeated, against all-different
  creations.
- **Evidence:** E9c, **confirmed** (neighbourhood distinct ratio 0.76 against 1.00 in 8 of 8 layouts). **Source:**
  the human/LLM text gap appears with instruction tuning, not pretraining (Reinhart et al. 2025, PNAS).
- **Implication:** the human text distribution is evidence about human objectives, not about how efficiently humans
  learn. Human learning is not "RL against a frozen scalar, done more efficiently". RL against a reward computed from
  the learner's and the audience's own model states is not ruled out (`PREFERENCE.md` §1).

## B. Intrinsic objectives (rewards computed from the agent's own model)

### L5. A reward over the agent's own changing knowledge has no fixed argmax
- **Finding:** an agent rewarded by the compression progress of its own view never repeats itself: a copy of what
  it has learned pays nothing.
- **Evidence:** E9c, **confirmed** (1.00 distinct in 8 of 8 layouts). The surprise agent also never repeats
  (also 1.00).
- **Implication:** non-stationarity of the reward, not the number of principles in it, is what prevents collapse
  (the prediction E10 is designed to test). This property is shared by *every* learning-dependent reward; what
  separates good ones from bad ones is L6.

### L6. Single-signal intrinsic rewards fail in the classic ways; learning progress avoids both
- **Finding:**
  - fluency alone (predictability of the view) seeks the dark room;
  - surprise alone (prediction error) seeks the noisy TV;
  - compression progress avoids both.
- **Evidence:** E9–E9c, **confirmed:**
  - fluency's view was all blank 50–62% of the time;
  - surprise watched the screen 69–81% of the time;
  - the appreciator 2.6% blank and 6% screen.
- **Implication:** the noisy TV is solved by measuring what *can be learned* (the fall of code length over time),
  not by uncertainty, and the dark room by measuring *change*, not level. Earlier lesson from the TBT line: a
  *rectified* learning-progress rate re-admits noise. Use the signed area for accounting and a prospective
  "not yet confident" level for exploration.
- **Caveat:** the appreciator's lingering time is set by its progress window τ (16 steps), not discovered.

### L7. Liking over experience outputs attention; making is not disinterested
- **Finding:** the appreciator spends a third of its time looking (lingering), but its creating consumes existing
  structure: it painted over *more* nature than the maximizer.
- **Evidence:** E9c, **confirmed** for lingering (WAIT 0.33 against 0.12). **Exploratory but consistent over three
  runs** for consuming nature (17/18/27% against 8/16/15%).
- **Implication:** Kant's disinterested liking covers the looking, not the making. Seeking new things to learn is an
  action on the world, and learned structure is its richest material. "Reward over experience" is not safe by
  itself: protecting existing beauty needs its own node (L8).

## C. Combining objectives (the preference graph)

### L8. A content edge works: preserve what was appreciated
- **Finding:** a preservation node whose content is written by appreciation protects nature and moves creation to
  blank space and to the canvas margins.
- **Evidence:** E10b, **confirmed:**
  - nature painted over 27% → 3.5% (exchange rate) and 0% (veto);
  - painting on blank 46% → 85% and 100%;
  - **exploratory:** 55% of painting on the canvas border, against 19%. A stroke there disturbs the fewest valued
    structures.
- **Implication:** one node's experience can define another node's content, so preferences acquire content from
  experience (Kant's "interest attaches afterwards"). Preservation shapes *where* the agent creates, not only how
  much.

### L9. A veto on everything valued becomes low impact; veto only the irreplaceable
- **Finding:** every act of creation destroys something. A preservation veto that protects all appreciated structure,
  including the commonest, forbids almost all creation, and the agent becomes a pure looker.
- **Evidence:** E10b, **confirmed** as a failure of the pre-registered prediction:
  - creation fell to 3% under the veto and 40% under the exchange rate;
  - one stroke on blank cost 1.6–17.7 bits against a 0.5-bit tolerance, because the start-up inflated blank's value
    to 76 bits.
- **Implication:**
  - Vetoes should act on the *extinction* of valued structure (its last instances); marginal erosion of common
    structure goes through an exchange rate (two tiers).
  - Value should account for rarity.
  - Preservation, harm avoidance and low impact are different nodes, and a badly scoped preservation node collapses
    into low impact.
- **Open:** E10c tests the two-tier design.

### L10. A fixed weighted sum of nodes is still one objective
- **Finding:** combining nodes with fixed weights gives one fixed argmax (L1–L2). Optimization amplifies any fixed
  bias term (L3).
- **Evidence:** **theory;** E10's exchange-rate arm is such a sum and still behaves as one objective.
- **Implication:** a preference graph needs at least one of the following, or it reduces to the maximizer
  (`PREFERENCE_GRAPH.md`):
  - state-dependent urgencies (satiation);
  - lexicographic vetoes;
  - non-stationary content edges.

### L11. One-shot judgments measure only the fast nodes
- **Finding:** a side-by-side rating sees fluency (immediate) but not boredom (built over exposures) or usefulness
  (per episode). A frozen reward model trained on such ratings optimizes the rising side of an inverted U forever.
- **Evidence:** **theory**, plus L3's source (the typicality bias). Not tested here.
- **Implication:** objective nodes need time scales (per-node discounts), and judging should happen over time, not
  once.

## D. Training objectives (the world-model and learning-rule lines, E2–E8)

### L12. Rationalization is overcounting a datum
- **Finding:** learning a refuted rule by repeating the refuting datum until it fits overfits and forgets. Bayesian
  updating counts the datum once against the old posterior.
- **Evidence:** E7, **confirmed** (Bayesian R3: one-refutation learning without forgetting; held-out 0.71–0.86,
  stable over 600 iterations, against plain rehearsal 0.86 → 0.18).
- **Implication:** the objective for learning from one datum is the posterior: MAP of (old posterior × the datum
  counted once), with precision accumulated by the Fisher information.

### L13. Prequential code length is the right score for a learner, and calibrating the prior pays
- **Finding:**
  - Bayesian in-world adaptation beats a one-size model on the whole-world code length, and the gap grows with
    training;
  - an empirically calibrated prior variance (closed-form EM) improves it further and removes most of the start-up
    cost.
- **Evidence:** E8b–E8c, **confirmed:** 7.80 against 15.83 bits/step on new worlds; the start-up cost fell from
  1.2–1.4 to 0.15–0.23 bits.
- **Implication:** score learners by the code length of their predictions *as they learn* (prequential), not by the
  final fit: start-up costs and adaptation speed are then in the number.

### L14. Averaging over the prior's samples did not pay
- **Finding:** posterior-predictive averaging over samples of the prior made predictions worse, in two
  configurations.
- **Evidence:** E8b and E8c, **confirmed** as refuted (10 bits worse early). The diagnostic suggests training
  through the sample mixture degraded the prior mean.
- **Implication:** a principled-looking objective term can cost more than it gives at small scale. Keep a term only
  when it beats its own cost in the measured objective.

## E. Apparatus lessons for objective experiments

### L15. How objective experiments go wrong
- **A penalty must be zero when the agent does nothing.** E10a: the noisy screen leaked into the preservation loss
  (3–5 bits per step with the agent only walking) and made the veto agent erratic. Measure every penalty with an
  idle agent first.
- **A weak optimizer hides an objective's failure modes.** E9: random shooting barely optimized, so the maximizer
  looked harmless. Check that the optimizer is strong enough before judging the objective.
- **Keep the best checkpoint when training is unstable.** E11b: a tokenizer reached 97% exact reconstruction, then
  degraded to 34%, and the run saved only the final state.
- **Summary statistics miss what the generated artifacts show.** E9b: the pre-registered copy measure could not see a
  sparse tiling, and the polka-dot hack was found by looking at the canvases. View the outputs before trusting
  any measure of them.
