# E13: the recurring-worlds stream: what is episodic memory worth?

**Question (the user's plan, 2026-10-23):** start with a recurring-worlds stream, before switching the world model's
attention to Spotlight. E12 found in-world learning near-ideal (2×) but nothing kept between worlds. When a world
comes back, how much faster could an ideal learner with memory be, and how close does a simple memory get?

## Pre-registration (2026-10-26)

**The stream** (`lab/e13/e13_stream.py`):
- 8 independent streams of 48 episodes, each 512 steps of random play (as E8's data), each on a **fresh layout**;
- per stream, **12 distinct physics** (training-split physics, seeds unseen by E8), each **recurring 4 times** in a
  random order;
- a physics recurs on new maps, so what can be remembered is the physics (the action permutation, the object codes,
  the parameters), not the layout.

**Four predictors, each scored by encounter** (the 1st to the 4th time a physics is met in the stream), on the same
move and turn instances and 8-step chunks as E12:

| predictor | what it is |
|---|---|
| **ideal, no memory** | E12's exact observer of robot motion, fresh every episode |
| **ideal, memory** | the same observer, starting each episode from the stream's exact recurrence prior: a new physics with probability (copies of unseen physics left) / (episodes left), else a physics already seen, in proportion to its copies left, represented by its earlier episodes' final posteriors. It is told which earlier episodes shared a physics: an upper bound on what memory can do |
| **learner, no memory** | E8c's L-Ac (adaptation with the calibrated prior), fresh every episode, exactly as in E8 |
| **learner, library** (a hippocampus-lite) | L-Ac plus a library of every finished episode's adapted fast weights. Each chunk is predicted by the Bayesian mixture of the fresh learner and the library entries, weighted by a Chinese-restaurant prior (α = 1: a new world with probability 1 / (n + 1) after n episodes) and by each candidate's evidence on this episode so far. The fresh learner keeps adapting; library entries are frozen; no labels are given |

**Measures:**
- transitions to 95% move and turn accuracy, per encounter;
- the learners' prequential bits per step (whole episode, and the first 2 chunks);
- the ideal observers' prediction cost for robot motion;
- for the library, the posterior mass on entries with the same physics, per chunk (recall).

**Predictions:**

| # | prediction |
|---|---|
| M1 | **Memory is worth a lot to an ideal learner:** on recurrences (encounters 2–4), the ideal with memory reaches 95% move accuracy within the first chunk (8 transitions); without memory it needs 24 (as in E12) |
| M2 | **The library recovers much of it:** on recurrences it reaches 95% move accuracy within 16 transitions (the fresh learner: about 48), and its mean bits per step are ≤ 0.8× the fresh learner's |
| M3 | **False recall costs little:** on first encounters the library's mean bits per step are ≤ 1.05× the fresh learner's |
| M4 | **Recall is fast:** on recurrences the library puts ≥ 90% posterior mass on same-physics entries by the end of chunk 2 |

**Control (not a prediction):** the fresh learner's curves should be the same for every encounter, since it has no
memory.

**Commands** (from `experiments/mechint_alignment`; each stage ≤ 10 min):

    python -m lab.e13.e13_stream data
    python -m lab.e13.e13_stream ideal
    python -m lab.e13.e13_stream learner
    python -m lab.e13.e13_stream report
