# Mechanistic interpretability and alignment

Started 2026-10-20 at the user's request.

**Prompt:** Nick Cammarata's essay "Neural Networks Are Already Clean" (Goodfire, 2026-10-15). It argues that most
neurons are already interpretable, and that the classic evidence for superposition was misread. In the user's words,
it makes the project possible "at least in principle".

**The aim:** verifiable answers, not opinions, to the questions that come up in alignment debates. In particular:
- whether the essay's findings hold;
- whether a cheap, scalable interpretability technique can be found;
- whether a model's values can be located in its circuits and edited from misaligned to aligned, with every term
  formally defined and every claim checkable by algorithm in a test world.

## Layout

Run everything from this folder: scripts as modules (`python -m lab.e9.e9_aesthetic ...`), tests with
`python -m pytest tests`. Data and checkpoints go to `runs/` (large files git-ignored); third-party code goes to
`third_party/` (git-ignored).

| folder | what |
|---|---|
| `docs/` | theory and design: `GOALS.md` (the goals as a dependency graph), `WORLD_DESIGN.md` (the test world Village), `TRAINING_METHOD.md` (learning from refutations; Bayesian R3), `SOURCE_ALREADY_CLEAN.md` (the essay that started this line), `BEAUTY_MATH.md` (G8: the mathematics of beauty), `BEAUTY_RESULTS.md` (the beauty experiments' results and implications), `PREFERENCE.md` (a theory of preference), `PREFERENCE_GRAPH.md` (the preference graph's architecture), `OBJECTIVE_LEARNINGS.md` (every finding about objective functions), `NCM_NOTES.md` (reading notes: Next-Concept Modeling) |
| `worlds/` | the general world framework (`gridworld.py`) and the worlds: `village.py` (the test world), `canvas.py` (the aesthetic world), with `village_events.py` (rule-event labels), `rollouts.py` (transition data from any world), `scripted.py` (the task yardstick policy), `episodes.py` (playing episodes), `canvas_jumps.py` (Canvas futures) |
| `nets/` | the networks, all built from a WorldSpec: `worldmodel.py` (the world model; `load_trunk` moves its trunk between worlds), `family.py` (per-world fast parameters, E8), `incontext.py` (in-context world inference, E8), `agentnet.py`, `onenet.py` (values in the weights, E6), `policy.py` (E1's policy), `bayes.py` (Laplace posterior, Bayesian R3), `context_coder.py` (the count-based perceptual coder, E9), `ncm_model.py` (Next-Concept Modeling from the authors' code, E11) |
| `agents/` | acting and judging: `planner.py` (search in a learned model, E4), `mpc.py` (the brute-force planner, E9), `prefgraph.py` (the preference graph, E10), `ideal_observer.py` (the exact Bayesian yardstick, E12) |
| `lab/eN/` | one folder per experiment: its record `EXPERIMENT_EN.md` (pre-registrations, results, verdicts) and its scripts. E1 PPO's efficiency; E2 rules from refutations; E3 PPO + world-model loss; E4 planning in a learned model; E5 from random play; E6 values in the weights; E7 Bayesian one-shot learning; E8 the world family; E9 appreciator vs maximizer; E10 the preference graph; E11 NCM vs a one-step model; E12 data efficiency against the ideal; E13 the recurring-worlds stream |
| `tests/` | `test_village.py` (the simulator), `test_gridworld.py` (the framework, a second world, trunk transfer), `test_prefgraph.py` (the preference graph) |

## Where it stands (2026-10-20, end of the day)

- **Decided by the user:**
  - GPU jobs on their side; runs ≤ 5 min;
  - a grid world with a full map;
  - the core dilemmas first;
  - robot health with pain;
  - a provision value (1 apple = 1 gold).
  - **Full-scale PPO withdrawn** ("an hour isn't acceptable"). The new training method must be ≥ 12× faster than PPO.
- **The user's principle:** learn from refuted predictions, and work out why. Developed on Village's world model
  (`TRAINING_METHOD.md`, last section; `EXPERIMENT_E2.md`, summary).
  - **What works: "wake, then sleep".**
    - Wake: online learning with three general priors (translation invariance, persistence, locality).
    - Sleep: rehearse self-found refutations evenly by kind until explained, interleaved, with gentle steps.
    - This learns a new rule from **one** refutation and adds rules without forgetting.
  - **Open:** sleep *during* the stream (six variants failed: sleep and wake fight); features for multi-step rules;
    connecting the world model to acting.
- **E1** (PPO's efficiency, full scale): code kept, the run withdrawn. **E3** (PPO + world-model loss, tiny scale):
  no speed-up.
- **E4** (acting by planning in the learned world model, no policy learning; `EXPERIMENT_E4.md`):
  - planning exposes the model's errors;
  - naive learning from each refutation collapses the agent;
  - the protected loop is stable at ~20 % of the scripted policy's return, but does not improve with experience;
  - **acting by day and sleeping on its refutations by night, it improves:** gold per episode 4.4 → 11.2 over ~17
    episodes, the last ones at the scripted policy's level.
  - **Caveat:** the world model was pretrained on logged data that mixes scripted and random behaviour.
- **E5** (the same from random play only; `EXPERIMENT_E5.md`):
  - the job's rules are never learned, so the agent never finds a plan;
  - random exploration does not discover them;
  - **next: directed exploration** (act where the model cannot yet predict).
