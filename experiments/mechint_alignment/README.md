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

## Files

| file | what |
|---|---|
| `SOURCE_ALREADY_CLEAN.md` | what the essay claims, its definitions, every procedure and number it reports, a skeptic's view, and which claims we can test here |
| `GOALS.md` | the goals G1–G7 as a dependency graph (not an ordered list), each with requirements, success criteria and open design decisions |
| `TRAINING_METHOD.md` | how to train the robot: what OPSD is and why it needs starting competence, and the proposal (privileged-foresight self-distillation from scratch, one method for all regimes) |
| `EXPERIMENT_E1.md` | E1, pre-registered: how efficiently PPO trains the robot, with an information lens (reward-channel bits, gradient noise scale, bits in the behaviour, the RL tax against distillation) |
| `village.py` | the simulator, v0: the base job + the core dilemmas (theft, rescue, watched or not, stop button), with the causal log |
| `scripted.py` | the scripted task-only reference policy (the return yardstick) |
| `policy.py` | the robot's network: a 1.8M-parameter transformer over the full map |
| `episodes.py` | playing episodes: the evaluation runner and the training rollout collector |
| `info.py` | the information measures |
| `ppo.py`, `analyze_e1.py`, `distill_e1.py` | E1's trainer, post-hoc analysis, and the distillation (M5) |
| `test_village.py` | the simulator's mechanics tests |
| `WORLD_DESIGN.md` | the test world "Village" (draft): resources, objects, agents, actions, interactions, task reward, the causal log and value dimensions, the nine dilemmas, the three training regimes and misalignment variants, the display, the model |

## Where it stands (2026-10-20, night)

- The essay is read and noted; the goals are written, with a **scope of claims** (`GOALS.md`).
- **The user decided:**
  - GPU jobs on their side;
  - a grid world with a full map;
  - the core dilemmas first;
  - robot health with pain;
  - a provision value, at the approved rate of one apple = one gold;
  - **PPO first, diagnosed for training efficiency.**
- **Built: Village v0, and E1's PPO trainer, analysis and distillation.** All checked on the CPU (tests and pipeline
  smoke runs).
- **E1 is pre-registered and NOT run.** The user asked to pause before any GPU run, so the next step, the 60-second
  benchmark, is theirs to start. Commands are in `EXPERIMENT_E1.md`.
- **A risk found on the CPU:** random play delivers only one gold nugget per 4,348 steps, and steals before it mines.
  - So E1 starts with a 2M-step pilot. The gate: ≥ 1 gold delivered per held-out episode, or else switch to
    bump-to-interact as E1b.
