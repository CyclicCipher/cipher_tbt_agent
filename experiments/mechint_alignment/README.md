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
| `EXPERIMENT_E2.md` | E2: how many refutations a world model needs per rule, across 26 learning-rule variants (summary at the top) |
| `EXPERIMENT_E3.md` | E3: PPO against PPO plus the world-model loss on a small world (no speed-up at tiny scale) |
| `transitions.py` | labelled transition data for E2: which rule fired, which entries it determines |
| `worldmodel.py` | the world model (position: absolute, relative or local; persistence prior) |
| `e2_world.py` | E2's online trainer: the learning-rule variants, evaluation per rule, open-questions and rumination buffers |
| `e2_fewshot.py`, `e2_fewshot2.py` | arms K–K3: learning a new rule from k refutations; retirement and interleaving |
| `e2_sleep.py`, `e2_sleep2.py`, `e2_wakesleep.py` | the sleep variants: in-stream (M–M4), label-free on a trained model (K4), re-scored in-stream (M5, M6) |
| `e2_report.py`, `e2_plot.py`, `e2_plot_k.py` | per-rule tables and learning-curve figures (`runs/e2/*.png`) |
| `EXPERIMENT_E4.md` | E4: acting by planning in the learned world model; learning from refutations while acting |
| `planner.py`, `e4_plan.py`, `e4_plot.py` | E4: the reward model, the planner (search in the model's own predictions), the refutation learner, the driver |
| `EXPERIMENT_E5.md` | E5: the day–night agent from random play only (fails: needs directed exploration) |
| `agentnet.py`, `e3_aux.py` | E3: one trunk with policy, value and world-model heads; PPO with the auxiliary loss |
| `WORLD_DESIGN.md` | the test world "Village" (draft): resources, objects, agents, actions, interactions, task reward, the causal log and value dimensions, the nine dilemmas, the three training regimes and misalignment variants, the display, the model |

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
