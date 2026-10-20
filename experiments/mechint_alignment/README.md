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
| `WORLD_DESIGN.md` | the test world "Village" (draft): resources, objects, agents, actions, interactions, task reward, the causal log and value dimensions, the nine dilemmas, the three training regimes and misalignment variants, the display, the model |

## Where it stands (2026-10-20)

- The essay is read and noted, and the goals are written, with a **scope of claims** (`GOALS.md`): success here would
  not mean alignment is solved, nor say which values to align to.
- **The user decided:** GPU jobs on their side for training (runs much longer than 2 minutes); a grid world; full map;
  the core dilemmas first (D2, D3, D6, D7); robot health kept, with pain; a provision value (more apples for the
  villagers).
- **`WORLD_DESIGN.md`** has these folded in.
- **`TRAINING_METHOD.md`** proposes the training method, for the user's review.
- Nothing built or run yet.
