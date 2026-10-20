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
| `WORLD_DESIGN.md` | the test world "Village" (draft): resources, objects, agents, actions, interactions, task reward, the causal log and value dimensions, the nine dilemmas, the three training regimes and misalignment variants, the display, the model |

## Where it stands (2026-10-20)

- The essay is read and noted, and the goals are written.
- **The user decided:** GPU jobs on their side for training, and a grid world.
- **`WORLD_DESIGN.md` (draft) is up for the user's review,** with three open questions at its end: observability, which
  dilemmas first, and robot health.
- Nothing built or run yet.
