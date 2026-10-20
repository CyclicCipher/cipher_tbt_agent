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

## Where it stands (2026-10-20)

- The essay is read and noted, and the goals are written.
- **Nothing designed or run yet.**
- Two decisions for the user are open (`GOALS.md`, end):
  - the training budget for the world's models;
  - the world family.
- **The next step is a design document for the test world (G1)** and the definitions it grounds (G7).
