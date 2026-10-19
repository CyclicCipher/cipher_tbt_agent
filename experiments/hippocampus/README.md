# Hippocampus — fast storage without interference, for a standard transformer

Started 2026-10-18 at the user's request, as a change of approach after `../intelligent_constraints/` Experiment 1c.
There, a dense dagger MLP transferred nothing to reversed questions, even with matched codes.

## The idea

**The complementary-learning-systems design, built onto a transformer.**
- A fast store writes each surprising episode once, with no gradient step and no interference.
- It is read by content, from any part of the episode.
- Sleep moves what it holds into the weights, by replaying it together with the model's own in-context inferences.

**One substrate per job.**
- The store is a database (one row per episode).
- The index is sparse binary hypervectors (HDC/SDR bundling: the union of the episode's token codes, so retrieval is
  direction-free).
- The read-out and consolidation are neural.

The full design and the reasons are in `PLAN.md`.

## Files

| file | what |
|---|---|
| `PLAN.md` | the problem, the architecture (database + HDC index + neural read-out and consolidation), §2b compression (the user's condition), measures, planned experiments H1–H6, risks, prior art |
| `store.py` | the store: a table of episodes indexed by sparse binary codes (union per episode); the dense control |
| `h1.py`, `run_h1.py` | Experiment H1 (arms A weights-only, B sparse store, C dense store) and its driver |
| `EXPERIMENT_H1.md` | H1: pre-registration, then results |
| `runs/` | results, one JSON per run |

## Standing rules

See `PLAN.md` §7. They are the same as in Dust 2 and intelligent constraints: ≤ 2-min runs, pre-registration, never
re-run, 16-bit, diverse tasks, no competitor-faithful arms, commits to `exp/mdl-vs-ntp`.

## Where it stands (2026-10-18)

The plan is written, with the compression mechanisms in §2b. **H1** (store plus read on synthetic facts) is
pre-registered. It reuses `../intelligent_constraints/` `data.py` and `model.py`.
