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
| `PLAN.md` | the problem, the architecture (database + HDC index + neural read-out and consolidation), measures, planned experiments H1–H6, risks, prior art |

## Standing rules

See `PLAN.md` §7. They are the same as in Dust 2 and intelligent constraints: ≤ 2-min runs, pre-registration, never
re-run, 16-bit, diverse tasks, no competitor-faithful arms, commits to `exp/mdl-vs-ntp`.

## Where it stands (2026-10-18)

The plan is written. Nothing has been run. H1 (store plus read on synthetic facts) is next, pending the user.
