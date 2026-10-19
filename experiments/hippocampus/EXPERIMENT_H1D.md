# Experiment H1d — can the reader learn to match by content, without a rank cue?

**Pre-registered 2026-10-18, before any H1d result run** (a 30-step mechanics check, disclosed). It continues the
user's "keep doing steps without pause". It addresses the open risk in `PLAN.md` §0, learning 12.

## Why

H1c's E1 readers are reliable (0.97–0.98 reverse) because retrieval ranks the fully matching episode first, and the
reader copies from that position. With the read order shuffled (E2, trained from scratch), readers reached only 0.20
reverse and 0.05 forward. Their errors were "right relation, wrong entity": they never learned to find the episode by
its *content*.

Harder reading will need content matching: several facts in one context, chains, real text where ranking is
imperfect. Two cheap routes are tested within the 2-minute rule.

## Arms (`run_h1d.py`)

Both use the B2 recipe with the expansion (4,500 steps, compile, cosine, familiarity gate), seeds 4 and 5.

| arm | start | read order in training |
|---|---|---|
| **W1** | **warm start** from the E1 reader of the same seed (`runs/h1c/reader_E1_s*.pt`) | shuffled |
| **M1** | from scratch | **mixed**: each question ranked or shuffled at random (50/50) |

**Evaluation:** each trained reader, and the E1 readers as reference, on world 0 under **both** read orders (ranked
and shuffled), with gated writes and the reverse error classes.

## Mechanics check (disclosed)

A 30-step mixed-order warm-start run plus both-order evaluation runs end to end (14 s + 2 s). Nothing was read from it.

## Predictions

- **D1 (warm start teaches content matching):** W1's shuffled-order reverse is **≥ 0.6** in both seeds. E2 from
  scratch: 0.20.
- **D2 (warm start keeps the ranked skill):** W1's ranked-order reverse is ≥ 0.9.
- **D3 (a mixed curriculum):** M1's shuffled-order reverse is ≥ 0.5, and its ranked-order reverse is ≥ 0.9.
- **D4 (E1 reference):** E1 readers evaluated with a shuffled order are below 0.3. They depend on the rank cue (their
  shuffled-order score has not been measured before).

## Cost

4 training runs of about 95 s, plus 6 evaluations of a few seconds.

---

## Results

*(to be filled in after the runs)*
