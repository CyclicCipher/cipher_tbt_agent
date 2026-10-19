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

## Results (2026-10-18; `runs/h1d/`, log `runs/h1d_log.txt`)

Training runs took 89–94 s; evaluations 2–4 s.

| reader | REV one-way clean: ranked / **shuffled** | fwd: ranked / shuffled | shuffled-order reverse errors |
|---|---|---|---|
| E1 s4 (reference) | 0.970 / 0.101 | 0.953 / 0.088 | 90% right relation, wrong entity |
| E1 s5 (reference) | 0.966 / 0.128 | 0.952 / 0.121 | 87% right relation, wrong entity |
| W1 s4 | **0.253** / **0.234** | 0.104 / 0.095 | 73% right relation, wrong entity; 4% not in context |
| W1 s5 | **0.215** / **0.226** | 0.092 / 0.107 | 73% right relation, wrong entity; 4% not in context |
| M1 s4 | 0.602 / **0.164** | 0.674 / 0.099 | 80% right relation, wrong entity |
| M1 s5 | 0.628 / **0.197** | 0.633 / 0.115 | 77% right relation, wrong entity |

**Against the predictions:**
- **D1 (W1 shuffled ≥ 0.6): REFUTED.** 0.23, close to E2 from scratch (0.20).
- **D2 (W1 keeps the ranked skill ≥ 0.9): REFUTED.** Ranked reverse collapsed from 0.97 to 0.22–0.25, and forward to
  0.10. Shuffled-order training *erased* the copying skill without replacing it.
- **D3 (M1 ≥ 0.5 shuffled and ≥ 0.9 ranked): REFUTED** on both (0.16–0.20 and 0.60–0.63). The mixed order made the
  ranked skill harder to learn and taught no content matching.
- **D4 (E1 depends on the rank cue, < 0.3 shuffled): holds.** 0.10–0.13.

**What H1d establishes:**
- This reader does not learn content matching, whether warm-started or given a mixed curriculum, within the budget.
- Every arm's errors are "right relation, wrong entity": it finds episodes with the asked relation (via r° → r) but
  cannot pick the one whose object is the question's entity.

**A hypothesis for why** (not yet tested): the operation is "find the read episode whose *object* equals the question's
entity b, then copy the token *two positions earlier*".
- With **relative** positions (rotary or ALiBi), "two positions earlier" is a single fixed offset: an induction-head
  pattern.
- With the reader's **learned absolute** positions over 51 slots, it must be learned separately at each of the 12
  episode positions.

**Next (H1e):** the same reader with rotary position encoding (RoPE), trained from scratch with a shuffled order (E2's
setting).
