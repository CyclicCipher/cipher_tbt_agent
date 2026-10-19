# Experiment H1e — rotary positions for content matching

**Pre-registered 2026-10-18, before any H1e result run.** It continues the user's "keep doing steps without pause".

## Why

H1d: without a rank cue, the reader never learns to find the read episode whose *object* is the question's entity and
copy its *subject*. Every error is "right relation, wrong entity", and neither a warm start nor a mixed order helped.

The hypothesis is the position encoding. The reader uses learned **absolute** positions over 51 slots, so "the token
two positions before the matched one" must be learned separately at each of the 12 episode positions. With **rotary**
positions (RoPE), a fixed offset is one operation everywhere: the induction-head pattern.

## Change

`Net(rope=True)`, in the shared `../intelligent_constraints/model.py`:
- rotary position encoding on queries and keys in every attention layer (base 10,000), replacing the learned absolute
  position embedding;
- existing configurations are unchanged (the option is off by default, and no random draws change).

## Arms (`run_h1e.py`)

The B2 recipe with the expansion (4,500 steps, compile, cosine, familiarity gate), from scratch, seeds 4 and 5:

| arm | positions | read order in training |
|---|---|---|
| **R_sh** | RoPE | shuffled (E2's setting) |
| **R_rk** | RoPE | ranked (E1's setting) |

Each trained reader is evaluated on world 0 under **both** orders, with the reverse error classes, as in H1d.

References, not re-run:
- E2 (absolute positions, shuffled): 0.20 shuffled;
- E1 (absolute, ranked): 0.97 ranked, 0.10–0.13 shuffled.

## Predictions

- **P1 (relative positions enable content matching):** R_sh's shuffled-order reverse is **≥ 0.6** in both seeds.
  - If it is ≤ 0.3, the position encoding is not the bottleneck.
- **P2 (RoPE keeps the copy route):** R_rk's ranked-order reverse is ≥ 0.9.
- **P3 (reported):** R_rk evaluated shuffled, against E1's 0.10–0.13. Does a RoPE reader trained on ranked order also
  match by content?

## Cost

4 training runs of about 95 s, plus evaluations.

---

## Results

*(to be filled in after the runs)*
