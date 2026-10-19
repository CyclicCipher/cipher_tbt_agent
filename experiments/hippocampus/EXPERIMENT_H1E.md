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

## Results (2026-10-18; `runs/h1e/`, log `runs/h1e_log.txt`)

**One run broke the 2-minute rule:** `R_sh s4` took 129 s, including the first compile of the RoPE graph. The other
three took 97–104 s.

| reader | REV one-way clean: ranked / **shuffled** | fwd: ranked / shuffled | shuffled-order reverse errors | reverse curve (training order) |
|---|---|---|---|---|
| R_sh s4 | 0.197 / **0.193** | 0.057 / 0.054 | 62% right relation, wrong entity; 17% not in context | 0.03 → 0.19 |
| R_sh s5 | 0.192 / **0.217** | 0.052 / 0.061 | 63% right relation, wrong entity; 14% not in context | 0.04 → 0.20 |
| R_rk s4 | **0.960** / 0.091 | 0.963 / 0.098 | 91% right relation, wrong entity | 0.87 → 0.96 |
| R_rk s5 | **0.966** / 0.116 | 0.948 / 0.125 | 88% right relation, wrong entity | 0.89 → 0.97 |

**Against the predictions:**
- **P1 (RoPE enables content matching, ≥ 0.6): REFUTED.** 0.19–0.22 shuffled, the same as absolute positions (E2:
  0.20). By the pre-registered line (≤ 0.3), RoPE's relative positions are not what is missing.
- **P2 (RoPE keeps the copy route ≥ 0.9): holds.** 0.96–0.97 ranked, and forward 0.95–0.96.
- **P3 (reported):** a RoPE reader trained on ranked order is as rank-dependent as E1 (0.09–0.12 shuffled).

**What H1e establishes:** relative positions alone do not unlock content matching. The failure persists as "right
relation, wrong entity".

**Next (H1f, the user's suggestion, 2026-10-18):** PoPE (arXiv:2509.10534), already implemented in the repo
(`../transformers/h1_lid.py`).
- PoPE's argument is that RoPE **entangles content and position**: its score carries content-dependent phases, so a
  head cannot match on *what* and offset by *where* independently.
- PoPE puts content in the magnitude (softplus) and position alone in the phase.
- Its reported gain is on an indexing diagnostic of this kind (11% → 95%).
