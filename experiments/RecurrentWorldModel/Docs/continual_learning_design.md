# Continual Learning — corrected understanding + redesign

> Step 4 of the temporal fork. This doc records a **negative result + a conceptual correction**
> (we mis-applied the wrong mechanism), then redesigns the experiment around what the source
> paper actually proposes. Status: design, for review before building.

## 1. The negative result (delta-encoding ≠ continual learning)

We hypothesised (from data point #2) that Δ-encoding, by making a distribution shift *invisible*,
would also prevent catastrophic forgetting. `train_continual.py`, A→B, forgetting =
`acc_A(end of A) − acc_A(end of B)` (1500 steps/phase):

| | offset shift | scale shift |
|---|---|---|
| **absolute** | +0.520 | +0.246 |
| **delta** | **+0.480** | +0.516 |

Prediction (`delta`+`offset` ≈ 0) **failed**. Reading it closely:
- `offset_delta` ends `acc_A 0.480 ≈ acc_B 0.469`, both collapsed from 0.961. The offset shift is
  invisible to delta, so phase B *is* phase A in delta space — this is not "forgot A to learn B,"
  it's the optimiser **drifting off a solution it already had** (flat-LR, no schedule — a harness
  flaw too).
- `scale_delta` (regimes genuinely differ): `acc_A 0.99 → 0.48` while `acc_B → 1.000` — real
  catastrophic forgetting.
- Delta **learns** A better (0.96–0.99 vs absolute 0.50–0.61) but **retains no better** — every arm
  forgets.

**Lesson: invisibility ≠ retention.** Making a shift invisible removes the *need* to relearn; it
provides no mechanism to stop the optimiser *overwriting* during continued training. And in the only
*meaningful* case (scale, where there's actually something to retain), delta forgets catastrophically.

## 2. The conceptual correction (we misread the paper)

Makushkin (`rand3289/ai2026/ai26.md`) does **not** propose derivative encoding for continual
learning. Its mechanism is **outcome-space partitioning**:

> "The system builds other conditional distributions **in parallel instead of modifying existing
> distributions**." New experiments are created in short-term memory and **merged during
> consolidation**; experiment boundaries are detected via **timestamps of state changes**;
> realisations that don't fit are **excluded**, so the existing distribution doesn't change.

That is an **allocation / memory** mechanism — *create new capacity for novel conditions, never
overwrite the old* — not a representation trick. We took the Stage-2 Δ idea (which solves
**generalisation**, data point #2) and mis-applied it to **retention**, a different problem.

**The clarifying irony:** the paper's mechanism must **detect** the shift (to partition — it uses
the absolute *timestamps of state changes*). Δ-encoding **hides** the shift. We built the conceptual
*opposite*. Generalisation wants the shift invisible; partition-based retention wants it visible.

**Meta-lesson logged:** read the source for the *idea*, not just the implementation. We read the
code for TBAF/SIGReg/LeWM but carried a half-remembered framing into the Step-4 *concept*.

## 3. Redesign — novelty-gated partitioning

Faithful to the paper and to the mainstream continual-learning literature (progressive / expandable
networks; mixture-of-experts with growth; complementary learning systems / hippocampal replay):
**on a novel regime, allocate a new expert and route to it; never overwrite an established expert;
consolidate later.**

**Components**
- **Shared trunk** + a **growable pool of expert heads** (start with one).
- **Novelty signal — general, NOT a handcoded rule** (bitter lesson): the prediction error /
  surprise of the best-matching expert. Novelty when `min_expert_error > μ + kσ` of recent
  in-regime error (the ML reading of the paper's "state-change timestamps" / the HTM-bursting
  signal we discussed). No "if v0 > 500" — the detector only sees its own error.
- **Routing:** each input goes to the lowest-error / highest-confidence expert; if none is confident
  → allocate a new one.
- **Protection:** freeze an expert's params once a newer one is allocated (or train only the active
  expert) — this is the non-overwriting that prevents forgetting.

**Two tiers (separate mechanism-proof from detection):**
1. **Oracle partition (upper bound):** separate head per regime, frozen at the *known* A→B boundary.
   Tests only: does non-overwriting → ≈0 forgetting? (It must — A's head is never touched.)
2. **Surprise-gated partition (the real test):** boundary detected automatically from the surprise
   signal. Tests: can the boundary be found and a head allocated *without supervision*?

**Baselines / arms:**
- `single_head` — the current `train_continual` (forgets: the negative we have).
- `oracle_partition` — expected ≈0 forgetting (mechanism upper bound).
- `surprise_partition` — ≈ oracle iff the detector fires at the boundary.

**Cross with encoding (the synthesis that redeems Δ):** partitioning retains regardless of encoding,
but *what the detector sees* depends on it:
- `absolute + offset`: surprise fires → allocate → **retain** (partitioning rescues the forgetful
  absolute model).
- `delta + offset`: shift invisible → no surprise → no allocation needed → **no forgetting anyway**.
- `either + scale`: surprise fires → allocate → **retain**.

So Δ-encoding and partitioning are **complementary, not competing**: Δ *reduces what must be
partitioned* (hides the shifts it can absorb), partitioning *handles the rest* (the visible ones).
You partition exactly when you are surprised, and you are surprised exactly when the shift is visible
to your representation. That is the real role of the Stage-2 result inside continual learning.

**Protocol:** A→B (and A→B→A to test routing *back* to A's expert), forgetting on held-out A; report
number of experts allocated and whether the surprise signal spiked at the boundary. **Add the WSD LR
schedule** (the omission that contaminated §1). Guardrail: the only place the boundary is supplied is
the explicitly-labelled `oracle` tier; everywhere else it must be detected from surprise.

**Predictions:** `single_head` forgets (have it); `oracle_partition` ≈0; `surprise_partition` ≈ oracle
if detection works; `delta+offset` trivially retains (nothing to partition).

## 4. Build plan (once approved)

1. A `PartitionedModel` wrapper: shared trunk + expert-head pool + surprise-gated allocation/routing +
   freezing. Default single-head (= current behaviour) so nothing else changes.
2. Extend `train_continual.py` with the three arms + WSD; reuse ShiftSeq offset/scale + the encoding
   cross.
3. Smoke tests (allocation fires on a synthetic regime jump; frozen heads stay frozen; oracle retains
   A in a tiny run).
4. Record outcome as the corrected Step-4 data point.
