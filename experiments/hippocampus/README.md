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
| `EXPERIMENT_H1B.md`, `run_h1b.py` | H1b: a stronger reader (compile, cosine, slot embedding) and the familiarity gate |
| `EXPERIMENT_H2.md`, `h2.py`, `run_h2.py` | H2: a continual stream; crowding; eviction by value against fifo, random and oracle |
| `EXPERIMENT_H2B.md`, `expand.py`, `run_h2b.py` | H2b: learned query expansion against crowding |
| `EXPERIMENT_H1C.md`, `run_h1c.py` | H1c: readers trained with the expansion (reliability), and streams with them |
| `EXPERIMENT_H3.md`, `h3.py`, `run_h3.py` | H3: consolidation (sleep with and without self-generated reverse inferences), then eviction |
| `EXPERIMENT_H3B.md`, `run_h3b.py` | H3b: longer sleep, 1:1 vs 3:1 mixes, self-checked eviction |
| `EXPERIMENT_H3C.md`, `h3c.py`, `run_h3c.py` | H3c: sleep inside the continual stream (keeping the long tail) |
| `EXPERIMENT_H3D.md` | H3d: the same with pseudo-rehearsal (the weights replay what they already know); arm `sleep_pr` of `h3c.py` |
| `EXPERIMENT_H3E.md` | H3e: the dose of self-replay, first-sleep fallback fixed; arms `sleep_pr_fix`, `sleep_pr_dose` |
| `EXPERIMENT_H5.md`, `h5.py` | H5: a Willshaw CA3 network (clipped Hebbian, autoassociative) against the table, by code sparsity |
| `EXPERIMENT_H5B.md` | H5b: the Willshaw network with slot-bound codes (`h5.py --slots`) |
| `runs/` | results, one JSON per run |

## Standing rules

See `PLAN.md` §7. They are the same as in Dust 2 and intelligent constraints: ≤ 2-min runs, pre-registration, never
re-run, 16-bit, diverse tasks, no competitor-faithful arms, commits to `exp/mdl-vs-ntp`.

## Where it stands (2026-10-18)

The plan is written, with the compression mechanisms in §2b.

**H1 (store plus read on synthetic facts): done.**
- One write per fact gives **reverse** access at 38–53%. Weights alone give ≤ 0.9% even after 64 exposures (chance
  0.12%).
- Forward: 1 write against 8–64 exposures for the weights.
- Retrieval recall is 1.00, so **the limit is the reader's skill** (reverse reading is still improving at the 3,500-step
  budget), not the store.
- The surprise gate writes every new fact once, but lets through 19–21% duplicates on re-presentation (one-to-many
  facts: probability spread over several answers). The fix is a familiarity gate from the index itself.

**H1b: done.**
- With a better-trained reader (compile, 4,500 steps, cosine decay), **one write per fact gives 83–93% reverse access
  and 95–96% forward**, against ≤ 0.9% reverse for the weights after 64 exposures.
- Readers plateau at about 0.5 (probably partly relation-blind), then jump as the learning rate decays.
- The slot embedding hurt (stuck at 0.50), so it is not carried forward.
- **The familiarity gate dedupes exactly: 0 duplicates**, against 14–15% for the surprise gate alone.
- B2's configuration is the reader for H2 and H3.

**H2 (a continual stream): done.**
- **Crowding** is the store's form of interference: reverse reads of old facts fall 2.5–3× as an uncapped store grows.
- **Eviction by value** (a usefulness rate from the reader's own answers) is within 0.03 of an oracle, beats fifo and
  random by 0.14–0.18, and **beats keeping everything at a quarter of the size**.
- One of the two new readers stayed on the reverse plateau, so plateau escape is the read path's main fragility.

**H2b (learned query expansion against crowding): done.**
- It fixes retrieval: reverse recall is 1.00 at 11,930 rows.
- But readers trained without it collapse (0.96 → 0.07 reverse): **the reader co-adapts to its retrieval**, and must be
  trained with it.
- Value eviction wins popularity-weighted, not per fact.

**H1c (readers trained with the expansion): done.**
- **4 of 4 seeds reach 0.97–0.98 reverse, with no plateau.**
- Crowding is fixed end to end.
- Without crowding, value eviction ties the full store on popularity-weighted accuracy at a quarter of the size, and
  loses the long tail.
- E1 is the reader from here.

**H3 (consolidation): done.**
- Sleep moves the store into the weights.
- **Forward-only replay brings the reversal curse back in the weights (0.985 / 0.000).**
- **Replaying the reader's own reverse inferences gives reverse 0.59–0.65 with no reverse labels.**
- Reading survives sleep.
- Eviction needs a self-check: a lax threshold cost 0.13.

**H3b (longer sleep, self-checked eviction): done.**
- The weights alone hold **forward 0.998 / reverse 0.966** after 5,000 sleep steps, the reverse entirely from the
  model's own inferences.
- **Self-checked eviction frees 64% of the store at ≤ 0.012 cost.**
- Reading survives.
- Complementary learning systems, working end to end.

**H3c (sleep inside the stream): done, negative.**
- **Catastrophic interference:** facts consolidated at the first sleep and then evicted are overwritten by later
  sleeps (retention about 0.2).
- The long tail barely improves (0.39 vs 0.35).
- Sleep must also replay what the weights already know (H3d).

**H3d (sleep with pseudo-rehearsal): done.**
- Long tail **0.60** (from 0.39) with a store **11× smaller** than the full one (full: 0.86).
- The weights alone hold 0.63.
- No online or reading cost.
- Forgetting is halved (retention 0.48), not stopped.

**H3e (self-replay dose, fallback fixed): done.**
- Long tail **0.60–0.63 at about 550 rows** (full store: 0.86 at 5,950).
- A larger replay share trades new for old: a stability–plasticity trade, not a gain.
- The total sleep compute binds.

**H5 (a Willshaw CA3 network instead of the table): done.**
- With sparse codes (k ≤ 20) a fixed 512 KB network holds all 11,930 episodes without saturating.
- It loses episode/role structure: union-coded cues blend every matching episode (top-1 0.11 at 11,930).
- H5b adds per-slot binding.

`PLAN.md` §0 holds the learnings so far.

H1 reuses `../intelligent_constraints/` `data.py` and `model.py`.
