# Where the H6 line was left (2026-10-20), and how to pick it up

The line paused here because the user opened a new experiment path (`../mechint_alignment/`). This file holds the last
exchange, so work can resume without re-deriving it.

## State

- **H6** (`EXPERIMENT_H6.md`), a character-level language model on real text: one-shot learning of a re-read passage,
  a gain under shift (Middle High German −0.43 bits/char), and +0.06 bits/char worse on non-repeating held-out Latin.
- **H6b** (`EXPERIMENT_H6B.md`), a retrieval-confidence gate:
  - a second reading becomes almost free (ratio 0.058–0.065, about 16× cheaper);
  - the Latin cost (+0.05) stays, although memory is shown on fewer than 3% of Latin windows. **The cost is in the
    weights:** 25% of training windows are re-reads, which are trivial copying;
  - most of the Middle High German gain is also a weights effect (better copying within the model's own context).
- **Proposed next step, H6c:** additive copy training, with re-read windows added on top of a full batch of ordinary
  text. Not pre-registered, not run.

## The user's question, and my answer (2026-10-20)

**The user:** "But, is this proposed next step actually going to make it work on Latin? Why doesn't it already? What is
it teaching the model?"

**Short answer: no.** H6c would at best remove the cost on Latin, not produce a gain. It tests where the cost comes
from; it does not make the store useful on ordinary prose.

**Why the store does not help on held-out Latin:**
1. **There is almost nothing to retrieve.** The store holds only what the model read earlier in the same
   50,000-character stream, and Latin prose rarely repeats.
   - Measured before H6b: only about 4% of Latin windows have an earlier passage containing ≥ 30% of the cue's
     idf-weighted words, and 0.7% reach ≥ 50%.
   - In Middle High German verse, about 60% of windows reach ≥ 30%.
2. **Partial matches do not help this model.** Showing the best partial match on every window (H6, and the ungated arm
   of H6b) made Latin worse. A shared name or rare word in a loosely related passage is mostly noise for a model this
   small.
3. **In distribution, the weights already know the language.** A store can only add information that is neither in the
   weights nor in the 128-character context. For more Cicero and Livy that is very little; for Middle High German,
   which the weights never saw, it is a lot. This is ZipLearn's E37 pattern: adaptation was worth +0.049 bits/char in
   distribution and +6.04 under shift.

**What the training teaches:**
- **Re-read windows (25%) teach exact copying:** if the memory holds the continuation of your context, copy it. That was
  learned extremely well (second reading 16× cheaper).
- **Ordinary windows teach nothing about memory under the gate**, since memory is hidden there 97–99% of the time.
  Without the gate they teach "a loosely related passage is mostly useless", learned imperfectly, which is the ungated
  arms' extra cost.
- **Two side effects in the weights:**
  - **The cost:** a quarter of training is trivial copying, so less goes into ordinary Latin (the +0.05).
  - **A benefit:** the copy circuit also improves copying within the model's own context, which repetitive verse
    rewards. That is most of the Middle High German gain, with memory shown on only 1.8% of its windows.

**So H6c** would probably bring the Latin cost to about zero and keep the one-reading learning. It is a useful control,
but the store would be neutral on ordinary Latin, not helpful.

## What would make it help on Latin: give it something relevant to retrieve

- **Option 1 (recommended):** store everything the model has read, including the training corpus, as a hippocampus
  that keeps its experience until consolidation.
  - Held-out Cicero and Livy could then retrieve similar constructions and formulae from the 10.7M training
    characters. That is the kNN-LM/RETRO result, whose gains are largest for small models.
  - The store would be about 333k chunks × 14 bytes ≈ 4.7 MB. Retrieval over it is heavier but feasible.
  - It is the honest test of whether episodic memory helps in-distribution prose. It may still not: the weights may
    already capture most of it.
- **Option 2:** test where episodic memory should matter: a full book of Livy with recurring names and events, rather
  than 50,000-character excerpts.
- **Either way:** fold the additive copy training into the same run, to remove the cost confound at the same time.

**Open question for the user when the line resumes:** the "store everything" test, the longer-document test, or both?
