# A hippocampus module for a standard transformer — plan

Started 2026-10-18. The user's words: "Let's change our approach. See 'fast storage without interference' … If you had
to design a 'hippocampus module' for a standard transformer, how would you do it?", then "Make a hippocampus experiments
folder with its own planning doc. What will the architecture of this hippocampus be? A neural net? A database? Some kind
of VSA/HDC system?"

Every experiment is pre-registered in its own `EXPERIMENT_*.md` before it runs.

---

## 0. Status and learnings (updated 2026-10-18, after H1 … H5c and H1d–H1f)

**Results so far:**

| experiment | result |
|---|---|
| **H1** (`EXPERIMENT_H1.md`) | One write per fact gives reverse access at 38–53%. Weights alone give ≤ 0.9% even after 64 exposures, with the converse tokens taught on the same world (chance 0.12%). Forward: 1 write against 8–64 exposures. Retrieval recall is 1.00, so the reader is the limit. The surprise gate re-wrote 19–21% of facts on a second presentation (one-to-many facts). |
| **H1b** (`EXPERIMENT_H1B.md`) | The reader with `torch.compile`, 4,500 steps and cosine decay (**B2**) gives **83–93% reverse and 95–96% forward from one write**. The familiarity gate gives **0 duplicates**. The slot embedding (B3) hurt (stuck at 0.50). |
| **H1h** (`EXPERIMENT_H1H.md`) | Non-causal attention does **not** enable forward content matching (0.06) and slows reverse (0 of 4 vs causal 1 of 4). Causal masking was not the obstacle. Forward content matching remains unlearned across about 20 readers (absolute, RoPE, PoPE; causal or not); its errors include 30–40% answers not in context. All runs within the 2-minute rule. |
| **H1g** (`EXPERIMENT_H1G.md`) | PoPE reverse content matching is **reliable with more updates**: 4 of 4 seeds at batch 128 × 8,000 (0.96–0.98, shuffled order), against 2 of 4 at batch 256 × 4,000, on the same data. **Forward is never learned** (10 PoPE readers, ≤ 0.16). Hypothesis: causal masking. ⚠ All four 8,000-step runs took 125–156 s, over the 2-minute rule. |
| **H1f** (`EXPERIMENT_H1F.md`) | **PoPE** (the user's suggestion; the repo's implementation): **one of two seeds learned content matching, 0.971 reverse with a shuffled read order (0.977 ranked)**, order-independent and the first such reader in the line (10 shuffled-order readers otherwise ≤ 0.23). The transition is seed-dependent (seed 5: 0.16), and forward was not learned (0.09–0.14), unexplained. PoPE keeps the copy route (0.97). |
| **H1e** (`EXPERIMENT_H1E.md`) | RoPE does not unlock content matching (0.19–0.22 shuffled, the same as absolute positions); it keeps the copy route (0.96–0.97 ranked). One run took 129 s, over the 2-minute rule. |
| **H1d** (`EXPERIMENT_H1D.md`) | Content matching without a rank cue is **not learned** by this reader. A warm start plus shuffled training gives 0.23 shuffled and *erases* the ranked skill (0.97 → 0.23). A mixed order gives 0.18 shuffled and 0.61 ranked. Errors are always "right relation, wrong entity". Hypothesis: learned absolute positions make "copy two positions before the matched object" hard; H1e tests rotary positions. |
| **H5c** (`EXPERIMENT_H5C.md`) | **Conjunctive (dentate-gyrus-like) units** (hashed pair units → third-slot Hebbian maps): **top-1 1.000** in both directions at every size up to 11,930, one-shot. The price is memory: 50 MB (about 34 kbit per episode) against the table's 14 bytes per episode. A random conjunctive expansion must be large; an allocated one (a row per episode) is about 300× cheaper. |
| **H5b** (`EXPERIMENT_H5B.md`) | Slot-bound codes (a fixed permutation per slot, with learned slot maps) remove the role confusion (@12 **0.99** at 11,930). But top-1 still falls (0.24): **a pairwise Hebbian matrix cannot store a conjunction.** An impostor paired with the entity by one episode and with the relation by another gets full support. Conjunctive (per-episode or dentate-gyrus-like) units are needed; the table's rows are such units. |
| **H5** (`EXPERIMENT_H5.md`) | A **Willshaw CA3 network** (clipped Hebbian, autoassociative, one-shot) on the same sparse codes. With **k ≤ 20 active bits it holds all 11,930 episodes without saturating** (k = 40 saturates). What it loses against the table is **episode structure**: a union-coded cue completes every episode containing its tokens, in either role, blended together (top-1 0.11, @12 0.83 at 11,930). |
| **H3e** (`EXPERIMENT_H3E.md`) | The first-sleep fallback fixed: long tail **0.60–0.63** at about 550 rows (full store 0.86 at 5,950). A larger self-replay share is a **stability–plasticity trade**: old facts are kept slightly better (+0.05–0.08) but fewer new ones are consolidated, and the long tail drops. The total sleep compute is the binding constraint within the 2-minute rule. |
| **H3d** (`EXPERIMENT_H3D.md`) | Sleep with **pseudo-rehearsal** (the weights replay their own confident answers): long tail **0.60** (vs 0.39 plain sleep, 0.35 no sleep, 0.86 uncapped) with an **11× smaller store** (about 550 vs 5,950 rows); weights alone 0.63; no online or reading cost. First-sleep retention 0.48 (vs 0.2): forgetting halved, not stopped. One implementation deviation (a spurious first-sleep pseudo set). |
| **H3c** (`EXPERIMENT_H3C.md`) | Sleep inside a stream (5,957 facts, a 1,500-row store, 4 sleeps): **catastrophic interference.** Facts consolidated and evicted at the first sleep are overwritten by later sleeps (retention 0.18–0.24). The long tail improves only +0.04 (0.39 vs 0.35; uncapped 0.86). Reading survives. |
| **H3b** (`EXPERIMENT_H3B.md`) | With 5,000 sleep steps, the weights alone hold **forward 0.998 and reverse 0.966** (reverse entirely from self-generated inferences). **Self-checked eviction** (the weights' top answer equals the stored answer, both directions) frees **64%** of the store at ≤ 0.012 cost. Reading survives (±0.008). 1:1 and 3:1 mixes tie. |
| **H3** (`EXPERIMENT_H3.md`) | Sleep (empty-memory replay plus rehearsal of reading) consolidates the store into the weights. **Forward-only replay brings the reversal curse back in the weights** (fwd 0.985, rev **0.000**). **Replaying the reader's own reverse inferences** (98% correct, no labels) gives the weights reverse 0.59–0.65 (fwd 0.61). Reading survives sleep (±0.01, held-out world too). Eviction at p ≥ 0.1 was too lax: 41% of rows evicted, the combined system −0.13. |
| **H1c** (`EXPERIMENT_H1C.md`) | Readers trained **with** the expansion (E1): **4 of 4 seeds at 0.968–0.979 reverse**, with no plateau (0.86–0.89 at step 500). A shuffled read order (E2) gives 0.20: reliability comes from retrieval ranking by full match, with the reader copying from the top. Crowding is fixed end to end (old facts keep 0.83–0.89× of reverse accuracy, against 0.31–0.40×). Without crowding, value eviction ties the full store popularity-weighted (0.80–0.81) at a quarter of the size, but loses the long tail per fact (0.64 vs 0.78). |
| **H2b** (`EXPERIMENT_H2B.md`) | Learned query expansion (converse token → its relation, from index co-occurrence) makes reverse recall 1.00 at any store size. But the H2 readers, trained without it, fall to 0.07–0.13 reverse: **reader–retrieval co-adaptation**. The plateau is not simply relation-blind reading. Per-fact scoring: value eviction wins only popularity-weighted. |
| **H2** (`EXPERIMENT_H2.md`) | A stream of 11,930 facts with Zipf queries. **Crowding** is the store's form of interference: uncapped, the first phase's reverse accuracy falls 2.5–3× as the store grows (recall 1.00 → 0.42). **Eviction by value** (a usefulness rate from the reader's own answers) beats fifo and random by 0.14–0.18, comes within 0.004–0.027 of an oracle, and **beats keeping everything** (0.64 vs 0.58, 0.82 vs 0.73) at a quarter of the size. Two new B2 readers: one escaped the plateau (0.96), one did not (0.475). Three of the four B2 readers so far escaped. |

**Learnings that change the design:**
1. **The reversal curse is a property of where knowledge is stored.** A store read by attention does not have it: the
   in-context mechanism, plus a direction-free index. This is the module's main justification so far.
2. **The write gate needs recognition before surprise.** Surprise is the wrong measure for set-valued answers: a reader
   that correctly spreads probability over k answers has surprise ≥ log₂ k. The familiarity check (pattern completion on
   the index, verified against the stored tokens) dedupes exactly and cheaply, and is the hippocampus's own recognition
   signal. The write rule is now: **write iff not familiar AND surprise > τ.**
3. **Readers stall at about 0.5 reverse, then break through as the learning rate decays.**
   - The plateau is consistent with a partly relation-blind reader (a random episode with b as object scores 0.386).
   - The breakthrough came only with the decaying learning rate and more steps.
   - The reader's training schedule matters as much as its architecture.
   - Save predictions in future runs, to classify the errors.
4. **Factoring position into (episode, slot) did not help here.** That is not a general verdict on factored position;
   it did not pay in this reader.
5. **At 3,000 episodes, the sparse and the dense index perform alike.** Pattern separation has not yet been tested at a
   scale where it should matter. Retrieval *crowding* is expected first: an entity with more episodes than the R = 12
   read slots. H2 tests this.
6. **Compression tiers 1–2 (§2b) work:** exact dedupe, and 14-byte rows. Tiers 3–5 are untested.
7. **The store's interference is crowding, not overwriting** (H2). When an entity has more episodes than the R read
   slots, its correct episode competes for a slot. This bounds how far an uncapped store can grow before reads degrade.
   It is now the first thing to fix on the store side (§5, risk 8).
8. **Forgetting by value is better than remembering everything, while crowding exists** (H2). A capped store that
   evicts by a usefulness rate is smaller and more accurate. Compression and accuracy point the same way, and
   ZipLearn's rule transfers with labels-free evidence.
9. **The reader's plateau escape is the read path's main fragility** (3 of 4 seeds). Before H3 builds on a reader,
   check that it escaped (reverse ≥ 0.8 on world 0). H2's seed 3 reader (0.963) qualifies; seed 2 (0.475) does not.
10. **Streams with Zipf(1) popularity concentrate ~10% of queries on one fact.** Report several stream seeds and a
    per-fact score next time. *(Done in H2b: "value beats uncapped" holds popularity-weighted, but not per fact.)*
11. **The reader co-adapts to its retrieval** (H2b).
    - Learned query expansion (each converse token learns its relation from co-occurrence) fixed crowding at retrieval:
      reverse recall stayed 1.00 at 11,930 rows.
    - But readers trained without it collapsed to about 1/12 reverse accuracy. The good reader matched the relation
      itself and left entity matching to retrieval.
    - **A read path is reader + retrieval; any retrieval change must be trained with.** H1c trains readers with the
      expansion.
12. **A reliable read path = retrieval that ranks by full match + a reader that copies from the top** (H1c). This
    removed the plateau entirely (4 of 4 seeds).
    - The reader cannot yet match content without a rank cue (shuffled: 0.20). Harder reading (several facts, chains)
      will need that skill. This is a future risk.
13. **H2's "forgetting beats remembering" was a crowding artefact** (H1c). Without crowding, value eviction is a pure
    compression trade: the same popularity-weighted accuracy at a quarter of the size, and the long tail lost.
    Consolidation (H3) is how to keep the long tail without the rows.

14. **The reversal curse is a property of how weights are trained, not of the knowledge** (H3). One-way replay of a
    store that reads both ways gives one-way weights. **Self-generated inferences** (the reader's own reverse answers)
    carry reverse into the weights without reverse data: the complementary-learning-systems mechanism, working.
15. **Rehearsal makes sleep safe for reading** (H3): no loss on the consolidated world or a held-out one.
16. **Eviction must be self-checked** (H3). Evict an episode only if the weights' *top* answer equals the stored answer
    in both directions. The store holds the answer, so no label is needed. A probability threshold (p ≥ 0.1) evicted
    facts the weights get wrong (−0.13).

17. **Consolidation without interleaving overwrites** (H3c). Self-checked eviction is loss-free at the moment of
    eviction, but it removes the replay source. Later sleeps then erase what was consolidated (retention about 0.2).
    Sleep must replay what the weights already know (pseudo-rehearsal / generative replay): H3d.

18. **Self-replay (pseudo-rehearsal) is the store-free answer to consolidation's interference** (H3d). It halves the
    forgetting and lifts the long tail by +0.25, but does not yet stop it (retention 0.48). The dose of self-replay is
    the next lever.

19. **Within a fixed sleep budget, the replay mix only trades old against new** (H3e). Sleep compute per fact is the
    lever. The best stream configuration keeps about 70% of the full store's per-fact accuracy at about 9% of its rows.

20. **A neural CA3 has the capacity, but needs role/slot binding** (H5). With sparse codes (k ≤ 20) a 512 KB
    Willshaw net stores ~12k episodes unsaturated. Unordered union codes, though, blend every episode that matches a
    cue. Separate rows plus an ordered reader (the table) keep the structure the network loses. H5b tests per-slot
    permutation binding.

21. **Relational episodes need conjunctive units, not a pairwise matrix** (H5b). With slot binding, a Willshaw net
    finds the right answers among 12 (0.99) but cannot single them out (top-1 0.24). Its order, not its capacity, is
    the limit. A neural store should be a layer of conjunctive units: per episode (the table with soft reads / a dense
    associative memory) or random-sparse (a dentate gyrus, the open version).

22. **Neural substrate, summed up** (H5–H5c): a pairwise CA3 has the capacity but no conjunctions; slot binding fixes
    roles; conjunctive units fix conjunctions (top-1 1.000). The table is the cheapest implementation (about 300×
    smaller than a random conjunctive layer); the conjunctive layer is the neural one.

23. **Content matching needs content and position disentangled** (H1d–H1f). Absolute positions and RoPE never learned
    it (≤ 0.23). PoPE did, in one seed of two (0.97, order-independent), as PoPE's own argument predicts. Not yet
    reliable: seed-dependent, and forward not learned in that run.

24. **Reverse content matching is learnable (PoPE, about 8,000 updates); forward content matching is not, so far**
    (H1d–H1h). For single-fact questions this does not matter: the ranked read path answers both directions at
    0.96–0.97. It matters once retrieval cannot rank (several relevant episodes, chains).
    - **Code review (the user asked): no bug; the cause is the layout.** With the question after the memory and causal
      attention, forward must match the question's entity against a *copied* subject, while reverse matches the
      object's *own* token.
    - Forward predictions are objects of the wrong episode (57%) or blends (32%).
    - **Fix to test when needed: question before the memory.**

**Decisions (the user):**
- **2026-10-18: the store stays a table; the neural substrate (H5–H5c) is not used.** In the user's words, "it's an
  interesting experiment but let's not use the neural approach because of the 50 MB cost". The conjunctive layer
  matched the table (top-1 1.000) at about 300× its memory (50 MB against 14 bytes per episode). H5's results stand as
  evidence of what a neural CA3 would need; the line does not build on it.

**Reader configuration carried forward (updated):** **E1.** That is B2 plus the learned expansion in every store,
ranked read order, readers `runs/h1c/reader_E1_s{4..7}.pt`.

**Reader configuration carried forward:** B2. That is 4 layers, d = 128, read R = 12, θ = 20 bits, τ = 2 bits,
compile + fused AdamW, 4,500 steps, cosine to 10%, and the familiarity gate.

---

## 1. What it is for

A transformer's permanent knowledge enters only through slow, dense, interfering gradient steps. Four failures follow
from that:

| failure | evidence |
|---|---|
| **slow absorption** | our tiny model needed about 85 exposures per fact to memorise it, about 0.1 bit per exposure (`../intelligent_constraints/`, E1c; reach99 750 steps × 512 / 4,523 prompts). At scale: about 1,000 exposures for 2 bits per parameter (Allen-Zhu & Li); accuracy tracks how many documents contain a fact (Kandpal et al.) |
| **the reversal curse** | `../intelligent_constraints/`: E1–E1c. Even a memory whose reverse read shares the forward weights transferred nothing; the transpose path stored paired reverse facts *separately* |
| **interference** | ZipLearn E6b: a gradient transformer learning tasks in sequence fell below chance on the earlier ones |
| **no fast adaptation** | ZipLearn E33/E37: an online store that writes each observation immediately reaches 2.13 bits/char after 1k characters, against 4.5 when trained and frozen. Under distribution shift, adaptation was worth +6.04 bits/char (only +0.049 in-distribution) |

**The brain's answer** is complementary learning systems (McClelland, McNaughton & O'Reilly 1995):
- a fast store (the hippocampus) writes an episode in one shot, with sparse codes so episodes don't interfere;
- it then teaches the slow store (the neocortex) by interleaved replay.

**ZipLearn's §9 rules are the same design in a compressor** (`../ziplearn/DESIGN.md`):
1. never overwrite: add an entry, and let the evidence decide;
2. the context selects an entry, and a new one is created only on refutation;
3. running out of capacity forces consolidation, and what to forget is chosen by value.

**The key observation.** A transformer already has a hippocampus-like store: the keys and values it caches for the
tokens in its context.
- Whittington, Warren & Behrens (2022) showed that a transformer with recurrent position encodings is formally the
  Tolman-Eichenbaum model of the hippocampal formation.
- This is why reversal works in context. Attention retrieves stored token states by content, and the stored state at
  "Jason" in "Joseph Momoa is the father of Jason Momoa" already encodes "child of Joseph".

What the cache lacks is persistence beyond one context window, scale, a rule for what to write, protection against
interference, and consolidation. The module supplies these.

---

## 2. The architecture: a database, a neural net, or VSA/HDC?

**Each part is built from what its job requires,** so the answer is all three, each in one place:

| part | job | substrate | why this substrate |
|---|---|---|---|
| **store** (CA3) | hold episodes: one-shot, exact, no interference, with the bookkeeping that eviction needs | **a database**: an append-only table, one row per episode | one-shot exact writes, zero interference and per-row metadata are what a table gives for free. It is the limit of CA3 with perfect pattern separation (one slot per episode). |
| **index codes** (dentate gyrus + CA3 index) | find the episodes relevant to a cue, from any part of the episode | **HDC/SDR**: sparse binary hypervectors. Each episode's index is the *union* (bundle) of its tokens' codes | a union is an unordered set, so retrieval is direction-free by construction: the converse constraint, for free. Sparse codes keep similar episodes apart (pattern separation). Overlap is counted exactly and cheaply with an inverted index. |
| **content** | carry the episode's order and roles | **the episode's tokens** | order and roles are read by ordinary attention, which handles both directions in context. No role vectors are needed. |
| **read-out** | turn retrieved episodes into better predictions | **neural**: the transformer's own attention over a memory segment, plus a learned gate | the only trained part of the read path; a general skill learned once, like using context |
| **consolidation** (sleep) | move what has been learned into the weights | **neural**: fine-tuning on replayed episodes plus the model's own in-context inferences | complementary learning systems: interleaved replay is where the shared structure (rules) is extracted |

**Why not one substrate for everything:**

- **A neural net alone** (a memory MLP written by gradient, as in Titans, or a dense associative memory):
  - writes cost several exposures and interfere, which is exactly what the module exists to avoid;
  - E1c showed that a dense, superposed store cannot make the selective reverse read: values are blurred sums of many
    answers, not detectors.
- **VSA/HDC alone** (episodes bound as role ⊗ filler and summed into memory vectors):
  - a summed memory vector holds only on the order of D bindings before crosstalk swamps it;
  - unbinding is noisy, and the clean-up step needs a table of item vectors anyway, which is a database;
  - role vectors require a schema chosen by hand: the rigid parse again.
  - So only HDC's *bundling* is used, not binding.
- **A database with dense learned keys alone** (standard RAG with sentence embeddings):
  - a key computed causally at the end of a sentence is not set-like, so retrieval can depend on direction;
  - dense keys have no pattern separation;
  - learned keys drift as the model trains.

**Methodological reason to start from the table.** The table is the *ideal* hippocampus: perfect separation and
unlimited capacity. Starting there isolates the question "does fast, separate, direction-free storage plus consolidation
give one-shot, reversible, non-interfering learning?". The separate question, "how well can a neural substrate (a sparse
Hopfield or HTM-style CA3) approximate the table?", is planned as H5, where what is lost against the ideal can be
measured.

### Data structures (first version)

- **Token codes:** every token id gets a fixed, seeded, random sparse binary code: D = 2048 bits, k = 40 active (≈2%,
  the HTM regime). These are identity codes from layer 0, which are stable and do not drift.
  - **Contextual codes,** from layer-ℓ states through a fixed random expansion plus fixed sparsity, are planned as H4.
- **The store (one row per episode):**
  - the tokens (int array);
  - the index (the set of active bit ids, the union of the tokens' codes);
  - the write surprise (bits);
  - the evidence count (times retrieved *and* useful);
  - bits saved (a running total);
  - the last-use step;
  - a consolidated flag.
- **Inverted index:** bit id → episode ids. A query's overlap with an episode is the number of shared active bits.

### The four operations

1. **Read** (every forward pass that uses memory):
   - the query code is the union of the codes of the current prompt's tokens (or of the last W tokens);
   - retrieve the best R episodes (R = 1–4) whose overlap is at least θ;
   - insert their tokens as a memory segment before the prompt: `[MEM] ep₁ [SEP] ep₂ … [/MEM] prompt`. The transformer
     attends to it as to ordinary context.
   - **Version 2 (H4):** insert stored layer-ℓ keys and values instead of re-reading the tokens. This is cheaper, but the
     stored states go stale when the weights change.
2. **Write.** After a segment is processed, first ask the store whether it **recognises** each episode. Recognition is
   pattern completion on the index (a row whose index contains the episode's whole code), verified against that row's
   tokens. Recognised episodes are not written. For the rest, compute the surprise, meaning the loss under the system
   *with* the current memory. Write only if it exceeds τ: only if it refutes everything the system already knows
   (ZipLearn's "create only on refutation").
   - *Learned in H1/H1b:* surprise alone re-wrote 14–21% of facts on a second presentation, because set-valued answers
     look surprising to a correct reader. Recognition dedupes exactly.
3. **Sleep** (consolidation), every S writes:
   - replay episodes by priority: surprise × (1 + evidence) × recency;
   - for each one, put it in context and let the model generate its in-context inferences: the reverse statement, and
     the questions it can answer;
   - keep only the inferences it makes confidently and consistently;
   - fine-tune the weights on episode plus inferences, interleaved with rehearsal of older material;
   - then measure surprise *without* memory, and mark episodes the weights now predict as consolidated.
4. **Evict.** Over capacity N_max, drop consolidated episodes first, then the episode with the least evidence × bits
   saved (ZipLearn's "forgetting chosen by value": the cheapest to lose goes first).

**Why this should succeed where the dagger MLP failed:**
- **Reversal:** the read uses the in-context mechanism, and the index is direction-free.
- **One-shot facts:** one write per fact, with no gradient step.
- **No interference:** separate rows, and sparse codes for episodes that look alike.
- **Consolidation:** this is the brain's "duplicate as a cache generated from a symmetric master copy". The weights learn
  both directions from the model's *own* reverse inferences, so no reversed data is needed.

---

## 2b. Compression: how the store stays small (the user's condition, 2026-10-18)

The user approved the plan "as long as there is a solution for efficiently compressing knowledge in the hippocampus". A
table of raw episodes grows with everything written, unless five mechanisms keep it down. Each one is tied to the
experiment that measures it.

| # | mechanism | what it saves | measured in |
|---|---|---|---|
| 1 | **Write gate: store only refutations.** First recognition (familiarity: already stored?), then surprise (> τ bits). What the weights or the store already hold costs nothing. This is ZipLearn's "create only on refutation". | duplicates and predictable material | **H1/H1b, WORKS:** surprise alone let through 19–21% duplicates; with recognition, **0** |
| 2 | **Minimal rows.** A row is the episode's token ids plus about 8 bytes of metadata. The index codes are a fixed function of the token ids, so they cost nothing to store. The overlap matrix is a cache, and an inverted index over token ids is the scalable replacement. | index storage | **H1, WORKS:** 14 bytes per fact (the cache is 256 bytes per row) |
| 3 | **Consolidation, then eviction.** Sleep moves an episode's content into the weights (with its self-generated inferences). An episode the weights then predict without memory is dropped. The store becomes a *buffer* of not-yet-compressed knowledge: its steady-state size tracks the rate of new information × the consolidation lag, not the total knowledge. | everything consolidated | **H2** (eviction by value without consolidation); **H3/H3b, WORKS:** with 5,000 sleep steps the weights hold both directions at ~0.97 (self-generated inferences), and self-checked eviction frees 64% of the store at ≤ 0.012 cost |
| 4 | **Merge into schemas** (ZipLearn rule 3, MERGE). Episodes that differ in one slot become one schema plus a list of slot values. Repeated structure is stored once. | shared structure | **H2** (designed when it is built) |
| 5 | **Entropy-coded rows** (optional). Store an episode as its arithmetic code under the model, about −log₂ p bits, in place of raw token ids. It can be decoded only with the weights that encoded it, so rows are re-encoded at sleep. | per-row bits, down to the surprise | **H4** option |

**The accounting.** The quantity to minimise is the bits in the weights plus the bits in the store. A good hippocampus
lowers the total by holding only what the weights cannot yet predict, for only as long as it takes to consolidate. This
is the same two-part code (description + exceptions) that ZipLearn priced.

---

## 3. Measures (common to the experiments)

| measure | definition |
|---|---|
| **one-shot accuracy** | hits@1 on facts written once, for forward and reverse questions, answered with memory |
| **exposures to criterion** | exposures per fact needed to reach a given accuracy: memory against weights only (the baseline is E1's ~85) |
| **retention** | accuracy on the first facts written, after N further writes (continual learning) |
| **false retrieval** | the fraction of reads whose top episode does not contain the answer |
| **consolidated accuracy** | accuracy after sleep with the store emptied: forward and reverse, with and without self-generated inferences |
| **cost** | memory reads per token, tokens added to the context, sleep compute in forward-pass equivalents |

---

## 4. Experiments (each pre-registered before it runs)

- **H1 — store plus read on synthetic facts. DONE** (§0). Arm D (causal-state index) was dropped from the pre-registration
  as a RAG-like arm; it may come back if direction-free retrieval needs defending.
- **H1b — a stronger reader and the familiarity gate. DONE** (§0).
  - Reuse `../intelligent_constraints/data.py`, with entity codes frozen random so the test facts' entities are as
    representable as the training ones.
  - **Train the read skill** on training facts: their episodes are in the store, and questions are asked in both
    directions.
  - **Test facts** are written once, forward only.
  - **Arms:**
    - (A) no memory, weights trained with 1 exposure and with k exposures;
    - (B) memory with the sparse union index;
    - (C) memory with a dense, order-free index (summed embeddings, cosine similarity), which tests pattern separation;
    - (D) memory with a causal-state index (the episode's last-token state, as in RAG), which tests direction-free
      retrieval.
  - **Main predictions:** B answers reverse questions about once-written facts far above A. D's reverse retrieval is
    worse than B's.
- **H2 — continual stream** (`EXPERIMENT_H2.md`). **DONE** (§0).
  - 11,930 facts (worlds 100–103) arrive in 24 phases, interleaved with 2,000 queries per phase drawn by Zipf
    popularity.
  - The store is uncapped, or capped at 3,000 rows with fifo, random, value or oracle eviction.
  - **Value** = (evidence + 1) / (age + 2,000): a usefulness rate with a prior, where evidence counts the times a row
    supplied the reader's answer.
  - **Measures:** crowding (interference without weight interference), forgetting by value, and online accuracy.
- **H3 — consolidation.** Sleep with and without self-generated reverse inferences, then empty the store and test the
  weights. Replay order (parts first vs random) is a factor, following ZipLearn E14.
- **H4 — contextual codes and stored states.** Codes from layer-ℓ states, and stored keys/values instead of re-reading
  tokens: cost against staleness.
- **H5 — a neural substrate. DONE (H5–H5c); not adopted** (the user, 2026-10-18: the 50 MB cost). The table stays the store.
- **H6 — real text.** The Latin corpus, long-tail facts, and distribution shift (Middle High German), where ZipLearn
  found the big gains. Diverse tasks are a standing rule.

---

## 5. Open problems and risks

1. **Learning to use the memory.** Reading retrieved episodes is a skill that must be trained with the memory in the
   loop, on entities other than the test ones.
2. **False retrievals inject noise.** The overlap threshold θ and the false-retrieval rate are measured in every
   experiment.
3. **Stale states** (H4). Consolidation changes the weights, so stored layer-ℓ states drift. The remedies are to store
   tokens and re-encode them during sleep, or to freeze the layers up to ℓ.
4. **Wrong inferences** (H3). An error generated during sleep is written into the weights by consolidation. The guard
   is to keep only inferences that are confident and consistent.
5. **Entity identity across mentions.** With real text, "his father" and "Joseph Momoa" must reach the same code. Token
   identity codes can't do that; contextual codes might. This is `../intelligent_constraints/IDEAS.md` C4: create a new
   entity only on refutation.
6. **It does not learn rules by itself.** The store is memorisation done right. Rules come from consolidation, where
   interleaved replay lets gradient descent find what episodes share.
7. **Expected size of the gains** (from ZipLearn E37): large on rare, new and shifted data, small on bulk text the
   weights already know.
8. **Retrieval crowding** (seen in H2's smoke test): when an entity has more episodes than the R read slots, a reverse
   cue overlaps all of them equally. Recall then falls (1.00 → 0.42 as the store grows to 11,930 rows). Candidate fixes:
   - a query code that includes the converse relation's *learned* partner;
   - a two-stage read (a coarse set, then reranking by the reader);
   - a larger R;
   - eviction or consolidation that keeps an entity's live episodes few.
9. **Readers stall, then break through** (H1b). Budgets and schedules must allow for the late transition. Seeds can
   differ in when it comes.

---

## 6. Prior art

| work | what it does | what this plan adds |
|---|---|---|
| Memorizing Transformers (Wu et al. 2022) | stores keys and values past the context window, retrieved by nearest neighbour | a direction-free sparse index; surprise-gated writes; eviction by value; consolidation |
| Larimar (Das et al. 2024) | one-shot writes to a distributed episodic memory, for knowledge editing | a table instead of a distributed memory (the ideal first); reversal and continual-learning tests |
| Titans (Behrouz et al. 2025) | a memory network written by gradient at test time, driven by surprise | writes without gradients, so no interference |
| HippoRAG (Gutiérrez et al. 2024) | retrieval over a knowledge graph, after the hippocampal indexing theory | no extracted graph or parse; the transformer reads raw episodes |
| EM-LLM (Fountas et al. 2024) | segments episodes at surprise peaks | consolidation into the weights |
| Whittington, Warren & Behrens 2022 | transformer ≈ the hippocampal formation (TEM-t) | the motivation for treating the context cache as the store to extend |

---

## 7. Standing rules

- Every test run takes ≤ 2 minutes.
- Pre-register every run; label exploratory work.
- Never re-run an unchanged experiment, controls included; log full curves the first time.
- Use 16-bit wherever it is accurate enough.
- Diverse tasks before conclusions.
- No compute on competitor-faithful arms.
- No heavy training on this machine.
- Commits go to `exp/mdl-vs-ntp`.
