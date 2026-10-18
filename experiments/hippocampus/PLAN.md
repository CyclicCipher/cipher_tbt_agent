# A hippocampus module for a standard transformer — plan

Started 2026-10-18. The user's words: "Let's change our approach. See 'fast storage without interference' … If you had
to design a 'hippocampus module' for a standard transformer, how would you do it?", then "Make a hippocampus experiments
folder with its own planning doc. What will the architecture of this hippocampus be? A neural net? A database? Some kind
of VSA/HDC system?"

Nothing here has been run. Every experiment below is planned, and each will be pre-registered in its own
`EXPERIMENT_*.md` before it runs.

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
2. **Write.** After a segment is processed, compute each episode's surprise, meaning its loss under the system *with* the
   current memory. Write it only if the surprise exceeds τ, that is, only if it refutes everything the system already
   knows (ZipLearn's "create only on refutation").
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

## 4. Experiments (planned; each pre-registered before it runs)

- **H1 — store plus read on synthetic facts.**
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
- **H2 — continual stream.** Facts arrive in sequence and the store is capped. Measure retention and interference, with
  eviction by value against first-in-first-out.
- **H3 — consolidation.** Sleep with and without self-generated reverse inferences, then empty the store and test the
  weights. Replay order (parts first vs random) is a factor, following ZipLearn E14.
- **H4 — contextual codes and stored states.** Codes from layer-ℓ states, and stored keys/values instead of re-reading
  tokens: cost against staleness.
- **H5 — a neural substrate.** Replace the table with a sparse Hopfield or HTM-style CA3 network, and measure capacity
  and interference against the table, which is the ideal.
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
