# Hierarchical Sequential Prediction Engine — Roadmap

## The Core Insight

Broca's area (BA44/45) was named for its role in language. But it activates equally for:
- **Action sequencing** — tool use, motor plans, cooking steps
- **Music** — harmonic expectation, rhythmic parsing, melodic structure
- **Mathematical reasoning** — parsing equation structure
- **Goal hierarchy** — "goals within goals" in planning (Fitch & Martins 2014)

The unifying principle is not language — it is **hierarchical sequential prediction**:
any structured stream where a noisy/ambiguous surface level is explained by
a cleaner, more predictable latent category level.

This is exactly the architecture of our engine:
- Surface predictions (next word, next action, next note) are **statistical** → use `freq_consolidate()`
- Latent categories (POS tag, goal type, harmonic role) are **near-deterministic** → use `synthesize()`
- Composition is **exact** → CTKG prerequisite chain

The same engine, synthesiser, ExampleStore, and CTKG infrastructure handles all domains.
Only the `.ctkg` domain file and the modality change.

---

## Phase L — Probabilistic Primitives (this session)

**Goal:** Make the engine able to consolidate statistical concepts into frequency distributions,
not just deterministic rules. This closes the gap between what `synthesize()` can handle
(deterministic) and what language/action/music require (distributional).

### Changes

**`memory.py`** — new distributional methods:
- `freq_predict(inputs)` — mode of the empirical distribution for these inputs
- `freq_dist(inputs)` — full empirical distribution (dict: output → probability)
- `build_full_freq_table()` — map all seen inputs to their output distributions
- `mean_log_likelihood(dist_fn)` — proper distributional evaluation metric

**`engine.py`** — distributional extension:
- `_dist_tables: Dict[str, Dict]` — stores freq tables for distributional concepts
- `freq_consolidate(concept_name)` — build freq table from ExampleStore, mark concept
- `ask()` — new priority 3: freq distribution prediction (before returning None)
- `kl()` — for distributional concepts, use mean cross-entropy vs. dist table
- `ask_dist(concept_name, inputs)` — return full probability distribution

**Why this matters:**
Before Phase L, the engine answered statistical questions with None (no exact match).
After Phase L, it answers with the most likely output from the empirical distribution.
KL for a distributional concept correctly reports the residual entropy — the fundamental
irreducibility of the task, not a synthesis failure.

---

## Phase M — Language Hierarchy (validation)

**Goal:** Demonstrate that the 4-stage POS hierarchy enables synthesis where flat bigrams cannot.
This is the first concrete test of the general architecture on a real-world task.

### The 4-stage CTKG chain

```
word_pos          (word -> POS tag)              deterministic  → synthesise
    |
next_pos          (pos_bigram -> next POS)        near-determ.  → synthesise
    |
word_given_pos    (context + next_pos -> word)   distributional → freq_consolidate
    |
next_word         (word_bigram -> word)           composed       → lookup chain
```

### Storage comparison
| Model | Storage | Entries (V=10K vocab, C=15 POS) |
|-------|---------|--------------------------------|
| Flat bigram | O(V²) | 100,000,000 |
| POS hierarchy | O(V·C + C²) | 150,225 |
| **Compression** | | **~665x** |

### Changes

**New `domains/language.ctkg`** — 4 concepts with prerequisite chain.

**`modalities/language.py`** — add POS tagging:
- `pos_of(word)` — returns POS tag (built-in corpus: lookup table; real corpus: spaCy)
- `pos_tag_examples()` — yields (word,) → (pos_tag,) pairs
- `next_pos_examples(context_size)` — yields (pos_seq,) → (next_pos,) pairs
- `word_given_pos_examples()` — yields (pos_seq, next_pos) → (word,) pairs

**`agent_loop.py`** — add `--hierarchy` flag:
- When `--hierarchy`: trains each level of the 4-concept chain separately
- Reports KL at each layer: word_pos → next_pos → word_given_pos → next_word
- Shows compression gain vs. flat bigram

### Expected experimental results

| Layer | Synthesis? | Expected test accuracy |
|-------|-----------|----------------------|
| word_pos | YES (deterministic) | ~95% (ambiguous words) |
| next_pos | YES or freq_consolidate | ~65-75% (top-1 POS bigram) |
| word_given_pos | freq_consolidate | measured by log-likelihood |
| next_word (composed) | lookup chain | > flat bigram on unseen contexts |

The key result: `next_word` via the hierarchy generalizes to **unseen bigram contexts**
(because it routes through POS, which was seen) where the flat bigram returns None.

---

## Phase N — Cross-Domain Formalization

**Goal:** Show the same infrastructure handles two more domains without engine changes.

### N.1 — Music hierarchy

```
note_role         (note -> harmonic role)         deterministic
    |
next_role         (role_seq -> next role)          near-determ.
    |
note_given_role   (key + next_role -> note)        distributional
    |
next_note         (note_seq -> note)               composed
```

CTKG file: `domains/music.ctkg`. Modality: `MusicModality` (MIDI stream input).
The engine code does not change — only the domain file and modality.

### N.2 — Minecraft action hierarchy

```
observe_state     (frame -> state_category)        deterministic (vision.ctkg)
    |
next_goal         (state + drives -> goal_type)    near-determ. (drive-modulated)
    |
action_given_goal (goal + context -> action)       distributional
    |
choose_action     (observation -> action)          composed
```

This is Phase K (dxcam/pynput/RCON) slotted into the hierarchy.
The metabolic drives (U_pain, U_hunger) bias `next_goal` at the distributional layer.

### N.3 — Motor control hierarchy

```
proprioception    (joint angles -> body_state)     deterministic
    |
movement_type     (body_state -> movement_cat)     near-determ.
    |
trajectory_given_type (goal + type -> trajectory)  distributional
    |
muscle_activation (observation -> activations)     composed
```

---

## Phase O — Unsupervised Category Discovery

**Goal:** The engine discovers latent categories without being told what they are.
This is the hardest phase — genuine concept formation without human-specified abstractions.

### Mechanism

1. **Observe surface examples** — accumulate (context, next_word) pairs
2. **Detect high residual KL** — flat bigram can't explain the data
3. **Cluster ExampleStore by output similarity** — words with similar distributions
   over next-words cluster together (distributional hypothesis: Firth 1957)
4. **Propose latent categories** — name clusters C₀, C₁, ..., Cₖ
5. **Add to CTKG dynamically** — `engine.add_concept()` for each cluster
6. **Re-synthesis with latent variables** — `next_pos` over discovered categories
7. **Confirm via KL drop** — if clustering helps, KL drops; else refine

### Key measurement: compression ratio

A discovered category is valid if and only if:
```
H(surface | category) + H(category) < H(surface)
```
i.e., adding the intermediate level reduces total description length (MDL principle).
This is the sheaf cohomology criterion: H¹ = 0 iff local observations glue cleanly.

### Implementation sketch

**`synthesis.py`** — new method `discover_categories(store, n_clusters)`:
- Compute output distribution for each unique input
- Cluster inputs by JS-divergence between their output distributions
- Return a mapping: input → cluster_id

**`engine.py`** — new method `induce_hierarchy(concept_name)`:
1. Calls `discover_categories()` on the ExampleStore
2. Adds cluster concepts to the CTKG dynamically
3. Adds prerequisite edges: cluster → concept
4. Trains `word_pos` (cluster assignment) from examples
5. Trains `next_pos` (cluster bigram) from examples
6. Re-measures KL — if improved, keep; else discard

---

## Phase P — Recursive Self-Improvement

**Goal:** The engine applies its own hierarchical synthesis to its own process language.

This is the Level 3 primitive layer from CTKG DESIGN.md:
- `quote(process_lines)` → represent a process as data
- `match(pattern, target)` → structural matching
- `substitute(pattern, replacement, target)` → rewriting
- `rewrite(rules, expr)` → apply rewriting rules repeatedly

With Level 3, the engine can:
1. Observe: "fold(b, a, succ) + carry solved all addition examples"
2. Abstract: "the fold-with-accumulator pattern solves counting-based operations"
3. Synthesise: a meta-process that generates fold templates for new domains
4. Apply: when shown a new counting-based problem, generate the right template automatically

This is the transition from **template search** (Phase O) to **template generation** (Phase P).

---

## Cross-Domain Infrastructure Map

| Component | Phase L | Phase M | Phase N | Phase O | Phase P |
|-----------|---------|---------|---------|---------|---------|
| `memory.py` | freq_predict, freq_dist | - | - | cluster_by_dist | - |
| `engine.py` | freq_consolidate, ask_dist | - | - | induce_hierarchy | self_improve |
| `synthesis.py` | - | - | - | discover_categories | meta_synthesize |
| `interpreter.py` | - | - | - | - | quote, match, substitute |
| `.ctkg` files | - | language.ctkg | music.ctkg, minecraft2.ctkg | auto-generated | auto-generated |
| Modalities | - | language.py (POS) | MusicModality | - | - |

---

## Why This Achieves General Intelligence

The claim: a system that can learn hierarchical sequential prediction across arbitrary domains,
with unsupervised category discovery and recursive self-improvement, is sufficient for
general intelligence in any environment that has compositional structure.

Evidence that most of cognition has compositional structure:
- Language: universal across human cultures (Chomsky, Hauser, Fitch 2002)
- Action planning: goals within goals (Miller, Galanter, Pribram 1960)
- Perception: hierarchical feature detectors (Hubel & Wiesel 1968; LeCun et al. 2015)
- Reasoning: Gentner (1983) structural mapping — analogy IS hierarchical relabelling
- Social cognition: theory of mind IS hierarchical prediction of agent goals

The engine's three-layer structure maps onto biological hierarchy:
- `ExampleStore` = hippocampus (episodic, verbatim)
- CTKG processes = neocortex (semantic, compressed)
- Distributional layer = cerebellum (forward model, probabilistic)
- `induce_hierarchy()` = prefrontal-hippocampal theta coupling (category formation)
- `self_improve()` = prefrontal metacognition (learning to learn)

---

## Comparison to Neural Approaches

| Property | Neural LM (Mamba3) | Symbolic Hierarchy |
|----------|--------------------|--------------------|
| Parameters | 1.26M | 0 |
| Training examples (addition) | 100 | 2 |
| Test accuracy (addition) | ~45% | 100% |
| Generalizes to unseen contexts | No (memorization) | Yes (via latent) |
| Interpretable process | No (black box) | Yes (CTKG DSL) |
| Catastrophic forgetting | Yes | No (ExampleStore) |
| Cross-domain transfer | Via fine-tuning | Via shared engine |
| New domain cost | O(data + compute) | O(ctkg file lines) |

The symbolic hierarchy is not a replacement for neural approaches — it is the
**scaffolding** that neural approaches fail to learn spontaneously (Mistake #44).
The long-term goal: hybrid system where symbolic hierarchy provides the skeleton and
neural modules fill in the distributional/perceptual details.
