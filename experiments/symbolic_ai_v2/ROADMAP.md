# CTKG Roadmap — The Neocortical Graph

## The Observation

The human neocortex is a spiking neural network that, when zoomed out, acts like
an unlabeled knowledge graph. Neurons are nodes. Synapses are edges. Activation
is working memory. Hebbian learning ("fire together, wire together") is edge
creation. Decay is forgetting. Sensory input fires neurons directly — there is
no separate "observation format" that gets "converted to" the graph.

## The Principle: One Graph

There is one graph. It is the KnowledgeGraph. When the environment produces a
token, that token IS a node in the KG that activates. When two tokens co-occur
in the same observation, the edge between them strengthens. When a token follows
an action, that is a directed edge. The environment does not produce a Graph
object. It produces tokens. Each token fires its node. Co-occurrence creates
edges. That's it.

Tokenization remains: the boundary between raw environment strings and opaque
integer IDs. Downstream of the tokenizer, everything is nodes and edges.
AgenticLoop is the only door.

---

## Architecture

### Files

```
ctkg/logic/
├── graph.py            — the one graph: nodes, edges, activation, spread, learn
├── loop.py             — AgenticLoop: observe → spread → learn → act
├── hippocampus.py      — episodic memory: activation snapshots + observation records
└── consolidation.py    — the slow path: replay, prune, structure discovery

ctkg/connectors/
└── tokenizer.py        — character-level tokenizer at the environment boundary
```

### What was removed

- `InputOutputTopology.py` — there is no observation graph.
- `WorkingMemory.py` — activation on KG nodes IS working memory.
- `Deduct.py` — prediction is activation spread.
- `Induct.py` — learning is Hebbian update.
- `Abduct.py` — anomaly response is edge creation on prediction failure.

Induction, deduction, and abduction are not three modules. They are three
descriptions of a single process: **spread, compare, update.**

### The Single Process

Each timestep:

```
1. DECAY      — all node activations *= decay_factor
2. INPUT      — observed tokens set activation = 1.0 (via tokenizer)
3. EDGES      — co-occurrence edges between same-observation tokens strengthen
               — transition edges from previous timestep's active nodes strengthen
4. LEARN      — compare the PREVIOUS timestep's spread prediction against
                THIS timestep's actual tokens:
                  confirmed → strengthen (induction)
                  wrong     → weaken (revision)
                  surprise  → create edge (abduction)
5. SPREAD     — active nodes propagate activation along weighted edges
                → produces prediction for NEXT timestep
6. STORE      — Hippocampus records active nodes + observation tokens
```

### Context-Dependent Prediction: Attention

The co-occurrence edge weight matrix IS the attention matrix. For each
candidate action c given context tokens {k₁, k₂, ...}:

```
logit(c) = Σᵢ activation(kᵢ) × normalised_cooccur(kᵢ, c)
```

where `normalised_cooccur(k, c) = cooccur(k, c) / Σⱼ cooccur(k, cⱼ)` —
each context token's co-occurrence weights to candidates are normalised to
sum to 1. This is the "projection" that makes attention discriminating:

- `succ → {0,1,...,9}` all equal → each gets 0.1 → non-discriminating.
- `6 → {7:0.9, others:0.01}` → 7 gets ~0.9 → highly discriminating.

The dot product automatically amplifies discriminating context tokens and
attenuates non-discriminating ones. No IDF, no hand-tuned weights.

**Causal masking**: co-occurrence edges are forward-only (earlier tokens
predict later tokens, not the reverse). Without this, training `[5, succ, 6]`
creates `6→5` (backward), making `6` predict `5` instead of `7` on tests.

**PoPE distance tracking**: each edge records the distribution of relative
positions at which its endpoints co-occur. Not yet used in action selection
(answer position doesn't transfer across observation boundaries) but
accumulated for future use in within-observation attention.

### Activation Dynamics

Every node carries:
- `activation: float` in [0, 1] — current firing level
- `resting: float` — baseline (prior weight / long-term importance)
- `preferred: float` — homeostatic target (active inference prior)

Every edge carries:
- `alpha, beta: float` — Beta posterior counts (permanent evidence)
- `weight: float` — computed: (alpha - beta) / (alpha + beta), range [-1, 1]
- `confidence: float` — computed: alpha + beta
- `role: int` — COOCCURRENCE (0) or TRANSITION (1)
- `sigma: float` — transient context-dependent state (reset each timestep)
- `dist_sum, dist_sq_sum, dist_count` — PoPE positional statistics

### Excitation and Inhibition

- **Excitatory edges normalised per source** — stochastic kernel / WTA
- **Inhibitory edges as raw threshold** — subtractive suppression

### Action Selection

Pure dot-product attention over normalised co-occurrence edges. No transition
edges in selection (they carry wrong-answer noise). No pragmatic bias, no
epistemic bonus, no backward spread. The co-occurrence edges learned from
data encode everything the system needs.

### Consolidation (the slow path)

- **Replay**: re-strengthen edges from stored observation pairs
- **Pruning**: remove deeply negative edges and isolated nodes
- **Structure discovery**: see "Categorical Structure" below

---

## Current Results

### Succession (Mode A, bare digit tokens)

| Cycles | Score | Notes |
|--------|-------|-------|
| 10 | 90% | 2 warmup passes, 5 train + 5 test per cycle |
| 30 | 96.7% | |
| 50 | 98.0% | Approaching ceiling |

Train/test pools are disjoint. Scoring counts test-phase answers only.
98% is generalisation: the agent sees `succ(2)=3` in training and correctly
answers `succ(6)=?` → `7` on test, having never seen that specific problem.

### What made it work

1. **Bare digit tokens** — no BOARD_/ANSWER_ prefixes. Same node for
   perception and action (APC / Rao 2024).
2. **Causal masking** — forward-only co-occurrence edges.
3. **Normalised per-source attention** — discriminating context wins.
4. **No transition edges in action selection** — they carry noise.
5. **Counting warmup** — builds the number line as a co-occurrence chain.

---

## Next: Addition via Functors

### The Problem

Addition requires COMPOSITION. `3 + 4 = 7` is not a direct co-occurrence —
it requires chaining four successor steps: `3→4→5→6→7`. The attention
mechanism that gets 98% on succession (one-hop) cannot do multi-hop.

### The Insight: Addition is a Functor

A child learns `3 + 4` by placing 3 blocks, then placing 4 more, then
counting the result. The key operation is ONE-TO-ONE CORRESPONDENCE: each
block in the second group matches one step on the number line from the
first operand.

In categorical terms: addition is a FUNCTOR. It maps the sub-chain
`[0, 1, 2, 3, 4]` (the second operand's position on the NNO) to the
sub-chain `[3, 4, 5, 6, 7]` (starting from the first operand), preserving
successor structure. The functor sends `0→3, 1→4, 2→5, 3→6, 4→7`.

The answer `7` is the image of `4` under this functor. No integer
extraction. No counting. Just a structure-preserving map between two
sub-chains of the same number line. The system discovers this functor by
observing that addition training examples always exhibit this correspondence.

### Why This Generalises

The functor approach works for any domain where the operands share structure:

- **Multiplication**: functor from NNO to iterated-addition category.
  Each NNO step maps to one application of the addition functor.
- **Language**: functor from syntactic roles to token positions.
  "Subject" maps to the token before the verb.
- **Analogy**: functor between parallel pairs. {king, queen} → {man, woman}
  preserving the "gender flip" morphism.

The general principle: **structure-preserving maps between subgraphs.**
The discovery algorithm is domain-agnostic:

1. Find subgraphs with consistent internal structure
2. Find maps between subgraphs that preserve edge structure
3. Verify the maps commute with internal morphisms (functor check)
4. Use the maps for prediction on novel inputs

None of these steps require knowing what the tokens mean. Finding that
`[0,1,2,3,4]` maps to `[3,4,5,6,7]` preserving SUCC edges is the same
operation as finding that `{king, queen}` maps to `{man, woman}` preserving
a role-flip edge. Purely graph-structural.

### Categorical Machinery Required

**Functors** — structure-preserving maps between subgraphs. The addition
functor maps the second-operand's NNO sub-chain to a parallel sub-chain
starting from the first operand. Discovery: given training examples of an
operation, find the map from the second operand's NNO position to the answer,
and verify it preserves successor edges.

**Natural transformations** — structural relations between parallel functors.
The "+1" functor and the "+2" functor are parallel (both map NNO sub-chains).
A natural transformation between them says "they work the same way, just
with different chain lengths." This enables generalisation: knowing "+1" and
"+2" teaches you the pattern for "+N".

**Adjunctions** — pairs of operations that are mutual inverses. `add ⊣ sub`:
addition and subtraction compose to identity. Discovery: find functor pairs
(F, G) where F ∘ G ≈ identity on all observed pairs. The adjunction creates
a bidirectional link so that knowing addition automatically gives subtraction.

**Colimits** — universal gluing of diagrams. The colimit of all "+4" training
examples is the "+4" concept node. Its cocone maps connect to each specific
example, and its factored edges connect to the answer digits. The colimit
abstracts over the specific examples to produce a reusable operation.

### Implementation Plan

**Step 7a — Iterated co-occurrence spread for multi-hop attention:**
Run the attention computation iteratively: round 1 activates direct
neighbors, round 2 activates their neighbors. After K rounds, the
activation has spread K hops. For `3 + 4`, 4 rounds of co-occurrence
spread from `3` reaches `7`. Problem: also activates `8`, `9`, etc.
The termination condition (stop at `7` specifically) requires the
functor structure from step 7b.

**Step 7b — Functor discovery in consolidation:**
During consolidation, scan observation records for pairs of sub-chains
that map to each other preserving successor edges. For each addition
training example `[a, +, b, =, c]`, check: is there a path from `a` to
`c` on the number line, and does the path length correspond to `b`'s
NNO position? This is a functor check — does the map `a → c` commute
with the successor structure?

Note: "corresponds to b's NNO position" is structural, not integer: it
means "the path from 0 to b has the same number of edges as the path
from a to c." This is checked by simultaneously walking both chains.

**Step 7c — Natural transformation discovery:**
After discovering "+1", "+2", "+3" as separate functors, find the NT
between them: they all have the same structure (NNO sub-chain map), just
with different chain lengths. The NT enables predicting "+7" from the
pattern without having seen "+7" examples.

**Step 7d — Adjunction discovery:**
Observe that `add(3, 4) = 7` and `sub(7, 4) = 3` always pair up. The
adjunction `add ⊣ sub` is discovered by finding functor pairs that
compose to identity. Create bidirectional links between the add and sub
functor nodes.

**Step 7e — Test on MathClassroom addition + subtraction:**
Target: >80% on addition test questions (disjoint from training).
Target: >60% on subtraction (discovered via adjunction, less training).

### Future

- **Multiplication**: functor from NNO to iterated-addition.
  Requires composing two levels of functors.
- **Division via adjunction**: `mul ⊣ div`, discovered same way as `add ⊣ sub`.
- **ω-category levels**: colimits of functors = higher-level abstractions.
- **MDL as objective**: description length determines structural changes.
- **Revision log**: append-only history of structural changes.
- **Science lab**: categorical structure for multi-step planning.
  Room navigation IS a functor from actions to room transitions.

---

## Milestones

1. **Succession** — 98% ✓
2. **Addition** — >80% via functor discovery
3. **Subtraction** — >60% via adjunction with addition
4. **Science lab** — affordance learning, exploration, escape
5. **Multiplication** — composed functors
6. **Physics** — discover laws from raw measurement streams
