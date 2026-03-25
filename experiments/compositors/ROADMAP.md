# Compositor Roadmap

## Lessons learned (v0.1 through v0.5)

Everything below was verified empirically. Each experiment isolated one
variable. The results are unambiguous.

### What we tried and what it proved

| Version | Change | Train | Test | Verdict |
|---------|--------|-------|------|---------|
| v0.1 | Bilinear relation heads | 72% | 27% | Graph learned nothing. Redundant attention. |
| v0.2 | Explicit A[K,N,N] + identity + softmax | 72% | 27% | Identity + softmax killed off-diagonal gradients. |
| v0.3 | Sigmoid edges, no identity, weight tying | 72% | 29% | Node embeddings differentiate. Edges still ignored. |
| v0.3 + seeding | Non-gradient edge seeding from data | 72% | 29% | Graph has correct edges. Model ignores them. |
| v0.3 + oracle | Perfect successor/predecessor edges | 72% | 29% | **Architecture can't use graph knowledge.** |
| v0.3.1 | Dual-stream SwiGLU (separate comp input) | 72% | 28% | FFN uses comp_out but comp_out is noise. |
| v0.4 | Hard node lookup, node embs as values | 72% | 28% | Clean retrieval. FFN still scrambles it. |
| v0.4 + RMSNorm | Scale-matched comp_out | 72% | 28% | Scale isn't the issue. out_proj scrambles. |
| v0.4 - out_proj | Weighted average, no projection | 72% | 28% | Attention bypass lets model ignore graph. |
| v0.5 | Attention→Composition→FFN (no bypass) | 67% | 30% | **Model IS forced to use graph. Overfits.** |

### Root cause chain

1. **v0.1-v0.2**: Graph had no structure. Fixed by sigmoid + seeding.
2. **v0.3-v0.4**: Graph had structure but model ignored it. Attention + FFN
   solved training without the graph. The composition layer was an optional
   side channel that got optimized away.
3. **v0.5**: Blocking the attention bypass forced the model through the graph.
   It learned (67% train), but the FFN came AFTER composition, so it could
   only post-process an already-completed traversal. It couldn't decide
   WHICH traversal to perform.
4. **The FFN was in the wrong position.** It should WRAP the composition
   layer, not follow it.
5. **No multi-digit structure.** The graph operates on single characters.
   Multi-digit numbers need products, functors, and place-value structure
   that the graph has no mechanism to represent.
6. **No test-time learning.** The graph is frozen at test time. But its
   structural operations don't need backprop — they should run on the
   test input prefix before prediction.

### Design insights from discussion

7. **The FFN's width is irrelevant.** The bottleneck is the interface
   between the FFN and the composition layer (K relation weights), not the
   FFN's internal capacity. Making the FFN wider computes K numbers more
   expressively but cannot express richer graph queries.
8. **The FFN needs to see the graph before querying it.** A pre-read of the
   current node's adjacency row shows what edges are available. Without it,
   the FFN formulates blind queries — it doesn't know what the graph contains.
9. **Single-step traversal cannot implement addition.** Addition requires
   iterating successor N times. This is depth (looping), not width.
10. **Iterated operations should consolidate into direct edges.** When the
    loop computes 3+5=8 via 5 successor steps, a structural operation should
    cache the result as a direct edge. Next time, one lookup replaces five.
    This is the Kan extension: compressing a chain of morphisms into a single
    composed morphism.

---

## Phase 6 — Looped FFN-Composition sandwich [NEXT]

**Goal:** The FFN wraps the composition layer, and the pair iterates.
Each iteration: the FFN reads the current state (including the previous
graph result), selects a graph operation, the composition layer executes
it, and the result feeds back for the next iteration.

### Architecture

```
Input: x (attention-enriched position states), input_ids

# Pre-read: show the FFN what the graph looks like from here
adj_row = adj[:, input_ids, :]          # (K, B*S, N) — available edges
pre_read = summarize(adj_row)           # (B, S, K) or richer

state = x                              # initial state from attention

for step in range(max_steps):
    # FFN first half: what do I want from the graph?
    h = norm(state)
    navigation = w1(h) * silu(w_gate(h))     # (B, S, d_hidden)

    # Relation selection: informed by pre-read + navigation
    rel_weights = relation_selector(navigation, pre_read)  # (B, S, K)

    # Composition: execute one graph traversal step
    graph_result = compose(input_ids, adj, rel_weights, graph.nodes)
                                              # (B, S, d_model)

    # Update state for next iteration
    # The graph result is in embedding space — project to d_hidden,
    # combine with navigation, project back to d_model
    graph_hidden = graph_proj(graph_result)   # (B, S, d_hidden)
    combined = navigation * graph_hidden      # gated combination
    state = state + w2(combined)              # residual update

    # Update node for next hop: which node did we arrive at?
    # The graph result is a weighted sum of node embeddings.
    # The strongest-matching node becomes the new lookup point.
    input_ids = (graph_result @ graph.nodes.T).argmax(dim=-1)

    # Halting: FFN can learn to produce zero-norm navigation when done
    if navigation.norm() < threshold:
        break

Output: state (added to residual)
```

### Why this is different from all previous attempts

1. **The FFN sees the graph before querying it** (pre-read). It knows what
   edges are available, not just what context attention provided.
2. **The FFN controls the traversal** (relation selection from navigation).
   It decides which relation to follow based on context AND graph structure.
3. **The graph result feeds back** (state update + node update). Multi-hop
   traversal emerges from iteration, not from fixed n_compose_hops.
4. **Node identity updates between iterations.** After traversing succ(7)=8,
   the next iteration starts from node 8, not node 7. The loop walks along
   the graph's edges.

### What each iteration computes

- Iteration 1: "I'm at node 7, context says succession, pre-read shows
  succ edge to 8. Select successor. Graph returns embedding(8)."
- Iteration 2: "I'm now at node 8, context still says succession. Pre-read
  shows succ edge to 9. Select successor. Graph returns embedding(9)."
- Iteration 3: "I'm at node 9, halting condition met (context says predict
  only one successor). Stop."

For addition (3 + 5):
- Iterations 1-5: walk from 3→4→5→6→7→8 via successor
- Iteration 6: halt (counter reached 5, the second operand)

### The halting problem

The FFN must learn WHEN to stop iterating. Options:
- **Learned halt:** The FFN's navigation intent includes a halt signal.
  When the gate produces near-zero output, the loop stops.
- **Counter from context:** Attention provides the number of steps (e.g.,
  the second operand in addition). The loop runs that many times.
- **Fixed max_steps with soft halt:** Always run max_steps iterations,
  but the FFN learns to produce identity-like updates after it's done.
  This is simpler to implement and avoids dynamic control flow.

Start with fixed max_steps + soft halt. The FFN learns to "do nothing"
on iterations past the useful ones.

### Validation

Oracle graph + looped sandwich. Test succession first:
- Single-digit: succ(7) should work in 1 iteration.
- Verify the node update: after traversing succ(7)=8, is the next
  iteration's lookup correctly starting from node 8?

Then test addition:
- 3+5: does the loop walk 5 steps of successor from node 3?
- This requires max_steps ≥ 5 and the FFN learning to count.

---

## Phase 7 — Composition memoisation

**Goal:** When the loop computes a result via N iterations, cache it
as a direct edge. Next time, one lookup replaces N iterations.

### What this is (categorically)

This is **morphism composition followed by memoisation**. The loop
computes succ∘succ∘succ(3)=6 via three compositions. Caching the result
as a direct edge 3→6 is storing the composite morphism. This is an
operation that exists in the definition of any category — it is NOT a
Kan extension.

A Kan extension would be Phase 8b (pattern extraction) — extending a
*partial* functor to a *total* one. See Phase 8b.

### The lifecycle of a single fact

```
1. COMPUTATION (Phase 6 loop):
   3+5 → iterate succ 5 times: 3→4→5→6→7→8
   Slow: 5 graph reads.

2. MEMOISATION (this phase):
   Record trace: (start=3, relation=succ, steps=5, end=8)
   Create direct edge: A[memo_rel, 3, 8] = 6.0
   Structural op, no backprop.

3. DIRECT LOOKUP (post-memo):
   Next time 3+5 is queried, the FFN sees the direct edge from 3→8
   in the pre-read. Selects the memo relation. One graph read returns 8.
   Fast: 1 graph read.
```

### Limitation: this only helps for EXACT REPEATS

Memoising 3+5=8 gives you 3+5=8 next time. It does NOT give you 4+5=9.
Every (a,b) pair requires its own computation and its own memo edge.
For single-digit addition, that's 100 memo edges (10×10). This is a
lookup table, not understanding. It is useful as a speedup but does not
generalise.

The generalisation step is Phase 8b.

### Implementation

```python
def memoise_traces(graph, traces, min_count=2):
    """Cache repeated loop results as direct edges.

    This is morphism composition + memoisation. NOT a Kan extension.

    traces: list of (start_node, relation, n_steps, end_node)
    """
    pair_count = defaultdict(int)
    for start, rel, steps, end in traces:
        pair_count[(start, end)] += 1

    # Use a dedicated relation slot for memoised edges
    memo_rel = graph.n_relations - 1
    for (start, end), count in pair_count.items():
        if count >= min_count:
            logit = min(6.0, math.log(count))
            graph.A.data[memo_rel, start, end] = logit
```

### When to memoise

- **During training:** every N epochs, same as composition closure.
- **At test time:** after the loop computes a result, immediately record
  the trace. If the same (start, end) pair has been seen before,
  memoise.

---

## Phase 8 — Two kinds of learning at test time

### Phase 8a — Observation (from input prefix)

**Goal:** The graph updates its structure from the test input prefix
before predicting. No backprop required.

The `seed_graph_from_data()` function already parses sequences and sets
edge logits from statistical regularities. At test time, run the same
parsing on the input prefix:
- Detect token transitions across delimiters
- Strengthen edges that match observed transitions
- Run composition closure

### Phase 8b — Law discovery via the parametrized NNO

**Goal:** Given a collection of memoised facts, discover the recursion
scheme they instantiate, store it as a law node, and use it to compute
results for ANY inputs — including unseen ones.

This is the step that turns a lookup table into understanding.

#### The categorical construction: parametrized NNO

The Natural Number Object has a universal property: given any object A
(the "parameter"), a base-case morphism `f: A → B`, and a step morphism
`g: B → B`, there exists a **unique** morphism `h: ℕ × A → B` satisfying:

```
h(0, a)       = f(a)          — base case
h(succ(n), a) = g(h(n, a))    — recursive step
```

**Addition** is the textbook instance:
- f = identity on ℕ (base: add(0, b) = b)
- g = successor (step: add(succ(a), b) = succ(add(a, b)))

**Multiplication** composes on top:
- f = zero morphism (base: mul(0, b) = 0)
- g = add(b, −) — the curried ADD law node (step: mul(succ(a), b) = add(b, mul(a, b)))

**Exponentiation** composes on top of that:
- f = constant-1 morphism (base: exp(0, b) = 1)
- g = mul(b, −) — the curried MUL law node

The Paré-Roman theorem guarantees that EVERY primitive recursive function
of k variables is realizable as a morphism ℕᵏ → ℕ in any category with
finite products and a parametrized NNO. One construction, any arity.

#### What to store in the graph

For each discovered parametric operation, create a **law node** with
exactly three outgoing edges:

```
Law node: ADD
  ├── BASE_CASE  →  identity morphism on ℕ    (what to return when induction var = 0)
  ├── STEP       →  succ morphism              (how to transform the result each step)
  └── RECURSE_ON →  arg position 0             (which argument is the induction variable)

Law node: MUL
  ├── BASE_CASE  →  zero morphism              (mul(0, b) = 0)
  ├── STEP       →  ADD law node (curried)     (mul(succ(a), b) = add(b, mul(a, b)))
  └── RECURSE_ON →  arg position 0

Law node: EXP
  ├── BASE_CASE  →  constant-1 morphism        (exp(0, b) = 1)
  ├── STEP       →  MUL law node (curried)     (exp(succ(a), b) = mul(b, exp(a, b)))
  └── RECURSE_ON →  arg position 0
```

Three edges per law, regardless of arity. The STEP edge of one law can
point to another law node — this is how operations compose.

#### Why arity is uniform

A k-ary operation `f(a₁, ..., aₖ)` always has exactly ONE induction
variable (indicated by RECURSE_ON). The remaining k−1 arguments form
the parameter object A — a product/tuple. The evaluator doesn't know
or care about k:

```
evaluate(law_node, induction_val, param_tuple):
    result = BASE_CASE(param_tuple)
    for i in range(induction_val):
        result = STEP(result)      # STEP may itself invoke another law
    return result
```

Same code for unary (predecessor: STEP=pred, BASE=0), binary (add),
ternary, or any arity. The parameter tuple is "everything except the
induction variable."

#### Currying: partial application for free

A morphism `add: ℕ × ℕ → ℕ` can be curried to `add_: ℕ → (ℕ → ℕ)`.
This means `add(3, −)` is a first-class morphism ℕ → ℕ — "add 3 to
whatever input you give me." The STEP edge of MUL points to exactly
this curried form. No need to store 10 separate "add-k" morphisms;
currying derives them from the single ADD law node.

#### How discovery works (from memo table to law node)

The memo table contains entries grouped by loop trace structure:

```
Memo table (grouped by step morphism used):
  Using succ as step:
    (induction=0, param=5) → 5
    (induction=1, param=5) → 6
    (induction=2, param=5) → 7
    (induction=3, param=5) → 8
    (induction=0, param=3) → 3
    (induction=1, param=3) → 4
    (induction=2, param=3) → 5
    ...
```

The discovery algorithm checks two conditions:

**1. Consistent base case?** For all entries where induction_val = 0,
does `f(0, b) = g(b)` for some morphism g already in the graph?
Above: `f(0, b) = b` for all b → g is the identity. ✓

**2. Consistent step?** For all entries where both `f(n, b)` and
`f(n+1, b)` exist, does `f(n+1, b) = h(f(n, b))` for some morphism h
already in the graph? Above: `f(n+1, b) = succ(f(n, b))` → h is
successor. ✓

If both hold, create: `LawNode(BASE_CASE=g, STEP=h, RECURSE_ON=0)`.

This check is purely structural. It does not know what addition is.
It detects that the memo table satisfies the NNO universal property
for some (g, h) pair, and stores them. The same check discovers
multiplication (g=zero, h=curried-ADD), exponentiation (g=one,
h=curried-MUL), and any other primitive recursive function.

#### Minimum data for discovery

The algorithm needs entries for at least two distinct values of the
induction variable (to detect the step pattern) and at least two
distinct values of the parameter (to confirm the base case is
parametric, not a coincidence). In practice, ≥ 10 entries with ≥ 3
distinct induction values and ≥ 3 distinct parameter values.

#### How the loop uses a discovered law

Once the ADD law node exists, the Phase 6 loop changes behaviour:

**Before discovery:** The FFN selects relation=succ repeatedly,
walking 3→4→5→6→7→8 in 5 iterations. Slow.

**After discovery:** The evaluator sees the ADD law node in the graph.
Given add(5, 3): read RECURSE_ON=0, so induction_val=5, param=3.
Apply BASE_CASE(3) = 3. Apply STEP=succ 5 times: 3→4→5→6→7→8. Return 8.

Wait — this is still 5 iterations. The speedup comes from TWO sources:

1. **The loop doesn't need to DECIDE what to do at each step.** Without
   the law node, the FFN must independently select relation=succ at
   every iteration, and the model must learn when to stop. With the law
   node, the evaluator knows the step morphism and the iteration count
   upfront. No FFN decisions needed. Deterministic execution.

2. **Composition compresses further.** Once ADD exists, MUL's step is
   `add(b, −)` — which is itself a law-node invocation. The evaluator
   calls ADD as a subroutine. And when enough addition results are
   memoised, the evaluator can shortcut individual additions via memo
   lookup. The tower of laws + memo cache converges toward O(1) for
   any previously-seen sub-computation.

#### What this is NOT

This is not a Kan extension. The Kan extension would be the step where
the system, given a partial functor defined on observed facts, extends
it to ALL of ℕ × ℕ via a universal property. What we are doing here is
more concrete: we detect the NNO recursion scheme in the data, and store
(BASE_CASE, STEP) — the two morphisms that the NNO universal property
says uniquely determine the operation. The operation is then defined for
all inputs by the universal property itself, without needing to store
individual facts.

The relationship to Kan extensions: the NNO universal property IS a
special case of a left Kan extension (extending along the inclusion
{0, succ} ↪ ℕ). But calling it by its specific name — the parametrized
NNO — is more precise and more useful for implementation.

---

## Phase 8c — Online learning during generation

**Goal:** Combine observation (8a), memoisation (7), and law discovery
(8b) into a single test-time loop.

```python
def predict_with_online_learning(model, prompt, max_new_tokens):
    # 1. Observe the prompt (Phase 8a)
    observe_sequence(model.graph, prompt)
    composition_closure(model.graph)

    # 2. Generate with trace recording
    traces = []
    for i in range(max_new_tokens):
        token, trace = generate_one_token(model, prompt)
        traces.append(trace)
        prompt += token

        # 3. Memoise repeated results (Phase 7)
        memoise_traces(model.graph, traces)

        # 4. Discover law nodes when enough memo entries exist (Phase 8b)
        #    Check: do the memo entries satisfy the NNO universal property
        #    for some (BASE_CASE, STEP) pair already in the graph?
        if enough_memo_entries(model.graph):
            discover_laws(model.graph)

    return prompt
```

The lifecycle during generation:
1. **Early tokens:** computed via the Phase 6 loop (slow, FFN decides
   each step). Loop traces are recorded.
2. **After repeats:** individual results memoised (Phase 7). Exact
   repeats become O(1) lookups.
3. **After enough data:** law node discovered (Phase 8b). ALL inputs
   become deterministic — the evaluator reads (BASE_CASE, STEP,
   RECURSE_ON) and executes without FFN decisions.
4. **After laws compose:** MUL's STEP points to ADD law node. Nested
   operations are evaluated by recursive law-node invocation. The memo
   cache shortcuts previously-seen sub-computations.

---

## Phase 9 — Products (chunking)

**Goal:** Multi-digit numbers become single entities in the graph.

### The problem

"47" is two characters. The graph has no node for "47". The loop can
walk succ(7)=8, but it doesn't know the '7' is a units digit or that
the tens digit '4' should be preserved.

### Categorical construction: products

Create concept node C_47 with projections:
- π₁: C_47 → '4' (tens)
- π₂: C_47 → '7' (units)

Successor on C_47: succ(C_47) = C_48, with:
- π₁(C_48) = '4' (tens preserved)
- π₂(C_48) = succ('7') = '8'

### Trigger

Detect character sequences that consistently appear between delimiters.
Count (char, char, delimiter) triples across training data. Frequent
triples → product nodes. This is measurable without task knowledge.

### Impact on the loop

With product nodes, the loop can operate at the NUMBER level:
- Start at C_47, apply number-level successor, arrive at C_48
- The projections decompose C_48 back into characters for output

The loop at the number level is ONE step (succ(47)=48) instead of the
character-level loop (copy '4', succ '7' → '8', detect no carry).

---

## Phase 10 — Functors

**Goal:** Succession on multi-digit numbers decomposes into operations
on individual digits via a structure-preserving map.

### Categorical construction

A functor F: Numbers → DigitPairs maps:
- F(47) = (4, 7)                        — object map
- F(succ) = (id, succ)    when units ≠ 9 — morphism map (no carry)
- F(succ) = (succ, wrap)  when units = 9 — morphism map (carry)

### Discovery

Given product nodes and successor edges, detect the pattern:
1. succ(C_47)=C_48: tens preserved, units = succ(units). No carry.
2. succ(C_49)=C_50: tens incremented, units wrapped 9→0. Carry.

Induce the functor as a natural transformation with two cases,
selected by the condition "units digit = 9."

### Comparison and ordering

With functors, comparison becomes lexicographic:
- Compare(47, 23): compare tens first (4>2), done.
- Compare(47, 43): tens equal (4=4), compare units (7>3), done.

This IS a functor from number-ordering to digit-tuple ordering.

---

## Phase 11 — Categorical constraints

**Goal:** Enforce axioms so the graph is a well-formed category.

### 11a. Composition consistency

If A→B→C and A→C both exist, their values should agree.
Soft loss term during training.

### 11b. Functor preservation

If F is a discovered functor and f: A→B is a morphism, verify
F(f): F(A)→F(B) exists and is consistent.

### 11c. Path independence

If the loop computes A→C via two different paths (A→B₁→C and A→B₂→C),
the memoised direct edge should be the same. This is the requirement
that composition is well-defined — a basic categorical axiom, not a
Kan extension.

---

## Phase 12 — Scaling and evaluation

### 12a. Parameter-matched comparison

Compositor vs baseline transformer at ~700K params.
Measure: OOD accuracy, data efficiency, compositional transfer.

### 12b. Harder tasks

Subtraction (predecessor iteration), multiplication (nested loops),
multi-digit arithmetic (products + functors + carry).

### 12c. Non-arithmetic domains

Syntax parsing, pattern completion, any compositional task.

---

## Success criteria (revised)

| Phase | Question | Pass condition |
|-------|----------|----------------|
| 6 | Can the looped sandwich navigate the graph? | Oracle single-digit succession > 50% OOD |
| 6 | Can the loop implement addition? | Oracle 3+5=8 via iterated successor |
| 7 | Does memoisation speed up repeated facts? | Second computation of same sum uses fewer iterations |
| 8a | Does observation from prefix help? | OOD accuracy improves with prompt context |
| 8b | Does law discovery work? | ADD law node created from memo table with correct (BASE_CASE=id, STEP=succ) |
| 8b | Does the law generalise? | Unseen (a,b) pairs computed correctly via law-node evaluation |
| 8b | Does composition work? | MUL law node's STEP correctly points to ADD law node |
| 9 | Do products enable multi-digit? | succ(47)=48 works |
| 10 | Do functors generalise? | Succession works for unseen number ranges |
| 11 | Do constraints help or hurt? | Compare with/without |
| 12 | Does Compositor beat transformer? | Higher OOD at same param count |

## Priority order

Phase 6 is the blocker. The looped sandwich is the core mechanism —
if the FFN can't navigate the graph via iteration, nothing else works.
Phase 7 is a speedup for exact repeats (useful but not deep). Phase 8b
is the first real intellectual milestone: discovering that the memo
table satisfies the NNO universal property for some (BASE_CASE, STEP)
pair, creating a law node, and using it to compute results for ANY
inputs — including those never seen in training. The same discovery
algorithm finds addition, multiplication, and exponentiation (each
law's STEP can reference another law node). Phases 9-10 are the
categorical machinery for multi-digit generalisation.

## What is NOT on this roadmap

- Resurrecting the attention bypass (v0.1-v0.4 proved it lets the model
  ignore the graph).
- Separate V embeddings (node embeddings as values work and have direct
  gradient from loss).
- Soft node lookup (hard lookup from input_ids eliminates ambiguity).
- Fixed n_compose_hops (replaced by the loop with dynamic node updates).
- Wider FFN (the bottleneck is the interface, not the capacity).
- Calling memoisation a Kan extension (it's morphism composition + caching).
