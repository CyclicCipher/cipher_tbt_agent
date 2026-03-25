# Compositor Design Log

Permanent record of design decisions, hypotheses, and results.

---

## Entry 001 — Initial toy model (2026-03-21)

### Hypothesis

Replacing nn.Embedding with a categorical graph structure (where embeddings
are pointers into a graph, and a composition layer queries the graph for
morphisms between tokens) will produce better compositional generalisation
than a standard transformer of equal size, trained on the same data.

### Design decisions

**Vocabulary:** 16 character-level tokens: `0-9`, `+`, `=`, `<`, `>`, `,`, ` `.
No BPE, no subword. The model must discover multi-digit numbers from characters.

**Graph structure:** The graph has N_nodes learned node embeddings (d_model each).
- First V nodes (V=vocab size) correspond to vocabulary tokens.
- Remaining N_nodes - V are "concept slots" — free nodes the model can use
  for learned abstractions (multi-digit numbers, operations, etc.).
- Morphisms are NOT stored as an N×N tensor (O(N²), doesn't scale).
  Instead, morphisms are computed via K bilinear relation heads:
  `morph_k(i, j) = (nodes[i] @ W_k_src) · (nodes[j] @ W_k_tgt)`
  This is O(N × d × K), scales with N not N².
- This is structurally similar to multi-head attention over graph nodes,
  which is intentional — it means we can use the same efficient primitives.

**Composition layer:** The novel sub-layer. For each sequence position:
1. Compute soft attention over graph nodes ("where does this position point?")
2. For pairs of positions that attention identified as related, retrieve
   the morphism type and strength from the graph.
3. Project the morphism information back into d_model space.

This is deliberately simple for the toy. We're testing whether the graph
structure helps AT ALL before optimising the composition mechanism.

**Architecture:** Attention → Composition → SwiGLU per block.
- d_model = 128
- n_heads = 4 (attention)
- n_layers = 4
- N_nodes = 64 (16 vocab + 48 concept slots)
- K = 8 relation types in composition layer
- SwiGLU hidden = 256
- PoPE positional encoding

**Parameter budget (target 1-2M):**
- Graph: 64 × 128 = 8,192 (node embeddings)
- Per layer attention: 4 × 128² = 65,536
- Per layer composition: ~40,000 (relation heads + projections)
- Per layer SwiGLU: 3 × 128 × 256 = 98,304
- Per layer norms: 3 × 128 = 384
- 4 layers total: ~816,000
- Output head: 128 × 16 = 2,048
- **Total estimate: ~830K params** (under budget — room to grow)

**Training data:** Character-level sequences from the concrete design target:
- Number line: "1 , 2 , 3 , 4 , ..." (succession)
- Comparisons: "1 < 2", "4 > 3", etc.
- Addition: "1 + 1 = 2", "2 + 3 = 5", etc.

**Loss:** Next-token prediction (cross-entropy on the result tokens).

**Evaluation:**
- Succession: can the model predict the next number for unseen numbers?
- Comparison: can the model predict < or > for unseen pairs?
- Addition: can the model compute a + b for unseen (a, b) pairs?

### What we expect to learn

1. Does the graph structure learn any meaningful morphisms, or does the
   gradient just route everything through attention + SwiGLU?
2. Does the composition layer contribute anything over a standard FFN?
3. Do the concept slots (free graph nodes beyond vocab) get used?
4. Does the model generalise to unseen numbers/pairs at all?

### Baseline comparison

We should also train a standard transformer (same size, same data, no graph,
no composition layer) as a control. If the Compositor doesn't beat the
baseline, the graph structure isn't helping.

### Results (v0.1)

The graph learned NOTHING. All morphism scores ~0. No succession signal.
No concept slot differentiation. The ~2% OOD improvement over baseline was
from extra parameters, not graph structure. See Phase 1 inspection report.

---

## Entry 002 — v0.2: Identity + Composition (2026-03-21)

### Hypothesis

The v0.1 graph failed because it had no categorical structure: bilinear
relation heads are redundant attention with worse initialisation. Adding
actual category theory machinery (identity morphisms, transitive composition
via adjacency matmul) will make the graph learn because it gives the
composition layer a capability attention CANNOT replicate in a single layer.

### Design changes from v0.1

1. **Explicit adjacency tensor:** `A[K, N, N]` replaces bilinear relation
   heads (W_src, W_tgt). Direct score of "relation k from node i to node j".
2. **Identity morphisms:** Structural bias `A + eye(N)` ensures every node
   maps to itself under all relation types. Not learned.
3. **Multi-hop composition:** `softmax(A+I) @ softmax(A+I)` gives 2-hop
   transitive closure. Combined: `result = (A+I) + softmax(A+I)@softmax(A+I)`.
   This is differentiable, and something attention cannot do in one layer.
4. **Morphism values:** `V[K, N, D]` — each relation×node has a value vector.
   When a morphism fires, this is the information it carries.
5. **Composition layer flow:** position_states -> soft node lookup ->
   morphism profile via composed adjacency -> value retrieval -> project out.

### Parameters

1,290,240 total (vs 661,376 for baseline). ~2x larger. The adjacency tensor
alone is 8 × 64 × 64 = 32,768 params, plus V[8, 64, 128] = 65,536 params.

### Results

- Train acc: 72.4% (similar to v0.1)
- Test acc: 27.2% (similar to v0.1)
- Generation: some succession works (16,17,18 -> 19,20,21), some addition
  works (7+5=12), but fails on larger numbers (45,46 -> 7,8,9,10) and
  multi-digit addition (12+19=22, should be 31).

### Graph inspection

**The graph still learned nothing meaningful:**
- Raw adjacency values: [-0.05, 0.035] — noise level.
- No succession signal: ratio of succ_score to non_succ_score ranges 0.71-1.63
  across relation types (noise).
- Identity morphisms dominate: diagonal ~1.01, off-diagonal ~0.01-0.05.
- Composition effect: negligible. Softmax over 64 nodes produces ~1/64 per
  entry; composed matrices are nearly uniform with identity spike.
- Node embeddings: random cosine similarity, no ordered structure.

### Diagnosis

The identity bias is TOO STRONG relative to the learned adjacency. After
softmax, each row of the adjacency is ~0.984 on the diagonal and ~0.00025
off-diagonal. Composing two such matrices produces a matrix that's even MORE
concentrated on the diagonal. The composition step effectively does nothing.

The gradient CAN'T learn through this because:
1. Off-diagonal signals are 4000x weaker than diagonal
2. Composition amplifies this disparity (squaring near-0 values)
3. The value retrieval is dominated by self-value (identity morphism)
4. Net effect: composition layer ≈ linear projection of own node value

**Root cause:** The fixed identity of 1.0 on the diagonal combined with
softmax normalisation kills all off-diagonal signal. The learned adjacency
(~0.01 scale) can never compete.

### Next steps to try

- Remove or reduce identity bias (e.g. 0.1 instead of 1.0)
- Use sigmoid instead of softmax (so off-diagonal scores aren't suppressed)
- Separate identity from composition (identity handled structurally, not
  through the same tensor that composition operates on)
- Scale up the adjacency init to compete with identity

---

## Entry 003 — v0.3: Sigmoid + Weight Tying + Seeding (2026-03-21)

### Design changes from v0.2

1. **Sigmoid instead of softmax:** Independent per-edge probabilities. 60x larger
   gradients at init (~0.018 vs ~0.0003). Multiple outgoing edges can coexist.
2. **No identity in adjacency:** Diagonal forced to -20 (sigmoid -> 0). The
   residual connection IS the identity morphism.
3. **Init A at -4:** sigmoid(-4) ~ 0.018. Sparse start, edges grow as needed.
4. **Output weight tying:** `logits = x @ graph.nodes[:V].T * scale`. Graph node
   embeddings get direct gradient from the cross-entropy loss.
5. **Dedicated node query:** Each composition layer has its own `node_query`
   projection for graph lookup, not shared.
6. **Separate optimizer groups:** Zero weight decay on graph.A (weight decay
   pushes A toward 0, which means sigmoid(A) toward 0.5 — densifies the graph).
7. **Phase 4 structural ops:** `seed_graph_from_data()` sets initial edge logits
   from training data statistics. Two strategies:
   - Strategy 1 (relations 0-1): Raw token co-occurrence at offsets 1-2.
   - Strategy 2 (relations 2-3): Parsed value-level transitions (successor
     detection from comma-separated number sequences, operator-operand detection).
8. **Composition closure:** Periodic transitive closure strengthens A->C when
   A->B and B->C both exist above threshold.

### Results (v0.3 with seeding)

- Train acc: ~72% (similar to all previous versions)
- Test acc: ~29% (marginal improvement, not significant)
- Graph edges now EXIST: ~38 strong succession edges in relation 2
- Node embeddings differentiate: digits ~0.8-0.93, operators ~0.6-0.9, concepts ~0.2
- Digit cosine similarity shows real structure: 5-9 cluster, 1-3 cluster

### Diagnosis

The graph has correct edges but the neural network doesn't USE them for
prediction. The composition layer outputs meaningful relational context, but
attention + SwiGLU still dominate because the composition output is added to
the residual stream where it mixes indistinguishably with attention output.
The FFN can't tell what came from the graph vs what came from attention, so
it treats the graph as noise and routes around it.

---

## Entry 004 — v0.3.1: Dual-Stream SwiGLU (Phase 3a) (2026-03-21)

### Hypothesis

The composition layer's output is being ignored because it's added to the
residual stream (mixing with attention output) before the FFN sees it. The
FFN can't distinguish graph information from attention information, so it
learns to ignore the weaker signal. Giving the FFN SEPARATE access to both
streams will force it to develop a navigation strategy for graph information.

### Design change

**DualStreamSwiGLU:** The SwiGLU FFN receives two inputs:
- `x_residual` (d_model) — what attention found (from the residual stream)
- `x_relational` (d_model) — what the graph says (from composition layer)

These are concatenated to 2×d_model and fed to `w1` and `w_gate`. The gate
can learn to selectively amplify or suppress graph information per-position.

**Critical architectural change:** The composition layer's output does NOT
go through a residual connection. ALL graph information must flow through
the FFN's gating mechanism. This forces the SwiGLU to be the sole arbiter
of how graph knowledge enters the residual stream.

```
# Old (v0.3): graph output mixed into residual before FFN sees it
x = x + self.attn(self.norm_attn(x), mask=mask)
x = x + self.comp(self.norm_comp(x), adj=adj, mask=mask)  # mixed in
x = x + self.ffn(self.norm_ffn(x))

# New (v0.3.1): graph output passed separately to FFN
x = x + self.attn(self.norm_attn(x), mask=mask)
comp_out = self.comp(self.norm_comp(x), adj=adj, mask=mask)  # separate
x = x + self.ffn(self.norm_ffn(x), comp_out)  # FFN gates it
```

### Parameters

1,615,489 total (+262,144 from v0.3 due to w1 and w_gate input doubling
from d_model to 2×d_model in 4 layers). Baseline unchanged at 661,376.

### What we expect

If the graph's seeded edges contain useful information (which they should —
succession edges like 3->4, 4->5 are directly relevant to predicting the
next number), and the FFN can now separately access that information, then:
1. The FFN's gate should learn to activate on positions where graph context
   is predictive (e.g., the last digit before a comma in succession sequences).
2. Test accuracy should improve because the graph's succession edges
   generalise to OOD numbers (the edge 8->9 works regardless of whether
   the number is 8 or 48).
3. The composition layer's gradient should be larger because the FFN is
   explicitly attending to its output, not averaging it away.

---

## Entry 005 — v0.4/v0.5: Hard lookup, attention bypass removal (2026-03-21)

### v0.4: Hard node assignment + node embeddings as values

Replaced soft node lookup with hard `input_ids` indexing. Replaced V[K,N,D]
with graph.nodes (weight-tied to output). This gives composition clean
retrieval with direct gradient from loss. Still didn't help: attention
bypass let model ignore graph entirely. Same 28% test.

### v0.5: Attention → Composition → FFN (no bypass)

Blocked the attention bypass: FFN only sees composition output, not
attention output. Added context-dependent relation selector (attention
output → K relation weights). Results: 67% train, 30% test. First time
the model was FORCED to use the graph. But the FFN came AFTER composition
— it could only post-process, not control traversal. Overfits.

### Root cause identified

The FFN needs to be in the WRONG position. It should WRAP the composition
layer, not follow it. The FFN should formulate graph queries, not process
graph answers. This led to the Phase 6 looped sandwich design.

---

## Entry 006 — v0.6: Looped FFN-Composition Sandwich (Phase 6) (2026-03-21)

### Hypothesis

The FFN must CONTROL graph traversal, not just process its output. The
looped sandwich architecture:
1. Pre-read: show FFN what edges exist from current node
2. FFN first half: formulate navigation intent (what to query)
3. Relation selection: select which edge type to follow
4. Composition: execute one graph hop
5. State update: integrate graph result
6. Node update: hop to the arrived-at node for next iteration

### Design changes from v0.5

1. **Pre-read projection:** K×N adjacency row → d_model summary. FFN
   knows what edges are available BEFORE it decides which to follow.
2. **Looped iteration:** max_steps iterations per block. Each iteration
   performs one graph hop with full FFN control. Multi-hop traversal
   emerges from iteration, not fixed n_compose_hops.
3. **Node identity update:** After each hop, argmax of graph_result
   similarity to node embeddings gives the new node ID. The loop
   WALKS ALONG graph edges — succ(7)=8, then succ(8)=9.
4. **Gated combination:** navigation × graph_hidden. The FFN's intent
   gates the graph result, not the other way around.
5. **Soft halting:** Fixed max_steps, FFN learns to produce near-zero
   updates when done (no dynamic control flow).
6. **DualStreamSwiGLU removed** — the looped sandwich subsumes it.
   The pre-read + navigation + graph integration is a strictly more
   expressive version of the dual-stream concept.
7. **get_composed_adjacency() bypassed** — the loop replaces fixed
   multi-hop with dynamic iteration using only 1-hop P.

### Parameters

1,361,025 total (up from 702K in v0.5). Main costs:
- pre_read_proj: 65K per block (K×N → d_model)
- w1/w_gate doubled: 65K each (input is 2×d_model from [state, pre_read])
- graph_proj: 32K per block (d_model → d_hidden)

### Validation plan

1. Oracle graph + max_steps=3: single-digit succession should work in 1 step
2. Verify node update: after succ(7)=8, next iteration starts from node 8 ✓
3. Addition (3+5): does the loop walk 5 successor steps from 3? Needs max_steps≥5

### Expected behaviour

With oracle graph, the loop should learn to:
- Select relation 0 (successor) when context says "next number"
- Walk along edges: 7→8→9→... until halting
- Produce near-zero updates after the useful hops are done

The pre-read is critical: it tells the FFN "from node 7, successor leads
to node 8 with P=0.9975." Without it, the FFN would be blind to graph
structure and have to guess which relation to select.
