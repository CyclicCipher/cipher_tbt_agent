# Spotlight Memory (Percepta, 2026) — reference notes

**Sources, all read in full 2026-10-03:**
- Blog: *Spotlight Memory: Growing Memory for Long-Context Modeling at Linear Cost*, Christos Tzamos, Guoqing Zheng,
  Athul Jacob and Percepta, 2 Oct 2026, https://percepta.ai/blog/spotlight-memory. © Percepta, all rights reserved —
  so these are notes in our words with short quotes, not a copy. (The page renders client-side; the text was read
  from its Next.js content chunk.)
- Companion blog: *Growing Intelligence Beyond the Weights* (slug `can-llms-grow-their-own-capabilities`) — a
  hand-built Python interpreter running inside a Spotlight transformer.
- Code: Hugging Face `percepta-ai/spotlight-vm` (Apache-2.0, created 2026-09-29) — the hand-built model from the
  companion post: `spotlight.c` (a 300-line C reference implementation), `spotlight.py` (a `transformers` wrapper),
  `intelligence/` (weights, < 100K parameters), `memory/` (MicroPython v1.28.0 + two packages, as cells).
- Earlier, same team: *Can LLMs Be Computers?* (11 Mar 2026) — a WebAssembly interpreter in transformer weights with
  2-D attention heads ("HullKVCache").

## The claim

Three properties at once, which the post says no prior architecture has combined:
1. **Growing state** — the addressable memory grows with the sequence;
2. **Sparse access** — each token reads and writes a CONSTANT number of cells per step;
3. **Learned addressing** — where to read and write is learned end-to-end from the task loss.

Attention has (1) and (3) at O(T²); linear attention / (Gated) DeltaNet have (2) and (3) with a fixed-size state;
NTM/DNC/Sparse Access Memory had fixed or slow-growing memories. The companion post's framing: "separates
intelligence from memory, allowing knowledge and skills to grow without changing the model's weights" — and,
against MoE, Spotlight is "arbitrarily sparse": it touches the same number of cells however large memory grows.

## The mechanism

- **A 2-D lattice of cells per head.** Keys and queries are projected to continuous 2-D ADDRESSES. A key WRITES
  to the small neighbourhood of cells around its address; a query READS the neighbourhood around its own. Each cell
  is a fixed-size recurrent state, **allocated on the first write** to that address, updated by later keys. All cells
  share the same learned projection and update rule, so the rules are reused as memory grows.
- **A compact, differentiable kernel.** In 1-D, with integer cell indices z, the weight of cell z for address x is
  `φ_z(x) = b(x − z) / sqrt(Σ_j b(x − j)²)`, and the induced kernel between a query and a key is
  `D_1(q, k) = Σ_z φ_z(q) φ_z(k)`. The post uses the bump `b(t) = cos²(πt/3)` for |t| < 3/2, else 0 — it and its
  derivative vanish at the edge, so a cell's contribution moves smoothly as the address moves (gradients for small
  address changes), and only the three nearest cells are non-zero. In 2-D the kernel is the product
  `D(q, k) = D_1(q_x, k_x)·D_1(q_y, k_y)`: **every read and write touches exactly 3 × 3 cells**. In d routing
  dimensions it would be 3^d cells; the post finds 2-D sufficient. It is presented as a compact replacement for the
  global RBF that softmax attention is (for equal-norm q, k, `exp(qᵀk) ∝ exp(−‖q − k‖²/2)`).
- **Routing separated from content.** Splitting `exp(qᵀk) = exp(q₁ᵀk₁)·exp(q₂ᵀk₂) ≈ exp(q₁ᵀk₁)(1 + q₂ᵀk₂)`: the 2-D
  part (q₁, k₁) becomes the lattice kernel that SELECTS cells; the content part (q₂, k₂) is linear and lives INSIDE
  the cell — in the trained variant **each cell is a d_k × d_v DeltaNet state per head**, updated by the delta rule.
  "Our architecture therefore learns to read and write across a growing collection of DeltaNet states."
- **Cost.** Memory grows with allocated cells; total work O(T). The complexity argument: a fixed-state model cannot
  even copy an arbitrary input without precision growing with its length (Jelassi et al., ICML 2024; Siegelmann &
  Sontag 1995), while attention has the memory but pays O(T²).

### What the released code does (the hand-built VM, `spotlight.c`)

Per layer: `proj = qkv_w·x + qkv_b`, 8 numbers per head — key address (x, y), query address (x, y), a write gate
`β = σ(·)`, an erase factor `α = σ(−·)`, a value v (d_v = 2). Per head and per token, in this order:
1. **write** — for the 3 × 3 cells around the key address, with weight `w = φ_x·φ_y·β`:
   `S ← S + w·(v − α·S)` (an erase-then-write, delta-rule-like update, scaled by the kernel weight);
2. **read** — `out = Σ φ_x·φ_y·S` over the 3 × 3 cells around the query address (an unallocated cell reads zero).
Then `out_w` maps the heads' reads into the residual (d_model = 16), a ReGLU MLP (48 wide), 8 layers, 8 heads, and a
rank-3 output head. Cells live in one hash map keyed by `(layer·H + head, x, y)`, allocated on first write.
Differences from the blog's trained variant: the kernel is sharper — `b(t) = cos²(πt/2)` for |t| < 1, so an INTEGER
address puts all its weight on one cell (exact, addressable RAM; program bytes are poked at `spacing = 2.0`) — and a
cell holds a 2-vector, with no in-cell content key (`d_rest = 0`), where the trained models hold DeltaNet states.
It runs MicroPython (parser, compiler, bytecode VM, GC — unmodified) out of memory at ~120–144K tokens/s on one CPU
core, with constant work per step over traces of 6.5M–47.6M tokens; recursion 2,000 calls deep; a package's source
in the context is written into memory and used from then on; updating a fact or a function overwrites cells in
place; an MNIST classifier with more parameters than the transformer runs by being loaded from memory.

## Results reported (blog)

- **MQAR** (Zoology; one mixer layer, d_model = 64, 2 heads, short conv1d): extrapolation trained on ≤ 64 pairs,
  tested at 256; capacity at 2¹⁷ = 131,072 pairs (a 524K-token context); a forgetting variant where half the keys are
  overwritten and the LATEST value must come back. Spotlight "near-perfect" on all three, stale-value rate zero;
  Gated DeltaNet degrades past its capacity; attention recalls at 131K but returns stale values when keys are
  rewritten, and with RoPE collapses beyond 16K keys. (The extended settings used a seven-stage curriculum, 4,000
  steps per stage.)
- **MAD** (6 synthetic tasks, 66 settings × 6 runs): highest mean vs GDN, GDN-2 and attention; near-perfect in-context
  and noisy recall. **PathFinder** (LRA, 1,024-pixel sequences, 4 layers, no positional encodings): 0.769, above
  attention and the fixed-state models.
- **Language models** — 140M / 280M / 670M, FineWeb-Edu at 8K context, 10.5B / 21B / 50B tokens, parameter-matched to
  0.1% against attention, GDN and GDN-2: held-out loss close to the fixed-state models and below attention (at 670M
  equal to GDN-2, GDN slightly better); lm-eval average at 670M 33.1 (GDN-2 33.8, attention 33.3, GDN 34.1).
- **Length extrapolation** (8K-trained, RULER single needle at 16K–128K): Spotlight 93–100% at 128K across sizes;
  fixed-state ≤ 5.6%; attention 0% from 16K up, with or without position rescaling. PG19 (8K–32K): better than GDN
  (2.791) and GDN-2 (2.821) at 670M.
- **After 1.17B tokens of 128K midtraining:** lowest held-out loss at every length from 32K to 128K for all sizes
  (670M at 128K: 1.440 nats/token, ~0.1 below the fixed-state models); needle recall 97.8% (140M) and 100% (670M)
  where the baselines reach ≤ 15.6%; HELMET TREC-coarse in-context classification rising 63% → 85% as prompts grow
  8K → 128K (670M), above every baseline.

## Not disclosed (what an implementation has to decide)

How the 2-D addresses are parameterised and kept in range (raw linear projections in the VM); how training is
parallelised (the per-token write-then-read is a recurrence — chunked like DeltaNet, or sequential?); how cells are
allocated in a batched GPU implementation (a hash, a bounded grid, collisions?); whether the in-cell DeltaNet has
gating or decay; layer/head counts and positional handling in the LM runs; the MQAR table values (held in a figure
component that was not extracted).

## What it means for the Neural Turing Architecture (see `../BRAINSTORM.md` §2)

- It is implementable now, from the post plus the 300-line reference: per head, two 2-D address projections, a
  write gate, an erase factor, a value; a cos² bump over 3 × 3 cells; a hash map of cell states.
- **In a looped core** every pass of every token writes then reads: the memory becomes a persistent scratchpad shared
  by the depth axis (passes) and the sequence axis (thoughts). Whether tied passes share one lattice per head, or
  each pass gets its own table, is a design choice with consequences (a pass can overwrite what an earlier pass of
  the same token wrote).
- **Coconut thoughts no longer need appended positions with an ever-growing cache:** the sequence mixer is a
  recurrence with O(1) work per position, and a thought can OVERWRITE its own earlier notes — the mutable working
  memory the brainstorm asked for.
- **Search needs branching.** A search node that adds one thought changes at most 9 · H · L cells, so forking memory
  is a copy-on-write overlay of those cells — cheap, unlike copying a KV cache (see
  `../CURRICULUM_LESS_COCONUT_AND_SEARCH.md`).
- **Hand-built ↔ trained.** The VM's sharp kernel makes integer addresses exact RAM; the trained variant's wider
  kernel makes them smooth. The same machinery spans both regimes.
