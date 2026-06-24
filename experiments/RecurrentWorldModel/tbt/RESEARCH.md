# Research notes — thalamo-cortical architecture

Living research log backing `THALAMO_CORTICAL_ARCHITECTURE.md`. Each entry: a load-bearing question,
what the literature says, and the implication for the build. Honest about what's solved vs still open.

---

## R1 — Online / single-shot learning of the task graph (the §9 unknown)

**Question.** TEM learns its structural code offline over many environments; on the ARC replica a task
dependency ("key unlocks door") must be inferred from a **single** playthrough, mid-level. How does
single-shot relational-structure learning work?

**Answer: a fast/slow weight decomposition.** This is the recurring theme across every relevant model:
- **SLOW weights** (SGD/EM, learned across many environments): the structural **prior** — the *space* of
  possible structures / causal schemas, and *how* to bind a new instance. Meta-learned, offline.
- **FAST weights** (Hebbian, one-shot, within-episode): bind **this** environment's specific rules in a
  single shot.

You do **not** learn structure from scratch online. You meta-learn the structure *space* offline and bind
the *instance* one-shot online. (Munkhdalai & Trischler 2018, *Metalearning with Hebbian Fast Weights*:
fast Hebbian weights do one-shot binding; slow SGD weights learn the representations *and the binding
rule*. TEM is the same shape — slow structural `g`, fast Hebbian memory `M`.)

**The slow substrate — schema spaces:**
- **CSCG** (Clone-Structured Cognitive Graph, George et al. 2021, Nat. Commun.): a cloned HMM — multiple
  latent "clones" per observation give context-dependent states; learns a latent graph from action-
  observation sequences (EM/Baum-Welch). Yields **transferable schemas, transitive inference, hierarchical
  abstraction, and vicarious (simulated) planning**. This is the task-space-map substrate the hippocampal–
  PFC model is built on.
- **Schema Networks** (Kansky/George 2017, ICML): an object-relational **generative causal** model of
  dynamics; learns causal rules (entity attrs → effects), reasons backward from goal through causes, and
  **transfers zero-shot**. This is the causal-rule (key→door) substrate.

**The fast trigger — exafference → a causal edge:** the **prediction-error / exafference** (the world
changing in a way self-motion did not predict — door vanished after stepping on the key) is exactly the
signal to bind a **new causal edge one-shot** (cause = recent action/state, effect = the change) via fast
Hebbian weights, into the task column's graph. The task graph then grows **incrementally, one edge per
surprising effect**. This reuses the thalamus's deviance signal and the exafference our agent already
separates.

**So: single-shot structure learning =** fast Hebbian binding of causal edges (triggered by exafference)
**into** a meta-learned schema space (slow). Slow = the schema/causal prior (or, cold, a minimal generic
"actions cause local effects / contact causes change" prior); fast = one-shot binding of this level's
specific rules.

**Honest residual hard parts (NOT solved by these papers):**
1. **Cold start.** A genuinely novel structure with no meta-learned prior leaves only the generic causal
   prior. ARC deliberately supplies novel structures — so how rich the generic prior must be is open.
2. **One-shot causal attribution.** *Which* recent cause produced the effect? (the blicket-detector
   ambiguity). One co-occurrence is weak evidence; needs a causal prior and/or a few confirmations.
3. **Full dependency structure.** Binding one edge is one-shot; inferring *ordering* and *conjunction*
   ("block AND pad AND goal") from few observations is harder than a single edge.

**Implication for the build.** The **task column** = a CSCG/schema-style latent graph + **fast Hebbian
edge insertion on exafference**, over a (slow, meta-learned or generic) schema prior. Slow schema-space
learning across levels/games + fast online instance binding. Ties to: thalamus deviance (the trigger),
exafference (already separated in `agent/column`), reward model (values which rule-sequence yields score).

Sources: [CSCG, George 2021](https://www.nature.com/articles/s41467-021-22559-5) ·
[Schema Networks, Kansky 2017](https://arxiv.org/abs/1706.04317) ·
[Metalearning w/ Hebbian Fast Weights, Munkhdalai & Trischler 2018](https://arxiv.org/abs/1807.05076) ·
TEM (Whittington 2020). See memory `reference_hierarchy_substrate`, `reference_exploration_replay`.

### R1 correction (the number-line reframing) — fast/slow was the WRONG framing for our substrate

The fast/slow, offline-meta requirement above is a property of **gradient/EM-trained models** (TEM, CSCG,
Schema Networks need many passes to move weights), **not of structure-learning itself.** Our column does
**not** learn structure by SGD — it **binds** it (grid + operator + Hebbian conjunction), which is
one-shot. Proof: the column learned the **number line from scratch** (successor transitions) and
generalised to all arithmetic. So:
- **Local cause→effect is 1-shot online** — a key→door edge is the same as a successor edge; observe the
  transition, bind the operator. **No meta-prior; cold-start is NOT the blocker.**
- **The genuinely hard part is LONG-RANGE credit assignment** — tracing a present failure to a distant past
  action. Mechanism = **reverse replay** (Mattar–Daw, `reference_exploration_replay`): replay the trajectory
  backward from the failure to find the action that mattered. It is a **learned** skill (replay is
  prioritised/learned), matching the intuition that one "had to learn how to go back and re-think."

**Revised open problem:** *long-range credit assignment via learned replay*, **not** cold-start
meta-learning. This is more optimistic and more correct.

---

## R2 — The location code: one SR-eigenvector frame for ANY topology (and why the hex grid is a prior)

**Question.** The column placed entities two different ways: a hard-coded universal **hex grid** (`learn_domain`)
and, online, a **metric embedding** discovered from inverse-action pairs (approach B). B placed line / ring /
2-D grid from transitions but **broke on a tree** (non-metric: no consistent axis displacement embeds it —
18/28, 27/60). A fallback (detect the metric collision, switch to distinct codes) was rejected as too
special-cased. So: one general mechanism, or accept a switch?

**Answer (biology decides): the grid IS the SR eigenbasis — compute it, don't hard-code it.** Stachenfeld
2017: place cells = the successor representation; grid cells = its **eigenvectors**. Hexagonal *only because*
open 2-D space has that transition structure — on a 1-D track, a hairpin, or a tree the eigenbasis is not
hexagonal, it is whatever the topology dictates. The grid also encodes **traversability**, not Euclidean
distance, so it re-tunes to the task (rescales — Barry 2007; fragments — Derdikman 2009; warps around
barriers — Carpenter 2015; warps to reward — Boccara/Butler 2019). So the hard-coded hex grid was only ever
the SR of the *default* (open-2-D) environment. Compute the SR frame of the **actual** transition graph and
one mechanism covers line / ring / 2-D / **tree** with no switch. Caveat (honest): the SR-eigenvector reading
(predictive map) competes with the continuous-attractor reading (Burak & Fiete 2009: hex as rigid hardwired
path-integration) — truth ≈ a scaffold + a learned SR; that hybrid is our prior-vs-learned tension. Full
biology log: memory `reference_grid_sr_eigenbasis`.

**Built (this is the implemented resolution).**
- `consolidate` and `learn_domain` now BOTH place via the **SR-eigenvector frame** of the transition graph
  (`_sr_frame`: symmetric-normalized adjacency → `eigh` → drop the trivial mode → near-orthonormal node
  codes). ONE code source — no parallel systems. Per-relation operators (L5) read off any relation.
- Verified: line 11/11, arithmetic 1806/1806, 2-D 164/164, **tree 28/28 & 60/60** (was B's hard limit),
  carry 60/60, factored 200/200, `unified_demo` all 1.000, 80 smoke tests.
- The hard-coded **hex grid is demoted to an innate metric PRIOR** (vector-nav to unvisited goals — Bush
  2015; CRT error-correcting code — Sreenivasan & Fiete 2011), switched off, to add back when a task needs
  those metric superpowers (they are *intrinsically* metric — you cannot vector-navigate a tree).
- **Emergent finding:** structure-specific frames AUTO-SEPARATE different domains (no-remap interference
  vanished, 0.18→1.0); only *identical* structures still need orthogonal slots (two rings: 0.500 → 1.000).

**A's real limitations (worked around / logged, not papered over).**
1. **Codes span only the explored structure** — no free address space for the unexplored. This is what
   forced `factored.py` to give POSITIONS their own discovered frame (a second column) instead of plucking
   grid cells — i.e., place value genuinely *needs* multi-column. (For non-metric structure this is correct:
   there is nothing to extrapolate to.)
2. **Batch `eigh()`** every consolidate. NEXT internal step: an incremental, persistent **TD-learned SR**
   (biological — modules persist and adapt gradually; engineering — dissolves the eigendecomposition cost).
3. **Cross-environment transfer** of a learned frame (TEM's generalization) — the thing that earns the word
   "universal" without a hard-coded lattice — is **not built**. Open.

**Implication.** Multi-column factorization is no longer assumed; it is forced by A's limitation #1. Emergent
column allocation (stage 6, `multicolumn.py`) and, after it, the goal-state control loop (§5/§6) sit on top
of this one frame.

Sources: [Stachenfeld 2017](https://www.nature.com/articles/nn.4650) · Barry 2007 · Derdikman 2009 ·
Carpenter 2015 · Boccara 2019 · Butler 2019 · Bush 2015 · Sreenivasan & Fiete 2011 · Burak & Fiete 2009 ·
TEM (Whittington 2020). Memory: `reference_grid_sr_eigenbasis`.
