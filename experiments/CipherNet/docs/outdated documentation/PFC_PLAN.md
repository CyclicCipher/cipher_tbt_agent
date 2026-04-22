# Prefrontal Cortex Planning Document

## Purpose

The PFC subgraph is a structured prior that provides:
1. Working memory — hold intermediate results across computation steps
2. Sequencing — control the order of multi-step operations
3. Gating — decide when to update working memory vs maintain it
4. Goal maintenance — track what the overall task is

Without the PFC, CipherNet can perform single-step operations (3+4=7)
but cannot chain them (3+7+1+4=15). The PFC is the machinery that
enables multi-step computation.

## Biological Research

### The PBWM Model (O'Reilly & Frank, 2006)

The most influential computational framework for PFC working memory.

**Core mechanism:** The prefrontal cortex has STRIPES — groups of
neurons (~0.2-0.8mm wide) that can independently maintain persistent
activity. Each stripe is a working memory "slot" that holds one piece
of information.

**Gating via basal ganglia:** The basal ganglia controls which stripes
get updated and which stay maintained:
- **Go pathway (D1 MSNs):** Fires to OPEN the gate. Disinhibits the
  thalamus, which sends excitation to a PFC stripe, toggling its
  bistable state. The stripe updates with new information.
- **NoGo pathway (D2 MSNs):** Fires to KEEP the gate closed. The
  stripe maintains its current contents, resistant to interference.

The Go/NoGo competition at each stripe determines whether that slot
updates or holds. This is learned via dopamine reward prediction errors.

**Key insight:** The demands for rapid UPDATING and robust MAINTENANCE
are in direct conflict. You can't have both simultaneously. The gate
resolves this: open = update, closed = maintain. The learning problem
is figuring out WHEN to open which gate.

Ref: [O'Reilly & Frank 2006, Neural Computation](https://pubmed.ncbi.nlm.nih.gov/16378516/)
Ref: [O'Reilly et al., NCBI Bookshelf](https://pmc.ncbi.nlm.nih.gov/articles/PMC2440774/)

### PFC Stripes and Persistent Activity

PFC neurons maintain information through **persistent firing** sustained
by recurrent excitatory connections between neurons with similar tuning.
This persistent activity is organized in stripe-like columns (0.2-0.8mm).

**Cellular bistability:** Individual PFC neurons have two stable states
(quiescent and active), maintained by NMDA receptor dynamics. The slow
time constant of NMDA receptors keeps neurons depolarized for extended
periods. This makes working memory robust against noise and distraction.

**Capacity limits:** Working memory capacity (~4-7 items in humans) is
determined by the number of independent stripes that can simultaneously
maintain distinct representations. Each stripe = one slot.

Ref: [Constantinidis & Klingberg 2016, Frontiers](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2015.00181/full)
Ref: [Camperi & Wang, J Comp Neurosci](https://link.springer.com/article/10.1023/A:1008837311948)

### Activity-Silent Working Memory (2024)

Recent work challenges pure persistent-activity models. During some
memory maintenance periods, PFC neurons show NO elevated firing, yet
information is preserved (possibly in synaptic weights or calcium
dynamics). Periods of absent activity between gamma bursts are consistent
with models where information is maintained in short-term synaptic
plasticity rather than firing rates.

This suggests working memory has TWO mechanisms:
1. **Active maintenance:** persistent firing (high-energy, robust)
2. **Silent maintenance:** synaptic traces (low-energy, fragile)

The system might switch between them: active during manipulation,
silent during pure storage.

Ref: [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.06.03.597259v1.full)
Ref: [Decoding WM info, J Neurophysiology 2023](https://journals.physiology.org/doi/abs/10.1152/jn.00290.2023)

### Dopamine and Learning When to Gate

Dopamine plays a dual role:
1. **Gating signal:** Phasic dopamine burst → open the gate → update WM
2. **Learning signal:** The same dopamine burst trains the basal ganglia
   to recognize WHEN gating should happen (reward prediction error).

When the system performs a correct multi-step computation and receives
a reward, dopamine reinforces the gating decisions that led to it.
Over time, the system learns the task structure: "after computing
a + b, gate the result into WM; after seeing another +, gate the
next operand into the second slot."

Ref: [D'Ardenne et al. 2012, PNAS](https://pubmed.ncbi.nlm.nih.gov/23086162/)
Ref: [Starkweather et al. 2018, Neuron](https://pubmed.ncbi.nlm.nih.gov/29656872/)

### Hierarchical PFC Organization

Different PFC regions handle different levels of abstraction:
- **Caudal lateral PFC:** responds to external stimuli (what am I seeing?)
- **Mid-lateral PFC:** contextual rules (which operation to apply?)
- **Rostral PFC:** abstract goals from working memory (what's the overall task?)

This suggests a hierarchy: the most abstract level (task goal) is
maintained by the most anterior PFC, while the specific operands
and intermediate results are maintained by more posterior regions.

Ref: [Robbins et al. 2021, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8617292/)

### Cortico-BG-Thalamocortical Loops (2025)

Recent 2025 work emphasizes that behavior emerges from INTERACTING
loops, not a single PFC-BG circuit:
- Multiple parallel loops handle different aspects of a task
- Striatal integration hubs share similarities with self-attention
  in Transformer neural networks
- Goal-directed and habitual processes are deeply intertwined,
  not a strict dichotomy

Ref: [Trends in Neurosciences 2025](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(25)00192-4)

### Adaptive Chunking (2025)

The PFC-BG system also handles CHUNKING — grouping related items
into single WM slots. This explains how experts maintain more
information than novices: they chunk familiar patterns into single
representations.

For arithmetic: "3 + 7 = 10" might initially take 3 WM slots
(operand, operand, result). After learning, the whole expression
might be chunked into one slot ("the sum pattern"), freeing capacity
for additional operations.

Ref: [eLife 2025 — Adaptive chunking](https://elifesciences.org/reviewed-preprints/97894v2)

## Design for CipherNet

### PFC Subgraph Structure

```
┌─────────────────────────────────────────────────┐
│  Prefrontal Cortex Subgraph                     │
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Stripe 1│  │ Stripe 2│  │ Stripe 3│  ...     │
│  │ (slot)  │  │ (slot)  │  │ (slot)  │         │
│  │         │  │         │  │         │         │
│  │ content │  │ content │  │ content │         │
│  │ nodes   │  │ nodes   │  │ nodes   │         │
│  └────┬────┘  └────┬────┘  └────┬────┘         │
│       │            │            │               │
│  ┌────┴────────────┴────────────┴────┐          │
│  │         Gate Controller           │          │
│  │  (Go/NoGo per stripe)            │          │
│  └────┬──────────────────────────────┘          │
│       │                                         │
│  ┌────┴────────────────────────────┐            │
│  │      Sequencer                  │            │
│  │  (which stripe to read/write    │            │
│  │   next, based on task state)    │            │
│  └─────────────────────────────────┘            │
│                                                 │
│  Exposed nodes:                                 │
│    input  → receives values to store            │
│    output → provides current slot value         │
│    step   → advance to next computation step    │
│    done   → signal computation complete         │
└─────────────────────────────────────────────────┘
```

### Stripe (Working Memory Slot)

Each stripe is a small subgraph with:
- **Content nodes:** connected to the torus modules. When a stripe
  holds the value "10", the content nodes activate the phase vector
  for 10 on the torus. The stripe effectively "points to" a position
  in the latent space.
- **Bistability mechanism:** content nodes maintain their activation
  without external input (persistent activity). Implemented as a
  self-reinforcing edge (node → itself with temporal edge, weight > decay).
- **Gate input:** a node that, when activated, allows the content
  to be overwritten. When the gate is closed, the content is maintained
  even if new input arrives.

### Gate Controller (Go/NoGo)

For each stripe:
- **Go node:** when activated, opens the gate. New input overwrites
  the stripe's content.
- **NoGo node:** when activated (default), keeps the gate closed.
  Content is maintained.

The Go/NoGo balance is learned: initially random, trained by feedback
(did the computation succeed?). This is the dopamine RL mechanism.

### Sequencer

Controls the order of operations:
- Maintains a "program counter" — which step of the computation
  are we on?
- Routes values between stripes and the manifold.
- For "3 + 7 + 1 + 4":
  Step 1: load 3 into stripe 1, load 7 into stripe 2, compute via
          addition manifold, gate result (10) into stripe 1.
  Step 2: stripe 1 already has 10, load 1 into stripe 2, compute,
          gate result (11) into stripe 1.
  Step 3: stripe 1 has 11, load 4 into stripe 2, compute, gate
          result (15) into stripe 1.
  Done: stripe 1 has the answer.

The sequencer is a state machine. Its states correspond to "positions"
in the task. Transitions are triggered by:
- Seeing an operator (+) → prepare to load next operand
- Seeing = → stop, output the current content of stripe 1
- Completing a manifold computation → gate result, advance

### Key Design Questions

1. **How many stripes?** Humans have ~4-7 WM slots. For arithmetic,
   2 slots suffice (current accumulator + next operand). For more
   complex tasks (nested expressions), more slots needed. Start with
   4 and see.

2. **How does the gate learn when to open?** Dopamine-based RL. But
   CipherNet doesn't have a reward signal (yet). Initial approach:
   hard-coded gating policy for arithmetic (open after each +
   operation). Learn the policy later.

3. **How does the sequencer know the order?** From the linear order
   of tokens. The expression is read left-to-right. Each operator
   triggers a computation step. This can be encoded as temporal
   edges: token_at_position_n → token_at_position_n+1.

4. **What about nested expressions?** (3 + 4) * (2 + 1) requires
   two separate sub-computations before the multiplication. This
   needs a STACK, not just a sequence. The stripes can act as a
   stack: push onto unused stripes, pop when sub-expression completes.

5. **Activity-silent vs active maintenance?** Start with active
   (persistent firing). Add silent (synaptic traces) later if
   needed for energy efficiency.

## Implementation Plan

### Phase 1: Minimal PFC (hard-coded gating)

- 2 stripes (accumulator + operand)
- Hard-coded gating: open stripe 2 for each new operand, compute,
  gate result into stripe 1, repeat
- No learning — the gating policy is fixed
- Test: "3 + 7 + 1 + 4 = 15"

### Phase 2: Learned gating

- Add Go/NoGo nodes per stripe
- Dopamine-like RL: after correct answer, strengthen the gating
  decisions that led to it
- Test: can the system learn when to gate from examples?

### Phase 3: Task generalization

- Test: can the same PFC handle both addition chains and
  multiplication chains?
- Test: can it handle mixed operators (3 + 4 * 2)?
  (This requires learning operator precedence — a sequencer
  task, not a manifold task.)

### Phase 4: Nested expressions

- Add stack behavior (push/pop onto stripes)
- Test: "(3 + 4) * (2 + 1) = 21"

## Expected Failures and Iteration

1. **Hard-coded gating will be brittle.** It works for simple chains
   but breaks on anything unexpected. This is fine for Phase 1.

2. **The bistability mechanism might be too simple.** A self-loop edge
   might not maintain activation long enough, or might maintain it
   too rigidly (can't be overwritten). May need NMDA-like time
   constants (slow decay instead of instant switching).

3. **The sequencer will struggle with variable-length expressions.**
   "3 + 4" and "3 + 4 + 5 + 6 + 7" need different numbers of steps.
   The sequencer must handle arbitrary length — it can't hard-code
   the number of steps.

4. **Operator precedence is hard.** "3 + 4 * 2" requires computing
   4 * 2 first. This needs the PFC to look ahead, recognize *, and
   defer the addition. This is a planning problem — exactly what
   CatPlan was designed for. The PFC might delegate this to a
   planner-like mechanism.

5. **Integration with the torus manifold will be tricky.** The PFC
   stripes hold values as phase vectors, and the manifold queries
   take phase vectors as input. The interface between them needs
   careful design: how does a stripe "send" its content to the
   manifold's input axis?
