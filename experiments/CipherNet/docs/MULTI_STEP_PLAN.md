# Multi-Step Computation — Fully Graph-Driven

## The Problem

Given: `4 + 5 + 1 + 9`
The system must compute this using only the 2-operand addition column,
by breaking it into steps: `4+5=9, 9+1=10, 10+9=19`.

All sequencing, gating, routing, and memory management must happen
through graph dynamics — no Python control flow.

The system learns HOW to do this from example-driven training.

## What Needs to Be Learned

The graph already knows arithmetic (displacement learning in columns).
What it needs to learn is the GATING POLICY — a temporal pattern
that controls the PFC/BG/thalamus circuit.

### The gating policy (5 rules to learn)

1. **Number arrives, WM0 empty → load into WM0**
   BG learns: Go(stripe 0) fires when a number token activates
   and WM0:L23 is inactive.

2. **Operator arrives → set goal in WM2**
   BG learns: Go(stripe 2) fires when a non-number token activates.

3. **Number arrives, WM0 full, WM1 empty → load into WM1**
   BG learns: Go(stripe 1) fires when a number token activates
   and WM0:L23 is already active.

4. **Both operands ready → compute and gate result into WM0**
   The sequencer learns: when WM0, WM1, and WM2 are all active,
   trigger computation. The result replaces WM0.

5. **After compute → clear WM1**
   BG learns: after a computation completes, WM1 is released
   (gate opens with no input → decays to empty).

### How these rules are learned

From EXAMPLES. The system sees multi-step expressions with
intermediate states:

Training example 1: "3 + 4 = 7" (single step)
  Tokens: 3, +, 4, =, 7
  The system sees: load 3 → set + → load 4 → compute → result 7
  Dopamine: positive (correct answer)

Training example 2: "3 + 4 + 2 = 9" (two steps)
  Tokens: 3, +, 4, +, 2, =, 9
  Step 1: load 3 → set + → load 4 → compute → WM0 = 7
  Step 2: WM1 cleared → load 2 → compute → WM0 = 9
  Dopamine: positive (correct answer)

Training example 3: "1 + 2 + 3 + 4 = 10" (three steps)
  Dopamine: positive

The BG's D1/D2 weights update to strengthen gating decisions
that led to correct answers.

## Architecture Requirements

### Token representation

Each token (digit or operator) must have a node in the graph
that activates when that token is presented. These token nodes
connect to the PFC L4 inputs and to the BG D1/D2 nodes via
the thalamus.

### State detection

The sequencer needs to detect the PATTERN "both operands filled +
goal set." This is a conjunction of three conditions:
  WM0:L5 active AND WM1:L5 active AND WM2:L5 active

In the graph: the sequencer's L4 receives from all three WM L5
outputs (already wired). When all three are active, the combined
input to sequencer:L23 exceeds a threshold → sequencer fires →
triggers BG Go for the compute-and-gate operation.

### Computation routing

When the sequencer triggers, the thalamic relays for the operation
columns must open. The addition column receives the WM0 and WM1
values, computes, and sends the result back through the thalamus
to WM0.

This requires:
- WM0:L5 → thalamic relay_add → addition column L4:a
- WM1:L5 → thalamic relay_add → addition column L4:b
- Addition column L5 → thalamic relay_0 → WM0:L4

These edges need to exist in the graph (added via config.json
or during learning).

### Temporal order

The graph must handle temporal order: load, then compute, then
gate result. This happens naturally over multiple step() calls:
- Step N: WM0 and WM1 are loaded (gate opened by BG)
- Step N+1: Sequencer detects both filled, fires
- Step N+2: Sequencer output activates BG Go, which opens
  thalamic relay for computation
- Step N+3: Computation result arrives at WM0 through thalamus
- Step N+4: WM1 cleared, ready for next operand

The temporal spread across ~4-5 step() calls IS the sequencing.
No Python loop needed — the graph's activation propagation IS
the control flow.

## Training Procedure

1. Present a complete expression as a sequence of tokens
2. At each token, activate the token's node in the graph
3. Run step() to let the dynamics propagate
4. After '=' token, check WM0 against the expected answer
5. If correct: activate dopamine node (positive RPE)
   → D1 Go weights strengthened, D2 NoGo weakened
6. If wrong: negative RPE → opposite weight changes
7. Repeat with many expressions

## Key Questions

1. How do numeric VALUES flow through the graph? The WM stripes
   hold activations (0-1 floats), not actual numbers. How does
   "the number 4" get from a token to WM to the addition column?

   Answer: the displacement learner operates OUTSIDE the activation
   dynamics. The graph handles routing and gating. The actual
   arithmetic uses the displacement model. The bridge: when the
   sequencer triggers computation, Python reads the current WM
   values, passes them to the displacement model, and injects
   the result back into the graph.

   This is a hybrid: graph handles control flow, displacement
   handles computation. Eventually computation should be graph-
   native too, but for now this is the honest interface.

2. How does the BG learn from dopamine? The D1/D2 weights need
   to change based on the dopamine signal. This requires a learning
   rule that adjusts edge weights — something the current graph
   engine doesn't do (edges have fixed weights).

   Answer: add a learn() method to the graph that adjusts edge
   weights based on a reward signal. This is the dopamine-modulated
   Hebbian rule: strengthen edges from active sources to active
   targets when dopamine is positive.

3. How many training examples are needed? The gating policy has
   5 rules. Each rule needs to be reinforced by seeing it in
   context. Estimate: 10-20 multi-step expressions should suffice.

## Implementation Order

1. Add token nodes to the graph (digit 0-9, operators +, -, *, =)
2. Wire token nodes to PFC and BG through thalamus
3. Add edge weight learning (dopamine-modulated Hebbian)
4. Implement the training loop: present tokens, run steps, check answer, reward
5. Test: can the system learn to solve 2-step expressions from 1-step training?
6. Test: can it generalize to 3-step and 4-step?
