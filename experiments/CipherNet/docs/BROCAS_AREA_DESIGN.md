# Broca's Area Design — Hierarchical Structure Builder

## What Merge Does

Merge takes two syntactic objects A and B and creates {A, B}.
That's it. One operation. Applied recursively, it builds all
hierarchical structure: expressions, sentences, action plans.

Ref: [Merge (linguistics), Wikipedia](https://en.wikipedia.org/wiki/Merge_(linguistics))
Ref: [Mathematical Structure of Syntactic Merge, MIT Press](https://direct.mit.edu/books/oa-monograph/6001/Mathematical-Structure-of-Syntactic-MergeAn)

Key properties:
1. **Binary**: always combines exactly two elements
2. **Bottom-up**: builds from leaves to root
3. **Recursive**: output of Merge can be input to another Merge
4. **Workspace**: Merge operates on a "computational scratchpad"
   that holds partially-built structures

Example — "3 + 4 * 2":
```
Step 1: Merge(4, *) → {4, *}     "4 times something"
Step 2: Merge({4,*}, 2) → {4,*,2}  "4 times 2" = [mul 4 2]
Step 3: Merge(3, +) → {3, +}     "3 plus something"
Step 4: Merge({3,+}, {4,*,2}) → {3,+,{4,*,2}}  "3 plus (4*2)" = [add 3 [mul 4 2]]
```

The order of Merges determines precedence: * merges before +,
so * binds tighter. Learning PEMDAS = learning which Merges to
do first.

## Biology: BA44 and BA45

**BA44 (posterior Broca's):** hierarchical structure building.
Implements Merge. Active for ALL hierarchical structure —
language, music, math, action sequences. This is where the
actual combining happens.

**BA45 (anterior Broca's):** semantic/controlled retrieval.
Selects which elements to Merge based on meaning. "Should I
merge * first or + first?" — BA45 provides the ranking.

**The workspace** is syntactic working memory — similar to PFC
WM but specialized for partially-built structures rather than
individual values. It holds the tree being constructed.

Ref: [Neural basis for human syntax, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2352154617301286)
Ref: [BA44 hierarchization, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0278262620302542)

## Mapping to CipherNet Graph

### Merge as column creation

Merge(A, B) → {A, B} maps to:
1. Columns A and B are both active
2. Broca's area creates a NEW column C that represents {A, B}
3. Column C connects to A and B via structural edges
4. Column C becomes active; A and B may deactivate (incorporated)

This IS our column factory: `graph.create_column()`. Merge = create
a column that represents the combination of two existing columns.

### The workspace as a stack of active columns

The workspace holds partially-built structures. In graph terms:
active columns in Broca's area = the workspace contents.

When a new token arrives:
1. Its column activates
2. Broca checks: can I Merge this with the top of the workspace?
3. If yes: Merge (create new column, connect to children)
4. If no: push this column onto the workspace (wait for more input)

This is bottom-up parsing: tokens arrive left to right, Merge
happens when a complete structure is recognized.

### Operator precedence as Merge priority

"3 + 4 * 2": the tokens arrive as 3, +, 4, *, 2.

Without precedence: Merge(3, +) → Merge({3,+}, 4) → Merge({3,+,4}, *)
→ Merge({3,+,4,*}, 2) → left-to-right evaluation = 14. WRONG.

With precedence: * has higher priority than +. So:
- 3 arrives → push
- + arrives → push (can't Merge yet, need right operand)
- 4 arrives → push
- * arrives → * has higher priority than + → push ABOVE +
- 2 arrives → Merge(4, *, 2) → [mul 4 2] → push result
- No more tokens → Merge(3, +, [mul 4 2]) → [add 3 [mul 4 2]] = 11

The priority determines Merge ORDER. This is learned from examples:
the system sees "4*2=8" and "3+4*2=11" and discovers that * must
Merge before +.

### How priority is represented in the graph

Each operator column has a PRIORITY WEIGHT. Higher priority =
Merge sooner. The weights are learned (Hebbian: when Merging *
before + leads to correct answers, the priority weight of *
increases relative to +).

Concretely: the Broca BA44 node that triggers Merge receives
activation from operator columns. The most activated operator
triggers Merge first. * has higher activation than + (learned),
so * Merges first.

## Subgraph Design

```
Broca's area subgraph:

  BA44 — Merge executor
    merge_trigger:  fires when two Mergeable elements are on workspace
    merge_priority: ranks which Merge to do first
    merge_output:   the newly created column from Merge

  BA45 — Semantic selector
    select_input:   receives from active columns and workspace
    select_output:  signals which elements to Merge

  Workspace — syntactic WM (like PFC WM but for structures)
    ws_slot_0:  first workspace position (stack top)
    ws_slot_1:  second workspace position
    ws_slot_2:  third workspace position
    ws_slot_3:  fourth workspace position
    (4 slots = depth-4 nesting capacity)

  Each ws_slot has the same column structure (L4, L23, L5, L6)
  with strong self-loops (persistent, like PFC WM).
```

## Connection to Other Areas

```
Tokens → Thalamus → Broca's BA45 (semantic selection)
                  → Broca's BA44 (merge trigger)
                  → Workspace (push new token)

Workspace → BA44 (what's available to Merge)
BA45 → BA44 (which elements to Merge, what priority)
BA44 → Column factory (create merged column)
BA44 → Workspace (push merged result, pop children)

Workspace → PFC (send completed structure for execution)
PFC → BG → Thalamus → Computation columns (execute the plan)
```

## Implementation Plan

### Step 1: Broca's area JSON prior

Create `broca.json` with BA44, BA45, and 4 workspace slots.
Wire to thalamus and PFC in config.json.

### Step 2: Merge operation in the graph engine

Add a `merge()` method that:
1. Reads the workspace (what's active)
2. Determines Merge priority (from BA45/operator weights)
3. Creates a new column representing the merged structure
4. Updates the workspace (pop children, push result)

### Step 3: Token processing with Merge

Extend the Brain to use Broca's area:
1. Token arrives → activate column → push to workspace
2. After each push, check if Merge is possible
3. If yes, Merge. If no, wait for more tokens.
4. After '=', the workspace should contain ONE structure = the
   complete parsed expression. Send to PFC for execution.

### Step 4: Learn operator precedence

Train on examples with different operator orders.
Hebbian learning adjusts priority weights.
Test: does it learn PEMDAS from examples?

## What This Gives Us

- **Math parsing**: "3+4*2" → [add 3 [mul 4 2]] → 11
- **Nested expressions**: "(3+4)*2" → [mul [add 3 4] 2] → 14
- **Sentence parsing**: "the cat sat" → [S [NP the cat] [VP sat]]
- **Action planning**: "click button then type text" → [seq [click button] [type text]]

All from the SAME Merge operation. Domain-general hierarchy building.
