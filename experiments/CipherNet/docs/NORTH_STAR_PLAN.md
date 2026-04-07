# North Star Plan — From Arithmetic to Danganronpa

## The Goal

Build an AI that can play Danganronpa: Trigger Happy Havoc from start
to finish, using only vision, audio, keyboard, and mouse. A generally
intelligent KVM agent.

## The Path

Start with text/token domains to validate the architecture. Text is
useful both for testing AND for deployment (processing codebases,
documents). Build toward vision and motor control incrementally.

## Foundation (Built)

| Component | Status | Description |
|-----------|--------|-------------|
| Core graph | Done | Nodes, edges (spatial/temporal/binding/gate), subgraphs |
| Update rule | Done | Gated recurrence (Mamba3-inspired), inhibition, bistable WM |
| ANS | Done | Magnitude comparison (JSON prior) |
| PFC | Done | 3 WM stripes, sequencer, monitor (JSON prior) |
| Basal ganglia | Done | Go/NoGo/GPe/STN/GPi/dopamine (JSON prior) |
| Thalamus | Done | Relay + reticular (JSON prior) |
| Column factory | Done | Dynamic column creation with auto thalamic relay |
| Number line | Done | Graph-native 0-50, dual-wave co-termination arithmetic |
| Hebbian learning | Done | Dopamine-modulated edge weight updates |
| Experience engine | Done | Brain class: learns digits, operators, computes expressions |
| Displacement learning | Done | Isometry group identification for arithmetic |

## What Needs to Change

### 1. Token-sequential processing (Mamba3-style)

Currently: Python code explicitly parses expressions and routes tokens.
Needed: tokens flow in one at a time, the graph state evolves, answers
emerge from the state. Like Mamba3's recurrent state.

Implementation: each token activates its column. The graph runs step().
The PFC state evolves. After enough tokens, the answer is readable
from the graph state. No Python parsing.

This is required for generality: the same mechanism must handle
"3+4=", "The butler did it", and game dialogue.

### 2. Character-level input everywhere

Currently: single-digit numbers are single characters, which works.
Multi-digit numbers (307) must be processed as '3','0','7' and the
system must discover place value and carry rules.

This is NOT special-cased for arithmetic. It's the same as processing
any multi-character sequence: 'c','a','t' → "cat". The system learns
that character ORDER matters and that different orderings mean
different things.

### 3. Broca's area — hierarchical structure builder

Currently missing entirely. Broca's area is the brain's system for:

**Merge operation**: combining two elements into a hierarchical structure.
"3" + "+" → [addition expression with first operand 3].
"the" + "cat" → [noun phrase].

**Recursive nesting**: structures within structures.
"(3 + 4) * 2" = [multiply [add 3 4] 2].
"The cat that the dog chased ran away" = nested relative clause.

**Sequential planning**: ordering actions for execution.
Planning keyboard inputs to type a response in the game.

**Domain-general**: Broca's area processes hierarchy in language,
music, math, action sequences, and abstract reasoning. It's not
language-specific — it's hierarchy-specific.

Ref: [Broca's area as supramodal hierarchical processor]
(https://www.sciencedirect.com/science/article/abs/pii/S0010945208703848)
Ref: [Broca's area and hierarchical organization of behavior]
(https://www.sciencedirect.com/science/article/pii/S0896627306004053)
Ref: [Broca's area processes hierarchical organization of observed action]
(https://pmc.ncbi.nlm.nih.gov/articles/PMC3894456/)

**For CipherNet:** Broca's area is a subgraph (like PFC, BG, thalamus)
that takes sequential input and builds hierarchical structure. It's
the PARSER — it discovers that "3+4*2" has structure [add 3 [mul 4 2]],
not [mul [add 3 4] 2]. It handles operator precedence, nested
expressions, sentence structure, and action planning.

Broca's area interacts with the PFC: PFC holds working memory,
Broca builds structure from the token stream, PFC sequences the
execution of the built structure.

### 4. Wernicke's area — meaning/semantics

Broca handles structure (syntax). Wernicke handles meaning (semantics).
"The cat sat on the mat" — Broca parses the syntax, Wernicke understands
what it MEANS (a cat, sitting, on a mat).

For Danganronpa: Broca parses the dialogue, Wernicke extracts the
meaning (who said what, what clues are being revealed, what's suspicious).

Implementation: Wernicke's area columns connect stimuli to learned
concepts. The "concept" of murder, alibi, evidence, etc. Each concept
is a column. Wernicke activates concept columns based on semantic
content of the input.

### 5. Vision (V1 → V2 → V4 → IT)

For Danganronpa: the system needs to see the screen. The visual
hierarchy:
- V1: edge detection, orientation (low-level features)
- V2: texture, simple shapes
- V4: color, complex shapes
- IT (inferotemporal): object recognition

Each level is columns that learn features at increasing abstraction.
TBT applies: each visual column learns a model of visual objects
by sensing features at locations.

Not needed for text domains. Needed for the game.

### 6. Motor output (keyboard + mouse)

For Danganronpa: the system needs to press keys and move/click the mouse.
Motor cortex columns plan and execute actions.

The PFC decides WHAT to do. Motor cortex decides HOW.
BG gates motor actions (Go/NoGo for action execution).

## Validation Criteria (must-pass before game integration)

Before attempting Danganronpa, three things must work:

### Criterion 1: Token I/O like an LLM

**Input**: tokens arrive sequentially, activate columns. Like tokens
entering a context window. Character-level tokenization.

**Output**: a dedicated output area (motor cortex analog) that the
system actively drives activations INTO when it produces a response.
The output area has one node per possible output token. The most
active output node IS the next output token. Like an LLM's output
logits projected from a specific head — not reading the whole brain.

The output area is:
- A fixed set of columns (one per output token: digits, letters,
  operators, special tokens)
- Connected FROM the PFC, Broca's, and computation columns
- The system LEARNS which output to produce by receiving reward
  for correct outputs (dopamine → strengthens the edges that led
  to the correct output activation)
- Output is generated token by token, autoregressively: produce
  one token, feed it back as input, produce the next

Test: given "3+4=", output "7". Given "12+38=", output "5","0".
Given "The capital of France is ", output "P","a","r","i","s".

### Criterion 2: Broca's + Wernicke's areas fully working

**Broca's area**: parses hierarchical structure from token stream.
Handles operator precedence, nested expressions, sentence structure.
Implements Merge (combine two active columns into a structure).
Domain-general: same mechanism for math, language, action planning.

**Wernicke's area**: extracts meaning/semantics. Connects parsed
structures to learned concepts. Understands WHAT things mean, not
just HOW they're structured.

Together: Broca parses "3+4*2=" into [add 3 [mul 4 2]], Wernicke
knows that "add" means the addition displacement and "mul" means
repeated addition. The PFC sequences execution.

For language: Broca parses "The butler was in the kitchen" into
[sentence [NP the butler] [VP was [PP in [NP the kitchen]]]],
Wernicke extracts: butler = person, kitchen = location, "was in" =
location-relation. Stored as a fact.

Test: "3+4*2=" → "11" (not "14"). "What is 3+4*2?" → "11".
Can parse and answer questions about short text passages.

### Criterion 3: Cortico-thalamic loop verified

The full loop works end to end:
- Cortical columns process input
- Thalamus routes between columns (gated by BG)
- BG learns when to gate (from dopamine)
- PFC holds working memory across steps
- Broca builds structure
- Output area produces response
- Reward signal trains the whole system

Test: the system can learn a NEW task from examples (not just
arithmetic — something it's never seen before) using the same
architecture. For example: learn to capitalize the first letter
of each word, or learn to reverse a string, or learn simple
ciphers. If the architecture is general, it should handle novel
tasks with the same machinery.

## Implementation Phases

### Phase 1: Output area + token I/O

Build the output area as a prior subgraph (like motor cortex).
One output node per character token. Implement autoregressive
output: most active output node → produced token → fed back as
input.

Replace Python-parsed expression processing with pure token I/O:
tokens in → graph processing → tokens out.

### Phase 2: Multi-digit numbers

Process "307+456=" as character stream: '3','0','7','+','4','5','6','='.
System discovers place value. Computes column-by-column with carries.
Number line only needs 0-18. PFC sequences digit positions.

Outputs "7","6","3" (ones first, or learns to output in reading order).

### Phase 3: Broca's area

Build Broca's as a JSON prior subgraph. Implements Merge.
Learns operator precedence from examples.
Learns sentence structure from text examples.

### Phase 4: Wernicke's area

Build semantic concept columns. Train on text.
System learns word meanings from context.
Can answer questions about passages.

### Phase 4.5: PEMDAS validation

After Broca's and Wernicke's are working, test whether the system
can learn operator precedence from examples alone:
- Train on: "3+4*2=11", "2*3+1=7", "5+6*0=5", etc.
- Test: "1+2*3+4=11" (never seen, requires correct precedence)
- The system must parse this as [add 1 [add [mul 2 3] 4]] = 11,
  NOT as [add [add [mul [add 1 2] 3] 4]] = 13.

This is the litmus test for Broca's area: if it can learn PEMDAS
from examples without hardcoded rules, hierarchical Merge works.

### Phase 5: Verify criterion 3 (novel task learning)

Test the architecture on tasks it wasn't designed for.
If it can learn novel tasks from examples, the architecture is general.
If it can't, identify what's missing and fix it.

### Phase 6+: Vision, motor, game integration

Only after criteria 1-3 are met. These add new input/output modalities
but don't change the core architecture.

## Broca's Area — Design Notes

### What it IS

A set of cortical columns (same architecture as everything else)
that specialize in hierarchical structure building. Like PFC but
for STRUCTURE rather than MAINTENANCE.

Subregions:
- **BA44 (posterior)**: syntactic structure, Merge operation,
  hierarchical processing. Active for ALL hierarchical structure —
  language, music, math, action planning.
- **BA45 (anterior)**: semantic integration, thematic structure,
  reanalysis. Active when structure needs to be revised.

### How Merge works as graph operations

Merge takes two active columns and creates a NEW column that
represents their combination:

Input: column for "3" is active, column for "+" is active.
Merge: create a new column "expr:3+" that represents "addition
expression with first operand 3."

Input: "expr:3+" is active, column for "4" is active.
Merge: create a new column "expr:3+4" that represents "complete
addition expression 3+4."

Each Merge creates a NEW column (or activates an existing one if
this structure has been seen before). The new column connects to
its children via structural edges.

This IS the column factory — Merge creates columns dynamically
from active inputs. The hierarchical structure is the GRAPH of
columns connected by structural edges.

### Operator precedence from learning

The system doesn't need hardcoded precedence rules. It learns them
from examples:
- Sees "3+4*2=11": learns that * binds tighter than +
- Sees "(3+4)*2=14": learns that parentheses override precedence
- The Broca columns that handle * develop stronger activation
  than those for + at the Merge level, causing * to Merge first.

This is learned the same way everything else is learned: from examples
with dopamine reward when correct.

## Key Insight

Every component of this system is the SAME THING: cortical columns
connected through the thalamus, gated by the BG, sequenced by the PFC.
Vision, language, arithmetic, reasoning, motor control — all columns.
The only things that differ are:
1. What inputs the columns receive
2. What other columns they connect to
3. What displacement models they've learned
4. What hierarchical structures Broca has built from them

The universality of the cortical column IS the universality of
intelligence.
