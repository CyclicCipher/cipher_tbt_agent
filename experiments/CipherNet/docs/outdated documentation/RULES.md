# CipherNet Rules

Non-negotiable design constraints.

## Tokenization

ALL input is tokenized at the level of individual characters.

- The number 307 is three tokens: '3', '0', '7'
- The expression "3 + 4 = 7" is: '3', ' ', '+', ' ', '4', ' ', '=', ' ', '7'
- A word like "apple" is: 'a', 'p', 'p', 'l', 'e'

The system must learn what groups of characters mean.
Multi-digit numbers, words, and operators are NOT pre-tokenized.

## No input/output distinction

No node, token, or value is labeled as "input" or "output,"
"operand" or "operator," "argument" or "result."

Every token is a position. Every relationship is a manifold.
Prediction is projection onto the manifold from any direction.

## Priors are subgraphs

Structured prior knowledge is provided as pre-built subgraphs
in the priors/ folder. A config file describes how they connect.
New priors = new subgraph files. The system works without priors
(slower learning, same capabilities).

Priors represent SUBCORTICAL or ARCHITECTURAL structure that is
innate (present at birth). The neocortex is LEARNED — no neocortical
structure should be hand-coded as a prior.

## Manifolds are probabilistic

Every discovered rule is a probability distribution, not a sharp
surface. Deterministic rules are the special case where the
distribution is tightly concentrated. Uncertainty is always
represented.

## Local-first training

New concepts are learned in isolated subgraphs first (fast, cheap),
then integrated into the main graph (slow, global). This mirrors
hippocampal fast learning + cortical slow consolidation.

## Geometric structure first, categorical structure second

Manifold learning discovers the shape of rules within a domain.
Categorical structure (composition, adjunctions, functors) discovers
how rules relate — both within and across domains. Both are needed.
But geometry comes first: you can't compose what you haven't discovered.

## No Python orchestration of brain regions

Code must NEVER say "use the thalamus" or "use Broca's area" or
"route to the addition column." All behavior must emerge from pure
graph dynamics — graph.step() propagating activations along edges.

The only Python code allowed is:
- graph.step() — the one update rule
- graph.settle() — prospective configuration (repeated step with clamps)
- Feeding input tokens (activating a column's L4)
- Reading output tokens (checking the output cortex)
- Learning (graph.learn() — error-driven weight updates)
- Setting clamps (desired input + output for settle)

Everything else — parsing, computing, sequencing, routing,
deciding which brain region does what — must come from the graph
structure and edge weights. If the graph can't do it, the graph
structure is wrong, not the rule.

## Predictive coding is the learning rule

The brain is simultaneously a recognition model (bottom-up errors)
and a generative model (top-down predictions).

- Feedforward (bottom-up): prediction ERRORS from L2/3 to L4 of higher area
- Feedback (top-down): PREDICTIONS from L5 to L2/3 of lower area (skip L4)
- Learning: delta_w = lr * target.error * source.activation
- Zero error = zero weight change (prevents catastrophic forgetting)

## Brain oscillations are functional

Different frequency bands separate message streams:
- Gamma (30-100 Hz): feedforward errors (L2/3, fast)
- Beta (13-30 Hz): feedback predictions (L5/L6, slow)
- Theta (4-8 Hz): WM maintenance, episodic sequencing (PFC)

In graph.step(), different decay rates per frequency band create
this spectral separation. Not cosmetic — functionally necessary.

## Context-dependent gating (Mamba selection)

Context must change the effective connectivity, not just add
activation. This is Mamba's selective state space mechanism:
delta (retention), B (input selection), C (output selection) all
vary with context. Without this, the same input always produces
the same output regardless of context.

Context is LEARNED, not hardcoded. The BG learns through reward
that '+' requires different gating patterns than '-'. The system
discovers which tokens create which contexts — we never tell it
"operators are gate sources." Synaptogenesis may create GATE edges
when a gating pattern reduces prediction error.
