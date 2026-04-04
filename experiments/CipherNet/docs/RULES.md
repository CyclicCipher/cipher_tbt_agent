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
