"""Expected free energy G(pi): epistemic + pragmatic value, policy selection.

Active inference (Friston et al.) selects actions by minimising the expected
free energy G(pi) of a policy pi.  Here a "policy" is a single action (one-step
lookahead), consistent with the reactive agent architecture.

G(pi) = -(epistemic_value + pragmatic_value)

  epistemic_value(a, c) = H(P(next|c)) - H(P(next|c+[a]))
      Information gain: how much does taking action a reduce uncertainty about
      the next observation?  Measured by the reduction in Shannon entropy of
      the next-token distribution.

  pragmatic_value(a, c, goal) = P(goal[0] | c + [a])
      Goal proximity: how likely is the first goal token immediately after
      taking action a?  This is a single-step approximation of the full KL
      divergence from prior to posterior under the goal distribution.

Both values are non-negative (epistemic value >= 0 always; pragmatic value
in [0,1]).  G(pi) = -(sum) so that lower G = more preferred action.

The Predictor is queried with `predict_next(context)` which returns a
distribution over the next token.  We append the action token to the context
and call `predict_next(context + [action])` to get the posterior.

Entropy: H(p) = -sum_x p(x) log2 p(x).

Reference: Parr, Pezzulo & Friston (2022), "Active Inference: The Free Energy
Principle in Mind, Brain and Behaviour."  MIT Press.

See CTKG_ARCHITECTURE.md §ActiveInference and ROADMAP.md Stage 5, Step 5.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(dist: dict[str, float]) -> float:
    """Shannon entropy H(dist) in bits.

    Parameters
    ----------
    dist:
        Probability distribution (values sum to approximately 1.0; zero
        values are silently skipped).

    Returns
    -------
    H in bits (>= 0).
    """
    h = 0.0
    for p in dist.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PolicyScore:
    """Score for one candidate action.

    Attributes
    ----------
    action:
        The action string evaluated.
    epistemic_value:
        Expected information gain (H_prior - H_posterior).  >= 0.
    pragmatic_value:
        P(goal[0] | context + [action]).  In [0, 1].
    G:
        -(epistemic_value + pragmatic_value).  Lower = more preferred.
    """

    action: str
    epistemic_value: float
    pragmatic_value: float
    G: float

    def __repr__(self) -> str:
        return (
            f"PolicyScore(action={self.action!r}, "
            f"epi={self.epistemic_value:.3f}, "
            f"prag={self.pragmatic_value:.3f}, "
            f"G={self.G:.3f})"
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def epistemic_value(
    predictor: Predictor,
    context_tokens: list[str],
    action: str,
) -> float:
    """Expected information gain from taking action in context.

    H(P(next|context)) - H(P(next|context + [action]))

    Parameters
    ----------
    predictor:
        Fitted Predictor.
    context_tokens:
        Current token sequence (the prefix seen so far).
    action:
        The candidate action token.

    Returns
    -------
    float >= 0 — entropy reduction in bits.
    """
    prior = predictor.predict_next(context_tokens)
    posterior = predictor.predict_next(list(context_tokens) + [action])
    return max(0.0, _entropy(prior) - _entropy(posterior))


def pragmatic_value(
    predictor: Predictor,
    context_tokens: list[str],
    action: str,
    goal_tokens: list[str],
) -> float:
    """Probability of reaching the first goal token after taking action.

    P(goal_tokens[0] | context + [action])

    Parameters
    ----------
    predictor:
        Fitted Predictor.
    context_tokens:
        Current token sequence.
    action:
        The candidate action token.
    goal_tokens:
        Goal token sequence.  Only the first token is used for the
        single-step approximation.  Empty goal_tokens returns 0.0.

    Returns
    -------
    float in [0, 1].
    """
    if not goal_tokens:
        return 0.0
    extended = list(context_tokens) + [action]
    dist = predictor.predict_next(extended)
    return dist.get(goal_tokens[0], 0.0)


def score_policy(
    predictor: Predictor,
    context_tokens: list[str],
    actions: list[str],
    goal_tokens: list[str],
) -> list[PolicyScore]:
    """Score all candidate actions by G(pi).

    Parameters
    ----------
    predictor:
        Fitted Predictor.
    context_tokens:
        Current observation sequence.
    actions:
        List of candidate action strings.
    goal_tokens:
        Goal token sequence; used for pragmatic value computation.

    Returns
    -------
    List of PolicyScore objects sorted by G ascending (best first = lowest G).
    Empty list if actions is empty.
    """
    scores: list[PolicyScore] = []
    for action in actions:
        epi  = epistemic_value(predictor, context_tokens, action)
        prag = pragmatic_value(predictor, context_tokens, action, goal_tokens)
        G    = -(epi + prag)
        scores.append(PolicyScore(
            action=action,
            epistemic_value=epi,
            pragmatic_value=prag,
            G=G,
        ))
    scores.sort(key=lambda s: s.G)
    return scores


def select_action(
    predictor: Predictor,
    context_tokens: list[str],
    actions: list[str],
    goal_tokens: list[str],
) -> str:
    """Return the action with lowest G(pi).

    Parameters
    ----------
    predictor:
        Fitted Predictor.
    context_tokens:
        Current observation sequence.
    actions:
        Non-empty list of candidate action strings.
    goal_tokens:
        Goal token sequence.

    Returns
    -------
    str -- selected action.  Returns actions[0] if actions has length 1;
    returns '' if actions is empty.
    """
    if not actions:
        return ''
    if len(actions) == 1:
        return actions[0]
    scored = score_policy(predictor, context_tokens, actions, goal_tokens)
    return scored[0].action if scored else actions[0]
