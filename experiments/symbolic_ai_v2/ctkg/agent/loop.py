"""Perception-action loop: observe -> perceive -> learn -> evaluate -> act.

The AgentLoop wires together the Environment with the CTKG inference modules:
  - EpisodicStore: records high-surprise steps for replay
  - SelfModel: provides reflective meta-awareness of knowledge quality
  - TheoryOfMind: tracks other agents' behaviour patterns
  - Predictor: next-token prediction for active inference
  - active_inference.score_policy / select_action: G(pi) policy scoring

Loop architecture per step:
  1. observe(env) -> (tokens, etypes)
  2. flatten tokens into context buffer
  3. estimate prediction_error from predictor (if available)
  4. episodic_store.add_event if surprising
  5. [online_learning] every token: compute_gradients + apply_gradients
  6. [online_learning] every update_interval tokens: SelfModel.update (stub for em_loop)
  7. if step >= random_until and predictor fitted:
       action = select_action(predictor, context, available_actions, goal)
     else:
       action = random.choice(available_actions)
  8. env.act(action)
  9. append action token to context buffer

The context buffer is capped at CONTEXT_MAX tokens to bound memory.

Usage::

    env  = TextWorldEnv()
    store = EpisodicStore()
    loop = AgentLoop(env, predictor=None, episodic_store=store,
                     goal_tokens=['HOLD_gem'])
    summary = loop.run(max_steps=200)
    print(summary)

See ROADMAP.md Stage 5, Step 5.5 for design decisions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.environment import Environment
from experiments.symbolic_ai_v2.ctkg.core.episodic_store import (
    EpisodicStore, EpisodicEvent,
)
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.self_model import SelfModel
from experiments.symbolic_ai_v2.ctkg.core.theory_of_mind import TheoryOfMind
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor
from experiments.symbolic_ai_v2.ctkg.inference.active_inference import (
    select_action, score_policy, PolicyScore,
)
from experiments.symbolic_ai_v2.ctkg.learning.lens_update import (
    compute_gradients, apply_gradients,
)


# Maximum tokens kept in the rolling context buffer
CONTEXT_MAX = 50


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    """Information about one completed agent step.

    Attributes
    ----------
    step:
        Global step index (0-based).
    action:
        Action string executed at this step.
    tokens:
        Flat token list from the observation BEFORE the action.
    prediction_error:
        1 - max_prob at this step (0.0 if no predictor available).
    won:
        True if the agent achieved the goal after this step.
    done:
        True if the episode terminated (failure or win).
    """

    step: int
    action: str
    tokens: list[str]
    prediction_error: float
    won: bool
    done: bool

    def __repr__(self) -> str:
        return (
            f"StepInfo(step={self.step}, "
            f"action={self.action!r}, "
            f"pe={self.prediction_error:.3f}, "
            f"done={self.done}, won={self.won})"
        )


@dataclass
class RunSummary:
    """Summary of a complete episode.

    Attributes
    ----------
    n_steps:
        Number of steps executed.
    won:
        True if the agent achieved the goal.
    G_history:
        G(pi) value of the selected action at each step (nan if random).
    pe_history:
        Prediction error at each step.
    episodes_stored:
        Number of high-surprise events stored in the episodic store.
    """

    n_steps: int
    won: bool
    G_history: list[float]
    pe_history: list[float]
    episodes_stored: int

    def __repr__(self) -> str:
        avg_G  = sum(self.G_history) / len(self.G_history) if self.G_history else float('nan')
        avg_pe = sum(self.pe_history) / len(self.pe_history) if self.pe_history else float('nan')
        return (
            f"RunSummary(n_steps={self.n_steps}, won={self.won}, "
            f"avg_G={avg_G:.3f}, avg_pe={avg_pe:.3f}, "
            f"stored={self.episodes_stored})"
        )


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """Full perception-action loop for CTKG agents.

    Parameters
    ----------
    env:
        Any Environment subclass.
    predictor:
        Fitted Predictor for next-token prediction and active inference.
        If None, the agent acts randomly for all steps (useful for warm-up).
    episodic_store:
        Pre-constructed EpisodicStore; episodes are added in-place.
    goal_tokens:
        Token sequence representing the goal.  Used for pragmatic value
        in active inference.  E.g. ['HOLD_gem'].
    theory_of_mind:
        Optional TheoryOfMind tracker (populated but not used for action
        selection in this implementation — reserved for Stage 6).
    self_model:
        Optional SelfModel for reflective meta-awareness.
    random_until:
        Act randomly for the first `random_until` steps (exploration phase).
        Afterwards, use active inference if predictor is available.
    seed:
        Random seed for action sampling.
    online_learning:
        If True, the CTKG is updated online during the loop.  Requires
        `morphism_graph` to be provided.

        Every token: compute_gradients + apply_gradients (Phase 8 lens update).
        Every `update_interval` steps: SelfModel.update (stub for em_loop /
        mdl_prune / meta_fca batch pass — to be wired in Stage 7).
    morphism_graph:
        The live MorphismGraph to update when `online_learning=True`.
        Ignored when `online_learning=False`.
    update_interval:
        Number of steps between SelfModel meta-updates when online learning
        is enabled.  Default 10.
    """

    def __init__(
        self,
        env: Environment,
        predictor: Optional[Predictor],
        episodic_store: EpisodicStore,
        goal_tokens: list[str],
        theory_of_mind: Optional[TheoryOfMind] = None,
        self_model: Optional[SelfModel] = None,
        random_until: int = 10,
        seed: int = 0,
        online_learning: bool = False,
        morphism_graph: Optional[MorphismGraph] = None,
        update_interval: int = 10,
    ) -> None:
        self._env            = env
        self._predictor      = predictor
        self._store          = episodic_store
        self._goal           = list(goal_tokens)
        self._tom            = theory_of_mind
        self._self_model     = self_model
        self._random_until   = random_until
        self._rng            = random.Random(seed)
        self._online_learning = online_learning
        self._mg             = morphism_graph
        self._update_interval = update_interval

        self._step_count: int = 0
        self._token_count: int = 0
        self._context: list[str] = []
        self._G_history: list[float] = []
        self._pe_history: list[float] = []
        self._episodes_stored: int = 0

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def context_tokens(self) -> list[str]:
        """Return the current rolling context buffer (last CONTEXT_MAX tokens)."""
        return list(self._context)

    def _extend_context(self, tokens: list[str]) -> None:
        """Append tokens to the context buffer and trim to CONTEXT_MAX."""
        self._context.extend(tokens)
        if len(self._context) > CONTEXT_MAX:
            self._context = self._context[-CONTEXT_MAX:]

    # ------------------------------------------------------------------
    # Prediction error
    # ------------------------------------------------------------------

    def _prediction_error(self, tokens: list[str]) -> float:
        """Estimate prediction error for the first token of `tokens`.

        Returns 1 - max_prob, where max_prob is the probability assigned
        to the actual first token by the predictor given the current context.
        Returns 0.5 (neutral surprise) if the predictor is unavailable or
        the context is empty.
        """
        if self._predictor is None or not self._context or not tokens:
            return 0.5
        try:
            dist = self._predictor.predict_next(self._context)
            first_tok = tokens[0]
            max_prob = dist.get(first_tok, 0.0)
            return 1.0 - max_prob
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _select(self, available: list[str]) -> tuple[str, float]:
        """Select an action; return (action, G_value).

        Uses active inference if the predictor is available and
        step_count >= random_until; otherwise picks uniformly at random.
        G_value is float('nan') for random actions.
        """
        if not available:
            return 'wait', float('nan')

        use_ai = (
            self._predictor is not None
            and self._step_count >= self._random_until
        )

        if use_ai:
            scored = score_policy(
                self._predictor,
                self._context,
                available,
                self._goal,
            )
            if scored:
                return scored[0].action, scored[0].G

        return self._rng.choice(available), float('nan')

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> StepInfo:
        """Execute one perception-action step.

        Returns
        -------
        StepInfo describing what happened.
        """
        # 1. Observe
        obs = self._env.observe()                     # list[(token, etype)]
        tokens = [tok for tok, _ in obs]

        # 2. Estimate prediction error
        pe = self._prediction_error(tokens)
        self._pe_history.append(pe)

        # 3. Store episode if surprising
        evt = self._store.add_event(self._step_count, tokens, pe)
        if evt is not None:
            self._episodes_stored += 1

        # 4. Extend context with observation tokens
        self._extend_context(tokens)

        # 4b. Online CTKG update (Phase 8 lens) — every token
        if self._online_learning and self._mg is not None and self._predictor is not None:
            for tok in tokens:
                self._token_count += 1
                try:
                    grads = compute_gradients(self._mg, self._context, tok)
                    apply_gradients(self._mg, grads)
                except Exception:
                    pass  # never crash the loop on a learning error
            # Meta-update (self-model) every update_interval steps
            if self._self_model is not None and self._step_count % self._update_interval == 0:
                try:
                    self._self_model.update(self._mg.morphisms())
                except Exception:
                    pass

        # 5. Select action
        available = self._env.available_actions()
        action, G_val = self._select(available)
        self._G_history.append(G_val)

        # 6. Execute action
        self._env.act(action)

        # 7. Append action token to context
        self._extend_context([action])

        # 8. Increment step counter
        self._step_count += 1

        return StepInfo(
            step=self._step_count - 1,
            action=action,
            tokens=tokens,
            prediction_error=pe,
            won=self._env.won,
            done=self._env.done or self._env.won,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, max_steps: int = 200) -> RunSummary:
        """Run until the episode terminates or max_steps are reached.

        Parameters
        ----------
        max_steps:
            Hard cap on the number of steps.

        Returns
        -------
        RunSummary with episode statistics.
        """
        for _ in range(max_steps):
            info = self.step()
            if info.done:
                break

        return RunSummary(
            n_steps=self._step_count,
            won=self._env.won,
            G_history=list(self._G_history),
            pe_history=list(self._pe_history),
            episodes_stored=self._episodes_stored,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the loop state (not the environment)."""
        self._step_count = 0
        self._token_count = 0
        self._context = []
        self._G_history = []
        self._pe_history = []
        self._episodes_stored = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AgentLoop(step={self._step_count}, "
            f"predictor={'set' if self._predictor else 'none'}, "
            f"goal={self._goal!r})"
        )
