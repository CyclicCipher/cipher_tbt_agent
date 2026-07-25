"""refutation.py — capture the moment a game REFUTES the agent's believed win condition.

That instant is where rule hypothesis generation is actually needed: a win condition that has been correct for a level or two
stops paying, the model is exactly as informed as it managed to become, and something has to say what ELSE might be required.
Replaying a game to reach it costs tens of seconds; capturing it turns iteration on a PROPOSER into milliseconds, with the
proposer as the only thing that varies.

DETECTION IS DOMAIN-FREE. A refutation is a drop in the contingency of a condition the agent BELIEVED — it credited that
condition with weight ≥ `eps`, the condition held again, and no reward came, so the weight fell. Nothing here reads a goal
tile, a block or a pad; it watches the agent's own reward model being wrong, which is a thing every game can do.

WHAT THIS IS FOR — and the warning that comes with it. A fixture like this makes it very easy to tune a proposer until it
scores perfectly on the captured moment, which validates nothing: four filters fitted to one example inside a rule grammar
chosen to fit that example is not a discovery. The fixtures are here to DISPROVE proposers, and the games are deliberately
drawn from different rule shapes so that a proposer which only handles one shape is visibly exposed:
  * LockPath L2   — the win is a CONJUNCTION of two spatial conditions (block on pad AND agent on goal).
  * CollectAll    — the win is a TOUR: visit every item, each consumed on contact. No target cell at all.
  * Toggle        — the win is reaching a goal, but the mechanic is a switch that CLOSES the door on the short path; the
                    rule that matters is "do not touch that tile", which is not an arrangement of objects.
A proposer that reports the truth on the first and nothing usable on the others has not generalised — it has a schema.
"""

from __future__ import annotations

from .harness import Environment


def capture(agent_factory, game_factory, budget: int = 600, quiet: int = 40):
    """Play until the agent's reward model is shown to be WRONG, and return that moment.

    TWO triggers, because refutation alone cannot start. Losing a belief presupposes having one, and a game that has never
    paid gives the agent nothing to lose — so on exactly the games where a rule most needs proposing, a refutation detector
    is silent. The second trigger comes from the EPISTEMIC drive, which needs no prior reward:

      * REFUTED    — a condition the agent believed held again and nothing was paid, so its contingency fell.
      * EXHAUSTED  — there is nothing left to learn (`_unlearned_cells` is empty: the model is confident about every
                     feature it can perceive) and still no reward. The
                     agent's implicit expectation is not "condition X pays" but the more basic one that a world it has
                     LEARNED is a world it can succeed in; having modelled everything it can perceive and been paid
                     nothing refutes that. In EFE terms both terms have gone to zero — no pragmatic return and no epistemic
                     return — which leaves no reason to act, and that state is itself a proof that the reward model is
                     missing something (`reference_efe_and_epiplexity`).

    `quiet` is how many consecutive exhausted steps count as settled.

    Returns `{agent, game, frame, step, condition, before, after}` — the live agent with everything it had learned, the game
    (snapshot-able), and which condition lost how much contingency — or None if no refutation occurred inside `budget`.
    The agent and game are returned LIVE rather than pickled: the learned state is large, and a caller that wants to branch
    can `game.snapshot()`/`restore()` around whatever it tries."""
    game = game_factory()
    env = Environment(game)
    frame = env.reset()
    agent = agent_factory()
    settled, best = 0, frame.score
    for step in range(budget):
        believed = {c: w for c, w in agent.goal_mem.w.items() if w >= agent.goal_mem.eps}
        action, coords = agent.step(frame)
        frame = env.step(action, coords)
        for cond, before in believed.items():                      # a BELIEVED condition that just lost contingency
            after = agent.goal_mem.w.get(cond, 0.0)
            if after < before - 1e-9:
                return dict(agent=agent, game=game, frame=frame, step=step, trigger="refuted",
                            condition=cond, before=before, after=after)
        objs = agent.transduce(frame.grid)                         # nothing left to learn, and nothing being paid. No
        exhausted = not agent._unlearned_cells(agent._positions(objs))   # threshold on the learning RATE: "the model is
        settled = settled + 1 if (exhausted and frame.score <= best) else 0   # confident everywhere" is already crisp,
        #   and a cut-off on a decaying rate would be exactly the arbitrary constant this is trying to avoid.
        best = max(best, frame.score)
        if settled >= quiet:                                       # learned everything available, still not paid
            return dict(agent=agent, game=game, frame=frame, step=step, trigger="exhausted",
                        condition=None, before=None, after=None)
        if frame.is_terminal() or frame.is_win():
            break
    return None
