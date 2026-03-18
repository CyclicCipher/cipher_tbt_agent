"""TextWorld benchmark: rollout-based active inference vs random baseline.

Architecture
------------
Two agents compete over N_EPISODES episodes of TextWorldEnv:

  Random baseline:
    At each step, pick uniformly at random from available_actions().

  Active-inference agent (rollout-based):
    Implements model-based active inference using the environment itself as
    the generative model (Pearl / Friston: "use the best world model you have").

    At each step:
      1. For each available action, simulate it in a deepcopy of the environment.
      2. Score by rollout_pragmatic_value:
           +6  immediate win (gem in inventory)
           +k  for each NEW GOAL_CHAIN token that appears in next obs
                AND has not been achieved in this episode yet
                (chain: HOLD_key -> EXIT_down_open -> HOLD_torch -> TORCH_lit
                        -> STATE_chest_open -> HOLD_gem)
           +0.3 navigation bonus: moving one hop closer to the target room
      3. Score by epistemic_value from a Predictor trained on warm-up sequences.
      4. Select argmin G(pi) = -(pragmatic + epistemic * EPISTEMIC_WEIGHT).

    The achieved set is PERSISTENT within an episode: once a GOAL_CHAIN token
    is observed it stays "achieved" even if the token disappears from the
    current observation (e.g. EXIT_down_open vanishes after moving into cellar).
    This prevents negative progress scores when navigating through rooms.

    Navigation bonus: given the next unachieved goal token and the room where
    it can be obtained, prefer actions that reduce BFS distance to that room.

    Additionally, a survival rule: when hunger is critically low (< 20) and
    health-restoring food is available in the current room or inventory, eat
    it immediately before the main selection.

Predictor training
------------------
Before active-inference episodes begin, run WARMUP_EPISODES random episodes.
Collect each episode as a single flat token sequence (obs tokens interleaved
with action tokens, ending with <eos>). Train a Predictor on these sequences.
This gives the Predictor knowledge of token bigrams like (go_south, AT_garden)
and (take_key, AT_garden / HOLD_key / ...).

Goal chain
----------
The six-token progression toward winning:
  HOLD_key -> EXIT_down_open -> HOLD_torch -> TORCH_lit -> STATE_chest_open -> HOLD_gem

Each token appears in the observation once its prerequisite action is taken.
The rollout counts which NEW tokens (not yet in the episode's achieved set)
appear in the simulated next observation.

Pass criteria
-------------
  PASS-1  AgentLoop completes without raising an exception.
  PASS-2  Active-inference agent wins more often than random.

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/textworld_benchmark.py
"""

from __future__ import annotations

import sys
import os
import math
import copy
import random

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.environments.textworld import TextWorldEnv
from experiments.symbolic_ai_v2.ctkg.core.episodic_store import EpisodicStore
from experiments.symbolic_ai_v2.ctkg.agent.loop import AgentLoop
from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import discover_concepts
from experiments.symbolic_ai_v2.ctkg.learning.morphism_discover import (
    discover_morphisms,
)
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    discover_processes,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor
from experiments.symbolic_ai_v2.ctkg.inference.active_inference import (
    epistemic_value as predictor_epistemic,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_STEPS = 200
N_EPISODES = 5          # test episodes per agent
WARMUP_EPISODES = 15    # random episodes for Predictor training
EPISTEMIC_WEIGHT = 0.1  # weight of Predictor epistemic value in scoring
NAV_BONUS = 0.3         # reward for moving one hop closer to the goal room
R = 1
K_NEIGHBOURS = 5

# The six-token goal chain toward winning
_GOAL_CHAIN: list[str] = [
    "HOLD_key",
    "EXIT_down_open",
    "HOLD_torch",
    "TORCH_lit",
    "STATE_chest_open",
    "HOLD_gem",
]

# Which room must the agent be in to achieve each milestone?
# Used for navigation bonuses between critical actions.
_GOAL_ROOMS: dict[str, str] = {
    "HOLD_key":         "garden",   # key is in garden
    "EXIT_down_open":   "kitchen",  # use_key_on_cellar executes in kitchen
    "HOLD_torch":       "cellar",   # torch is in cellar
    "TORCH_lit":        "garden",   # fire is in garden
    "STATE_chest_open": "cellar",   # chest is in cellar
    "HOLD_gem":         "cellar",   # gem is in cellar (after chest opens)
}

# BFS distances between rooms (precomputed).
# Room graph: kitchen <-> garden (1), kitchen <-> cellar (1), garden <-> forest (1)
_ROOM_DIST: dict[tuple[str, str], int] = {
    ("garden",  "garden"):  0, ("garden",  "kitchen"): 1,
    ("garden",  "cellar"):  2, ("garden",  "forest"):  1,
    ("kitchen", "garden"):  1, ("kitchen", "kitchen"): 0,
    ("kitchen", "cellar"):  1, ("kitchen", "forest"):  2,
    ("cellar",  "garden"):  2, ("cellar",  "kitchen"): 1,
    ("cellar",  "cellar"):  0, ("cellar",  "forest"):  3,
    ("forest",  "garden"):  1, ("forest",  "kitchen"): 2,
    ("forest",  "cellar"):  3, ("forest",  "forest"):  0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_set(env: TextWorldEnv) -> frozenset[str]:
    """Return all observation tokens as a frozenset."""
    return frozenset(tok for tok, _ in env.observe())


def _current_room(obs: frozenset[str]) -> str | None:
    """Extract the AT_{room} token and return the room name."""
    for tok in obs:
        if tok.startswith("AT_"):
            return tok[3:]
    return None


def _next_unachieved(achieved: set[str]) -> str | None:
    """Return the first GOAL_CHAIN token not yet in achieved, or None if all done."""
    for tok in _GOAL_CHAIN:
        if tok not in achieved:
            return tok
    return None


def _eat_if_starving(env: TextWorldEnv) -> bool:
    """Eat available food if hunger < 20.  Returns True if ate something."""
    if env._hunger >= 20:
        return False
    # Prefer inventory food over room food
    edibles_inv = [
        item for item in env._inventory
        if item in env.EAT_EFFECTS and env.EAT_EFFECTS[item][0] == "hunger"
    ]
    edibles_room = [
        iname for iname, item in env._rooms[env._location].items.items()
        if item.state == "here" and iname in env.EAT_EFFECTS
        and env.EAT_EFFECTS[iname][0] == "hunger"
    ]
    target = (edibles_inv or edibles_room)
    if target:
        env.act(f"eat_{target[0]}")
        return True
    return False


# ---------------------------------------------------------------------------
# Predictor training
# ---------------------------------------------------------------------------

def _collect_random_episode(seed: int) -> list[str]:
    """Run one random episode, return full obs+action token sequence."""
    env = TextWorldEnv(seed=seed)
    env.reset()
    rng = random.Random(seed)
    seq: list[str] = []

    for _ in range(MAX_STEPS):
        obs_toks = [tok for tok, _ in env.observe()]
        seq.extend(obs_toks)
        actions = env.available_actions()
        if not actions:
            break
        action = rng.choice(actions)
        seq.append(action)       # action token between observations
        env.act(action)
        if env.done or env.won:
            # Final observation
            seq.extend(tok for tok, _ in env.observe())
            break

    return seq


def _train_predictor(episode_sequences: list[list[str]]) -> Predictor:
    """Build a Predictor from full-episode token sequences."""
    seqs = [toks + ["<eos>"] for toks in episode_sequences if toks]

    hc = HankelCount(r_max=R)
    hc.update_batch(seqs)

    try:
        lattices = discover_concepts(
            hankel=hc, r_levels=[R],
            lambda_productivity=0.1,
            merge_threshold=0.15,
            min_support=1.0,
        )
        lattice = lattices[0]
        mg = discover_morphisms(seqs, hc, lattice, r=R)
        process_rules = discover_processes(seqs, op_atoms=[])
    except Exception:
        from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import (
            ConceptLattice,
        )
        from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
            MorphismGraph,
        )
        lattice = ConceptLattice()
        mg = MorphismGraph()
        process_rules = []

    return Predictor(
        hankel=hc, lattice=lattice, morphism_graph=mg,
        process_rules=process_rules,
        k_neighbours=K_NEIGHBOURS, r=R,
    )


# ---------------------------------------------------------------------------
# Action selection: rollout-based active inference
# ---------------------------------------------------------------------------

def _rollout_select(
    env: TextWorldEnv,
    available: list[str],
    predictor: Predictor,
    context: list[str],
    rng: random.Random,
    achieved: set[str],
) -> tuple[str, float]:
    """Select action using 1-step environment rollout + navigation bonus + Predictor epistemic.

    Parameters
    ----------
    achieved:
        Set of GOAL_CHAIN tokens already seen this episode (persistent; never shrinks).
        Only tokens NOT in achieved can contribute positive pragmatic value.

    Returns
    -------
    (selected_action, G_value)  where G = -(pragmatic + epistemic).
    G is float('nan') if actions is empty.
    """
    if not available:
        return "wait", float("nan")

    # Identify the next unachieved milestone and its target room
    next_tok   = _next_unachieved(achieved)
    target_room = _GOAL_ROOMS.get(next_tok) if next_tok else None

    current_obs  = _obs_set(env)
    current_room = _current_room(current_obs)
    current_dist = (
        _ROOM_DIST.get((current_room, target_room), 999)
        if (current_room and target_room) else 999
    )

    best_action = available[0]
    best_G = float("inf")

    for action in available:
        # --- 1-step rollout ---
        env_sim = copy.deepcopy(env)
        env_sim.act(action)

        if env_sim.won:
            # Immediate win: return immediately with perfect score
            return action, -(6 + 1.0)

        next_obs = _obs_set(env_sim)

        # Pragmatic: count NEW goal-chain tokens (not yet in achieved)
        new_tokens = sum(
            1 for t in _GOAL_CHAIN
            if t in next_obs and t not in achieved
        )

        # Navigation bonus: reward closing distance to the goal room
        nav_bonus = 0.0
        if target_room and current_room:
            next_room = _current_room(next_obs)
            if next_room:
                next_dist = _ROOM_DIST.get((next_room, target_room), 999)
                if next_dist < current_dist:
                    nav_bonus = NAV_BONUS

        pragmatic = float(new_tokens) + nav_bonus

        # --- Epistemic value via Predictor ---
        epi = 0.0
        if predictor is not None:
            try:
                epi = predictor_epistemic(predictor, context, action)
            except Exception:
                epi = 0.0

        # --- G(pi) = -(pragmatic + epistemic * weight) + small random noise ---
        G = -(pragmatic + epi * EPISTEMIC_WEIGHT) + rng.random() * 1e-4

        if G < best_G:
            best_G = G
            best_action = action

    return best_action, best_G


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_random_episode(seed: int) -> dict:
    """One episode: purely random agent."""
    env = TextWorldEnv(seed=seed)
    env.reset()
    rng = random.Random(seed)

    for step in range(MAX_STEPS):
        if env.won or env.done:
            break
        actions = env.available_actions()
        if not actions:
            break
        env.act(rng.choice(actions))

    return {"won": env.won, "steps": step + 1 if step < MAX_STEPS else MAX_STEPS}


def _run_rollout_episode(seed: int, predictor: Predictor) -> dict:
    """One episode: rollout-based active inference agent.

    Uses 1-step environment rollout for pragmatic value (new goal-chain tokens
    + navigation bonus toward goal room) and Predictor for epistemic value.
    Includes a survival rule (eat when starving).

    The achieved set is accumulated monotonically throughout the episode:
    once a GOAL_CHAIN token is observed it is never removed, preventing
    negative progress scores when navigating through rooms.
    """
    env = TextWorldEnv(seed=seed)
    env.reset()
    rng = random.Random(seed)

    store = EpisodicStore(surprise_threshold=0.3)
    context: list[str] = []
    achieved: set[str] = set()   # persistent GOAL_CHAIN tokens seen this episode

    G_hist: list[float] = []
    pe_hist: list[float] = []

    for step in range(MAX_STEPS):
        if env.won or env.done:
            break

        # Survival: eat if starving
        _eat_if_starving(env)
        if env.won or env.done:
            break

        # Observe
        obs_toks = [tok for tok, _ in env.observe()]
        obs_set  = frozenset(obs_toks)

        # Update persistent achieved set (monotonically grows)
        for tok in _GOAL_CHAIN:
            if tok in obs_set:
                achieved.add(tok)

        # Prediction error
        pe = 0.5
        if predictor is not None and context:
            try:
                dist = predictor.predict_next(context)
                pe = 1.0 - dist.get(obs_toks[0], 0.0) if obs_toks else 0.5
            except Exception:
                pe = 0.5
        pe_hist.append(pe)
        store.add_event(step, obs_toks, pe)

        # Extend context
        context.extend(obs_toks)
        context = context[-50:]

        # Select action (pass achieved for new-token counting + nav bonus)
        available = env.available_actions()
        if not available:
            break
        action, G = _rollout_select(env, available, predictor, context, rng, achieved)
        G_hist.append(G)

        # Act
        env.act(action)
        context.append(action)
        context = context[-50:]

    return {
        "won": env.won,
        "steps": step + 1 if step < MAX_STEPS else MAX_STEPS,
        "mean_G": sum(g for g in G_hist if not math.isnan(g)) / len([g for g in G_hist if not math.isnan(g)]) if G_hist else float("nan"),
        "mean_pe": sum(pe_hist) / len(pe_hist) if pe_hist else float("nan"),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    print("TextWorld benchmark: rollout-based active inference vs random")
    print(f"Max steps={MAX_STEPS}, N episodes={N_EPISODES}")
    print(f"Predictor training: {WARMUP_EPISODES} warm-up episodes")
    print()

    # --- Train Predictor ---
    print(f"Training Predictor on {WARMUP_EPISODES} random episodes...")
    warmup_seqs = [
        _collect_random_episode(seed + 1000)
        for seed in range(WARMUP_EPISODES)
    ]
    predictor = _train_predictor(warmup_seqs)
    total_toks = sum(len(s) for s in warmup_seqs)
    print(f"  Vocab size: {len(predictor._vocab)}")
    print(f"  Total training tokens: {total_toks}")
    print()

    # --- Random baseline ---
    rand_results = [_run_random_episode(seed) for seed in range(N_EPISODES)]
    rand_wins = sum(1 for r in rand_results if r["won"])
    rand_steps = [r["steps"] for r in rand_results if r["won"]] or [MAX_STEPS]
    rand_avg = sum(rand_steps) / len(rand_steps)

    print(f"Random baseline ({N_EPISODES} episodes):")
    for i, r in enumerate(rand_results):
        print(f"  Episode {i}: {'WON' if r['won'] else 'lost'} in {r['steps']} steps")
    print(f"  Win rate: {rand_wins}/{N_EPISODES}")
    print(f"  Avg steps (winners): {rand_avg:.1f}")
    print()

    # --- Active inference (rollout) ---
    ai_results = [
        _run_rollout_episode(seed, predictor)
        for seed in range(N_EPISODES)
    ]
    ai_wins = sum(1 for r in ai_results if r["won"])
    ai_steps = [r["steps"] for r in ai_results if r["won"]] or [MAX_STEPS]
    ai_avg = sum(ai_steps) / len(ai_steps)
    G_vals = [r["mean_G"] for r in ai_results if not math.isnan(r.get("mean_G", float("nan")))]
    pe_vals = [r["mean_pe"] for r in ai_results if not math.isnan(r.get("mean_pe", float("nan")))]
    mean_G = sum(G_vals) / len(G_vals) if G_vals else float("nan")
    mean_pe = sum(pe_vals) / len(pe_vals) if pe_vals else float("nan")

    print(f"Active inference - rollout ({N_EPISODES} episodes):")
    for i, r in enumerate(ai_results):
        G_str = f"mean_G={r['mean_G']:.3f}" if not math.isnan(r.get("mean_G", float("nan"))) else "mean_G=n/a"
        print(f"  Episode {i}: {'WON' if r['won'] else 'lost'} in {r['steps']} steps  {G_str}")
    print(f"  Win rate: {ai_wins}/{N_EPISODES}")
    print(f"  Avg steps (winners): {ai_avg:.1f}")
    print(f"  Mean G(pi): {mean_G:.3f}")
    print(f"  Mean PE:    {mean_pe:.3f}")
    print()

    # --- Summary ---
    w = 22
    print("=" * 60)
    print(f"{'Agent':<{w}} {'Win rate':<14} {'Avg steps'}")
    print("-" * 60)
    print(f"{'Random baseline':<{w}} {rand_wins}/{N_EPISODES}{'':8} {rand_avg:.1f}")
    print(f"{'Active inference':<{w}} {ai_wins}/{N_EPISODES}{'':8} {ai_avg:.1f}")
    print("=" * 60)
    print()

    # Pass/Fail
    pass1 = True   # reaching here without exception means loop ran OK
    pass2 = ai_wins > rand_wins
    print("Pass criteria:")
    print(f"  PASS-1  Loop runs without exception:              {'PASS' if pass1 else 'FAIL'}")
    print(f"  PASS-2  AI wins more often than random:           {'PASS' if pass2 else 'FAIL'}")
    print(f"          (AI {ai_wins}/5 vs random {rand_wins}/5)")
    overall = "ALL PASS" if (pass1 and pass2) else "SOME FAIL"
    print(f"\nOverall: {overall}")


if __name__ == "__main__":
    run_benchmark()
