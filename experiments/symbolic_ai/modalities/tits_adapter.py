"""tits_adapter.py — Phase S.2a: TiTS world model + FEP drives + goal factory.

Contains:
  TiTSWorldModel  — maintains a running summary of TiTS game state
  build_state()   — converts a raw obs dict → normalised state dict for planning
  make_tits_drives() — HP, Lust, and Explore FEP drives
  make_tits_goals()  — FEPGoal list for the DecisionEngine

All lambdas read from ``state`` dynamically — no static closures.  The
planning engine itself remains domain-agnostic (planning.py).

Design principle: THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TiTSWorldModel — tracks game state across steps
# ---------------------------------------------------------------------------

@dataclass
class TiTSWorldModel:
    """Running world model for TiTS.

    Updated each step by calling ``world.update(obs)``.  Exposes normalised
    fields used by Drive.measure lambdas and Goal.condition callables.

    All numeric fields are in [0, 1] unless noted otherwise.
    """
    # Character vitals (normalised)
    hp_frac:   float = 1.0    # hp / hp_max
    lust_frac: float = 0.0    # lust / lust_max  (high = bad)
    level:     int   = 1      # character level (1+)
    credits:   float = 0.0    # raw credit count

    # Exploration state
    locations_visited: int = 0   # unique locations seen this episode
    steps_this_episode: int = 0  # total steps taken
    narrative_history: List[str] = field(default_factory=list, repr=False)

    # Combat state
    in_combat: bool = False
    combat_steps: int = 0    # consecutive steps in combat

    # Episode state
    done: bool = False
    score: float = 0.0

    def update(self, obs: Dict[str, Any]) -> None:
        """Update world model from a raw obs dict (output of TiTSModality.build_obs)."""
        hp     = float(obs.get("hp",     1.0))
        hp_max = float(obs.get("hp_max", max(1.0, hp)))
        lust     = float(obs.get("lust",     0.0))
        lust_max = float(obs.get("lust_max", max(1.0, lust)))

        self.hp_frac   = hp   / hp_max   if hp_max > 0 else 1.0
        self.lust_frac = lust / lust_max if lust_max > 0 else 0.0
        self.level     = int(float(obs.get("level",   self.level)))
        self.credits   = float(obs.get("credits", self.credits))

        self.in_combat = bool(obs.get("in_combat", False))
        if self.in_combat:
            self.combat_steps += 1
        else:
            self.combat_steps = 0

        self.done  = bool(obs.get("done",  False))
        self.score = float(obs.get("score", self.score))
        self.steps_this_episode += 1

        narrative = obs.get("text", "")
        if narrative and (not self.narrative_history or narrative != self.narrative_history[-1]):
            self.narrative_history.append(narrative)
            if len(self.narrative_history) > 200:
                self.narrative_history = self.narrative_history[-100:]

    def reset(self) -> None:
        """Reset for a new episode."""
        self.hp_frac   = 1.0
        self.lust_frac = 0.0
        self.in_combat = False
        self.combat_steps = 0
        self.done  = False
        self.score = 0.0
        self.steps_this_episode = 0
        self.locations_visited = 0
        self.narrative_history.clear()

    def to_state(self) -> Dict[str, Any]:
        """Return a flat state dict suitable for Drive.measure and Goal.condition."""
        return {
            "hp_frac":      self.hp_frac,
            "lust_frac":    self.lust_frac,
            "level":        self.level,
            "credits":      self.credits,
            "in_combat":    self.in_combat,
            "combat_steps": self.combat_steps,
            "done":         self.done,
            "score":        self.score,
            "locations":    self.locations_visited,
            "steps":        self.steps_this_episode,
        }


# ---------------------------------------------------------------------------
# build_state() — obs dict → normalised state dict
# ---------------------------------------------------------------------------

def build_state(obs: Dict[str, Any], world: TiTSWorldModel) -> Dict[str, Any]:
    """Update world model from obs and return normalised state dict.

    This is the bridge between raw game observations (pixels → text → stats)
    and the planning layer (Drive.measure / Goal.condition callables).

    Parameters
    ----------
    obs   : dict returned by TiTSModality.build_obs()
    world : TiTSWorldModel instance (mutated in place)

    Returns
    -------
    state : dict with normalised keys for planning
    """
    world.update(obs)
    state = world.to_state()

    # Pass through text and admissible actions for goal condition checks
    state["text"]       = obs.get("text", "")
    state["admissible"] = obs.get("admissible", [])
    state["screen"]     = obs.get("screen", None)

    return state


# ---------------------------------------------------------------------------
# make_tits_drives() — homeostatic drives
# ---------------------------------------------------------------------------

def make_tits_drives() -> Dict[str, "Drive"]:
    """Return the three canonical TiTS homeostatic drives.

    Returns a dict keyed by drive name for easy reference in goal factories.

    Drive summary
    -------------
    hp_drive    : urgency=1.0; deficit when hp_frac < 1.0, critical when < 0.25
    lust_drive  : urgency=0.65; deficit when lust_frac > 0.30 (high lust is bad)
    explore_drive: urgency=0.35; always-on mild pressure to discover new areas
    """
    try:
        from planning import Drive  # type: ignore[import]
    except ImportError:
        from experiments.symbolic_ai.planning import Drive  # type: ignore[import]

    hp_drive = Drive(
        name     = "hp",
        measure  = lambda s: s.get("hp_frac", 1.0),
        setpoint = 1.0,
        urgency  = 1.0,
    )

    # Lust is inverted: high lust = bad. measure = 1 - lust_frac.
    # At lust_frac=0.0, measure=1.0, deficit=0 (satisfied).
    # At lust_frac=1.0, measure=0.0, deficit=0.65 (critical).
    # Starts firing meaningfully around lust_frac > 0.30.
    lust_drive = Drive(
        name     = "lust",
        measure  = lambda s: 1.0 - s.get("lust_frac", 0.0),
        setpoint = 0.70,   # deficit kicks in when lust_frac > 0.30
        urgency  = 0.65,
    )

    # Explore: always a small pressure. measure = tanh(locations/10).
    # At 0 locations: measure≈0, deficit=0.35.  At 10+ locations: measure≈1, deficit≈0.
    explore_drive = Drive(
        name     = "explore",
        measure  = lambda s: float(min(1.0, s.get("locations", 0) / 10.0)),
        setpoint = 1.0,
        urgency  = 0.35,
    )

    return {
        "hp":      hp_drive,
        "lust":    lust_drive,
        "explore": explore_drive,
    }


# ---------------------------------------------------------------------------
# make_tits_goals() — FEPGoal list for DecisionEngine
# ---------------------------------------------------------------------------

def make_tits_goals(world: TiTSWorldModel) -> List[Any]:
    """Build the goal list for a TiTS episode.

    Goals (highest priority first when drives are fully active)
    -----------------------------------------------------------
    1. survive      (safety)   — in combat + low HP → flee/defend/item
    2. manage_lust  (safety)   — lust critical → seek relief
    3. heal         (FEP/hp)   — HP < 50% → use item / rest
    4. combat_act   (FEP/hp+lust) — in combat → offensive action
    5. explore      (FEP/explore) — not in combat → advance narrative

    All achieve functions are intentionally simple random-from-admissible
    selectors filtered by keywords.  Better planning emerges from the beam
    search in AIFEngine rather than hard-coded heuristics here.
    """
    try:
        from planning import Drive, FEPGoal  # type: ignore[import]
    except ImportError:
        from experiments.symbolic_ai.planning import Drive, FEPGoal  # type: ignore[import]

    drives = make_tits_drives()
    hp_drive      = drives["hp"]
    lust_drive    = drives["lust"]
    explore_drive = drives["explore"]

    # ------------------------------------------------------------------
    # Achieve helpers — select from admissible actions by keyword
    # ------------------------------------------------------------------

    def _pick(state: dict, keywords: tuple, exclude: tuple = ()) -> Optional[Tuple[str, str]]:
        """Pick a random admissible action matching one of ``keywords``."""
        admissible = state.get("admissible", [])
        candidates = [
            a for a in admissible
            if any(kw in a.lower() for kw in keywords)
            and not any(ex in a.lower() for ex in exclude)
        ]
        if candidates:
            choice = random.choice(candidates)
            return choice, f"keyword-match: {keywords[0]}"
        return None

    def _any_admissible(state: dict) -> Optional[Tuple[str, str]]:
        """Fallback: pick any admissible action at random."""
        admissible = state.get("admissible", [])
        if admissible:
            return random.choice(admissible), "random-fallback"
        return None

    # ------------------------------------------------------------------
    # 1. Survive (safety goal) — flee / defend when HP critical in combat
    # ------------------------------------------------------------------
    def survive_condition(s: dict) -> bool:
        return bool(s.get("in_combat")) and s.get("hp_frac", 1.0) < 0.25

    def survive_achieve(s: dict, **kw) -> Optional[Tuple[str, str]]:
        # Prefer flee; fall back to defend, then item
        return (
            _pick(s, ("flee",))
            or _pick(s, ("defend", "guard"))
            or _pick(s, ("item", "heal", "potion"))
            or _any_admissible(s)
        )

    survive_goal = FEPGoal(
        name          = "survive",
        drives        = [hp_drive],
        condition     = survive_condition,
        achieve       = survive_achieve,
        is_safety     = True,
    )

    # ------------------------------------------------------------------
    # 2. Manage lust (safety goal) — lust > 0.85 is game-threatening
    # ------------------------------------------------------------------
    def lust_condition(s: dict) -> bool:
        return s.get("lust_frac", 0.0) > 0.85

    def lust_achieve(s: dict, **kw) -> Optional[Tuple[str, str]]:
        return (
            _pick(s, ("resist", "struggle", "suppress"))
            or _pick(s, ("flee", "run"))
            or _any_admissible(s)
        )

    lust_goal = FEPGoal(
        name      = "manage_lust",
        drives    = [lust_drive],
        condition = lust_condition,
        achieve   = lust_achieve,
        is_safety = True,
    )

    # ------------------------------------------------------------------
    # 3. Heal — HP < 50% and not critically low (survive handles that)
    # ------------------------------------------------------------------
    def heal_condition(s: dict) -> bool:
        return s.get("hp_frac", 1.0) < 0.50 and not s.get("in_combat")

    def heal_achieve(s: dict, **kw) -> Optional[Tuple[str, str]]:
        return (
            _pick(s, ("heal", "rest", "sleep", "medic", "item", "potion"))
            or _any_admissible(s)
        )

    heal_goal = FEPGoal(
        name      = "heal",
        drives    = [hp_drive],
        condition = heal_condition,
        achieve   = heal_achieve,
    )

    # ------------------------------------------------------------------
    # 4. Combat act — in combat with acceptable HP
    # ------------------------------------------------------------------
    def combat_condition(s: dict) -> bool:
        return bool(s.get("in_combat")) and s.get("hp_frac", 1.0) >= 0.25

    def combat_achieve(s: dict, **kw) -> Optional[Tuple[str, str]]:
        # Prefer offensive actions: attack/special/tease
        return (
            _pick(s, ("attack", "special"), exclude=("flee",))
            or _pick(s, ("tease",), exclude=("flee",))
            or _any_admissible(s)
        )

    combat_goal = FEPGoal(
        name      = "combat_act",
        drives    = [hp_drive, lust_drive],
        condition = combat_condition,
        achieve   = combat_achieve,
    )

    # ------------------------------------------------------------------
    # 5. Explore — main quest / open-world navigation
    # ------------------------------------------------------------------
    def explore_condition(s: dict) -> bool:
        return not s.get("in_combat") and not s.get("done")

    def explore_achieve(s: dict, **kw) -> Optional[Tuple[str, str]]:
        # Prefer navigation and narrative advancement
        return (
            _pick(s, ("go ", "travel", "move", "enter", "leave", "exit"))
            or _pick(s, ("talk", "speak", "ask", "tell"))
            or _pick(s, ("take", "pick up", "grab"))
            or _any_admissible(s)
        )

    explore_goal = FEPGoal(
        name      = "explore",
        drives    = [explore_drive],
        condition = explore_condition,
        achieve   = explore_achieve,
    )

    return [survive_goal, lust_goal, heal_goal, combat_goal, explore_goal]
