"""tits_modality.py — Phase S.2a: Trials in Tainted Space screen adapter.

Subclasses GenericModality.  All text reading is done by the inherited
ScreenReader (GlyphReader + TextScanner); this class adds only the three
non-text signals that require TiTS-specific knowledge:

  1. Combat detection  — recognise combat layout from button labels
  2. Game-over detection — pattern-match known death/defeat text
  3. Score derivation — quest + exploration progress heuristic

All domain knowledge (stat names, keywords, button patterns) is loaded from
``tits.adapter.yaml``.  The YAML drives GenericModality; this file is ~100
lines of overrides only.

Design principle: THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK.
Knowledge in callables; engine remains domain-agnostic.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from modalities.generic_modality import GenericModality
from modalities.screen_reader import ScreenReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combat keywords — any 2+ of these in admissible buttons = combat layout
# ---------------------------------------------------------------------------
_COMBAT_KEYWORDS = (
    "attack", "tease", "flee", "item", "defend", "special",
    "struggle", "submit", "resist", "strip", "wait",
)

# ---------------------------------------------------------------------------
# Game-over / defeat text patterns (lower-case match)
# ---------------------------------------------------------------------------
_DEFEAT_PATTERNS = (
    "game over",
    "you have been defeated",
    "you have died",
    "you lose",
    "your journey ends here",
    "knocked unconscious",
    "you are dead",
    "fall unconscious",
)

# ---------------------------------------------------------------------------
# Known neutral / menu buttons (not combat, not navigation)
# ---------------------------------------------------------------------------
_MENU_BUTTONS = {
    "new game", "load game", "save game", "settings", "credits",
    "quit", "exit", "close",
}


class TiTSModality(GenericModality):
    """TiTS-specific ScreenModality.

    Usage
    -----
    ::

        import yaml
        from modalities.screen_reader import ScreenReader
        from modalities.tits_modality import TiTSModality

        cfg    = yaml.safe_load(open('tits.adapter.yaml'))
        reader = ScreenReader.load(cfg.get('glyph_reader', 'glyph_reader.pkl'))
        mod    = TiTSModality(cfg, reader)
        obs    = mod.connect()        # capture first frame; run new_game if configured

    The obs dict contains (on top of the GenericModality keys):
        in_combat   bool   — True during combat scene
        done        bool   — True on game-over / defeat
        score       float  — heuristic quest/exploration progress score
        hp          float  — current HP (if visible)
        hp_max      float  — max HP (if visible)
        lust        float  — current lust (if visible)
        lust_max    float  — max lust (if visible)
        level       float  — character level (if visible)
        credits     float  — current credits (if visible)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        reader: ScreenReader,
        **kwargs,
    ) -> None:
        super().__init__(config, reader, **kwargs)
        self._quest_score = 0.0        # accumulated exploration score
        self._locations_visited: set = set()
        self._in_combat = False

    # ------------------------------------------------------------------
    # connect() — launch new game if configured
    # ------------------------------------------------------------------

    def connect(self) -> dict:
        """Connect to TiTS and optionally run the new-game sequence."""
        if hasattr(super(), "connect"):
            frame = super().connect() if callable(getattr(super(), "connect", None)) else None
        # If the parent ScreenModality.connect() returned an obs dict, we handle it below.
        # Otherwise capture a frame ourselves.
        if hasattr(self, "_capture"):
            frame = self._capture()
        else:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        obs = self.build_obs(frame)

        if self._config.get("new_game"):
            logger.info("TiTSModality: running new-game sequence …")
            self._run_new_game_sequence(delay_s=self._config.get("new_game_delay", 1.5))
            if hasattr(self, "_capture"):
                frame = self._capture()
                obs = self.build_obs(frame)

        return obs

    # ------------------------------------------------------------------
    # build_obs() — add TiTS-specific signals on top of GenericModality
    # ------------------------------------------------------------------

    def build_obs(self, frame: np.ndarray, info: dict = {}) -> dict:
        """Extend GenericModality obs with combat / game-over / score signals."""
        obs = super().build_obs(frame, info)

        # 1. Combat detection from button labels
        obs["in_combat"] = self._detect_combat(obs.get("admissible", []))
        self._in_combat = obs["in_combat"]

        # 2. Game-over detection from narrative
        if self._detect_game_over(obs.get("text", "")):
            obs["done"] = True

        # 3. Score heuristic (accumulates; never decreases)
        obs["score"] = self._derive_tits_score(obs)

        # 4. Location tracking for exploration reward
        location = self._extract_location(obs.get("text", ""))
        if location and location not in self._locations_visited:
            self._locations_visited.add(location)
            self._quest_score += 1.0
            logger.debug(f"New location discovered: {location!r} (total {len(self._locations_visited)})")

        return obs

    # ------------------------------------------------------------------
    # Non-text signal detectors
    # ------------------------------------------------------------------

    def _detect_combat(self, buttons: List[str]) -> bool:
        """Return True if the current button set looks like a combat layout.

        Combat is inferred when ≥ 2 combat keywords appear in the admissible
        button list and none of the buttons are pure menu options.
        """
        lower = [b.lower() for b in buttons]
        hits = sum(
            any(kw in btn for kw in _COMBAT_KEYWORDS)
            for btn in lower
        )
        menu_count = sum(b in _MENU_BUTTONS for b in lower)
        return hits >= 2 and menu_count == 0

    def _detect_game_over(self, text: str) -> bool:
        """Return True if the narrative contains a known defeat/game-over pattern."""
        text_lower = text.lower()
        return any(pat in text_lower for pat in _DEFEAT_PATTERNS)

    def _derive_tits_score(self, obs: dict) -> float:
        """Heuristic score = quest_score + level_bonus + credit_bonus.

        All components are normalised so the total is on a human-interpretable
        scale without requiring hard-coded targets.
        """
        score = self._quest_score

        level = float(obs.get("level", 0))
        if level > 0:
            score += level * 5.0            # +5 per level gained

        credits = float(obs.get("credits", 0))
        score += credits / 1000.0          # +0.001 per credit

        return score

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract a room/area name from the narrative text (best-effort).

        TiTS typically shows "You are in <location>" or the location appears
        as a capitalised phrase near the start of the narrative.  This is a
        heuristic — incorrect extractions simply mean a location is recorded
        under a slightly wrong name, which does not affect correctness.
        """
        import re
        # Pattern: "You are in the ..." or "You are at ..."
        m = re.search(r"you are (?:in|at|on)\s+(?:the\s+)?([A-Z][a-z][\w\s]{2,30})", text)
        if m:
            return m.group(1).strip()
        # Fallback: any capitalised multi-word phrase (first 100 chars of narrative)
        snippet = text[:100]
        m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", snippet)
        if m:
            return m.group(1)
        return None

    # ------------------------------------------------------------------
    # Optional helpers for external code
    # ------------------------------------------------------------------

    @property
    def locations_visited(self) -> set:
        """Set of unique location strings discovered this episode."""
        return set(self._locations_visited)

    def reset_episode(self) -> None:
        """Reset exploration tracking for a new episode."""
        self._quest_score = 0.0
        self._locations_visited.clear()
        self._in_combat = False
        logger.debug("TiTSModality: episode reset")

    def __repr__(self) -> str:
        return (
            f"TiTSModality("
            f"locs={len(self._locations_visited)}, "
            f"combat={self._in_combat}, "
            f"score={self._quest_score:.1f})"
        )
