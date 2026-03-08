"""generic_modality.py -- Phase S.1a: Config-driven ScreenModality using ScreenReader.

Covers any text UI game with zero game-specific code. Subclass and override
build_obs() only for non-text visual signals (combat detection, scene type, etc.).

Design principle: THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK.
THE MODEL MUST BE GENERAL.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from modalities.screen_modality import ScreenModality  # type: ignore[import]
    _HAS_SCREEN_MODALITY = True
except ImportError:
    _HAS_SCREEN_MODALITY = False
    ScreenModality = object  # type: ignore[misc,assignment]

from modalities.screen_reader import ScreenReader
from modalities.text_scanner import FrameReading


class GenericModality(ScreenModality if _HAS_SCREEN_MODALITY else object):  # type: ignore[misc]
    """Config-driven modality using the learned ScreenReader.

    Parameters
    ----------
    config      Dict loaded from a .adapter.yaml file (or constructed manually).
    reader      Pre-trained ScreenReader instance.
    """

    def __init__(self, config: Dict[str, Any], reader: ScreenReader, **kwargs) -> None:
        self._config   = config
        self._reader   = reader
        self._last_fr: Optional[FrameReading] = None
        self._step_count = 0
        self._score = 0.0
        self._done  = False

        self._stat_patterns: Dict[str, re.Pattern] = {}
        for name, pattern in config.get("stats", {}).items():
            try:
                self._stat_patterns[name] = re.compile(pattern, re.IGNORECASE)
            except re.error:
                pass

        if _HAS_SCREEN_MODALITY:
            win_title = config.get("window_title", "")
            super().__init__(
                window_title=win_title,
                dry_run=kwargs.pop("dry_run", False),
                verbose=kwargs.pop("verbose", False),
                **kwargs,
            )

    def build_obs(self, frame: np.ndarray, info: dict = {}) -> dict:
        """Read frame -> state dict via ScreenReader.

        Subclasses override this to add non-text signals. Always call
        super().build_obs(frame, info) first.
        """
        reading = self._reader.read(frame)
        self._last_fr = reading
        stats = self._parse_stats(reading.stats_text + " " + reading.narrative)
        return {
            "screen":     frame,
            "text":       reading.narrative,
            "admissible": reading.buttons,
            "score":      self._derive_score(stats),
            "done":       self._is_done(stats, reading.narrative),
            **stats,
        }

    def _parse_stats(self, text: str) -> Dict[str, Any]:
        """Apply stat regex patterns from config to extracted text."""
        stats: Dict[str, Any] = {}
        for name, pattern in self._stat_patterns.items():
            m = pattern.search(text)
            if m:
                groups = m.groups()
                if len(groups) == 1:
                    try:
                        stats[name] = float(groups[0].replace(",", ""))
                    except ValueError:
                        stats[name] = groups[0]
                elif len(groups) == 2:
                    try:
                        stats[name]          = float(groups[0].replace(",", ""))
                        stats[name + "_max"] = float(groups[1].replace(",", ""))
                    except ValueError:
                        stats[name] = groups[0]
        return stats

    def _derive_score(self, stats: Dict[str, Any]) -> float:
        """Default: return the raw score stat if present, else 0.0."""
        return float(stats.get("score", self._score))

    def _is_done(self, stats: Dict[str, Any], narrative: str) -> bool:
        """Default: False (subclass overrides for game-over detection)."""
        return bool(stats.get("done", self._done))

    def send_text(self, text: str) -> None:
        """Click the button matching text, or type as keyboard input."""
        if self._last_fr is None:
            return
        target = text.strip().lower()
        for result in self._last_fr.all_reads:
            if result.region is None:
                continue
            if result.region.region_type != "button":
                continue
            label = result.text.strip().lower()
            if label == target or target in label or label in target:
                cx, cy = result.region.centroid
                self._click_pixel(cx, cy)
                time.sleep(0.1)
                return
        self._keyboard_type(text)

    def _click_pixel(self, x: int, y: int) -> None:
        """Click at pixel position (x, y) via pynput or stub."""
        try:
            from pynput.mouse import Button, Controller  # type: ignore[import]
            mouse = Controller()
            mouse.position = (x, y)
            time.sleep(0.05)
            mouse.click(Button.left)
        except ImportError:
            pass

    def _keyboard_type(self, text: str) -> None:
        """Type text via pynput keyboard controller."""
        try:
            from pynput.keyboard import Controller  # type: ignore[import]
            kb = Controller()
            kb.type(text)
        except ImportError:
            pass

    def _run_new_game_sequence(self, delay_s: float = 1.0) -> None:
        """Execute the new_game.steps list from config."""
        steps = self._config.get("new_game", {}).get("steps", [])
        for step in steps:
            if "click" in step:
                frame = self._capture() if hasattr(self, "_capture") else np.zeros((100, 100, 3), dtype=np.uint8)
                reading = self._reader.read(frame)
                label = step["click"]
                clicked = False
                for result in reading.all_reads:
                    if result.region and label.lower() in result.text.lower():
                        cx, cy = result.region.centroid
                        self._click_pixel(cx, cy)
                        clicked = True
                        break
                if not clicked:
                    self._keyboard_type(label + "\n")
            elif "type" in step:
                self._keyboard_type(step["type"])
            time.sleep(delay_s)

    def __repr__(self) -> str:
        return (
            f"GenericModality("
            f"game={self._config.get('name', '?')},"
            f" reader={self._reader.summary().split(chr(10))[0]})"
        )
