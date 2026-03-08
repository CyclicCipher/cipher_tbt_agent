"""MinecraftModality — future dxcam / pynput / RCON implementation.

STATUS: Stub.  MineDojo has been removed.  The planned implementation uses:

  Perception:  dxcam DirectX screen capture  → VisualCortex Gabor features
  Drives:      mcrcon TCP queries (health / food / XP) polled at 2 Hz
  Events:      server log tail for advancements + deaths
  Actions:     pynput keyboard + mouse events to the focused window

Server setup (one-time, vanilla Minecraft 1.20.1 — no mods required):
  1. Download minecraft_server.1.20.1.jar from launcher.mojang.com
  2. mkdir server && cd server
  3. echo 'eula=true' > eula.txt
  4. Add to server.properties:
       enable-rcon=true
       rcon.password=agent
       rcon.port=25575
       gamemode=survival
  5. java -Xmx2G -jar minecraft_server.1.20.1.jar nogui

Dependencies (when implementing Phase K):
  pip install dxcam mcrcon pynput pygetwindow

CTKG primitives (once implemented):
  Observation: mc_frame, mc_rgb, mc_health, mc_food, mc_xp
  Temporal:    mc_frame_diff, mc_frame_std
  Action:      mc_press_key, mc_release_key, mc_mouse_move, mc_click, mc_scroll
  Events:      mc_poll_events   → list of {type, name} dicts

Metabolic drives (once implemented):
  U_pain()    → float in [0,1]   (1.0 - health/20)
  U_hunger()  → float in [0,1]   (1.0 - food/20)
  U_sleep()   → currently unused for RCON-based implementation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modalities.base import Modality

# VisualCortex is kept — it will be used when Phase K is implemented.
try:
    from modalities.vision_cortex import VisualCortex
    _HAS_VC = True
except ImportError:
    _HAS_VC = False


# ---------------------------------------------------------------------------
# MinecraftModality stub
# ---------------------------------------------------------------------------

class MinecraftModality(Modality):
    """Stub MinecraftModality.  Returns null observations and no-ops actions.

    Replace this class body with the Phase K dxcam/mcrcon/pynput implementation.
    The interface (drive API, primitives dict, Modality base class) is stable.
    """

    # Null frame returned when no screen capture is available.
    _NULL_FRAME = np.zeros((160, 256, 3), dtype=np.uint8)

    def __init__(self, **kwargs) -> None:
        """Accept and ignore any keyword args for backwards compatibility."""
        self._health:  float = 20.0
        self._food:    float = 20.0
        self._xp:      float = 0.0
        self._frame:   np.ndarray = self._NULL_FRAME
        self._events:  List[Dict] = []

        self._vc = VisualCortex() if _HAS_VC else None

        self._primitives: Dict[str, Any] = {
            # --- observation (all return stub values until Phase K) ---
            'mc_frame':       self._get_frame,
            'mc_health':      lambda: self._health,
            'mc_food':        lambda: self._food,
            'mc_xp':          lambda: self._xp,
            'mc_frame_diff':  lambda: self._NULL_FRAME,
            'mc_frame_std':   lambda: 0.0,
            # --- action (no-ops until Phase K) ---
            'mc_press_key':   lambda key: None,
            'mc_release_key': lambda key: None,
            'mc_mouse_move':  lambda dx, dy: None,
            'mc_click':       lambda btn: None,
            'mc_scroll':      lambda dy: None,
            # --- events ---
            'mc_poll_events': lambda: [],
        }

        # Add VisualCortex primitives if available (used in Phase J concepts).
        if self._vc is not None:
            self._primitives.update({
                'vc_encode':            lambda f: self._vc.encode(f),
                'vc_saliency':          lambda f: self._vc.saliency(f, f),
                'vc_next_gaze':         lambda s: self._vc.next_gaze(s),
                'vc_foveal_mean':       lambda feat: self._vc.foveal_mean(feat),
                'vc_foveal_std':        lambda feat: self._vc.foveal_std(feat),
                'vc_brightness_gradient': lambda f: self._vc.brightness_gradient(f),
                'vc_looming':           lambda f: self._vc.looming(f, f),
                'vc_sky_fraction':      lambda f: self._vc.sky_fraction(f),
            })

    # ------------------------------------------------------------------
    # Modality interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'minecraft'

    @property
    def primitives(self) -> Dict[str, Any]:
        return self._primitives

    # ------------------------------------------------------------------
    # Metabolic drives (stub — always satisfied)
    # ------------------------------------------------------------------

    def U_pain(self) -> float:
        """Pain urgency: 0 = no pain, 1 = critical.  Stub returns 0."""
        return max(0.0, 1.0 - self._health / 20.0)

    def U_hunger(self) -> float:
        """Hunger urgency: 0 = full, 1 = starving.  Stub returns 0."""
        return max(0.0, 1.0 - self._food / 20.0)

    def U_sleep(self) -> float:
        """Sleep urgency.  Stub returns 0 (no day/night tracking yet)."""
        return 0.0

    def current_priority(self, engine) -> Tuple[str, float, str]:
        """Return (mode, urgency, target) for the planning loop."""
        if self.U_pain() > 0.5:
            return 'SURVIVE', self.U_pain(), 'health'
        if self.U_hunger() > 0.7:
            return 'EAT', self.U_hunger(), 'food'
        return 'WANDER', 0.1, 'none'

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_frame(self) -> np.ndarray:
        return self._frame

    # ------------------------------------------------------------------
    # Stubs for methods called by the old agent_loop (back-compat)
    # ------------------------------------------------------------------

    def update(self, obs: Any) -> None:
        """No-op: called by old MineDojo loop.  Remove when Phase K lands."""
        pass

    def forward(self, n: int = 1) -> None:  pass
    def back(self, n: int = 1) -> None:     pass
    def left(self, n: int = 1) -> None:     pass
    def right(self, n: int = 1) -> None:    pass
    def jump(self) -> None:                 pass
    def attack(self) -> None:               pass
    def use(self) -> None:                  pass
    def turn(self, yaw: float = 0.0, pitch: float = 0.0) -> None: pass
    def select_slot(self, n: int = 0) -> None: pass
