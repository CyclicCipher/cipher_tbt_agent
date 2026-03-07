"""screen_modality.py — Generic screen-capture + keyboard + mouse modality.

Architecture
============
``ScreenModality`` is an abstract base class for any game that requires:
  - Screen capture  (reading pixel output from the game)
  - Keyboard input  (sending keystrokes to the game)
  - Mouse input     (sending pointer events to the game)

It wraps three optional OS-level backends, selected by capability:

    Screen capture: dxcam (fastest, Windows) → mss (cross-platform) →
                    PIL.ImageGrab (fallback) → zeros (stub)
    Keyboard/mouse: pynput → logging stub (no-op)

All backends produce the same interface: ``capture() → np.ndarray``,
``key_tap(key)``, ``key_hold(key, duration)``, ``mouse_move(x, y)``,
``mouse_click(x, y, button)``, ``mouse_scroll(x, y, dx, dy)``.

Adapter contract
================
Game adapters **subclass** ``ScreenModality`` and override:

    build_obs(frame, info={}) → dict
        Convert the raw screen frame (H, W, 3 uint8) + any auxiliary info
        (score, done, etc.) into the agent's state dict.  This is the ONLY
        game-specific method the episode loop calls.

    get_reward() → float
        Read the current game score/reward.  Default: 0.0.

    get_done() → bool
        Return True if the episode is over.  Default: False.

    get_events() → List[dict]
        Return events since last step: {'type': 'acquired'/'lost', ...}.
        Default: [].

    get_admissible() → List[Action]
        Optional: return currently valid actions.  Default: [] (engine
        generates actions freely from its own model).

    send_text(text: str) → None
        Optional: handle text-command actions (for hybrid text+screen games).
        Default: type the text on the keyboard.

Usage
=====
::

    class MyGameModality(ScreenModality):
        def build_obs(self, frame, info={}):
            return {
                'screen':    frame,
                'location':  self._parse_location(frame),
                'score':     self.get_reward(),
                'done':      self.get_done(),
                'admissible': self.get_admissible(),
            }

    mod = MyGameModality(window_title='My Game', capture_fps=15)
    obs = mod.connect()        # grab first frame
    engine.decide(obs, rng)    # engine returns Action
    mod.dispatch(action)       # execute the action
    mod.step_time()            # wait for frame to update

Design principle
================
**THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST BE GENERAL.**

Zero game-specific logic in this file.  All game knowledge belongs in the
subclass's ``build_obs()``, ``get_reward()``, and ``get_done()`` methods.

See AIF_ROADMAP.md Phase R7 for the full adapter specification.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from action import (
    Action, KeyPress, KeyHold, MouseMove, MouseClick, MouseScroll,
    flatten, is_text_action,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------------

# Screen capture — try in order of quality/speed.
try:
    import dxcam                     # type: ignore
    _HAS_DXCAM = True
except ImportError:
    _HAS_DXCAM = False

try:
    import mss                       # type: ignore
    _HAS_MSS = True
except ImportError:
    _HAS_MSS = False

try:
    from PIL import ImageGrab        # type: ignore
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# Keyboard and mouse — pynput.
try:
    import pynput.keyboard           # type: ignore
    import pynput.mouse              # type: ignore
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False


# ---------------------------------------------------------------------------
# Internal helpers: key string → pynput Key
# ---------------------------------------------------------------------------

def _pynput_key(key_str: str):
    """Convert a key string to a pynput Key or character.

    Handles special keys (``'space'``, ``'enter'``, ``'f1'``, etc.) and
    modifier combos (``'ctrl+c'``, ``'ctrl+shift+z'``).

    Returns a list of (modifier_keys, main_key) suitable for
    ``pynput.keyboard.Controller.pressed()`` context or direct press.
    """
    if not _HAS_PYNPUT:
        return None, []

    kb = pynput.keyboard

    _SPECIAL: Dict[str, Any] = {
        'space':     kb.Key.space,
        'enter':     kb.Key.enter,
        'escape':    kb.Key.esc,
        'tab':       kb.Key.tab,
        'backspace': kb.Key.backspace,
        'delete':    kb.Key.delete,
        'up':        kb.Key.up,
        'down':      kb.Key.down,
        'left':      kb.Key.left,
        'right':     kb.Key.right,
        'shift':     kb.Key.shift,
        'ctrl':      kb.Key.ctrl,
        'alt':       kb.Key.alt,
        'cmd':       kb.Key.cmd,
        'home':      kb.Key.home,
        'end':       kb.Key.end,
        'page_up':   kb.Key.page_up,
        'page_down': kb.Key.page_down,
        **{f'f{i}': getattr(kb.Key, f'f{i}') for i in range(1, 13)},
    }

    _MODIFIER_MAP: Dict[str, Any] = {
        'ctrl':  kb.Key.ctrl,
        'shift': kb.Key.shift,
        'alt':   kb.Key.alt,
        'cmd':   kb.Key.cmd,
    }

    parts = [p.strip().lower() for p in key_str.split('+')]
    modifiers = []
    main = None

    for part in parts:
        if part in _MODIFIER_MAP:
            modifiers.append(_MODIFIER_MAP[part])
        elif part in _SPECIAL:
            main = _SPECIAL[part]
        else:
            main = part   # single character

    if main is None and modifiers:
        main = modifiers.pop()

    return main, modifiers


def _pynput_button(button_str: str):
    """Convert a button string to a pynput Button."""
    if not _HAS_PYNPUT:
        return None
    btn_map = {
        'left':   pynput.mouse.Button.left,
        'right':  pynput.mouse.Button.right,
        'middle': pynput.mouse.Button.middle,
    }
    return btn_map.get(button_str.lower(), pynput.mouse.Button.left)


# ---------------------------------------------------------------------------
# ScreenCapture  (backend selection)
# ---------------------------------------------------------------------------

class _ScreenCapture:
    """Internal screen-capture backend, selected at construction time."""

    def __init__(
        self,
        window_title: Optional[str] = None,
        region:       Optional[Tuple[int, int, int, int]] = None,  # (left, top, right, bottom) px
        width:        int = 1280,
        height:       int = 720,
    ) -> None:
        self._w = width
        self._h = height
        self._region = region
        self._camera = None
        self._sct     = None
        self._backend = 'zeros'

        if _HAS_DXCAM:
            try:
                self._camera = dxcam.create(output_color='BGR')
                if region:
                    self._camera.start(region=region, target_fps=30)
                else:
                    self._camera.start(target_fps=30)
                self._backend = 'dxcam'
                log.info('ScreenCapture: using dxcam')
                return
            except Exception as exc:
                log.warning(f'dxcam failed ({exc}), falling back')

        if _HAS_MSS:
            try:
                self._sct = mss.mss()
                self._backend = 'mss'
                log.info('ScreenCapture: using mss')
                return
            except Exception as exc:
                log.warning(f'mss failed ({exc}), falling back')

        if _HAS_PIL:
            self._backend = 'pil'
            log.info('ScreenCapture: using PIL.ImageGrab')
            return

        log.warning('ScreenCapture: all backends unavailable; returning zeros')

    def capture(self) -> np.ndarray:
        """Capture one frame.  Returns (H, W, 3) uint8 BGR array.

        Returns a zero array if no backend is available.
        """
        try:
            if self._backend == 'dxcam' and self._camera is not None:
                frame = self._camera.get_latest_frame()
                if frame is None:
                    return np.zeros((self._h, self._w, 3), dtype=np.uint8)
                return frame  # already (H, W, 3) BGR uint8

            if self._backend == 'mss' and self._sct is not None:
                mon = self._sct.monitors[1]  # primary monitor
                if self._region:
                    l, t, r, b = self._region
                    mon = {'left': l, 'top': t, 'width': r - l, 'height': b - t}
                img = np.array(self._sct.grab(mon))
                return img[:, :, :3]   # BGRA → BGR (drop alpha)

            if self._backend == 'pil':
                import PIL.Image
                box = self._region or None
                img = PIL.ImageGrab.grab(bbox=box).convert('RGB')
                return np.array(img)[:, :, ::-1]  # RGB → BGR

        except Exception as exc:
            log.warning(f'ScreenCapture.capture() failed: {exc}')

        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def close(self) -> None:
        """Release backend resources."""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ScreenModality  (base class)
# ---------------------------------------------------------------------------

class ScreenModality:
    """Generic base class for keyboard + mouse + screen-capture modalities.

    Subclass this for each new game.  Override ``build_obs()``, and
    optionally ``get_reward()``, ``get_done()``, ``get_events()``,
    ``get_admissible()``, ``send_text()``.

    Parameters
    ----------
    window_title    Target window title for focus management (optional).
    region          Screen region to capture: (left, top, right, bottom) in
                    absolute pixels.  None = full primary monitor.
    capture_w       Desired frame width after resizing (default 1280).
    capture_h       Desired frame height after resizing (default 720).
    frame_delay_s   Seconds to wait after dispatching an action before
                    capturing the next frame.  Allows the game to update.
    dry_run         If True, no OS events are dispatched (safe for testing).
    verbose         If True, log dispatched actions.
    """

    def __init__(
        self,
        window_title:  Optional[str]                  = None,
        region:        Optional[Tuple[int, int, int, int]] = None,
        capture_w:     int                            = 1280,
        capture_h:     int                            = 720,
        frame_delay_s: float                          = 0.1,
        dry_run:       bool                           = False,
        verbose:       bool                           = False,
    ) -> None:
        self.window_title  = window_title
        self.region        = region
        self.capture_w     = capture_w
        self.capture_h     = capture_h
        self.frame_delay_s = frame_delay_s
        self.dry_run       = dry_run
        self.verbose       = verbose

        self._cap = _ScreenCapture(window_title, region, capture_w, capture_h)

        # OS input controllers.
        self._kbd = pynput.keyboard.Controller() if _HAS_PYNPUT else None
        self._mse = pynput.mouse.Controller()    if _HAS_PYNPUT else None

        # Track held keys (KeyHold countdown).
        self._held: Dict[str, int] = {}   # key_str → remaining_steps

        # Current frame (updated by step()).
        self._frame: np.ndarray = self._cap.capture()

        # Auxiliary state that subclasses may update.
        self._score:  float     = 0.0
        self._done:   bool      = False
        self._events: List[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> dict:
        """Initialise the modality and return the first observation dict.

        Call this once at the start of each episode (before the first
        ``step()`` call).  Resets internal score/done/events state.
        """
        self._score  = 0.0
        self._done   = False
        self._events = []
        self._held   = {}
        self._frame  = self._cap.capture()
        return self.build_obs(self._frame)

    def reset(self) -> dict:
        """Alias for :meth:`connect` — satisfies the gym-style reset API."""
        return self.connect()

    def disconnect(self) -> None:
        """Release OS resources (camera, input controllers)."""
        # Release any held keys.
        for key in list(self._held.keys()):
            self._release_key(key)
        self._cap.close()

    # ------------------------------------------------------------------
    # Core step interface
    # ------------------------------------------------------------------

    def step(self, action: Action) -> Tuple[dict, float, bool, dict]:
        """Dispatch ``action``, capture a new frame, return (obs, reward, done, info).

        This is the standard Gym-style interface; ``agent.py`` calls it as::

            obs_new, reward, done, info = env.step(action)

        Parameters
        ----------
        action  Any :data:`Action` value.

        Returns
        -------
        obs     State dict from :meth:`build_obs`.
        reward  Current reward from :meth:`get_reward`.
        done    Episode-over flag from :meth:`get_done`.
        info    Auxiliary info dict (empty by default; subclass may populate).
        """
        self._events = []
        self.dispatch(action)
        self.step_time()
        self._frame = self._cap.capture()
        obs    = self.build_obs(self._frame)
        reward = self.get_reward()
        done   = self.get_done()
        return obs, reward, done, {}

    def step_time(self) -> None:
        """Wait ``frame_delay_s`` seconds and decrement held-key counters."""
        time.sleep(self.frame_delay_s)
        # Tick held keys; release those whose countdown reaches zero.
        for key in list(self._held.keys()):
            self._held[key] -= 1
            if self._held[key] <= 0:
                self._release_key(key)

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def dispatch(self, action: Action) -> None:
        """Translate an ``Action`` value to OS input events.

        Handles all :data:`Action` variants, including nested macros
        (lists of actions).  Logs dispatched actions when ``verbose=True``.
        """
        for atomic in flatten(action):
            if self.verbose:
                log.info(f'dispatch: {atomic!r}')
            if self.dry_run:
                continue
            self._dispatch_atomic(atomic)

    def _dispatch_atomic(self, action: Action) -> None:
        """Dispatch a single non-list action."""
        if isinstance(action, str):
            self.send_text(action)
        elif isinstance(action, KeyPress):
            self._key_tap(action.key, action.duration_s)
        elif isinstance(action, KeyHold):
            self._start_hold(action.key, action.steps)
        elif isinstance(action, MouseMove):
            self._mouse_move(action.x, action.y, action.relative, action.duration_s)
        elif isinstance(action, MouseClick):
            self._mouse_click(action.x, action.y, action.button, action.double)
        elif isinstance(action, MouseScroll):
            self._mouse_scroll(action.x, action.y, action.dx, action.dy, action.clicks)
        else:
            log.warning(f'Unknown action type: {type(action).__name__}')

    # ------------------------------------------------------------------
    # OS-level input primitives
    # ------------------------------------------------------------------

    def _norm_to_pixel(self, nx: float, ny: float) -> Tuple[int, int]:
        """Convert normalised [0,1] coordinates to absolute pixel coordinates."""
        if self.region:
            l, t, r, b = self.region
            px = int(l + nx * (r - l))
            py = int(t + ny * (b - t))
        else:
            import ctypes  # noqa: F401 (Windows-specific; graceful below)
            try:
                import ctypes
                user32 = ctypes.windll.user32   # type: ignore[attr-defined]
                sw = user32.GetSystemMetrics(0)
                sh = user32.GetSystemMetrics(1)
            except Exception:
                sw, sh = self.capture_w, self.capture_h
            px = int(nx * sw)
            py = int(ny * sh)
        return px, py

    def _key_tap(self, key_str: str, duration_s: float = 0.0) -> None:
        if self._kbd is None:
            log.debug(f'kbd stub: tap {key_str!r}')
            return
        main, mods = _pynput_key(key_str)
        if main is None:
            return
        try:
            for mod in mods:
                self._kbd.press(mod)
            self._kbd.press(main)
            if duration_s > 0.0:
                time.sleep(duration_s)
            self._kbd.release(main)
            for mod in reversed(mods):
                self._kbd.release(mod)
        except Exception as exc:
            log.warning(f'key_tap({key_str!r}) failed: {exc}')

    def _start_hold(self, key_str: str, steps: int) -> None:
        if self._kbd is None:
            log.debug(f'kbd stub: hold {key_str!r} × {steps}')
            return
        main, _ = _pynput_key(key_str)
        if main is None:
            return
        try:
            self._kbd.press(main)
            self._held[key_str] = steps
        except Exception as exc:
            log.warning(f'start_hold({key_str!r}) failed: {exc}')

    def _release_key(self, key_str: str) -> None:
        if self._kbd is None:
            return
        main, _ = _pynput_key(key_str)
        if main is None:
            return
        try:
            self._kbd.release(main)
        except Exception:
            pass
        self._held.pop(key_str, None)

    def _mouse_move(
        self, nx: float, ny: float, relative: bool, duration_s: float
    ) -> None:
        if self._mse is None:
            log.debug(f'mse stub: move ({nx:.3f}, {ny:.3f})')
            return
        try:
            if relative:
                dx = int(nx * self.capture_w)
                dy = int(ny * self.capture_h)
                self._mse.move(dx, dy)
            else:
                px, py = self._norm_to_pixel(nx, ny)
                self._mse.position = (px, py)
        except Exception as exc:
            log.warning(f'mouse_move failed: {exc}')

    def _mouse_click(
        self,
        nx:     Optional[float],
        ny:     Optional[float],
        button: str,
        double: bool,
    ) -> None:
        if self._mse is None:
            log.debug(f'mse stub: click {button!r} at ({nx}, {ny})')
            return
        try:
            if nx is not None and ny is not None:
                px, py = self._norm_to_pixel(nx, ny)
                self._mse.position = (px, py)
            btn = _pynput_button(button)
            clicks = 2 if double else 1
            for _ in range(clicks):
                self._mse.click(btn)
        except Exception as exc:
            log.warning(f'mouse_click failed: {exc}')

    def _mouse_scroll(
        self,
        nx:     Optional[float],
        ny:     Optional[float],
        dx:     float,
        dy:     float,
        clicks: int,
    ) -> None:
        if self._mse is None:
            log.debug(f'mse stub: scroll dy={dy}')
            return
        try:
            if nx is not None and ny is not None:
                px, py = self._norm_to_pixel(nx, ny)
                self._mse.position = (px, py)
            self._mse.scroll(int(dx), int(dy))
        except Exception as exc:
            log.warning(f'mouse_scroll failed: {exc}')

    # ------------------------------------------------------------------
    # Abstract / default methods for subclasses
    # ------------------------------------------------------------------

    def build_obs(self, frame: np.ndarray, info: dict = {}) -> dict:
        """Convert raw frame to the agent's state dict.

        **Override this in your game adapter.**

        The minimum required keys (for ``agent.py`` + ``AIFEngine``):
          - ``'screen'``      Raw frame (H, W, 3) uint8.
          - ``'score'``       Current game score (float).
          - ``'done'``        Episode over (bool).
          - ``'admissible'``  List of valid actions (may be empty list).

        Subclasses typically also add game-specific keys parsed from the
        screen (e.g., ``'location'``, ``'inventory'``, ``'health'``).

        Parameters
        ----------
        frame  Raw BGR uint8 frame from the screen capture backend.
        info   Optional auxiliary info dict (subclass may populate from
               side-channel data, e.g., RCON for Minecraft).
        """
        return {
            'screen':     frame,
            'score':      self.get_reward(),
            'done':       self.get_done(),
            'admissible': self.get_admissible(),
        }

    def get_reward(self) -> float:
        """Return the current game score/reward.  Override in subclass."""
        return self._score

    def get_done(self) -> bool:
        """Return True if the episode is over.  Override in subclass."""
        return self._done

    def get_events(self) -> List[dict]:
        """Return events accumulated since last step.  Override in subclass."""
        return list(self._events)

    def get_admissible(self) -> List[Action]:
        """Return currently valid actions.  Default: empty (engine decides freely).

        Override in subclass if the game exposes a set of valid actions
        (e.g., menu items read from screen via VisualSymbolLearner).
        """
        return []

    def send_text(self, text: str) -> None:
        """Type a text string on the keyboard (for text-command games).

        Default: type the text character-by-character via pynput.
        Override if the game accepts text via a dedicated API.
        """
        if self._kbd is None:
            log.debug(f'kbd stub: type {text!r}')
            return
        try:
            self._kbd.type(text)
        except Exception as exc:
            log.warning(f'send_text({text!r}) failed: {exc}')

    # ------------------------------------------------------------------
    # Convenience: current frame
    # ------------------------------------------------------------------

    @property
    def frame(self) -> np.ndarray:
        """The most recently captured frame (H, W, 3) uint8."""
        return self._frame

    def capture(self) -> np.ndarray:
        """Capture and return a fresh frame without dispatching any action."""
        self._frame = self._cap.capture()
        return self._frame

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        backend = self._cap._backend
        kr = 'pynput' if _HAS_PYNPUT else 'stub'
        dry = ' [dry_run]' if self.dry_run else ''
        return (
            f'ScreenModality(capture={backend}, input={kr}'
            f', {self.capture_w}×{self.capture_h}{dry})'
        )
