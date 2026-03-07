"""action.py — Structured action types for keyboard + mouse + text games.

Design
======
The engine (AIFEngine / DecisionEngine) produces ``Action`` values.
The modality (ScreenModality subclass) translates them to OS-level events.
Both sides are fully decoupled: changing the engine does not require changing
the modality, and vice versa.

Action hierarchy
================
::

    Action = Union[
        str,          # text command (text-adventure / menu selection)
        KeyPress,     # keyboard tap
        KeyHold,      # keyboard hold for N agent steps
        MouseMove,    # move cursor to absolute or relative position
        MouseClick,   # click at position
        MouseScroll,  # scroll wheel
        List[Action], # macro: ordered sequence of any of the above
    ]

``str`` actions are forwarded verbatim to the modality's ``send_text()``
method (text-adventure style).  Structured actions are translated to OS
events by ``ScreenModality.dispatch(action)``.

Screen coordinates
==================
All (x, y) fields use **normalised** coordinates in [0.0, 1.0]:
  - (0.0, 0.0) = top-left corner
  - (1.0, 1.0) = bottom-right corner

This makes actions resolution-independent: the same action works whether
the game window is 720p or 4K.  The modality converts to pixel coordinates
before dispatching.

Key string format
=================
Key strings follow the pynput ``Key`` naming convention plus modifiers:
  - Single keys:   ``'w'``, ``'a'``, ``'space'``, ``'enter'``, ``'escape'``
  - Function keys: ``'f1'`` .. ``'f12'``
  - Modifiers:     ``'shift'``, ``'ctrl'``, ``'alt'``, ``'cmd'``
  - Combos:        ``'ctrl+c'``, ``'ctrl+shift+z'``, ``'alt+f4'``

Design principle
================
**THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST BE GENERAL.**

This file contains zero game-specific logic.  Key names, mouse positions,
and action sequences are chosen by the engine at runtime.

See AIF_ROADMAP.md Phase R7 for the full adapter contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Atomic action types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyPress:
    """A keyboard key press (key-down followed immediately by key-up).

    Parameters
    ----------
    key         Key identifier string (see module docstring for format).
    duration_s  Seconds to hold the key before releasing.
                0.0 = instantaneous tap (default).
                Use 0.05–0.5 for keys that require sustained press.
    """
    key:        str
    duration_s: float = 0.0

    def __repr__(self) -> str:
        d = f', {self.duration_s:.2f}s' if self.duration_s else ''
        return f'KeyPress({self.key!r}{d})'


@dataclass(frozen=True)
class KeyHold:
    """Hold a key for a specified number of agent steps, then release.

    Intended for continuous movement: ``KeyHold('w', steps=10)`` keeps
    the forward key pressed for 10 agent ticks before releasing it.
    The modality handles the hold/release scheduling.

    Parameters
    ----------
    key     Key identifier string.
    steps   Number of agent steps to hold.  Must be >= 1.
    """
    key:   str
    steps: int = 1

    def __repr__(self) -> str:
        return f'KeyHold({self.key!r}, {self.steps})'


@dataclass(frozen=True)
class MouseMove:
    """Move the mouse cursor to a screen position.

    Parameters
    ----------
    x, y        Normalised screen coordinates in [0.0, 1.0].
    relative    If True, (x, y) are deltas from the current cursor position.
                Default False (absolute coordinates).
    duration_s  Time to spend moving (0.0 = instant jump; > 0 = smooth move).
    """
    x:          float
    y:          float
    relative:   bool  = False
    duration_s: float = 0.0

    def __repr__(self) -> str:
        kind = 'rel' if self.relative else 'abs'
        return f'MouseMove({self.x:.3f}, {self.y:.3f}, {kind})'


@dataclass(frozen=True)
class MouseClick:
    """A mouse button click (down + up) at an optional screen position.

    Parameters
    ----------
    x, y    Normalised screen coordinates.
            If None, click at the current cursor position (no move).
    button  Which button: ``'left'``, ``'right'``, or ``'middle'``.
    double  Emit a double-click instead of a single click.
    """
    x:      Optional[float] = None
    y:      Optional[float] = None
    button: str             = 'left'
    double: bool            = False

    def __repr__(self) -> str:
        pos = f'({self.x:.3f}, {self.y:.3f})' if self.x is not None else 'here'
        d = ', double' if self.double else ''
        return f'MouseClick({pos}, {self.button!r}{d})'


@dataclass(frozen=True)
class MouseScroll:
    """Scroll the mouse wheel at an optional screen position.

    Parameters
    ----------
    x, y    Normalised screen coordinates (or None = current cursor position).
    dx      Horizontal scroll delta (positive = right).
    dy      Vertical scroll delta (positive = up, negative = down).
    clicks  Number of scroll wheel clicks (integer steps).
    """
    x:      Optional[float] = None
    y:      Optional[float] = None
    dx:     float           = 0.0
    dy:     float           = -3.0   # negative = scroll down (most common)
    clicks: int             = 3

    def __repr__(self) -> str:
        dir_str = 'up' if self.dy > 0 else 'down'
        return f'MouseScroll({dir_str}, {self.clicks})'


# ---------------------------------------------------------------------------
# Composite type
# ---------------------------------------------------------------------------

#: The full Action type.  Accepted by ``ScreenModality.dispatch()`` and
#: produced by ``AIFEngine.decide()`` / ``DecisionEngine.decide()``.
Action = Union[
    str,
    KeyPress,
    KeyHold,
    MouseMove,
    MouseClick,
    MouseScroll,
    List['Action'],   # macro: sequence of any of the above
]


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def is_text_action(action: Action) -> bool:
    """Return True if ``action`` is a plain text command (``str`` type)."""
    return isinstance(action, str)


def is_keyboard_action(action: Action) -> bool:
    """Return True if ``action`` involves keyboard input."""
    return isinstance(action, (KeyPress, KeyHold))


def is_mouse_action(action: Action) -> bool:
    """Return True if ``action`` involves mouse input."""
    return isinstance(action, (MouseMove, MouseClick, MouseScroll))


def is_macro(action: Action) -> bool:
    """Return True if ``action`` is a composite macro (list of actions)."""
    return isinstance(action, list)


def flatten(action: Action) -> List[Action]:
    """Recursively flatten a (possibly nested) macro into a flat action list.

    Non-list actions are returned as a single-element list.

    Parameters
    ----------
    action  Any Action value.

    Returns
    -------
    List[Action]
        Flat list of atomic actions (str, KeyPress, KeyHold, MouseMove,
        MouseClick, or MouseScroll) in execution order.
    """
    if isinstance(action, list):
        result: List[Action] = []
        for sub in action:
            result.extend(flatten(sub))
        return result
    return [action]


# ---------------------------------------------------------------------------
# Factory helpers (convenience constructors)
# ---------------------------------------------------------------------------

def tap(key: str, duration_s: float = 0.0) -> KeyPress:
    """Convenience: create a :class:`KeyPress` for a single key."""
    return KeyPress(key=key, duration_s=duration_s)


def hold(key: str, steps: int = 1) -> KeyHold:
    """Convenience: create a :class:`KeyHold` for a single key."""
    return KeyHold(key=key, steps=steps)


def click(x: float, y: float, button: str = 'left', double: bool = False) -> MouseClick:
    """Convenience: create a :class:`MouseClick` at normalised coordinates."""
    return MouseClick(x=x, y=y, button=button, double=double)


def move(x: float, y: float, relative: bool = False) -> MouseMove:
    """Convenience: create a :class:`MouseMove` to normalised coordinates."""
    return MouseMove(x=x, y=y, relative=relative)


def scroll_down(clicks: int = 3, x: Optional[float] = None, y: Optional[float] = None) -> MouseScroll:
    """Convenience: scroll down ``clicks`` notches."""
    return MouseScroll(x=x, y=y, dy=-float(clicks), clicks=clicks)


def scroll_up(clicks: int = 3, x: Optional[float] = None, y: Optional[float] = None) -> MouseScroll:
    """Convenience: scroll up ``clicks`` notches."""
    return MouseScroll(x=x, y=y, dy=float(clicks), clicks=clicks)


def macro(*actions: Action) -> List[Action]:
    """Convenience: build a macro from a sequence of actions."""
    return list(actions)
