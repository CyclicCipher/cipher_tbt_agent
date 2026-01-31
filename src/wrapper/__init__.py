"""
Sensorimotor wrapper for game interaction.

Provides the agent with sensory access to the game (vision, audio) and
motor control over inputs (mouse, keyboard, gaze).
"""

# Core imports that work in headless environments
from .gaze import GazeController
from .sensory import ScreenCapture, AudioCapture
from .logger import GameplayLogger, GameplayPlayer

# Conditional imports for components requiring display/input access
# These will raise ImportError in headless environments
try:
    from .motor import MouseController, KeyboardController, EmergencyStop
    from .wrapper import SensorimotorWrapper
    _DISPLAY_AVAILABLE = True
except ImportError as e:
    _DISPLAY_AVAILABLE = False
    _DISPLAY_IMPORT_ERROR = e

    # Provide helpful error message when trying to use display-dependent features
    class _DisplayRequiredPlaceholder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"This component requires a graphical environment with display/input access.\n"
                f"Original error: {_DISPLAY_IMPORT_ERROR}\n\n"
                f"Solutions:\n"
                f"  - On Linux: Ensure X server is running and DISPLAY is set\n"
                f"  - On Windows: Run from native Windows environment, not WSL\n"
                f"  - For testing: Use headless-safe components (GazeController, ScreenCapture, etc.)"
            )

    MouseController = _DisplayRequiredPlaceholder
    KeyboardController = _DisplayRequiredPlaceholder
    EmergencyStop = _DisplayRequiredPlaceholder
    SensorimotorWrapper = _DisplayRequiredPlaceholder

__all__ = [
    'SensorimotorWrapper',
    'GazeController',
    'ScreenCapture',
    'AudioCapture',
    'MouseController',
    'KeyboardController',
    'EmergencyStop',
    'GameplayLogger',
    'GameplayPlayer',
]

__version__ = '0.1.0'
