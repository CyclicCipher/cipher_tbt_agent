"""
Sensorimotor wrapper for game interaction.

Provides the agent with sensory access to the game (vision, audio) and
motor control over inputs (mouse, keyboard, gaze).
"""

from .wrapper import SensorimotorWrapper
from .gaze import GazeController
from .sensory import ScreenCapture, AudioCapture
from .motor import MouseController, KeyboardController, EmergencyStop
from .logger import GameplayLogger, GameplayPlayer

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
