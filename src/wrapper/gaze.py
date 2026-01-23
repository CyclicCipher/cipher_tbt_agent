"""
Gaze position control system for foveal vision.

The gaze controller maintains an internal state representing where the agent
is "looking" on the screen. This determines the position of the high-resolution
foveal crop.

Gaze is controlled via active inference: the network predicts where gaze should be,
and prediction errors drive saccades to move gaze toward the predicted position.
"""

import numpy as np
from typing import Tuple, Optional


class GazeController:
    """
    Manages gaze position state and coordinate transformations.

    Gaze position is stored in normalized coordinates [0,1] x [0,1] where
    (0,0) is top-left and (1,1) is bottom-right.

    Attributes:
        position: Current gaze position in normalized coords (x, y)
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        fovea_size: Size of foveal crop in pixels (square)
        gain: Saccade gain factor (multiplier for gaze errors)
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        fovea_size: int,
        initial_position: Optional[Tuple[float, float]] = None,
        gain: float = 50.0
    ):
        """
        Initialize gaze controller.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            fovea_size: Size of foveal crop in pixels (square)
            initial_position: Initial gaze position in normalized coords.
                            Defaults to center (0.5, 0.5)
            gain: Saccade gain factor for converting prediction errors to movement
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fovea_size = fovea_size
        self.gain = gain

        # Initialize gaze at center if not specified
        if initial_position is None:
            self.position = np.array([0.5, 0.5], dtype=np.float32)
        else:
            self.position = np.array(initial_position, dtype=np.float32)
            self._clip_position()

    def get_position(self) -> np.ndarray:
        """
        Get current gaze position in normalized coordinates.

        Returns:
            Array of shape (2,) with normalized gaze position [x, y]
        """
        return self.position.copy()

    def get_pixel_position(self) -> Tuple[int, int]:
        """
        Get current gaze position in pixel coordinates.

        Returns:
            Tuple (x, y) of pixel coordinates
        """
        x = int(self.position[0] * self.screen_width)
        y = int(self.position[1] * self.screen_height)
        return (x, y)

    def get_foveal_crop_bounds(self) -> Tuple[int, int, int, int]:
        """
        Get bounding box for foveal crop centered at current gaze position.

        The crop is constrained to stay within screen bounds. If gaze is near
        the edge, the crop is shifted to remain fully on-screen.

        Returns:
            Tuple (left, top, right, bottom) in pixel coordinates
        """
        x_pixel, y_pixel = self.get_pixel_position()
        half_fovea = self.fovea_size // 2

        # Calculate crop bounds centered on gaze
        left = x_pixel - half_fovea
        top = y_pixel - half_fovea
        right = x_pixel + half_fovea
        bottom = y_pixel + half_fovea

        # Constrain to screen bounds
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > self.screen_width:
            left -= (right - self.screen_width)
            right = self.screen_width
        if bottom > self.screen_height:
            top -= (bottom - self.screen_height)
            bottom = self.screen_height

        # Ensure bounds are still valid after constraints
        left = max(0, left)
        top = max(0, top)
        right = min(self.screen_width, right)
        bottom = min(self.screen_height, bottom)

        return (left, top, right, bottom)

    def update_from_error(self, gaze_error: np.ndarray, dt: float = 1.0) -> None:
        """
        Update gaze position from prediction error via active inference.

        In active inference, motor commands are generated to minimize prediction
        errors. When the network predicts gaze should be at position P_pred but
        proprioception reports it's at P_current, the error signal drives a
        saccade to make reality match prediction.

        Args:
            gaze_error: Prediction error in normalized coordinates [dx, dy].
                       Positive values move gaze right/down.
            dt: Time step for integration (default 1.0)
        """
        # Apply gain and integrate error to update position
        # gaze_error = predicted_position - current_position
        # Movement is proportional to error
        movement = self.gain * gaze_error * dt
        self.position += movement
        self._clip_position()

    def set_position(self, position: Tuple[float, float]) -> None:
        """
        Directly set gaze position (for initialization or testing).

        Args:
            position: New gaze position in normalized coordinates (x, y)
        """
        self.position = np.array(position, dtype=np.float32)
        self._clip_position()

    def _clip_position(self) -> None:
        """Ensure gaze position stays within [0,1] x [0,1] bounds."""
        self.position = np.clip(self.position, 0.0, 1.0)

    def get_proprioception(self) -> np.ndarray:
        """
        Get proprioceptive feedback for the network.

        This is the sensory signal that reports current gaze position.
        The network compares its prediction of gaze position against this
        signal, and the resulting error drives saccades.

        Returns:
            Array of shape (2,) with normalized gaze position [x, y]
        """
        return self.get_position()

    def reset(self, position: Optional[Tuple[float, float]] = None) -> None:
        """
        Reset gaze to initial position.

        Args:
            position: Position to reset to. If None, resets to center (0.5, 0.5)
        """
        if position is None:
            self.position = np.array([0.5, 0.5], dtype=np.float32)
        else:
            self.position = np.array(position, dtype=np.float32)
            self._clip_position()

    def __repr__(self) -> str:
        x, y = self.position
        x_px, y_px = self.get_pixel_position()
        return (f"GazeController(position=({x:.3f}, {y:.3f}), "
                f"pixels=({x_px}, {y_px}), "
                f"screen={self.screen_width}x{self.screen_height}, "
                f"fovea_size={self.fovea_size})")
