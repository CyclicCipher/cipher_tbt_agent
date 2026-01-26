"""
Retinal preprocessing for vision input.

Mimics biological retinal processing:
- Edge detection (ON/OFF center-surround cells)
- Movement detection (temporal differences)
- Salience computation (guides attention/gaze)
- Compression/encoding for network input

Keeps foveal (high-res) and peripheral (low-res) separation.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class RetinalPreprocessor:
    """
    Biological-inspired retinal preprocessing.

    Transforms raw RGB pixels into processed visual features:
    - Edges (center-surround receptive fields)
    - Movement (temporal differences)
    - Salience (attention guidance)

    Maintains foveal + peripheral structure.
    """

    def __init__(
        self,
        fovea_size: int = 320,
        periphery_size: int = 96,
        enable_edges: bool = True,
        enable_movement: bool = True,
        enable_salience: bool = True,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize retinal preprocessor.

        Args:
            fovea_size: Size of foveal input (square)
            periphery_size: Size of peripheral input (square)
            enable_edges: Whether to compute edge features
            enable_movement: Whether to compute movement features
            enable_salience: Whether to compute salience map
            dtype: Output dtype
        """
        self.fovea_size = fovea_size
        self.periphery_size = periphery_size
        self.enable_edges = enable_edges
        self.enable_movement = enable_movement
        self.enable_salience = enable_salience
        self.dtype = dtype

        # Previous frames for movement detection
        self.prev_fovea = None
        self.prev_periphery = None

        # Salience map (normalized coordinates 0-1)
        self.salience_map = None

    def process(
        self,
        fovea: np.ndarray,
        periphery: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Process foveal and peripheral frames.

        Args:
            fovea: RGB foveal frame (fovea_size, fovea_size, 3), range [0,1]
            periphery: RGB peripheral frame (periphery_size, periphery_size, 3), range [0,1]

        Returns:
            Tuple of:
                - processed_fovea: Processed foveal features
                - processed_periphery: Processed peripheral features
                - salience_map: Attention guidance map (if enabled), shape (H, W)
        """
        # Convert to grayscale for processing
        fovea_gray = self._to_grayscale(fovea)
        periphery_gray = self._to_grayscale(periphery)

        # Edge detection (ON/OFF center-surround cells)
        if self.enable_edges:
            fovea_edges = self._detect_edges(fovea_gray)
            periphery_edges = self._detect_edges(periphery_gray)
        else:
            fovea_edges = fovea_gray
            periphery_edges = periphery_gray

        # Movement detection (temporal differences)
        if self.enable_movement:
            fovea_movement = self._detect_movement(fovea_gray, self.prev_fovea)
            periphery_movement = self._detect_movement(periphery_gray, self.prev_periphery)

            # Update previous frames
            self.prev_fovea = fovea_gray.copy()
            self.prev_periphery = periphery_gray.copy()
        else:
            fovea_movement = np.zeros_like(fovea_gray)
            periphery_movement = np.zeros_like(periphery_gray)

        # Stack features: [edges, movement, intensity]
        fovea_processed = np.stack([fovea_edges, fovea_movement, fovea_gray], axis=-1)
        periphery_processed = np.stack([periphery_edges, periphery_movement, periphery_gray], axis=-1)

        # Compute salience map for attention guidance
        if self.enable_salience:
            salience = self._compute_salience(fovea_edges, fovea_movement, periphery_edges, periphery_movement)
        else:
            salience = None

        return fovea_processed, periphery_processed, salience

    def _to_grayscale(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale using human luminance weights."""
        # Human eye sensitivity: R=0.299, G=0.587, B=0.114
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect edges using center-surround receptive fields.

        Mimics retinal ganglion cells with ON-center/OFF-surround organization.
        """
        # Simple approach: Sobel edge detection
        # (More sophisticated: DoG filters to mimic center-surround)
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Sobel in X and Y
        sobelx = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude
        edges = np.sqrt(sobelx**2 + sobely**2)

        # Normalize
        edges = np.clip(edges / 255.0, 0, 1).astype(self.dtype)

        return edges

    def _detect_movement(self, current: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
        """
        Detect movement via temporal differences.

        Mimics retinal motion detection (transient cells responding to changes).
        """
        if previous is None:
            # First frame, no movement
            return np.zeros_like(current)

        # Temporal difference
        movement = np.abs(current - previous)

        # Threshold to reduce noise
        movement = np.clip(movement * 2.0, 0, 1).astype(self.dtype)

        return movement

    def _compute_salience(
        self,
        fovea_edges: np.ndarray,
        fovea_movement: np.ndarray,
        periphery_edges: np.ndarray,
        periphery_movement: np.ndarray
    ) -> np.ndarray:
        """
        Compute salience map for attention guidance.

        Salience = where should the agent look?
        - High edges = interesting features
        - High movement = dynamic events

        Returns map in peripheral resolution for computational efficiency.
        """
        # Combine edge and movement information
        # Edges indicate structure (text, objects)
        # Movement indicates events (animations, changes)

        # Weight movement higher (motion attracts attention)
        periphery_salience = 0.3 * periphery_edges + 0.7 * periphery_movement

        # Apply spatial smoothing (salience spreads)
        kernel_size = max(3, self.periphery_size // 32)
        if kernel_size % 2 == 0:
            kernel_size += 1

        periphery_salience = cv2.GaussianBlur(
            periphery_salience,
            (kernel_size, kernel_size),
            sigmaX=2.0
        )

        # Normalize to [0, 1]
        if periphery_salience.max() > 0:
            periphery_salience = periphery_salience / periphery_salience.max()

        self.salience_map = periphery_salience

        return periphery_salience

    def get_attention_suggestion(self) -> Optional[Tuple[float, float]]:
        """
        Get suggested gaze position from salience map.

        Returns:
            Tuple (x, y) in normalized coordinates [0,1], or None if no salience
        """
        if self.salience_map is None:
            return None

        # Find peak of salience map
        peak_y, peak_x = np.unravel_index(
            np.argmax(self.salience_map),
            self.salience_map.shape
        )

        # Convert to normalized coordinates
        x_norm = peak_x / self.periphery_size
        y_norm = peak_y / self.periphery_size

        return (x_norm, y_norm)

    def reset(self):
        """Reset temporal buffers (call when starting new sequence)."""
        self.prev_fovea = None
        self.prev_periphery = None
        self.salience_map = None


def flatten_visual_input(
    fovea_processed: np.ndarray,
    periphery_processed: np.ndarray
) -> np.ndarray:
    """
    Flatten foveal + peripheral features into single vector for network input.

    Args:
        fovea_processed: Processed foveal features (H, W, C)
        periphery_processed: Processed peripheral features (H, W, C)

    Returns:
        Flattened vector of shape (fovea_features + periphery_features,)
    """
    fovea_flat = fovea_processed.flatten()
    periphery_flat = periphery_processed.flatten()

    # Concatenate: [foveal features, peripheral features]
    return np.concatenate([fovea_flat, periphery_flat])


def get_visual_input_size(fovea_size: int, periphery_size: int, channels: int = 3) -> int:
    """
    Compute total input size for network.

    Args:
        fovea_size: Foveal resolution (square)
        periphery_size: Peripheral resolution (square)
        channels: Number of feature channels (default 3: edges, movement, intensity)

    Returns:
        Total flattened size
    """
    fovea_features = fovea_size * fovea_size * channels
    periphery_features = periphery_size * periphery_size * channels
    return fovea_features + periphery_features


def retinal_preprocessing(rgb_patch: np.ndarray) -> np.ndarray:
    """
    Simple preprocessing for a single RGB patch (for testing).

    Applies edge detection, no movement (static patch).
    Returns 3-channel output: [edges, zeros, intensity]

    Args:
        rgb_patch: RGB image (H, W, 3), values in [0, 255]

    Returns:
        Processed features (H, W, 3), values in [0, 1]
    """
    # Normalize to [0, 1]
    if rgb_patch.max() > 1.0:
        rgb_patch = rgb_patch / 255.0

    # Convert to grayscale
    gray = 0.299 * rgb_patch[:, :, 0] + 0.587 * rgb_patch[:, :, 1] + 0.114 * rgb_patch[:, :, 2]

    # Edge detection (Sobel)
    gray_uint8 = (gray * 255).astype(np.uint8)
    sobelx = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.clip(edges / 255.0, 0, 1).astype(np.float32)

    # No movement detection (static image)
    movement = np.zeros_like(gray, dtype=np.float32)

    # Stack: [edges, movement, intensity]
    features = np.stack([edges, movement, gray], axis=-1)

    return features
