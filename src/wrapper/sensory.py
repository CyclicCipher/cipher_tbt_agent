"""
Sensory input systems: screen capture and audio capture.

Provides the agent with visual and auditory access to the game environment.
"""

import time
import numpy as np
import mss
import cv2
from typing import Tuple, Optional, Dict
from collections import deque
import threading
import queue

from .gaze import GazeController


class ScreenCapture:
    """
    Screen capture system with foveal and peripheral vision.

    Captures the screen and provides two outputs:
    - Foveal: High-resolution crop at gaze position
    - Peripheral: Downsampled full screen for context

    This mimics biological vision where the fovea has high acuity but
    limited field of view, while peripheral vision covers the full scene
    at lower resolution.

    Attributes:
        fovea_size: Size of foveal crop in pixels (square)
        periphery_size: Size of peripheral downsample in pixels (square)
        gaze_controller: GazeController for determining foveal crop position
        monitor: Monitor configuration for mss capture
    """

    def __init__(
        self,
        gaze_controller: GazeController,
        fovea_size: int = 320,
        periphery_size: int = 96,
        monitor_index: int = 0,
        normalize: bool = True,
        window_title: Optional[str] = None
    ):
        """
        Initialize screen capture system.

        Args:
            gaze_controller: GazeController instance for foveal positioning
            fovea_size: Size of foveal crop in pixels (square)
            periphery_size: Size of peripheral downsample in pixels (square)
            monitor_index: Index of monitor to capture (0 = primary)
            normalize: If True, normalize pixel values to [0, 1]
            window_title: Optional window title to capture. If specified, only
                        captures this window. For safety with non-fullscreen apps.
        """
        self.gaze_controller = gaze_controller
        self.fovea_size = fovea_size
        self.periphery_size = periphery_size
        self.normalize = normalize
        self.window_title = window_title

        # Initialize mss
        self.sct = mss.mss()

        # Try to get window-specific region if window_title specified
        if window_title is not None:
            self.monitor = self._get_window_region(window_title, monitor_index)
        else:
            self.monitor = self.sct.monitors[monitor_index + 1]  # 0 is "all monitors"

        # Update gaze controller with actual screen dimensions
        self.screen_width = self.monitor["width"]
        self.screen_height = self.monitor["height"]
        self.gaze_controller.screen_width = self.screen_width
        self.gaze_controller.screen_height = self.screen_height

        # Pre-allocate buffers for zero-copy capture
        self._full_screen_buffer = None
        self._fovea_buffer = np.zeros((fovea_size, fovea_size, 3), dtype=np.float32)
        self._periphery_buffer = np.zeros((periphery_size, periphery_size, 3), dtype=np.float32)

        # Timing statistics
        self.last_capture_time = 0.0
        self.capture_times = deque(maxlen=100)

    def _get_window_region(self, window_title: str, monitor_index: int) -> Dict:
        """
        Get screen region for a specific window by title.

        Args:
            window_title: Window title to search for (partial match)
            monitor_index: Fallback monitor index if window not found

        Returns:
            Dictionary with 'left', 'top', 'width', 'height' keys for mss
        """
        try:
            import pygetwindow as gw

            # Find windows matching title (partial match)
            windows = gw.getWindowsWithTitle(window_title)

            if windows:
                # Use first matching window
                window = windows[0]

                # Get window position and size
                return {
                    'left': window.left,
                    'top': window.top,
                    'width': window.width,
                    'height': window.height
                }
            else:
                print(f"Warning: Window '{window_title}' not found. Using full screen.")
                return self.sct.monitors[monitor_index + 1]

        except ImportError:
            print("Warning: pygetwindow not available. Install with: pip install pygetwindow")
            print("Falling back to full screen capture.")
            return self.sct.monitors[monitor_index + 1]
        except Exception as e:
            print(f"Warning: Could not get window region: {e}")
            print("Falling back to full screen capture.")
            return self.sct.monitors[monitor_index + 1]

    def capture(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Capture screen and return foveal and peripheral frames.

        Returns:
            Tuple of:
                - fovea: Array of shape (fovea_size, fovea_size, 3)
                - periphery: Array of shape (periphery_size, periphery_size, 3)
                - timestamp: Capture timestamp in seconds since epoch
        """
        start_time = time.time()

        # Capture full screen
        screenshot = self.sct.grab(self.monitor)

        # Convert to numpy array (BGR format from mss)
        full_screen = np.array(screenshot)[:, :, :3]  # Drop alpha channel

        # Convert BGR to RGB
        full_screen = cv2.cvtColor(full_screen, cv2.COLOR_BGR2RGB)

        # Extract foveal crop
        fovea = self._extract_fovea(full_screen)

        # Downsample for peripheral vision
        periphery = self._downsample_periphery(full_screen)

        # Normalize if requested
        if self.normalize:
            fovea = fovea.astype(np.float32) / 255.0
            periphery = periphery.astype(np.float32) / 255.0

        # Record timing
        capture_time = time.time() - start_time
        self.capture_times.append(capture_time)
        self.last_capture_time = start_time

        return fovea, periphery, start_time

    def _extract_fovea(self, full_screen: np.ndarray) -> np.ndarray:
        """
        Extract foveal crop from full screen at current gaze position.

        Args:
            full_screen: Full screen capture array

        Returns:
            Foveal crop of shape (fovea_size, fovea_size, 3)
        """
        left, top, right, bottom = self.gaze_controller.get_foveal_crop_bounds()

        # Crop from full screen
        fovea = full_screen[top:bottom, left:right, :]

        # Handle edge cases where crop might be smaller than expected
        if fovea.shape[0] != self.fovea_size or fovea.shape[1] != self.fovea_size:
            # Resize to expected size
            fovea = cv2.resize(fovea, (self.fovea_size, self.fovea_size))

        return fovea

    def _downsample_periphery(self, full_screen: np.ndarray) -> np.ndarray:
        """
        Downsample full screen for peripheral vision.

        Args:
            full_screen: Full screen capture array

        Returns:
            Downsampled array of shape (periphery_size, periphery_size, 3)
        """
        # Use area interpolation for best downsampling quality
        periphery = cv2.resize(
            full_screen,
            (self.periphery_size, self.periphery_size),
            interpolation=cv2.INTER_AREA
        )
        return periphery

    def get_fps(self) -> float:
        """
        Get average FPS over recent captures.

        Returns:
            Average frames per second
        """
        if len(self.capture_times) == 0:
            return 0.0
        avg_time = np.mean(self.capture_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_latency_ms(self) -> float:
        """
        Get average capture latency in milliseconds.

        Returns:
            Average latency in ms
        """
        if len(self.capture_times) == 0:
            return 0.0
        return np.mean(self.capture_times) * 1000.0

    def cleanup(self) -> None:
        """Clean up resources."""
        self.sct.close()

    def __repr__(self) -> str:
        fps = self.get_fps()
        latency = self.get_latency_ms()
        return (f"ScreenCapture(screen={self.screen_width}x{self.screen_height}, "
                f"fovea={self.fovea_size}x{self.fovea_size}, "
                f"periphery={self.periphery_size}x{self.periphery_size}, "
                f"fps={fps:.1f}, latency={latency:.1f}ms)")


class AudioCapture:
    """
    Audio capture system with mel spectrogram preprocessing.

    Captures system audio and converts it to mel spectrograms for the network.
    Audio chunks are synchronized with video frames via timestamps.

    Attributes:
        sample_rate: Audio sample rate in Hz
        chunk_duration_ms: Duration of each audio chunk in milliseconds
        mel_bands: Number of mel frequency bands
        buffer_size: Number of chunks to keep in buffer
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        chunk_duration_ms: int = 100,
        mel_bands: int = 64,
        buffer_size: int = 16,
        device_name: Optional[str] = None
    ):
        """
        Initialize audio capture system.

        Args:
            sample_rate: Audio sample rate in Hz (48000 for most Windows devices)
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            mel_bands: Number of mel frequency bands
            buffer_size: Number of chunks to keep in buffer
            device_name: Name of audio device to capture. If None, uses default.
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.mel_bands = mel_bands
        self.buffer_size = buffer_size

        # Calculate samples per chunk
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)

        # Audio buffer (circular)
        self.buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)

        # Threading for async capture
        self.capture_thread = None
        self.capture_queue = queue.Queue(maxsize=buffer_size * 2)
        self.running = False

        # NOTE: Actual soundcard initialization deferred until start()
        # to avoid issues if audio device not available
        self.device_name = device_name
        self.recorder = None

    def start(self) -> None:
        """
        Start audio capture in background thread.

        Uses soundcard library with WASAPI loopback for system audio capture.
        Converts raw audio to mel spectrograms in real-time.
        """
        try:
            import soundcard as sc
        except ImportError:
            raise ImportError(
                "Audio capture requires soundcard library. "
                "Install with: pip install soundcard\n"
                "Note: May require numpy.frombuffer patch (see docs/notes/setup_issues.md)"
            )

        # Get loopback device (captures system audio)
        # Loopback devices are exposed as microphones with the speaker's name
        if self.device_name is None:
            try:
                # Get default speaker name for loopback
                default_speaker = sc.default_speaker()
                speaker_name = default_speaker.name

                # Find matching loopback microphone
                loopback_mics = sc.all_microphones(include_loopback=True)
                loopback = None

                for mic in loopback_mics:
                    if speaker_name in mic.name or "loopback" in mic.name.lower():
                        loopback = mic
                        break

                if loopback is None:
                    # Fallback: try to get any loopback device
                    loopback = [m for m in loopback_mics if "loopback" in str(m).lower()]
                    if loopback:
                        loopback = loopback[0]
                    else:
                        raise RuntimeError("No loopback device found")

                self.loopback_device = loopback

            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize audio loopback: {e}\n"
                    f"Check audio device availability and permissions.\n"
                    f"Note: Ensure soundcard library has numpy.frombuffer patch applied."
                )
        else:
            # Find specific loopback device by name
            loopback_mics = sc.all_microphones(include_loopback=True)
            matching = [m for m in loopback_mics if self.device_name.lower() in m.name.lower()]
            if not matching:
                raise ValueError(f"Audio loopback device '{self.device_name}' not found")
            self.loopback_device = matching[0]

        self.running = True

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self) -> None:
        """Background thread for continuous audio capture."""
        with self.loopback_device.recorder(samplerate=self.sample_rate) as recorder:
            while self.running:
                try:
                    # Record one chunk
                    data = recorder.record(numframes=self.chunk_samples)

                    # Convert to mono if needed
                    if data.ndim > 1:
                        data = data.mean(axis=1)

                    # Convert to mel spectrogram (simplified - just return raw for now)
                    # TODO: Implement actual mel spectrogram conversion
                    mel_spec = self._simple_mel_spectrogram(data)

                    # Add to buffer with timestamp
                    timestamp = time.time()
                    self.buffer.append(mel_spec)
                    self.timestamps.append(timestamp)

                except Exception as e:
                    if self.running:  # Only print if not shutting down
                        print(f"Audio capture error: {e}")
                    break

    def _simple_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel spectrogram (simplified implementation).

        Args:
            audio_data: Raw audio samples

        Returns:
            Mel spectrogram of shape (mel_bands, time_steps)
        """
        # Simplified: Return power spectrum approximation
        # For MVP, just return energy bands
        # TODO: Implement proper mel filterbank

        # Compute FFT
        fft = np.fft.rfft(audio_data)
        power = np.abs(fft) ** 2

        # Downsample to mel_bands
        bins_per_band = len(power) // self.mel_bands
        mel_spec = np.array([
            power[i * bins_per_band:(i + 1) * bins_per_band].mean()
            for i in range(self.mel_bands)
        ])

        # Reshape to (mel_bands, 1) for consistency
        return mel_spec.reshape(-1, 1).astype(np.float32)

    def stop(self) -> None:
        """Stop audio capture."""
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)

    def get_latest_chunk(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the most recent audio chunk.

        Returns:
            Tuple of (mel_spectrogram, timestamp) or None if no data available
            mel_spectrogram has shape (mel_bands, time_steps)
        """
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1], self.timestamps[-1]

    def get_synchronized_chunk(self, video_timestamp: float, tolerance: float = 0.05) -> Optional[np.ndarray]:
        """
        Get audio chunk synchronized with a video frame.

        Finds the audio chunk whose timestamp is closest to the video timestamp
        within the specified tolerance.

        Args:
            video_timestamp: Timestamp of video frame to sync with
            tolerance: Maximum time difference in seconds

        Returns:
            Mel spectrogram array or None if no matching chunk found
        """
        if len(self.timestamps) == 0:
            return None

        # Find closest timestamp
        timestamps_array = np.array(self.timestamps)
        diffs = np.abs(timestamps_array - video_timestamp)
        min_idx = np.argmin(diffs)

        if diffs[min_idx] <= tolerance:
            return self.buffer[min_idx]
        return None

    def get_buffer(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get entire audio buffer.

        Returns:
            Tuple of:
                - chunks: Array of shape (buffer_size, mel_bands, time_steps)
                - timestamps: Array of shape (buffer_size,)
        """
        if len(self.buffer) == 0:
            return np.array([]), np.array([])

        chunks = np.array(list(self.buffer))
        timestamps = np.array(list(self.timestamps))
        return chunks, timestamps

    def __repr__(self) -> str:
        return (f"AudioCapture(sample_rate={self.sample_rate}Hz, "
                f"chunk_duration={self.chunk_duration_ms}ms, "
                f"mel_bands={self.mel_bands}, "
                f"buffer_size={self.buffer_size}, "
                f"chunks_captured={len(self.buffer)})")
