"""
Main sensorimotor wrapper integration.

Coordinates all sensory and motor systems to provide a unified interface
for the agent to interact with the game environment.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, Callable
from pynput import keyboard

from .gaze import GazeController
from .sensory import ScreenCapture, AudioCapture
from .motor import MouseController, KeyboardController, EmergencyStop
from .logger import GameplayLogger


class SensorimotorWrapper:
    """
    Main wrapper coordinating all sensorimotor systems.

    This is the primary interface between the agent and the game environment.
    It provides:
    - Sensory inputs (foveal/peripheral vision, audio, proprioception)
    - Motor outputs (gaze, cursor, clicks, keyboard)
    - Recording and playback capabilities
    - Safety features (emergency stop, rate limiting)

    The wrapper implements the active inference sensorimotor loop:
    1. Capture sensory inputs
    2. Network processes inputs and generates predictions
    3. Prediction errors drive motor outputs
    4. Motor actions affect environment
    5. Loop continues

    Attributes:
        config: Configuration dictionary
        gaze: GazeController instance
        screen: ScreenCapture instance
        audio: AudioCapture instance (optional)
        mouse: MouseController instance
        keyboard: KeyboardController instance
        logger: GameplayLogger instance (optional)
        emergency_stop: EmergencyStop instance
        running: Whether the main loop is running
        target_fps: Target frames per second
    """

    def __init__(
        self,
        config: Dict[str, Any],
        enable_audio: bool = False,
        enable_logging: bool = False,
        log_dir: str = "logs/gameplay"
    ):
        """
        Initialize sensorimotor wrapper.

        Args:
            config: Configuration dictionary with parameters from config.yaml
            enable_audio: Whether to enable audio capture
            enable_logging: Whether to enable gameplay logging
            log_dir: Directory for gameplay logs
        """
        self.config = config
        self.target_fps = 20  # Target FPS as specified in docs
        self.frame_time = 1.0 / self.target_fps

        # Extract configuration
        vision_cfg = config.get('vision', {})
        motor_cfg = config.get('motor', {})
        audio_cfg = config.get('audio', {})
        logging_cfg = config.get('logging', {})

        # Get screen dimensions (will be updated by ScreenCapture)
        # Default to 1920x1080 if not specified
        screen_width = 1920
        screen_height = 1080

        # Initialize gaze controller
        self.gaze = GazeController(
            screen_width=screen_width,
            screen_height=screen_height,
            fovea_size=vision_cfg.get('fovea_size', 320),
            gain=motor_cfg.get('gaze_gain', 50.0)
        )

        # Initialize screen capture
        self.screen = ScreenCapture(
            gaze_controller=self.gaze,
            fovea_size=vision_cfg.get('fovea_size', 320),
            periphery_size=vision_cfg.get('periphery_size', 96),
            normalize=True
        )

        # Initialize audio capture (optional)
        self.audio = None
        if enable_audio:
            self.audio = AudioCapture(
                sample_rate=audio_cfg.get('sample_rate', 16000),
                chunk_duration_ms=audio_cfg.get('chunk_duration_ms', 100),
                mel_bands=audio_cfg.get('mel_bands', 64),
                buffer_size=audio_cfg.get('buffer_size', 16)
            )

        # Initialize motor controllers
        self.mouse = MouseController(
            screen_width=self.screen.screen_width,
            screen_height=self.screen.screen_height,
            gain=motor_cfg.get('cursor_gain', 5.0),
            click_threshold=motor_cfg.get('click_threshold', 0.7)
        )

        self.keyboard_ctrl = KeyboardController()

        # Initialize emergency stop
        # Default: Ctrl+Shift+Esc to trigger emergency stop
        self.emergency_stop = EmergencyStop(
            kill_keys={keyboard.Key.ctrl, keyboard.Key.shift, keyboard.Key.esc},
            mouse_controller=self.mouse,
            keyboard_controller=self.keyboard_ctrl,
            callback=self._on_emergency_stop
        )

        # Initialize logger (optional)
        self.logger = None
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = GameplayLogger(log_dir=log_dir)

        # State
        self.running = False
        self.frame_count = 0
        self.start_time = None

        # Network interface (to be connected)
        self.network_step_fn = None

    def set_network(self, step_fn: Callable) -> None:
        """
        Set the network step function.

        The network step function should have signature:
            step_fn(sensory_dict) -> motor_dict

        Where:
            sensory_dict contains:
                - 'foveal': foveal frame
                - 'peripheral': peripheral frame
                - 'audio': audio chunk (if enabled)
                - 'gaze': gaze position
                - 'cursor': cursor position
                - 'timestamp': frame timestamp

            motor_dict should contain:
                - 'gaze_error': prediction error for gaze (2,)
                - 'cursor_error': prediction error for cursor (2,)
                - 'click_signal': click activation (scalar)
                - 'key_signals': dict of key activations

        Args:
            step_fn: Network step function
        """
        self.network_step_fn = step_fn

    def start(
        self,
        duration: Optional[float] = None,
        start_logging: bool = True,
        log_metadata: Optional[Dict] = None
    ) -> None:
        """
        Start the sensorimotor loop.

        Args:
            duration: Optional duration in seconds. If None, runs until stopped.
            start_logging: Whether to start logging if logging is enabled
            log_metadata: Optional metadata to include in log
        """
        if self.running:
            print("Warning: Already running.")
            return

        if self.network_step_fn is None:
            print("Warning: No network connected. Use set_network() first.")
            print("Running in manual mode (no motor control).")

        self.running = True
        self.start_time = time.time()
        self.frame_count = 0

        # Start audio capture if enabled
        if self.audio is not None:
            try:
                self.audio.start()
            except NotImplementedError:
                print("Audio capture not yet implemented. Continuing without audio.")
                self.audio = None

        # Start logging if enabled
        if self.enable_logging and start_logging and self.logger is not None:
            self.logger.start(metadata=log_metadata)

        print(f"Starting sensorimotor loop at {self.target_fps} FPS")
        print("Press Ctrl+Shift+Esc to trigger emergency stop")

        # Main loop
        try:
            self._main_loop(duration)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()

    def _main_loop(self, duration: Optional[float]) -> None:
        """
        Main sensorimotor loop.

        Args:
            duration: Optional duration in seconds
        """
        while self.running:
            loop_start = time.time()

            # Check duration
            if duration is not None:
                elapsed = time.time() - self.start_time
                if elapsed >= duration:
                    break

            # Check emergency stop
            if self.emergency_stop.triggered:
                break

            # Execute one step
            self.step()

            # Frame rate limiting
            loop_time = time.time() - loop_start
            sleep_time = self.frame_time - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            self.frame_count += 1

    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the sensorimotor loop.

        Returns:
            Dictionary with sensory data and motor outputs
        """
        # Capture sensory inputs
        foveal, peripheral, timestamp = self.screen.capture()

        # Get audio (if enabled)
        audio_chunk = None
        if self.audio is not None:
            audio_data = self.audio.get_latest_chunk()
            if audio_data is not None:
                audio_chunk, _ = audio_data

        # Get proprioception
        gaze_position = self.gaze.get_proprioception()
        cursor_position = self.mouse.get_proprioception()

        # Package sensory inputs
        sensory_dict = {
            'foveal': foveal,
            'peripheral': peripheral,
            'audio': audio_chunk,
            'gaze': gaze_position,
            'cursor': cursor_position,
            'timestamp': timestamp,
            'frame': self.frame_count
        }

        # Network processing (if connected)
        motor_dict = None
        if self.network_step_fn is not None:
            motor_dict = self.network_step_fn(sensory_dict)

            # Execute motor commands
            self._execute_motor_commands(motor_dict)

        # Logging
        if self.logger is not None and self.logger.recording:
            self.logger.log_frame(
                foveal_frame=foveal,
                peripheral_frame=peripheral,
                gaze_position=gaze_position,
                cursor_position=cursor_position,
                timestamp=timestamp
            )
            if audio_chunk is not None:
                self.logger.log_audio(audio_chunk, timestamp)

        return {
            'sensory': sensory_dict,
            'motor': motor_dict
        }

    def _execute_motor_commands(self, motor_dict: Dict[str, Any]) -> None:
        """
        Execute motor commands from network output.

        Args:
            motor_dict: Dictionary with motor commands
        """
        # Gaze control
        if 'gaze_error' in motor_dict:
            gaze_error = motor_dict['gaze_error']
            self.gaze.update_from_error(gaze_error)

        # Cursor control
        if 'cursor_error' in motor_dict and 'click_signal' in motor_dict:
            cursor_error = motor_dict['cursor_error']
            click_signal = motor_dict['click_signal']
            self.mouse.update_from_error(cursor_error, click_signal)

            # Log clicks
            if click_signal > self.mouse.click_threshold:
                if self.logger is not None and self.logger.recording:
                    self.logger.log_mouse_click('left', time.time())

        # Keyboard control
        if 'key_signals' in motor_dict:
            key_signals = motor_dict['key_signals']
            self.keyboard_ctrl.update_from_signals(key_signals)

            # Log key presses
            if self.logger is not None and self.logger.recording:
                for key_name, signal in key_signals.items():
                    threshold = self.keyboard_ctrl.key_thresholds.get(
                        key_name, self.keyboard_ctrl.default_threshold
                    )
                    if signal > threshold:
                        self.logger.log_key_press(key_name, time.time())

    def stop(self) -> None:
        """Stop the sensorimotor loop."""
        if not self.running:
            return

        self.running = False

        # Stop audio
        if self.audio is not None:
            self.audio.stop()

        # Stop logging
        if self.logger is not None and self.logger.recording:
            self.logger.stop()

        # Stop emergency stop listener
        self.emergency_stop.stop()

        # Cleanup
        self.screen.cleanup()

        # Print statistics
        if self.start_time is not None:
            duration = time.time() - self.start_time
            actual_fps = self.frame_count / duration if duration > 0 else 0
            print(f"\nSession statistics:")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Frames: {self.frame_count}")
            print(f"  Actual FPS: {actual_fps:.1f}")
            print(f"  Target FPS: {self.target_fps}")
            print(f"  Screen capture latency: {self.screen.get_latency_ms():.1f}ms")

    def _on_emergency_stop(self) -> None:
        """Callback for emergency stop."""
        self.running = False

    def get_sensory_shapes(self) -> Dict[str, tuple]:
        """
        Get shapes of sensory inputs for network initialization.

        Returns:
            Dictionary mapping input names to shapes
        """
        shapes = {
            'foveal': (self.screen.fovea_size, self.screen.fovea_size, 3),
            'peripheral': (self.screen.periphery_size, self.screen.periphery_size, 3),
            'gaze': (2,),
            'cursor': (2,),
        }

        if self.audio is not None:
            # Audio shape will depend on mel bands and time steps
            # This is a placeholder - actual shape determined by AudioCapture
            shapes['audio'] = (self.audio.mel_bands, None)  # Variable time dimension

        return shapes

    def __repr__(self) -> str:
        status = "RUNNING" if self.running else "IDLE"
        fps = self.screen.get_fps()
        latency = self.screen.get_latency_ms()
        return (f"SensorimotorWrapper(status={status}, "
                f"frames={self.frame_count}, "
                f"fps={fps:.1f}, "
                f"latency={latency:.1f}ms)")
