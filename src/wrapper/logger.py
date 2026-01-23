"""
Gameplay logging system for recording and replaying sessions.

Records all sensory inputs (screen, audio) and motor outputs (mouse, keyboard)
with timestamps for synchronized playback and analysis.
"""

import os
import time
import json
import numpy as np
import h5py
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime


class GameplayLogger:
    """
    Records gameplay sessions to disk for replay and analysis.

    Uses HDF5 format for efficient storage and random access. Each session
    is stored in a separate file with datasets for frames, audio, actions,
    and metadata.

    File structure:
        session_YYYYMMDD_HHMMSS.h5
        ├── foveal_frames: (N, H, W, 3) float32
        ├── peripheral_frames: (N, H, W, 3) float32
        ├── audio_chunks: (M, mel_bands, time_steps) float32
        ├── gaze_positions: (N, 2) float32
        ├── cursor_positions: (N, 2) float32
        ├── mouse_clicks: (K,) structured array [(button, timestamp)]
        ├── key_presses: (L,) structured array [(key, timestamp)]
        ├── timestamps: (N,) float64
        └── metadata: JSON string with session info

    Attributes:
        log_dir: Directory to save logs
        session_name: Name of current session
        file_path: Path to current log file
        recording: Whether recording is active
    """

    def __init__(
        self,
        log_dir: str = "logs/gameplay",
        session_name: Optional[str] = None,
        compress: bool = True
    ):
        """
        Initialize gameplay logger.

        Args:
            log_dir: Directory to save log files
            session_name: Optional session name. If None, uses timestamp
            compress: If True, use compression for HDF5 datasets
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress

        # Generate session name if not provided
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_name = session_name

        # File path
        self.file_path = self.log_dir / f"{session_name}.h5"

        # Recording state
        self.recording = False
        self.file = None

        # Buffers for accumulating data before writing
        self.foveal_buffer = []
        self.peripheral_buffer = []
        self.audio_buffer = []
        self.gaze_buffer = []
        self.cursor_buffer = []
        self.mouse_click_buffer = []
        self.key_press_buffer = []
        self.timestamp_buffer = []

        # Metadata
        self.metadata = {
            'session_name': session_name,
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'duration_seconds': 0.0
        }

    def start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start recording session.

        Args:
            metadata: Optional metadata to include in log
        """
        if self.recording:
            print("Warning: Already recording. Call stop() first.")
            return

        # Update metadata
        self.metadata['start_time'] = time.time()
        if metadata is not None:
            self.metadata.update(metadata)

        # Open HDF5 file
        self.file = h5py.File(self.file_path, 'w')

        self.recording = True
        print(f"Started recording to: {self.file_path}")

    def stop(self) -> None:
        """Stop recording and save to disk."""
        if not self.recording:
            print("Warning: Not currently recording.")
            return

        # Finalize metadata
        self.metadata['end_time'] = time.time()
        self.metadata['duration_seconds'] = (
            self.metadata['end_time'] - self.metadata['start_time']
        )
        self.metadata['total_frames'] = len(self.timestamp_buffer)

        # Write all buffered data
        self._write_buffers()

        # Write metadata
        self.file.attrs['metadata'] = json.dumps(self.metadata)

        # Close file
        self.file.close()
        self.file = None
        self.recording = False

        print(f"Stopped recording. Saved {self.metadata['total_frames']} frames "
              f"({self.metadata['duration_seconds']:.1f} seconds)")

    def log_frame(
        self,
        foveal_frame: np.ndarray,
        peripheral_frame: np.ndarray,
        gaze_position: np.ndarray,
        cursor_position: np.ndarray,
        timestamp: float
    ) -> None:
        """
        Log a single frame with sensory and proprioceptive data.

        Args:
            foveal_frame: Foveal vision frame (H, W, 3)
            peripheral_frame: Peripheral vision frame (H, W, 3)
            gaze_position: Gaze position in normalized coords (2,)
            cursor_position: Cursor position in normalized coords (2,)
            timestamp: Frame timestamp
        """
        if not self.recording:
            return

        self.foveal_buffer.append(foveal_frame)
        self.peripheral_buffer.append(peripheral_frame)
        self.gaze_buffer.append(gaze_position)
        self.cursor_buffer.append(cursor_position)
        self.timestamp_buffer.append(timestamp)

    def log_audio(self, audio_chunk: np.ndarray, timestamp: float) -> None:
        """
        Log an audio chunk.

        Args:
            audio_chunk: Mel spectrogram (mel_bands, time_steps)
            timestamp: Chunk timestamp
        """
        if not self.recording:
            return

        self.audio_buffer.append((audio_chunk, timestamp))

    def log_mouse_click(self, button: str, timestamp: float) -> None:
        """
        Log a mouse click.

        Args:
            button: Button name ('left', 'right', 'middle')
            timestamp: Click timestamp
        """
        if not self.recording:
            return

        self.mouse_click_buffer.append((button, timestamp))

    def log_key_press(self, key: str, timestamp: float) -> None:
        """
        Log a key press.

        Args:
            key: Key name
            timestamp: Press timestamp
        """
        if not self.recording:
            return

        self.key_press_buffer.append((key, timestamp))

    def _write_buffers(self) -> None:
        """Write all buffered data to HDF5 file."""
        if len(self.timestamp_buffer) == 0:
            print("Warning: No frames to write.")
            return

        # Compression settings
        compression = 'gzip' if self.compress else None
        compression_opts = 4 if self.compress else None

        # Write frame data
        self.file.create_dataset(
            'foveal_frames',
            data=np.array(self.foveal_buffer),
            compression=compression,
            compression_opts=compression_opts
        )
        self.file.create_dataset(
            'peripheral_frames',
            data=np.array(self.peripheral_buffer),
            compression=compression,
            compression_opts=compression_opts
        )

        # Write audio data if available
        if len(self.audio_buffer) > 0:
            audio_chunks = np.array([chunk for chunk, _ in self.audio_buffer])
            audio_timestamps = np.array([t for _, t in self.audio_buffer])
            self.file.create_dataset(
                'audio_chunks',
                data=audio_chunks,
                compression=compression,
                compression_opts=compression_opts
            )
            self.file.create_dataset('audio_timestamps', data=audio_timestamps)

        # Write proprioception data
        self.file.create_dataset('gaze_positions', data=np.array(self.gaze_buffer))
        self.file.create_dataset('cursor_positions', data=np.array(self.cursor_buffer))
        self.file.create_dataset('timestamps', data=np.array(self.timestamp_buffer))

        # Write action data (mouse clicks and key presses)
        if len(self.mouse_click_buffer) > 0:
            # Create structured array for mouse clicks
            click_dtype = np.dtype([('button', 'U10'), ('timestamp', 'f8')])
            clicks = np.array(self.mouse_click_buffer, dtype=click_dtype)
            self.file.create_dataset('mouse_clicks', data=clicks)

        if len(self.key_press_buffer) > 0:
            # Create structured array for key presses
            key_dtype = np.dtype([('key', 'U20'), ('timestamp', 'f8')])
            keys = np.array(self.key_press_buffer, dtype=key_dtype)
            self.file.create_dataset('key_presses', data=keys)

    def __repr__(self) -> str:
        status = "RECORDING" if self.recording else "IDLE"
        frames = len(self.timestamp_buffer)
        return (f"GameplayLogger(session={self.session_name}, "
                f"status={status}, frames={frames})")


class GameplayPlayer:
    """
    Plays back recorded gameplay sessions.

    Loads a recorded session and provides methods to iterate through frames
    and retrieve synchronized data for analysis or replay.

    Attributes:
        file_path: Path to log file
        file: HDF5 file handle
        metadata: Session metadata
        num_frames: Number of frames in session
    """

    def __init__(self, file_path: str):
        """
        Initialize gameplay player.

        Args:
            file_path: Path to recorded session file (.h5)
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")

        # Open file
        self.file = h5py.File(self.file_path, 'r')

        # Load metadata
        self.metadata = json.loads(self.file.attrs['metadata'])
        self.num_frames = len(self.file['timestamps'])

        # Current playback position
        self.current_frame = 0

    def get_frame(self, index: int) -> Dict[str, Any]:
        """
        Get data for a specific frame.

        Args:
            index: Frame index

        Returns:
            Dictionary with frame data:
                - foveal: Foveal frame
                - peripheral: Peripheral frame
                - gaze: Gaze position
                - cursor: Cursor position
                - timestamp: Frame timestamp
        """
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self.num_frames})")

        return {
            'foveal': self.file['foveal_frames'][index],
            'peripheral': self.file['peripheral_frames'][index],
            'gaze': self.file['gaze_positions'][index],
            'cursor': self.file['cursor_positions'][index],
            'timestamp': self.file['timestamps'][index]
        }

    def get_audio_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get audio chunk nearest to specified timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            Audio chunk or None if no audio recorded
        """
        if 'audio_chunks' not in self.file:
            return None

        audio_timestamps = self.file['audio_timestamps'][:]
        idx = np.argmin(np.abs(audio_timestamps - timestamp))
        return self.file['audio_chunks'][idx]

    def get_actions_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, list]:
        """
        Get all mouse/keyboard actions in a time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Dictionary with 'clicks' and 'keys' lists
        """
        actions = {'clicks': [], 'keys': []}

        # Get mouse clicks
        if 'mouse_clicks' in self.file:
            clicks = self.file['mouse_clicks'][:]
            mask = (clicks['timestamp'] >= start_time) & (clicks['timestamp'] <= end_time)
            actions['clicks'] = [
                (str(click['button']), float(click['timestamp']))
                for click in clicks[mask]
            ]

        # Get key presses
        if 'key_presses' in self.file:
            keys = self.file['key_presses'][:]
            mask = (keys['timestamp'] >= start_time) & (keys['timestamp'] <= end_time)
            actions['keys'] = [
                (str(key['key']), float(key['timestamp']))
                for key in keys[mask]
            ]

        return actions

    def iter_frames(self, start: int = 0, end: Optional[int] = None):
        """
        Iterator over frames.

        Args:
            start: Start frame index
            end: End frame index (exclusive). If None, iterates to end.

        Yields:
            Frame data dictionaries
        """
        if end is None:
            end = self.num_frames

        for i in range(start, end):
            yield self.get_frame(i)

    def close(self) -> None:
        """Close the log file."""
        self.file.close()

    def __repr__(self) -> str:
        return (f"GameplayPlayer(session={self.metadata['session_name']}, "
                f"frames={self.num_frames}, "
                f"duration={self.metadata['duration_seconds']:.1f}s)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
