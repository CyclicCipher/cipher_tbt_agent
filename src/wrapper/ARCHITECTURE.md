# Sensorimotor Wrapper Architecture

## Overview
The sensorimotor wrapper provides the agent with sensory access to the game and motor control over inputs. It operates at the OS level without requiring game modification.

## Module Structure

```
src/wrapper/
├── __init__.py          # Package exports
├── sensory.py           # Screen and audio capture
├── motor.py             # Mouse and keyboard control
├── gaze.py              # Gaze position state management
├── logger.py            # Gameplay recording system
└── wrapper.py           # Main integration class
```

## Component Specifications

### 1. Sensory System (`sensory.py`)

**ScreenCapture Class**
- Captures screen at configurable FPS (target: 20 FPS)
- Provides two-resolution output:
  - Foveal: High-res crop at gaze position (320x320 default)
  - Peripheral: Downsampled full screen (96x96 default)
- Uses `mss` for screen capture
- Window-specific capture when possible (safety feature)
- Performance target: <50ms latency per frame

**AudioCapture Class**
- Captures system audio via WASAPI loopback
- Outputs: 16kHz mono mel spectrogram
- Synchronized with video frames via timestamps
- Buffer management for temporal context (16 chunks default)
- Uses `soundcard` library

### 2. Motor System (`motor.py`)

**MouseController Class**
- Cursor movement via active inference
- Click triggering via threshold
- Coordinate normalization [0,1] → screen pixels
- Safety: Constrain to window bounds (when applicable)
- Uses `pynput.mouse`

**KeyboardController Class**
- Per-key threshold triggering
- Configurable key mappings
- Safety: Emergency kill switch
- Uses `pynput.keyboard`

### 3. Gaze System (`gaze.py`)

**GazeController Class**
- Maintains internal gaze position state [0,1] x [0,1]
- Converts gaze position to foveal crop coordinates
- Provides proprioceptive feedback (current gaze position)
- Saccade control via prediction error (gaze_predicted - gaze_current)

### 4. Logging System (`logger.py`)

**GameplayLogger Class**
- Records all sensory inputs (frames, audio)
- Records all motor outputs (mouse, keyboard)
- Records gaze position over time
- Timestamped entries for synchronization
- Compressed storage format (HDF5 or NPZ)
- Playback capability for debugging

### 5. Integration (`wrapper.py`)

**SensorimotorWrapper Class**
- Coordinates all components
- Main loop: sense → process → act
- Timing management for consistent FPS
- Thread safety for concurrent capture
- Graceful shutdown handling

## Data Flow

```
┌─────────────────┐
│   Game Screen   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────┐
│ ScreenCapture   │◄──────┤ GazeController│
│ (fovea+periph)  │       │  (crop pos)   │
└────────┬────────┘       └──────▲───────┘
         │                       │
         │                  gaze error
         │                       │
         ▼                       │
┌─────────────────┐       ┌─────┴────────┐
│  AudioCapture   │       │              │
│  (mel spectra)  │──────►│   NETWORK    │
└─────────────────┘       │  (to be      │
                          │ implemented)  │
┌─────────────────┐       │              │
│ Proprioception  │──────►│              │
│ (gaze, cursor)  │       └─────┬────────┘
└─────────────────┘             │
                          motor errors
                                │
         ┌──────────────────────┴────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│ MouseController │            │KeyboardController│
│  (moves cursor) │            │  (key presses)  │
└────────┬────────┘            └────────┬────────┘
         │                               │
         └──────────────┬────────────────┘
                        ▼
                 ┌──────────────┐
                 │     Game     │
                 └──────────────┘
```

## Safety Features

1. **Window Constraint**: Mouse/keyboard limited to target window when windowed
2. **Emergency Kill**: Specific key combo stops all control
3. **Rate Limiting**: Prevents runaway motor commands
4. **Logging**: All actions recorded for debugging
5. **Graceful Shutdown**: Cleanup on exit

## Performance Requirements

- **Frame Rate**: 20 FPS minimum (50ms per frame)
- **Latency**: <50ms from screen capture to motor output
- **Memory**: Sensory buffers ~20MB (16 frames foveal+peripheral)
- **CPU Usage**: <30% single core for wrapper overhead

## Testing Strategy

1. **Unit Tests**: Each component tested independently
2. **Performance Tests**: Verify FPS and latency targets
3. **Synchronization Tests**: Audio-video alignment verification
4. **Integration Tests**: Full sensorimotor loop
5. **Safety Tests**: Kill switch, boundary constraints

## Implementation Notes

### Screen Capture Optimization
- Use `mss` for zero-copy screen capture
- Cache monitor info to avoid repeated queries
- Preallocate numpy arrays for frame buffers

### Audio Synchronization
- Timestamp each audio chunk on capture
- Match with nearest video frame timestamp
- Buffer management for smooth streaming

### Thread Safety
- Separate threads for screen/audio capture
- Thread-safe queues for data transfer
- Lock-free reads where possible

### Coordinate Systems
- Gaze: Normalized [0,1] x [0,1] (0,0 = top-left)
- Screen: Absolute pixels
- Fovea: Centered crop around gaze position
- Periphery: Full screen downsampled

## Future Extensions (Deferred)

- Multi-monitor support
- Variable resolution fovea (zoom levels)
- Eye-tracker integration (replace predicted gaze)
- Hardware acceleration for preprocessing
- Distributed capture (separate machine)
