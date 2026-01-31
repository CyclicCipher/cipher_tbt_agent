# Predictive Coding Agent

A biologically-inspired AI system based on predictive coding principles for learning to play Danganronpa: Trigger Happy Havoc.

## Project Overview

This project investigates data efficiency and optimal training curricula for machine learning by comparing three experimental conditions:

- **Model A:** No pretraining - learns purely from game interaction
- **Model B:** Multimodal text pretraining → game exposure
- **Model C:** Game exposure → text pretraining → game exposure

The goal is to understand how the order of training experiences affects learning efficiency and capability development.

## Architecture

### Core Components

1. **Layered Backbone Network** (10-12 layers, ~1500-2000 neurons/layer)
   - Two-compartment neurons (apical/basal) with temporal convolution
   - Block tridiagonal structure for efficient prospective learning
   - Sparse overlay connections (2-5%) for long-range integration

2. **Foveal Vision System**
   - High-resolution fovea (320×320) at gaze position
   - Low-resolution periphery (96×96) for context
   - Active gaze control via prediction errors

3. **Hippocampal Sub-Network**
   - Salience-triggered episodic memory
   - Cue-based retrieval with pattern completion
   - Consolidation via simulated sleep/replay

4. **Active Inference Motor Control**
   - Gaze, cursor, and keyboard control
   - Motor outputs emerge from proprioceptive prediction errors
   - Emergency stop system for safety

5. **Sensorimotor Wrapper**
   - Screen capture (20+ FPS, <50ms latency)
   - Audio capture (future implementation)
   - Gameplay logging for analysis

## Hardware Requirements

- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop (4GB VRAM) or better
- **RAM:** 8GB+ system RAM
- **OS:** Linux (tested) or Windows
- **Python:** 3.8+

## Installation

```bash
# Clone repository
git clone https://github.com/CyclicCipher/predictive-coding-agent.git
cd predictive-coding-agent

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Configuration

Edit `configs/default.yaml` to customize:

- Network architecture (layers, neurons, temporal windows)
- Vision parameters (fovea size, periphery size)
- Learning rates (awake vs. consolidation)
- Hippocampus parameters (buffer size, salience threshold)
- Motor control gains

Key settings:
```yaml
device: "cuda"
dtype: "float16"  # FP16 for 4GB VRAM efficiency

network:
  num_layers: 10
  neurons_per_layer: 1500

vision:
  fovea_size: 320
  periphery_size: 96

hippocampus:
  buffer_size: 2000
  latent_dim: 512
```

## Current Status

### ✅ Phase 1 Complete: Sensorimotor Wrapper

- Gaze position control system
- Screen capture (foveal + peripheral vision)
- Window-specific capture (safety feature)
- Audio capture with soundcard integration
- Motor control (mouse + keyboard + emergency stop)
- Gameplay logging and replay (HDF5 format)
- Comprehensive test suite
- **Performance: 29 FPS @ 34ms latency** (exceeds 20 FPS / 50ms targets)

### 📋 Phase 2: Minimal Viable Network (Next)

- 5-layer predictive coding backbone
- Two-compartment neurons with temporal processing
- Prospective learning implementation
- Network-wrapper integration

### 📋 Future Phases

- Hippocampal memory system (Phase 5)
- Text pretraining pipeline (Phase 6)
- Experimental comparison Models A, B, C (Phase 7)

## Testing

The project includes a comprehensive test suite:

```bash
# Test screen capture performance
python tests/test_screen_capture.py

# Test motor control (will ask before moving mouse)
python tests/test_motor_control.py

# Test complete integration
python tests/test_integration.py
```

See `tests/README.md` for detailed test documentation.

## Usage Examples

### Basic Screen Capture

```python
from src.wrapper import GazeController, ScreenCapture

# Initialize
gaze = GazeController(screen_width=1920, screen_height=1080, fovea_size=320)
screen = ScreenCapture(gaze_controller=gaze)

# Capture frame
foveal, peripheral, timestamp = screen.capture()
print(f"Foveal: {foveal.shape}, Peripheral: {peripheral.shape}")
```

### Window-Specific Capture (Safety Feature)

```python
# Capture only a specific window (for non-fullscreen apps)
screen = ScreenCapture(
    gaze_controller=gaze,
    window_title="My Application"  # Partial match supported
)
# Falls back to full screen if window not found
```

### Sensorimotor Loop with Network

```python
import yaml
from src.wrapper import SensorimotorWrapper

# Load config
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# Initialize wrapper
wrapper = SensorimotorWrapper(config, enable_logging=True)

# Define network step function
def network_step(sensory_dict):
    # Process sensory inputs
    foveal = sensory_dict['foveal']
    peripheral = sensory_dict['peripheral']
    gaze = sensory_dict['gaze']
    cursor = sensory_dict['cursor']

    # Return motor commands (prediction errors)
    return {
        'gaze_error': np.array([0.0, 0.0]),
        'cursor_error': np.array([0.0, 0.0]),
        'click_signal': 0.0,
        'key_signals': {}
    }

# Connect network and run
wrapper.set_network(network_step)
wrapper.start(duration=10.0)  # Run for 10 seconds
```

### Replay Logged Session

```python
from src.wrapper import GameplayPlayer

with GameplayPlayer('logs/gameplay/session_20260123_120000.h5') as player:
    print(f"Session: {player.metadata['session_name']}")
    print(f"Duration: {player.metadata['duration_seconds']:.1f}s")
    print(f"Frames: {player.num_frames}")

    # Iterate through frames
    for frame_data in player.iter_frames():
        foveal = frame_data['foveal']
        gaze = frame_data['gaze']
        # Process frame...
```

## Safety Features

- **Emergency Stop:** Press `Ctrl+Shift+Esc` to immediately disable all motor control
- **Rate Limiting:** Prevents runaway motor commands
- **Coordinate Constraints:** Mouse movement bounded to screen/window
- **Logging:** All actions recorded for debugging
- **Disable by Default:** Motor control can be disabled during testing

## Documentation

- `docs/planning/` - Project planning and specifications
- `docs/notes/` - Development notes and issues
- `src/wrapper/ARCHITECTURE.md` - Detailed wrapper architecture
- `tests/README.md` - Test suite documentation

## Project Structure

```
predictive-coding-agent/
├── configs/              # Configuration files
│   └── default.yaml
├── docs/                 # Documentation
│   ├── notes/
│   └── planning/
├── logs/                 # Gameplay logs (created at runtime)
├── src/                  # Source code
│   ├── control/          # Control interface (hotkeys)
│   ├── hippocampus/      # Episodic memory (future)
│   ├── network/          # Neural network (future)
│   ├── pretraining/      # Text pretraining (future)
│   └── wrapper/          # Sensorimotor wrapper ✅
│       ├── gaze.py
│       ├── sensory.py
│       ├── motor.py
│       ├── logger.py
│       └── wrapper.py
├── tests/                # Test suite
│   ├── test_screen_capture.py
│   ├── test_motor_control.py
│   └── test_integration.py
├── requirements.txt
└── README.md
```

## Development Phases

**Phase 1: Sensorimotor Wrapper** ✅ (Current)
- Screen and audio capture
- Motor control
- Logging system

**Phase 2: Minimal Viable Network** (Next)
- 5-layer backbone
- Simple two-compartment neurons
- Prospective learning

**Phase 3: Temporal and Multimodal** (Week 6-9)
- Temporal convolution
- Multimodal integration

**Phase 4: Motor Integration** (Week 10-12)
- Active inference motor control
- Gaze and click learning

**Phase 5: Hippocampus** (Week 13-17)
- Episodic memory
- Sparse overlay connections

**Phase 6: Text Pretraining** (Week 18-22)
- Semantic, glyph, and phonetic grounding

**Phase 7: Experiments** (Week 23+)
- Train Models A, B, C
- Comparative analysis

## Contributing

This is a research project. Contributions are welcome, especially for:

- Performance optimization
- Additional test coverage
- Documentation improvements
- Bug fixes

## License

[To be determined]

## Acknowledgments

Built with:
- PyTorch (deep learning framework)
- mss (screen capture)
- pynput (input control)
- h5py (data logging)
- OpenCV (image processing)

## Contact

For questions or issues, please use the GitHub issue tracker.
