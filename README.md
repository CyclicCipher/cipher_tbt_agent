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

### Phase 1: Sensorimotor Wrapper — DONE
- Screen capture (foveal + peripheral), motor control, gameplay logging
- **Performance: 29 FPS @ 34ms latency** (exceeds targets)

### Phase 2: Baseline Predictive Coding — DONE
- Standard PC network: 7 layers, 95.14% MNIST test accuracy
- See `src/network/` and `train_mnist_pc.py`

### Phase 3: Bayesian Predictive Coding (BPC) — DONE
- Matrix Normal Wishart weight posteriors with Hebbian closed-form updates
- 4 layers, 93.5% MNIST test accuracy (1 epoch)
- See `experiments/BayesianPC/`

### Phase 4: Error-based Bayesian Predictive Coding (eBPC) — DONE
- Combines ePC error reparameterization with BPC Bayesian weight updates
- 4 layers, **95.74% MNIST test accuracy** (3 epochs) — exceeds both baselines
- See `experiments/eBPC/`

### Phase 5: eBPC-ResNet with Optimizations — IN PROGRESS
- Diagonal V/Psi approximation (6x parameter reduction) — debugging NaN
- ResNet-18 with ePC skip connections — pending
- bfloat16 mixed precision, adaptive T — pending
- CIFAR-10 target: ePC's 92.17%
- See `experiments/eBPC_ResNet/`

### Future Phases
- JEPA integration
- Network-wrapper integration for game interaction
- Experimental comparison Models A, B, C

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

- `MISTAKES.md` - Catalogued mistakes and lessons learned (critical reference)
- `experiments/eBPC_ResNet/TODO.md` - Current roadmap (Phase 1 core + Phase 2 research)
- `src/wrapper/ARCHITECTURE.md` - Sensorimotor wrapper architecture

## Project Structure

```
predictive-coding-agent/
├── experiments/
│   ├── BayesianPC/       # BPC implementation (93.5% MNIST)
│   ├── eBPC/             # eBPC implementation (95.74% MNIST)
│   └── eBPC_ResNet/      # Diagonal eBPC + ResNet (in progress)
├── src/
│   ├── network/          # Baseline PC network (95.14% MNIST)
│   ├── active_inference/ # Active inference module
│   ├── control/          # Hotkey control
│   └── wrapper/          # Sensorimotor wrapper (29 FPS)
├── tests/
├── MISTAKES.md           # Lessons learned (17 documented mistakes)
└── README.md
```

## Development Phases

1. **Sensorimotor Wrapper** — DONE
2. **Baseline PC** — DONE (95.14% MNIST)
3. **Bayesian PC (BPC)** — DONE (93.5% MNIST)
4. **Error-based BPC (eBPC)** — DONE (95.74% MNIST)
5. **eBPC-ResNet + Optimizations** — IN PROGRESS
6. **JEPA Integration** — Planned
7. **Network-Wrapper Integration** — Planned
8. **Experimental Comparison (Models A, B, C)** — Planned

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
