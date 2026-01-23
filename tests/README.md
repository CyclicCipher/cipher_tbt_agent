# Test Suite for Predictive Coding Agent

This directory contains test scripts for the sensorimotor wrapper and related components.

## Test Files

### `test_screen_capture.py`
Tests the screen capture system (foveal + peripheral vision).

**Tests:**
- Basic screen capture functionality
- Gaze control and foveal positioning
- Gaze movement via prediction errors
- Performance (latency and FPS)

**Requirements:**
- Latency < 50ms per frame
- FPS >= 20 frames per second

**Usage:**
```bash
python tests/test_screen_capture.py
```

### `test_motor_control.py`
Tests mouse and keyboard control via active inference.

**Tests:**
- Mouse proprioception
- Click threshold mechanism
- Keyboard controller basics
- Rate limiting
- Mouse movement (optional, requires user confirmation)

**⚠ Safety Warning:** This test will move your mouse cursor. The test is designed to be safe with limited movement, but be prepared before running.

**Usage:**
```bash
python tests/test_motor_control.py
```

### `test_integration.py`
Integration test for the complete sensorimotor loop.

**Tests:**
- Wrapper initialization
- Single sensorimotor step
- Loop with mock network
- Gameplay logging
- Sustained performance

**Usage:**
```bash
python tests/test_integration.py
```

## Running All Tests

To run all tests sequentially:

```bash
# Screen capture (safe)
python tests/test_screen_capture.py

# Motor control (will ask before moving mouse)
python tests/test_motor_control.py

# Integration (safe, motor control disabled)
python tests/test_integration.py
```

## Test Safety

All tests are designed with safety in mind:

1. **Motor control disabled by default** in integration tests
2. **User confirmation required** before actual mouse movement
3. **Rate limiting** prevents runaway commands
4. **Emergency stop** system (Ctrl+Shift+Esc) available during operation
5. **Bounded movements** in movement tests

## Expected Results

### Screen Capture
- **Latency:** < 50ms average
- **FPS:** >= 20 (typically 30-60 depending on hardware)
- All shape and value range checks should pass

### Motor Control
- Proprioception should return normalized coordinates [0, 1]
- Click threshold should correctly filter signals
- Rate limiting should prevent excessive updates
- Movement test requires visual verification by user

### Integration
- All component initialization checks should pass
- Mock network should run for 20 frames successfully
- Logging should create valid HDF5 files
- Sustained performance should maintain target FPS

## Troubleshooting

### Screen Capture Issues
- **Low FPS:** Check system load, close unnecessary applications
- **High latency:** Check for other screen recording software
- **Permission errors:** On some systems, screen capture requires elevated privileges

### Motor Control Issues
- **Mouse not moving:** Check if game/fullscreen app has focus
- **Permission errors:** On Linux, may need to be in input group
- **Rate limiting too aggressive:** Adjust `rate_limit_hz` parameter

### Integration Issues
- **Import errors:** Ensure you're running from project root
- **Config not found:** Check that `configs/default.yaml` exists
- **Memory errors:** Reduce buffer sizes if running on low-memory systems

## Future Tests

Tests still to be implemented:

- [ ] Audio capture and synchronization
- [ ] Network integration with actual predictive coding network
- [ ] Long-duration stability test (hours)
- [ ] Multi-monitor support
- [ ] Error recovery and failsafe mechanisms

## Performance Benchmarks

Expected performance on target hardware (RTX 3050 Ti Laptop, 4GB VRAM):

| Metric | Target | Expected |
|--------|--------|----------|
| Screen capture FPS | >= 20 | 30-60 |
| Frame latency | < 50ms | 15-30ms |
| Memory usage (wrapper) | < 200MB | 150-180MB |
| CPU usage (single core) | < 30% | 10-25% |

## Notes

- Tests are designed to run without requiring the full network implementation
- Mock network in integration test demonstrates the expected interface
- All tests clean up resources properly (close files, stop threads)
- Logs from integration tests are saved to `logs/test/`
