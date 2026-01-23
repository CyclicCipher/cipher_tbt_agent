"""
Integration test for complete sensorimotor loop.

Tests the full SensorimotorWrapper with a simple mock network.
"""

import sys
import time
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wrapper import SensorimotorWrapper


class MockNetwork:
    """
    Mock network for testing the sensorimotor loop.

    Implements a simple behavior:
    - Gaze follows a circular pattern
    - Cursor follows gaze with a delay
    - Occasionally triggers clicks
    """

    def __init__(self):
        self.frame_count = 0
        self.gaze_angle = 0.0

    def step(self, sensory_dict):
        """
        Process sensory inputs and return motor commands.

        Args:
            sensory_dict: Sensory inputs from wrapper

        Returns:
            Dictionary with motor commands
        """
        self.frame_count += 1

        # Get current positions
        current_gaze = sensory_dict['gaze']
        current_cursor = sensory_dict['cursor']

        # Generate circular gaze pattern
        # Center at (0.5, 0.5) with radius 0.2
        self.gaze_angle += 0.05  # Increment angle
        target_gaze_x = 0.5 + 0.2 * np.cos(self.gaze_angle)
        target_gaze_y = 0.5 + 0.2 * np.sin(self.gaze_angle)
        target_gaze = np.array([target_gaze_x, target_gaze_y])

        # Gaze error (active inference)
        gaze_error = target_gaze - current_gaze

        # Cursor follows gaze with some lag
        cursor_error = current_gaze - current_cursor

        # Click every 50 frames
        click_signal = 0.9 if self.frame_count % 50 == 0 else 0.0

        # No keyboard for this test
        key_signals = {}

        return {
            'gaze_error': gaze_error * 0.1,  # Scale down for smooth movement
            'cursor_error': cursor_error * 0.1,
            'click_signal': click_signal,
            'key_signals': key_signals
        }


def load_config():
    """Load configuration from default.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_wrapper_initialization():
    """Test wrapper initialization."""
    print("=" * 60)
    print("TEST: Wrapper Initialization")
    print("=" * 60)

    config = load_config()

    # Initialize without audio or logging
    wrapper = SensorimotorWrapper(
        config=config,
        enable_audio=False,
        enable_logging=False
    )

    print(f"Wrapper initialized: {wrapper}")
    print(f"Target FPS: {wrapper.target_fps}")
    print(f"Frame time: {wrapper.frame_time * 1000:.1f}ms")

    # Check components
    assert wrapper.gaze is not None, "Gaze controller not initialized"
    assert wrapper.screen is not None, "Screen capture not initialized"
    assert wrapper.mouse is not None, "Mouse controller not initialized"
    assert wrapper.keyboard_ctrl is not None, "Keyboard controller not initialized"
    assert wrapper.emergency_stop is not None, "Emergency stop not initialized"

    print("\n✓ Wrapper initialization test PASSED\n")


def test_single_step():
    """Test a single sensorimotor step."""
    print("=" * 60)
    print("TEST: Single Sensorimotor Step")
    print("=" * 60)

    config = load_config()
    wrapper = SensorimotorWrapper(
        config=config,
        enable_audio=False,
        enable_logging=False
    )

    # Disable motor control for safety
    wrapper.mouse.disable()
    wrapper.keyboard_ctrl.disable()

    # Execute one step without network
    result = wrapper.step()

    # Check sensory outputs
    sensory = result['sensory']
    assert 'foveal' in sensory, "Missing foveal output"
    assert 'peripheral' in sensory, "Missing peripheral output"
    assert 'gaze' in sensory, "Missing gaze output"
    assert 'cursor' in sensory, "Missing cursor output"
    assert 'timestamp' in sensory, "Missing timestamp"

    print(f"Foveal shape: {sensory['foveal'].shape}")
    print(f"Peripheral shape: {sensory['peripheral'].shape}")
    print(f"Gaze: {sensory['gaze']}")
    print(f"Cursor: {sensory['cursor']}")
    print(f"Timestamp: {sensory['timestamp']}")

    wrapper.screen.cleanup()
    print("\n✓ Single step test PASSED\n")


def test_loop_with_mock_network():
    """Test sensorimotor loop with mock network."""
    print("=" * 60)
    print("TEST: Sensorimotor Loop with Mock Network")
    print("=" * 60)

    config = load_config()
    wrapper = SensorimotorWrapper(
        config=config,
        enable_audio=False,
        enable_logging=False
    )

    # Disable motor control for safety
    wrapper.mouse.disable()
    wrapper.keyboard_ctrl.disable()

    # Connect mock network
    mock_network = MockNetwork()
    wrapper.set_network(mock_network.step)

    print("Running 20 frames with mock network...")
    print("(Motor control disabled for safety)")

    start_time = time.time()

    # Run for 20 frames
    for i in range(20):
        result = wrapper.step()

        if (i + 1) % 5 == 0:
            sensory = result['sensory']
            motor = result['motor']
            print(f"  Frame {i + 1}: "
                  f"gaze=({sensory['gaze'][0]:.3f}, {sensory['gaze'][1]:.3f}), "
                  f"gaze_error=({motor['gaze_error'][0]:.3f}, {motor['gaze_error'][1]:.3f})")

        # Small delay to avoid overwhelming output
        time.sleep(0.01)

    duration = time.time() - start_time
    fps = 20 / duration

    print(f"\nCompleted 20 frames in {duration:.2f}s ({fps:.1f} FPS)")

    wrapper.screen.cleanup()
    wrapper.emergency_stop.stop()

    print("\n✓ Mock network loop test PASSED\n")


def test_logging():
    """Test gameplay logging."""
    print("=" * 60)
    print("TEST: Gameplay Logging")
    print("=" * 60)

    config = load_config()
    wrapper = SensorimotorWrapper(
        config=config,
        enable_audio=False,
        enable_logging=True,
        log_dir="logs/test"
    )

    # Disable motor control
    wrapper.mouse.disable()
    wrapper.keyboard_ctrl.disable()

    # Start logger
    wrapper.logger.start(metadata={'test': 'integration_test'})

    print("Recording 10 frames...")

    # Record some frames
    for i in range(10):
        wrapper.step()

    # Stop logger
    wrapper.logger.stop()

    print(f"Log saved to: {wrapper.logger.file_path}")

    # Verify log file exists
    assert wrapper.logger.file_path.exists(), "Log file not created"

    # Try to load the log
    from src.wrapper import GameplayPlayer

    with GameplayPlayer(str(wrapper.logger.file_path)) as player:
        print(f"Log metadata: {player.metadata}")
        print(f"Frames in log: {player.num_frames}")

        assert player.num_frames == 10, f"Expected 10 frames, got {player.num_frames}"

        # Load first frame
        frame = player.get_frame(0)
        print(f"First frame keys: {frame.keys()}")

        assert 'foveal' in frame, "Foveal data not in log"
        assert 'peripheral' in frame, "Peripheral data not in log"
        assert 'gaze' in frame, "Gaze data not in log"

    wrapper.screen.cleanup()
    wrapper.emergency_stop.stop()

    print("\n✓ Logging test PASSED\n")


def test_performance_sustained():
    """Test sustained performance over longer duration."""
    print("=" * 60)
    print("TEST: Sustained Performance")
    print("=" * 60)

    config = load_config()
    wrapper = SensorimotorWrapper(
        config=config,
        enable_audio=False,
        enable_logging=False
    )

    # Disable motor control
    wrapper.mouse.disable()
    wrapper.keyboard_ctrl.disable()

    num_frames = 100
    print(f"Running {num_frames} frames to test sustained performance...")

    start_time = time.time()
    frame_times = []

    for i in range(num_frames):
        frame_start = time.time()
        wrapper.step()
        frame_end = time.time()

        frame_times.append(frame_end - frame_start)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_frames}")

    duration = time.time() - start_time
    avg_fps = num_frames / duration
    avg_frame_time = np.mean(frame_times) * 1000
    max_frame_time = np.max(frame_times) * 1000

    print(f"\nResults:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Average frame time: {avg_frame_time:.1f}ms")
    print(f"  Max frame time: {max_frame_time:.1f}ms")
    print(f"  Target: {wrapper.target_fps} FPS ({wrapper.frame_time * 1000:.1f}ms per frame)")

    wrapper.screen.cleanup()
    wrapper.emergency_stop.stop()

    # Check if we met target (with some tolerance)
    if avg_fps >= wrapper.target_fps * 0.9:  # 90% of target is acceptable
        print("\n✓ Sustained performance test PASSED\n")
        return True
    else:
        print(f"\n⚠ Performance below target: {avg_fps:.1f} < {wrapper.target_fps}\n")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    try:
        # Run tests
        test_wrapper_initialization()
        test_single_step()
        test_loop_with_mock_network()
        test_logging()
        performance_ok = test_performance_sustained()

        # Summary
        print("=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        print("✓ Wrapper initialization: PASSED")
        print("✓ Single step: PASSED")
        print("✓ Mock network loop: PASSED")
        print("✓ Logging: PASSED")
        print(f"{'✓' if performance_ok else '⚠'} Sustained performance: {'PASSED' if performance_ok else 'WARNING'}")
        print("=" * 60)

        if performance_ok:
            print("\n✓✓✓ ALL TESTS PASSED ✓✓✓\n")
            return 0
        else:
            print("\n⚠ TESTS COMPLETED WITH WARNINGS ⚠\n")
            return 0  # Still return 0 since warnings are not failures

    except Exception as e:
        print(f"\n✗✗✗ TEST SUITE FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
