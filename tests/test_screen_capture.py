"""
Test script for screen capture performance.

Verifies that the screen capture system meets performance requirements:
- Latency < 50ms per frame
- FPS >= 20 frames per second
- Proper foveal and peripheral vision output
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wrapper import GazeController, ScreenCapture


def test_screen_capture_basic():
    """Test basic screen capture functionality."""
    print("=" * 60)
    print("TEST: Basic Screen Capture")
    print("=" * 60)

    # Initialize
    gaze = GazeController(
        screen_width=1920,
        screen_height=1080,
        fovea_size=320
    )
    screen = ScreenCapture(gaze_controller=gaze)

    print(f"Screen dimensions: {screen.screen_width}x{screen.screen_height}")
    print(f"Fovea size: {screen.fovea_size}x{screen.fovea_size}")
    print(f"Periphery size: {screen.periphery_size}x{screen.periphery_size}")

    # Capture one frame
    foveal, peripheral, timestamp = screen.capture()

    # Verify shapes
    assert foveal.shape == (screen.fovea_size, screen.fovea_size, 3), \
        f"Foveal shape mismatch: {foveal.shape}"
    assert peripheral.shape == (screen.periphery_size, screen.periphery_size, 3), \
        f"Peripheral shape mismatch: {peripheral.shape}"

    # Verify value range (should be normalized to [0, 1])
    assert foveal.min() >= 0 and foveal.max() <= 1, \
        f"Foveal values out of range: [{foveal.min()}, {foveal.max()}]"
    assert peripheral.min() >= 0 and peripheral.max() <= 1, \
        f"Peripheral values out of range: [{peripheral.min()}, {peripheral.max()}]"

    print(f"✓ Foveal shape: {foveal.shape}")
    print(f"✓ Peripheral shape: {peripheral.shape}")
    print(f"✓ Value ranges correct")

    screen.cleanup()
    print("\n✓ Basic screen capture test PASSED\n")


def test_screen_capture_performance():
    """Test screen capture performance (latency and FPS)."""
    print("=" * 60)
    print("TEST: Screen Capture Performance")
    print("=" * 60)

    # Initialize
    gaze = GazeController(
        screen_width=1920,
        screen_height=1080,
        fovea_size=320
    )
    screen = ScreenCapture(gaze_controller=gaze)

    # Warm-up captures
    print("Warming up...")
    for _ in range(10):
        screen.capture()

    # Performance test
    num_frames = 100
    print(f"Capturing {num_frames} frames...")

    start_time = time.time()
    latencies = []

    for i in range(num_frames):
        frame_start = time.time()
        foveal, peripheral, timestamp = screen.capture()
        frame_end = time.time()

        latency_ms = (frame_end - frame_start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_frames} frames")

    end_time = time.time()
    duration = end_time - start_time

    # Calculate statistics
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    std_latency = np.std(latencies)
    achieved_fps = num_frames / duration

    # Report results
    print("\n" + "-" * 60)
    print("RESULTS:")
    print("-" * 60)
    print(f"Total duration: {duration:.2f}s")
    print(f"Frames captured: {num_frames}")
    print(f"Achieved FPS: {achieved_fps:.2f}")
    print(f"\nLatency Statistics:")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")

    # Check requirements
    print("\n" + "-" * 60)
    print("REQUIREMENT CHECKS:")
    print("-" * 60)

    latency_ok = avg_latency < 50.0
    fps_ok = achieved_fps >= 20.0

    print(f"{'✓' if latency_ok else '✗'} Latency < 50ms: {avg_latency:.2f}ms (target: <50ms)")
    print(f"{'✓' if fps_ok else '✗'} FPS >= 20: {achieved_fps:.2f} (target: >=20)")

    screen.cleanup()

    if latency_ok and fps_ok:
        print("\n✓ Performance test PASSED\n")
    else:
        print("\n✗ Performance test FAILED\n")
        if not latency_ok:
            print(f"  - Latency too high: {avg_latency:.2f}ms > 50ms")
        if not fps_ok:
            print(f"  - FPS too low: {achieved_fps:.2f} < 20")

    return latency_ok and fps_ok


def test_gaze_control():
    """Test gaze control and foveal crop positioning."""
    print("=" * 60)
    print("TEST: Gaze Control and Foveal Positioning")
    print("=" * 60)

    # Initialize
    gaze = GazeController(
        screen_width=1920,
        screen_height=1080,
        fovea_size=320
    )
    screen = ScreenCapture(gaze_controller=gaze)

    # Test different gaze positions
    test_positions = [
        (0.5, 0.5),  # Center
        (0.25, 0.25),  # Top-left quadrant
        (0.75, 0.75),  # Bottom-right quadrant
        (0.1, 0.5),  # Left edge
        (0.9, 0.5),  # Right edge
    ]

    print("Testing gaze positions:")
    for x, y in test_positions:
        gaze.set_position((x, y))
        foveal, peripheral, _ = screen.capture()

        # Verify capture succeeds and has correct shape
        assert foveal.shape == (screen.fovea_size, screen.fovea_size, 3)

        # Get crop bounds
        left, top, right, bottom = gaze.get_foveal_crop_bounds()
        crop_width = right - left
        crop_height = bottom - top

        print(f"  Position ({x:.2f}, {y:.2f}): "
              f"crop=({left}, {top}, {right}, {bottom}), "
              f"size={crop_width}x{crop_height}")

        # Verify crop is reasonable
        assert crop_width > 0 and crop_height > 0, "Crop has zero size"

    print("\n✓ Gaze control test PASSED\n")
    screen.cleanup()


def test_gaze_movement():
    """Test gaze movement via prediction errors."""
    print("=" * 60)
    print("TEST: Gaze Movement via Prediction Errors")
    print("=" * 60)

    gaze = GazeController(
        screen_width=1920,
        screen_height=1080,
        fovea_size=320,
        gain=1.0  # Lower gain for controlled movement
    )

    initial_pos = gaze.get_position()
    print(f"Initial position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")

    # Simulate prediction error: network predicts gaze should move right
    gaze_error = np.array([0.1, 0.0])  # Move right
    gaze.update_from_error(gaze_error, dt=1.0)

    new_pos = gaze.get_position()
    print(f"After right movement: ({new_pos[0]:.3f}, {new_pos[1]:.3f})")

    assert new_pos[0] > initial_pos[0], "Gaze did not move right"
    assert abs(new_pos[1] - initial_pos[1]) < 0.01, "Gaze moved vertically unexpectedly"

    # Move down
    gaze_error = np.array([0.0, 0.1])
    gaze.update_from_error(gaze_error, dt=1.0)

    final_pos = gaze.get_position()
    print(f"After down movement: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")

    assert final_pos[1] > new_pos[1], "Gaze did not move down"

    print("\n✓ Gaze movement test PASSED\n")


def main():
    """Run all screen capture tests."""
    print("\n" + "=" * 60)
    print("SCREEN CAPTURE TEST SUITE")
    print("=" * 60 + "\n")

    try:
        # Run tests
        test_screen_capture_basic()
        test_gaze_control()
        test_gaze_movement()
        performance_ok = test_screen_capture_performance()

        # Summary
        print("=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        print("✓ Basic functionality: PASSED")
        print("✓ Gaze control: PASSED")
        print("✓ Gaze movement: PASSED")
        print(f"{'✓' if performance_ok else '✗'} Performance: {'PASSED' if performance_ok else 'FAILED'}")
        print("=" * 60)

        if performance_ok:
            print("\n✓✓✓ ALL TESTS PASSED ✓✓✓\n")
            return 0
        else:
            print("\n⚠ SOME TESTS FAILED ⚠\n")
            return 1

    except Exception as e:
        print(f"\n✗✗✗ TEST SUITE FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
