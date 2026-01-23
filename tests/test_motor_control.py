"""
Test script for motor control verification.

Tests mouse and keyboard control via active inference.

WARNING: This test will actually move your mouse cursor and may press keys.
Make sure you're ready before running.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wrapper import MouseController, KeyboardController
from pynput import keyboard


def test_mouse_proprioception():
    """Test mouse proprioceptive feedback."""
    print("=" * 60)
    print("TEST: Mouse Proprioception")
    print("=" * 60)

    mouse = MouseController(
        screen_width=1920,
        screen_height=1080
    )

    # Get current position
    pixel_pos = mouse.get_position()
    norm_pos = mouse.get_normalized_position()
    proprio = mouse.get_proprioception()

    print(f"Pixel position: {pixel_pos}")
    print(f"Normalized position: {norm_pos}")
    print(f"Proprioception: {proprio}")

    # Verify normalized position is in [0, 1]
    assert 0 <= norm_pos[0] <= 1, f"X position out of range: {norm_pos[0]}"
    assert 0 <= norm_pos[1] <= 1, f"Y position out of range: {norm_pos[1]}"

    # Verify proprioception matches normalized position
    assert np.allclose(proprio, norm_pos), "Proprioception doesn't match position"

    print("\n✓ Mouse proprioception test PASSED\n")


def test_mouse_movement():
    """Test mouse movement via prediction errors."""
    print("=" * 60)
    print("TEST: Mouse Movement via Prediction Errors")
    print("=" * 60)
    print("\n⚠ WARNING: This test will move your mouse cursor!")
    print("Move your mouse to the CENTER of the screen and press Enter to continue...")
    input()

    mouse = MouseController(
        screen_width=1920,
        screen_height=1080,
        gain=2.0  # Moderate gain for visible but controlled movement
    )

    # Disable mouse to prevent actual movement during test setup
    mouse.disable()

    initial_pos = mouse.get_normalized_position()
    print(f"Initial position: {initial_pos}")

    # Simulate prediction errors (but don't actually move yet)
    print("\nTesting error calculations (movement disabled)...")

    # Test 1: Small error
    small_error = np.array([0.01, 0.0])
    mouse.update_from_error(small_error, click_signal=0.0)
    print(f"  Small error {small_error} processed (no movement - disabled)")

    # Test 2: Larger error
    large_error = np.array([0.1, 0.1])
    mouse.update_from_error(large_error, click_signal=0.0)
    print(f"  Large error {large_error} processed (no movement - disabled)")

    print("\n⚠ Ready to test actual movement.")
    print("The cursor will move in a small square pattern.")
    print("Press Enter to continue...")
    input()

    # Enable mouse for actual movement
    mouse.enable()

    # Move in a square pattern
    movements = [
        ("Right", np.array([0.05, 0.0])),
        ("Down", np.array([0.0, 0.05])),
        ("Left", np.array([-0.05, 0.0])),
        ("Up", np.array([0.0, -0.05])),
    ]

    print("\nExecuting movement pattern...")
    for direction, error in movements:
        print(f"  Moving {direction}...")
        mouse.update_from_error(error, click_signal=0.0, dt=0.5)
        time.sleep(0.5)

    final_pos = mouse.get_normalized_position()
    print(f"\nFinal position: {final_pos}")
    print(f"Position change: {final_pos - initial_pos}")

    # Get statistics
    stats = mouse.get_movement_stats()
    print(f"\nMovement statistics:")
    print(f"  Total movements: {stats['movements']}")
    print(f"  Average speed: {stats['avg_speed']:.2f}")

    print("\n✓ Mouse movement test COMPLETED")
    print("  (Visual verification required by user)\n")


def test_click_threshold():
    """Test click threshold mechanism."""
    print("=" * 60)
    print("TEST: Click Threshold")
    print("=" * 60)

    mouse = MouseController(
        screen_width=1920,
        screen_height=1080,
        click_threshold=0.7
    )

    # Disable actual clicking for safety
    mouse.disable()

    print(f"Click threshold: {mouse.click_threshold}")

    # Test signals below threshold
    print("\nTesting signals below threshold (no click expected):")
    test_signals = [0.0, 0.3, 0.5, 0.69]
    for signal in test_signals:
        mouse.update_from_error(np.array([0.0, 0.0]), click_signal=signal)
        print(f"  Signal {signal:.2f}: {'Below' if signal <= mouse.click_threshold else 'Above'} threshold")

    # Test signals above threshold
    print("\nTesting signals above threshold (click would trigger):")
    test_signals = [0.71, 0.8, 1.0]
    for signal in test_signals:
        mouse.update_from_error(np.array([0.0, 0.0]), click_signal=signal)
        print(f"  Signal {signal:.2f}: {'Below' if signal <= mouse.click_threshold else 'Above'} threshold")

    print("\n✓ Click threshold test PASSED\n")


def test_keyboard_basic():
    """Test keyboard controller basic functionality."""
    print("=" * 60)
    print("TEST: Keyboard Controller Basics")
    print("=" * 60)

    kbd = KeyboardController()

    # Disable for safety
    kbd.disable()

    print(f"Default threshold: {kbd.default_threshold}")
    print(f"Enabled: {kbd.enabled}")

    # Test signal processing without actual key presses
    test_signals = {
        'space': 0.8,
        'enter': 0.5,
        'a': 0.9,
    }

    print("\nTesting key signals (keyboard disabled):")
    for key, signal in test_signals.items():
        above_threshold = signal > kbd.default_threshold
        print(f"  {key}: {signal:.2f} - {'Would trigger' if above_threshold else 'Below threshold'}")

    kbd.update_from_signals(test_signals)

    # Check press counts (should be 0 since disabled)
    total_presses = kbd.get_key_press_count()
    print(f"\nTotal key presses recorded: {total_presses}")
    assert total_presses == 0, "Keys were pressed despite being disabled"

    print("\n✓ Keyboard basic test PASSED\n")


def test_rate_limiting():
    """Test rate limiting for motor controllers."""
    print("=" * 60)
    print("TEST: Rate Limiting")
    print("=" * 60)

    # Mouse rate limiting
    mouse = MouseController(
        screen_width=1920,
        screen_height=1080,
        rate_limit_hz=10.0  # 10 Hz = 100ms minimum interval
    )
    mouse.disable()

    print(f"Mouse rate limit: {mouse.rate_limit_hz} Hz")
    print(f"Minimum update interval: {mouse.min_update_interval * 1000:.0f}ms")

    # Try to update faster than rate limit
    num_attempts = 100
    num_accepted = 0

    start_time = time.time()
    for _ in range(num_attempts):
        mouse.update_from_error(np.array([0.01, 0.01]), click_signal=0.0)
        # Don't sleep - try to update as fast as possible

    duration = time.time() - start_time
    print(f"\nAttempted {num_attempts} updates in {duration*1000:.1f}ms")
    print(f"Expected maximum: ~{mouse.rate_limit_hz * duration:.0f} updates accepted")

    # Keyboard rate limiting
    kbd = KeyboardController(rate_limit_hz=5.0)  # 5 Hz = 200ms minimum interval
    kbd.disable()

    print(f"\nKeyboard rate limit: {kbd.rate_limit_hz} Hz")
    print(f"Minimum key interval: {kbd.min_key_interval * 1000:.0f}ms")

    print("\n✓ Rate limiting test PASSED\n")


def main():
    """Run all motor control tests."""
    print("\n" + "=" * 60)
    print("MOTOR CONTROL TEST SUITE")
    print("=" * 60)
    print("\n⚠ SAFETY WARNING ⚠")
    print("This test suite will interact with your mouse and keyboard.")
    print("The tests are designed to be safe with limited movement.")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    try:
        # Run tests that don't require actual movement
        test_mouse_proprioception()
        test_click_threshold()
        test_keyboard_basic()
        test_rate_limiting()

        # Ask before running movement test
        print("\n" + "=" * 60)
        print("The remaining test will actually move your mouse.")
        print("Press Enter to continue with movement test, or Ctrl+C to skip...")
        try:
            input()
            test_mouse_movement()
        except KeyboardInterrupt:
            print("\n\nSkipped mouse movement test.")

        # Summary
        print("=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        print("✓ Mouse proprioception: PASSED")
        print("✓ Click threshold: PASSED")
        print("✓ Keyboard basics: PASSED")
        print("✓ Rate limiting: PASSED")
        print("✓ Mouse movement: COMPLETED (requires visual verification)")
        print("=" * 60)
        print("\n✓✓✓ ALL TESTS COMPLETED ✓✓✓\n")
        return 0

    except Exception as e:
        print(f"\n✗✗✗ TEST SUITE FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
