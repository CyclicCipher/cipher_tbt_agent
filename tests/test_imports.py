"""
Test that all imports work correctly (headless-safe).

This test can run in environments without X display.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("IMPORT TEST (Headless-Safe)")
print("=" * 60)

try:
    print("\nTesting basic imports...")
    import numpy as np
    import cv2
    import h5py
    import yaml
    print("✓ numpy, cv2, h5py, yaml")

    print("\nTesting wrapper imports (excluding pynput-dependent modules)...")
    from src.wrapper import gaze
    print("✓ gaze module")

    from src.wrapper import sensory
    print("✓ sensory module")

    from src.wrapper import logger
    print("✓ logger module")

    # Test individual classes without initializing (to avoid X server requirement)
    print("\nTesting class definitions...")
    from src.wrapper.gaze import GazeController
    print("✓ GazeController class")

    from src.wrapper.sensory import ScreenCapture, AudioCapture
    print("✓ ScreenCapture, AudioCapture classes")

    from src.wrapper.logger import GameplayLogger, GameplayPlayer
    print("✓ GameplayLogger, GameplayPlayer classes")

    print("\nNote: motor and wrapper modules require X display (pynput)")
    print("Skipping motor.py and wrapper.py imports in headless environment")

    print("\n" + "=" * 60)
    print("IMPORT TEST: PASSED")
    print("=" * 60)
    print("\nAll imports successful in headless environment.")
    print("Full tests require graphical environment (X server on Linux, GUI on Windows)")

    sys.exit(0)

except Exception as e:
    print(f"\n✗ IMPORT TEST FAILED")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
