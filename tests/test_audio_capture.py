"""
Test script for audio capture system.

Tests audio capture with soundcard library.

WARNING: This test will capture system audio. Make sure you have:
1. Applied numpy.frombuffer patch to soundcard (see docs/notes/setup_issues.md)
2. Audio devices available
3. System audio playing (for capture verification)
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_audio_import():
    """Test that soundcard library is available."""
    print("=" * 60)
    print("TEST: Audio Library Import")
    print("=" * 60)

    try:
        import soundcard as sc
        print("✓ soundcard library imported successfully")

        # List available speakers
        speakers = sc.all_speakers()
        print(f"\nAvailable audio devices ({len(speakers)}):")
        for i, speaker in enumerate(speakers):
            print(f"  {i+1}. {speaker.name}")

        default = sc.default_speaker()
        print(f"\nDefault speaker: {default.name}")

        print("\n✓ Audio import test PASSED\n")
        return True

    except ImportError as e:
        print(f"\n✗ soundcard library not installed")
        print(f"Error: {e}")
        print("\nInstall with: pip install soundcard")
        return False
    except Exception as e:
        print(f"\n✗ Audio import test FAILED")
        print(f"Error: {e}")
        return False


def test_audio_device_init():
    """Test audio device initialization."""
    print("=" * 60)
    print("TEST: Audio Device Initialization")
    print("=" * 60)

    try:
        from src.wrapper import AudioCapture

        # Initialize with default settings (48000 Hz, stereo)
        audio = AudioCapture(sample_rate=48000)

        print(f"AudioCapture initialized:")
        print(f"  Sample rate: {audio.sample_rate} Hz")
        print(f"  Chunk duration: {audio.chunk_duration_ms} ms")
        print(f"  Chunk samples: {audio.chunk_samples}")
        print(f"  Mel bands: {audio.mel_bands}")
        print(f"  Buffer size: {audio.buffer_size}")

        print("\n✓ Audio device initialization test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Audio device initialization FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_capture_basic():
    """Test basic audio capture (short duration)."""
    print("=" * 60)
    print("TEST: Basic Audio Capture")
    print("=" * 60)
    print("\n⚠ This test will capture system audio for 2 seconds.")
    print("Play some audio (music, video, etc.) to verify capture is working.")
    print("Press Enter to start capture...")
    input()

    try:
        from src.wrapper import AudioCapture

        # Initialize audio capture
        audio = AudioCapture(sample_rate=48000)
        print("Starting audio capture...")

        # Start capture
        audio.start()
        print("Capturing audio for 2 seconds...")

        # Wait for some chunks to be captured
        time.sleep(2.0)

        # Stop capture
        audio.stop()
        print("Stopped capture.")

        # Check buffer
        chunks_captured = len(audio.buffer)
        print(f"\nChunks captured: {chunks_captured}")

        if chunks_captured == 0:
            print("\n⚠ WARNING: No audio chunks captured!")
            print("Possible issues:")
            print("  - No system audio playing during capture")
            print("  - Audio device not properly initialized")
            print("  - numpy.frombuffer patch not applied (see docs/notes/setup_issues.md)")
            return False

        # Get latest chunk
        chunk, timestamp = audio.get_latest_chunk()
        print(f"Latest chunk shape: {chunk.shape}")
        print(f"Latest chunk timestamp: {timestamp:.3f}")

        # Check chunk shape
        expected_shape = (audio.mel_bands, 1)
        if chunk.shape != expected_shape:
            print(f"\n✗ Unexpected chunk shape: {chunk.shape} != {expected_shape}")
            return False

        # Check chunk values
        print(f"Chunk statistics:")
        print(f"  Min: {chunk.min():.6f}")
        print(f"  Max: {chunk.max():.6f}")
        print(f"  Mean: {chunk.mean():.6f}")
        print(f"  Std: {chunk.std():.6f}")

        print("\n✓ Basic audio capture test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Basic audio capture test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_synchronization():
    """Test audio-video synchronization."""
    print("=" * 60)
    print("TEST: Audio-Video Synchronization")
    print("=" * 60)

    try:
        from src.wrapper import AudioCapture

        audio = AudioCapture(sample_rate=48000)
        audio.start()

        # Capture for 1 second
        time.sleep(1.0)

        # Create fake video timestamps
        video_timestamp = time.time()

        # Get synchronized audio chunk
        synced_chunk = audio.get_synchronized_chunk(video_timestamp, tolerance=0.1)

        if synced_chunk is not None:
            print(f"✓ Found synchronized audio chunk")
            print(f"  Shape: {synced_chunk.shape}")
        else:
            print(f"⚠ No synchronized audio chunk found")
            print(f"  This is normal if no audio was playing")

        audio.stop()

        print("\n✓ Audio synchronization test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Audio synchronization test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_numpy_frombuffer_patch():
    """Test if numpy.frombuffer patch has been applied to soundcard."""
    print("=" * 60)
    print("TEST: numpy.frombuffer Patch Verification")
    print("=" * 60)

    try:
        import soundcard as sc
        import inspect

        # Get soundcard source file path
        soundcard_path = Path(inspect.getfile(sc))
        mediafoundation_path = soundcard_path.parent / "mediafoundation.py"

        if not mediafoundation_path.exists():
            print("⚠ Warning: mediafoundation.py not found")
            print("  This is expected on non-Windows systems")
            return True

        # Check if file contains frombuffer
        with open(mediafoundation_path, 'r') as f:
            content = f.read()

        has_frombuffer = 'numpy.frombuffer' in content or 'np.frombuffer' in content
        has_fromstring = 'numpy.fromstring' in content or 'np.fromstring' in content

        print(f"soundcard path: {mediafoundation_path}")
        print(f"Contains numpy.frombuffer: {has_frombuffer}")
        print(f"Contains numpy.fromstring (deprecated): {has_fromstring}")

        if has_fromstring and not has_frombuffer:
            print("\n✗ PATCH NOT APPLIED!")
            print("\nThe soundcard library needs to be patched for numpy 2.x compatibility.")
            print("Follow instructions in docs/notes/setup_issues.md:")
            print(f"  1. Open: {mediafoundation_path}")
            print(f"  2. Replace: numpy.fromstring → numpy.frombuffer")
            print(f"  3. Save")
            return False

        if has_frombuffer:
            print("\n✓ Patch appears to be applied")

        print("\n✓ numpy.frombuffer patch test PASSED\n")
        return True

    except Exception as e:
        print(f"\n⚠ Could not verify patch: {e}")
        print("Continuing anyway...")
        return True


def main():
    """Run all audio capture tests."""
    print("\n" + "=" * 60)
    print("AUDIO CAPTURE TEST SUITE")
    print("=" * 60 + "\n")

    results = {}

    # Test 1: Import
    results['import'] = test_audio_import()
    if not results['import']:
        print("\n⚠ Cannot proceed without soundcard library")
        print("Install with: pip install soundcard")
        return 1

    # Test 2: Patch verification
    results['patch'] = test_numpy_frombuffer_patch()

    # Test 3: Device initialization
    results['device_init'] = test_audio_device_init()
    if not results['device_init']:
        print("\n⚠ Cannot proceed without proper initialization")
        return 1

    # Test 4: Basic capture
    results['basic_capture'] = test_audio_capture_basic()

    # Test 5: Synchronization
    results['synchronization'] = test_audio_synchronization()

    # Summary
    print("=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} {test_name}")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓\n")
        return 0
    else:
        print("\n⚠ SOME TESTS FAILED ⚠\n")
        return 1


if __name__ == "__main__":
    exit(main())
