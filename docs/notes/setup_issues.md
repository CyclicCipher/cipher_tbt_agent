# Setup Issues and Fixes

This document tracks issues encountered during setup and their solutions.

---

## Audio Capture: numpy/soundcard Incompatibility

**Date:** January 2026

**Problem:** 
The `soundcard` library uses `numpy.fromstring()` which was removed in numpy 2.0. This causes the error:
```
ValueError: The binary mode of fromstring is removed, use frombuffer instead
```

**Environment:**
- Python 3.11.9
- numpy 2.x (installed with PyTorch)
- soundcard (latest)

**Solution:**
Edit the soundcard library directly:

1. Open: `venv\Lib\site-packages\soundcard\mediafoundation.py`
2. Find and Replace: `numpy.fromstring` → `numpy.frombuffer`
3. Save

**Note:** This fix must be reapplied if soundcard is reinstalled or the venv is recreated.

**Alternative Solutions Attempted (Failed):**
- `sounddevice` library: Sample rate mismatch issues with Stereo Mix
- `pyaudio` with WASAPI loopback: Standard pip version lacks loopback support
- Downgrading numpy: Conflicts with other packages

---

## Stereo Mix Configuration

**Observation:**
Stereo Mix is available and shows audio levels (green bars visible in Windows Sound settings), but has driver-level compatibility issues with some audio APIs.

The soundcard library with WASAPI loopback mode bypasses Stereo Mix entirely and captures directly from the output device, which is more reliable.

**Audio Device Used:**
- Device: Headphones (Realtek(R) Audio)
- Sample Rate: 48000 Hz
- Channels: 2 (stereo)

---

## Future Considerations

- If recreating venv, remember to patch soundcard
- Consider forking soundcard with the fix if this becomes a repeated issue
- Monitor for soundcard library updates that may fix this upstream
```