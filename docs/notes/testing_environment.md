# Testing Environment Requirements

## Issue: Headless vs. Graphical Environments

The sensorimotor wrapper tests **require a graphical environment** because they interact with the display and input devices. Running tests in headless environments (like WSL without X server, Docker containers, or SSH sessions) will fail.

## Error Symptoms

If you see this error:
```
ImportError: this platform is not supported: ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
```

You're running in a headless environment without display access.

## Solutions

### Option 1: Run Tests in Windows Native Environment (Recommended)

Since you're developing on Windows, run the tests directly in Windows PowerShell or CMD, not in WSL or a Linux container.

```powershell
# In Windows PowerShell/CMD:
cd C:\path\to\predictive-coding-agent
python tests\test_screen_capture.py
python tests\test_motor_control.py
python tests\test_integration.py
```

### Option 2: WSL with X Server

If using WSL, you need to set up an X server:

1. Install VcXsrv or Xming on Windows
2. Start the X server
3. In WSL, set the DISPLAY variable:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
   ```
4. Run tests:
   ```bash
   python tests/test_screen_capture.py
   ```

### Option 3: Headless Testing (Limited)

For basic validation without full functionality:

```bash
# This works in any environment and tests imports/basic functionality
python tests/test_imports.py
```

## Test Requirements by Environment

| Test | Headless OK? | Requires |
|------|--------------|----------|
| test_imports.py | ✓ Yes | None (imports only) |
| test_screen_capture.py | ✗ No | Active display for screen capture |
| test_motor_control.py | ✗ No | Input device access (mouse/keyboard) |
| test_integration.py | ✗ No | Full graphical environment |

## Recommended Testing Workflow

### During Development (Windows)

1. **Quick validation** (in any environment):
   ```bash
   python tests/test_imports.py
   ```

2. **Full testing** (Windows native only):
   ```powershell
   python tests\test_screen_capture.py
   python tests\test_motor_control.py  # Will ask before moving mouse
   python tests\test_integration.py
   ```

### In Production/Deployment

The actual agent **must** run in a graphical environment since it needs to:
- Capture the game screen
- Control mouse and keyboard
- Display the game (Danganronpa)

## Dependencies Fixed

The following dependencies were missing and have been added:

- `opencv-python==4.10.0.84` - Image processing for screen capture
- `h5py==3.12.1` - HDF5 file format for logging
- `pynput==1.8.1` - Mouse/keyboard control
- `mss==10.1.0` - Screen capture
- `PyYAML==6.0.3` - Configuration files

These are now in `requirements.txt`.

## Why Did Setup Tests Work?

The original setup tests (in `docs/planning/PC_Agent_Setup_Instructions.md`) worked because they:

1. Only tested individual libraries (mss, soundcard) without the full wrapper
2. Didn't import the entire wrapper module
3. Were run in your native Windows environment with display access

The new integration tests require the complete wrapper system, which includes pynput (needs display) from the start.

## Solution Summary

**For your case (Windows development):**
1. Run tests from Windows PowerShell/CMD, not WSL
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests natively on Windows where you have display access

The code is correct - the environment just needs to support graphical operations.
