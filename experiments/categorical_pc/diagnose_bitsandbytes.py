"""
Diagnostic script to determine why bitsandbytes isn't loading.
Run this and share the output.
"""

import sys
import platform

print("=" * 60)
print("BITSANDBYTES DIAGNOSTIC")
print("=" * 60)

# System info
print("\n[1] System Information:")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version}")
print(f"  Platform: {platform.platform()}")

# PyTorch info
print("\n[2] PyTorch Information:")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ✗ CUDA not available (bitsandbytes requires CUDA)")
except ImportError as e:
    print(f"  ✗ PyTorch not installed: {e}")

# bitsandbytes info
print("\n[3] Bitsandbytes Installation Check:")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True,
        text=True
    )
    if "bitsandbytes" in result.stdout:
        for line in result.stdout.split('\n'):
            if 'bitsandbytes' in line.lower():
                print(f"  ✓ Found in pip list: {line.strip()}")
    else:
        print(f"  ✗ Not found in pip list")
        print(f"  → Try: pip install bitsandbytes")
except Exception as e:
    print(f"  ✗ Could not check pip list: {e}")

# Detailed import attempt
print("\n[4] Import Attempt (with full error):")
try:
    import bitsandbytes as bnb
    print(f"  ✓ Import successful!")
    print(f"  ✓ Version: {bnb.__version__}")

    # Test CUDA setup
    try:
        from bitsandbytes.cuda_setup.main import get_compute_capability
        cc = get_compute_capability()
        print(f"  ✓ Compute capability: {cc}")
    except Exception as e:
        print(f"  ⚠ CUDA setup warning: {e}")

except ImportError as e:
    print(f"  ✗ ImportError: {e}")
    print(f"\n  Common causes:")
    print(f"    - Not installed: pip install bitsandbytes")
    print(f"    - Wrong environment: Make sure you're in the right venv")

except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")
    import traceback
    print("\n  Full traceback:")
    traceback.print_exc()

# Windows-specific checks
if platform.system() == "Windows":
    print("\n[5] Windows-Specific Checks:")
    print("  ℹ bitsandbytes on Windows requires:")
    print("    - CUDA 11.7+ or 12.x")
    print("    - Visual Studio Build Tools")
    print("    - Recent bitsandbytes version (0.43.0+)")

    # Check if running in WSL
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                print("  ℹ Detected WSL - should work like Linux")
    except:
        pass

print("\n[6] Recommendations:")
print("  If bitsandbytes isn't installed:")
print("    → pip install bitsandbytes==0.45.0")
print("\n  If installed but import fails on Windows:")
print("    → Check CUDA version matches PyTorch CUDA version")
print("    → Try: pip install bitsandbytes --force-reinstall")
print("    → If still fails, bitsandbytes may not support your setup")
print("    → Fallback to FP32 works fine (just uses more memory)")

print("\n" + "=" * 60)
print("END DIAGNOSTIC")
print("=" * 60)
