"""Diagnose GPU environment for ePC-Mamba3 reproducibility issues.

Checks TF32 settings, cuDNN config, NVIDIA driver version, and runs
a quick Newton step precision test to detect environment changes.

Usage:
    python experiments/Mamba3/diagnose_env.py
"""

import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_environment():
    section("ENVIRONMENT")
    print(f"PyTorch:           {torch.__version__}")
    print(f"CUDA available:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:               {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"Compute cap:       {cap[0]}.{cap[1]}")
        is_ampere = cap[0] >= 8
        print(f"Ampere or newer:   {is_ampere}")
        print(f"CUDA version:      {torch.version.cuda}")
        print(f"cuDNN version:     {torch.backends.cudnn.version()}")

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"NVIDIA driver:     {result.stdout.strip()}")
        else:
            print("NVIDIA driver:     (nvidia-smi failed)")
    except Exception:
        print("NVIDIA driver:     (nvidia-smi not available)")


def check_precision_settings():
    section("PRECISION SETTINGS (DEFAULTS)")
    print(f"TF32 matmul:       {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cuDNN:        {torch.backends.cudnn.allow_tf32}")
    print(f"cuDNN benchmark:   {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"float32_matmul_precision: {torch.get_float32_matmul_precision()}")

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            print()
            print("*** WARNING: Ampere GPU detected! ***")
            if torch.backends.cuda.matmul.allow_tf32:
                print("*** TF32 is ENABLED for matmul — float32 ops use 10-bit mantissa ***")
                print("*** This reduces precision of ALL linear layers and gradient computation ***")
            if torch.backends.cudnn.allow_tf32:
                print("*** TF32 is ENABLED for cuDNN — convolutions use reduced precision ***")


def test_matmul_precision():
    """Test whether TF32 actually affects computation on this GPU."""
    if not torch.cuda.is_available():
        print("(skipped — no CUDA)")
        return

    section("TF32 MATMUL PRECISION TEST")

    device = torch.device('cuda')

    # Create matrices where TF32 precision loss is detectable
    torch.manual_seed(42)
    a = torch.randn(256, 256, device=device, dtype=torch.float32)
    b = torch.randn(256, 256, device=device, dtype=torch.float32)

    # Test WITH TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result_tf32 = torch.mm(a, b)

    # Test WITHOUT TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result_fp32 = torch.mm(a, b)

    diff = (result_tf32 - result_fp32).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / result_fp32.abs().clamp(min=1e-8)).mean().item()

    print(f"Max absolute diff:  {max_diff:.6e}")
    print(f"Mean absolute diff: {mean_diff:.6e}")
    print(f"Mean relative diff: {rel_diff:.6e}")

    if max_diff > 1e-6:
        print("*** TF32 IS ACTIVE: float32 matmul produces different results ***")
        print("*** This affects ALL linear layer forward/backward passes ***")
    else:
        print("TF32 has no effect (GPU may not support it)")

    # Restore
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def test_newton_step_precision():
    """Test Newton step with and without TF32 to see if results differ."""
    if not torch.cuda.is_available():
        print("(skipped — no CUDA)")
        return

    section("NEWTON STEP PRECISION TEST")
    print("Running 1-batch ePC inference with TF32 on vs off...")

    from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3Block, RMSNorm
    from experiments.Mamba3.epc_model import ePCMamba3LM

    device = torch.device('cuda')

    config = Mamba3Config(d_model=128, d_state=64, n_layer=4, chunk_size=32)

    results = {}

    for label, tf32_on in [("TF32=ON", True), ("TF32=OFF", False)]:
        torch.backends.cuda.matmul.allow_tf32 = tf32_on
        torch.backends.cudnn.allow_tf32 = tf32_on

        # Deterministic init
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        model = ePCMamba3LM(
            config, vocab_size=16, iters=2, damping=0.1,
            precision_mode='geometric', precision_base=3.0,
        ).to(device)

        # Deterministic input
        torch.manual_seed(123)
        inputs = torch.randint(1, 16, (4, 64), device=device)
        targets = torch.randint(0, 16, (4, 64), device=device)

        # Run inference (Newton steps)
        model.train()
        E_val = model(inputs, targets)
        diag = model.get_diagnostics()

        # Measure speed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(inputs, targets)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_per_call = (t1 - t0) / 10 * 1000

        results[label] = {
            'E_initial': diag['E_initial'],
            'E_final': diag['E_final'],
            'convergence': diag['convergence'],
            'error_norms': diag['error_norms'],
            'ms': ms_per_call,
        }

        print(f"\n  {label}:")
        print(f"    E_initial:    {diag['E_initial']:.4f}")
        print(f"    E_final:      {diag['E_final']:.4f}")
        print(f"    Convergence:  {diag['convergence']:.4f}")
        norms_str = ', '.join(f'{n:.6f}' for n in diag['error_norms'])
        print(f"    Error norms:  [{norms_str}]")
        print(f"    Speed:        {ms_per_call:.1f} ms/call")

        del model
        torch.cuda.empty_cache()

    # Compare
    print("\n  COMPARISON:")
    e_diff = abs(results["TF32=ON"]['E_final'] - results["TF32=OFF"]['E_final'])
    c_diff = abs(results["TF32=ON"]['convergence'] - results["TF32=OFF"]['convergence'])
    ms_diff = results["TF32=ON"]['ms'] - results["TF32=OFF"]['ms']

    print(f"    E_final diff:     {e_diff:.4f}")
    print(f"    Convergence diff: {c_diff:.4f}")
    print(f"    Speed diff:       {ms_diff:+.1f} ms (positive = TF32 slower)")

    if c_diff > 10.0:
        print("\n  *** SIGNIFICANT: TF32 substantially changes Newton convergence ***")
        print("  *** This is likely the cause of non-reproducibility ***")
    elif c_diff > 1.0:
        print("\n  ** MODERATE: TF32 has noticeable effect on Newton convergence **")
    else:
        print("\n  TF32 has minimal effect on Newton step")

    # Restore defaults
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def test_speed_benchmarks():
    """Benchmark forward pass speed with different settings."""
    if not torch.cuda.is_available():
        print("(skipped — no CUDA)")
        return

    section("SPEED BENCHMARK")

    from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3Block

    device = torch.device('cuda')
    config = Mamba3Config(d_model=128, d_state=64, n_layer=4, chunk_size=32)

    torch.manual_seed(42)
    block = Mamba3Block(config).to(device)
    x = torch.randn(32, 64, 128, device=device)

    configs = [
        ("TF32=ON, benchmark=ON", True, True),
        ("TF32=ON, benchmark=OFF", True, False),
        ("TF32=OFF, benchmark=ON", False, True),
        ("TF32=OFF, benchmark=OFF", False, False),
    ]

    for label, tf32, benchmark in configs:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = benchmark

        # Warmup
        for _ in range(5):
            _ = block(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(50):
            _ = block(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) / 50 * 1000

        print(f"  {label:35s}: {ms:.2f} ms/forward")

    # Restore
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def main():
    print("ePC-Mamba3 Environment Diagnostic")
    print("=" * 60)

    check_environment()
    check_precision_settings()
    test_matmul_precision()
    test_newton_step_precision()
    test_speed_benchmarks()

    section("RECOMMENDATIONS")
    print("""
If TF32 is causing different Newton convergence, add these lines
BEFORE model creation in train_epc.py:

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

If cuDNN benchmark is affecting speed, disable it:

    torch.backends.cudnn.benchmark = False

If the NVIDIA driver version is different from when the 99.2% result
was achieved, this is likely the root cause. Check Windows Update
history for recent GPU driver updates.
""")


if __name__ == '__main__':
    main()
