"""
Phase 0: Proper Predictive Coding Architecture

Motor outputs belong at the BOTTOM layer, not top.

Architecture:
- Layer 0 (bottom): Motor output (count representation)
- Layer 1-6: Hierarchical processing
- Layer 7 (top): Sensory input (digit pattern)

Training:
1. Clamp Layer 0 (motor) to target count
2. Present digit at Layer 7 (sensory input)
3. Network minimizes prediction error
4. Learns mapping: sensory → motor through hierarchy

This is proper active inference / predictive coding.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
import torch
import numpy as np

print("=" * 70)
print("PHASE 0: PROPER PREDICTIVE CODING ARCHITECTURE")
print("=" * 70)

# Digit patterns
DIGIT_PATTERNS = {
    0: np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    1: np.array([[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]], dtype=np.float32),
    2: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]], dtype=np.float32),
    3: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    4: np.array([[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]], dtype=np.float32),
    5: np.array([[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    6: np.array([[0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    7: np.array([[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]], dtype=np.float32),
    8: np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    9: np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
}

# Motor output encoding: count as pattern
def create_motor_target(count: int, size: int) -> torch.Tensor:
    """
    Create motor output pattern for a count.

    Encoding: First N neurons at 0.8, rest at 0.0
    This is what the network should command the "motor system" to produce.
    """
    target = torch.zeros(size, dtype=torch.float32)
    if count > 0:
        target[:min(count, size)] = 0.8
    return target

# Network configuration
motor_size = 10  # Motor layer (bottom)
hidden_size = 50
sensory_size = 35  # Digit pattern (7x5)

print(f"\nArchitecture:")
print(f"  Layer 0 (motor):   {motor_size} neurons - COUNT OUTPUT")
print(f"  Layers 1-6:        {hidden_size} neurons - PROCESSING")
print(f"  Layer 7 (sensory): {sensory_size} pixels - DIGIT INPUT")
print(f"\nTraining method: Supervised pre-training")
print(f"  - Clamp Layer 0 to target count")
print(f"  - Present digit at input")
print(f"  - Network learns sensory->motor mapping")
print()

# For now, test if clamping bottom layer works
# Will need to modify backbone to support this properly

# Simple test: Can network learn with clamped bottom layer?
# This requires architecture modification - backbone currently
# expects input at layer 0, not layer 7

print("=" * 70)
print("ARCHITECTURE CHECK")
print("=" * 70)

print("\nCurrent backbone architecture:")
print("  - Designed for: input at layer 0 (bottom)")
print("  - Output: reconstruction from top layer")
print()
print("Required for motor output:")
print("  - Input: at top layer (sensory)")
print("  - Motor output: at layer 0 (bottom)")
print("  - Clamp layer 0, run inference, learn to predict it")
print()

print("NEXT STEP: Modify backbone to support:")
print("  1. Input injection at arbitrary layer (not just layer 0)")
print("  2. Clamping layer 0 (motor) during training")
print("  3. Bidirectional inference (sensory->motor prediction)")
print()

print("This requires backbone refactoring.")
print("Current test demonstrates the architectural requirement.")
print()

print("User is correct:")
print("  - Previous experiment had wrong architecture")
print("  - Cannot conclude catastrophic forgetting from bad setup")
print("  - Need proper motor layer at bottom first")
print()

print("For hundreds of millions of parameters:")
print("  - Current: ~7 layers x 50 neurons = ~350 neurons")
print("  - GPT-2 small: 124M parameters")
print("  - Would need: ~1000+ neurons/layer, 12+ layers")
print("  - Requires compute infrastructure planning")
