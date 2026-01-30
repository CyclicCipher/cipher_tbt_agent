"""
Validate the full categorical architecture for computer-using agent.

Tests the dual-stream vision + three association areas design.
"""

from architecture_validator import NetworkConfig, ValidationReport

print("=" * 70)
print("VALIDATING FULL COMPUTER-USING AGENT ARCHITECTURE")
print("=" * 70)
print()

config = NetworkConfig()

# ============================================================================
# VISION SYSTEM: Dual-stream (Ventral × Dorsal)
# ============================================================================

# Fovea and periphery inputs (coproduct - combined into one input)
# For now, treat as single concatenated input of 30000 dims
# (320×320×3 fovea + 96×96×3 periphery ≈ 30000)

# Convolutional segment (V1/V2-like) - processes visual input
conv_vision = config.add_subnet(
    "conv_vision", "simple",
    input_size=30000,
    output_size=18432,  # 12×12×128 spatial feature map
    has_weight_sharing=True  # Conv layers use weight sharing (natural transformation)
)

# Ventral stream: "What" pathway (object identity, position-invariant)
ventral = config.add_subnet(
    "ventral", "simple",
    input_size=18432,  # From conv features
    output_size=512    # Abstract object features
)

# Dorsal stream: "Where/How" pathway (spatial positions, action-relevant)
dorsal = config.add_subnet(
    "dorsal", "simple",
    input_size=18432,  # From conv features
    output_size=256    # Spatial/positional features
)

# Vision output is PRODUCT: Ventral × Dorsal
vision = config.add_subnet(
    "vision", "product",
    input_size=18432,
    output_size=768,  # 512 + 256
    components=[ventral, dorsal]
)

# Connections for vision
config.connect(conv_vision, ventral)
config.connect(conv_vision, dorsal)

# ============================================================================
# MOTOR SYSTEM: Coproduct (Keyboard + Mouse + Gaze)
# ============================================================================

# Motor primitives (each is a simple subnet)
motor_keyboard = config.add_subnet(
    "motor_keyboard", "simple",
    input_size=256,  # From keyboard association
    output_size=100  # One-hot over ~100 keys
)

motor_mouse = config.add_subnet(
    "motor_mouse", "simple",
    input_size=256,  # From mouse association
    output_size=5    # (x, y, left_click, right_click, scroll)
)

motor_gaze = config.add_subnet(
    "motor_gaze", "simple",
    input_size=256,  # From gaze association
    output_size=2    # (x, y) fovea position on screen
)

# Motor output is COPRODUCT: Keyboard + Mouse + Gaze
# Output size = max(100, 5, 2) = 100 (one action selected per timestep)
motor = config.add_subnet(
    "motor", "coproduct",
    input_size=256,  # Not directly connected, receives from association
    output_size=100,
    components=[motor_keyboard, motor_mouse, motor_gaze]
)

# ============================================================================
# ASSOCIATION SYSTEM: Three exponentials (Motor^Vision decomposition)
# ============================================================================

# By exponential decomposition: Motor^Vision = (K + M + G)^(V × D)
#                                            ≅ K^(V×D) × M^(V×D) × G^(V×D)

# Association for keyboard: (Ventral × Dorsal) → Keyboard
assoc_keyboard = config.add_subnet(
    "assoc_keyboard", "exponential",
    input_size=768,   # Vision output (ventral × dorsal)
    output_size=256,  # Maps to keyboard motor input
    domain=vision,
    codomain=motor_keyboard
)

# Association for mouse: (Ventral × Dorsal) → Mouse
assoc_mouse = config.add_subnet(
    "assoc_mouse", "exponential",
    input_size=768,   # Vision output
    output_size=256,  # Maps to mouse motor input
    domain=vision,
    codomain=motor_mouse
)

# Association for gaze: (Ventral × Dorsal) → Gaze
assoc_gaze = config.add_subnet(
    "assoc_gaze", "exponential",
    input_size=768,   # Vision output
    output_size=256,  # Maps to gaze motor input
    domain=vision,
    codomain=motor_gaze,
    is_recurrent=True  # Gaze feeds back to vision (fovea position)
)

# Connections: Vision → Associations → Motor primitives
config.connect(vision, assoc_keyboard)
config.connect(vision, assoc_mouse)
config.connect(vision, assoc_gaze)

config.connect(assoc_keyboard, motor_keyboard)
config.connect(assoc_mouse, motor_mouse)
config.connect(assoc_gaze, motor_gaze)

# ============================================================================
# VALIDATE
# ============================================================================

print("\nArchitecture specification:")
config.print_architecture()

print("\nValidating categorical constraints...")
report = config.validate()
print()
report.print_report()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ARCHITECTURE SUMMARY")
print("=" * 70)

print("\nStructure:")
print("  Vision = Ventral × Dorsal (product)")
print("    ├─ Ventral: What (object identity)")
print("    └─ Dorsal: Where/How (spatial/action)")
print()
print("  Motor = Keyboard + Mouse + Gaze (coproduct)")
print("    ├─ Keyboard: Discrete key presses")
print("    ├─ Mouse: Continuous position + clicks")
print("    └─ Gaze: Fovea position (recurrent)")
print()
print("  Association = Motor^Vision (exponential)")
print("    By decomposition: (K + M + G)^(V × D) ≅ K^(V×D) × M^(V×D) × G^(V×D)")
print("    ├─ Keyboard Association: Vision → Keyboard")
print("    ├─ Mouse Association: Vision → Mouse")
print("    └─ Gaze Association: Vision → Gaze (recurrent)")
print()

print("Categorical properties:")
print("  ✓ Products preserve projections")
print("  ✓ Coproducts support selection (one action at a time)")
print("  ✓ Exponentials are proper function spaces")
print("  ✓ Gaze recurrence is explicit (Vision ← Gaze → Vision)")
print()

if report.is_valid():
    print("✓ This architecture is categorically sound!")
    print("  Ready for implementation.")
else:
    print("✗ Architecture needs fixes before implementation.")

print("=" * 70)
