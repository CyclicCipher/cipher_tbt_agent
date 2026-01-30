# Proprioception and Canonical Microcircuit Implementation

## 1. Proprioception: Cursor and Gaze Position

### The Problem

Computer-using agent needs to know:
- **Cursor position:** (x, y) on screen
- **Gaze position:** (x, y) fovea center
- **Maybe:** Scroll position, window focus, mouse button states

This is analogous to human proprioception (knowing where your limbs are without looking).

### Where Does This Fit?

**Type:** Granular sensory input (raw sensor data, not processed)

**But NOT like vision/audio:**
- Vision: 1920×1080×3 = 6.2M dims (spatial)
- Audio: 48kHz × 1 second = 48K dims (temporal)
- **Proprioception: ~10-20 dims** (continuous scalars)

**Architecture:**

```python
class ProprioceptionSubnet(GranularSubnet):
    """
    Small dense network for proprioceptive signals.

    Input:
    - Cursor: (x, y, left_click, right_click, scroll)
    - Gaze: (x, y, vergence)
    - Window: (focus_id, scroll_x, scroll_y)

    Total: ~10-15 dims
    """

    def __init__(self):
        # Layer 0: DIRECT SENSORY INPUT (granular)
        # But dense, not conv (no spatial structure)
        self.layer0 = DensePCLayer(
            input_size=15,  # Small vector
            output_size=64,  # Expand to feature space
            receives_sensory=True,  # Granular
            has_lateral=True
        )

        # Layer 1: Integration
        self.layer1 = DensePCLayer(
            input_size=64,
            output_size=64
        )

        # Layer 2: Output (to association)
        self.layer2 = DensePCLayer(
            input_size=64,
            output_size=64  # Proprioceptive features
        )
```

**Updated Position 0:**

```
Position 0: SENSORY (All Granular)
├─ Vision: Conv PC (768 dims out)
├─ Audio: Temporal PC (512 dims out)
├─ Proprioception: Dense PC (64 dims out)  ← NEW
└─ Motor: Dense PC (coproduct)
```

**Association receives:**
- Vision: 768 dims
- Audio: 512 dims
- Proprioception: 64 dims
- **Total: 1344 dims** (instead of 1280)

### Why Proprioception Matters

**For attention control:**
- "Look at suspicious item" requires knowing where you're currently looking
- Gaze shift = (target_x - current_gaze_x, target_y - current_gaze_y)

**For motor control:**
- "Click button" requires knowing where cursor is
- Motor command = (target_x - current_cursor_x, target_y - current_cursor_y)

**Brain analogy:**
- Dorsal stream uses proprioception for action guidance
- Parietal cortex integrates vision + proprioception
- "Where am I looking?" is critical for "where should I look next?"

---

## 2. Canonical Microcircuit and Neuron Design

### What Our Current Neuron Has

Looking at `src/network/layer.py`:

```python
class PredictiveCodingLayer:
    def __init__(self, layer_index, num_neurons, input_size_below, input_size_above):
        self.neurons = NeuronPopulation(...)

        # Bottom-up weights
        self.neurons.W_basal = nn.Parameter(...)

        # Top-down weights (for layers that receive feedback)
        self.neurons.W_apical = nn.Parameter(...) if input_size_above > 0 else None
```

**What we have:**
- ✓ Bottom-up connections (`W_basal`)
- ✓ Top-down connections (`W_apical`)
- ✓ State maintenance (`self.state`)

**What we're missing:**
- ✗ Lateral connections (within-layer)
- ✗ Layer-specific behavior (superficial vs deep)
- ✗ Explicit feedback targeting

### What Canonical Microcircuit Needs

From neuroscience:

1. **Lateral connections** (within superficial layers)
   - Layer II/III neurons connect to nearby neurons
   - Enables contextual modulation
   - In conv layers: already have this via convolution
   - In dense layers: need to add

2. **Layer-specific computation:**
   - Superficial (Layer 0-1): Error computation, manipulation
   - Deep (Layer 2): Output, feedback generation

3. **Feedback targeting:**
   - Feedback should target apical dendrites (Layer I in biology)
   - We model this implicitly with `W_apical`

### Proposed Changes to Neuron

**Option 1: Minimal Changes (Recommended)**

Keep current neuron simple, add lateral connections as needed:

```python
class PredictiveCodingLayer:
    """Enhanced with optional lateral connections."""

    def __init__(
        self,
        layer_index,
        num_neurons,
        input_size_below,
        input_size_above,
        has_lateral=False,  # ← NEW
        layer_type="middle"  # "superficial", "middle", "deep"
    ):
        super().__init__()

        self.layer_index = layer_index
        self.layer_type = layer_type
        self.num_neurons = num_neurons

        # Standard connections
        self.neurons = NeuronPopulation(...)
        self.neurons.W_basal = nn.Parameter(...)  # Bottom-up
        self.neurons.W_apical = nn.Parameter(...) if input_size_above > 0 else None  # Top-down

        # Lateral connections (optional)
        if has_lateral:
            self.W_lateral = nn.Parameter(
                torch.randn(num_neurons, num_neurons) * 0.01
            )
            # Sparse lateral (only nearby neurons in conv case)
            # Dense lateral (all-to-all in dense case)
        else:
            self.W_lateral = None

        self.state = nn.Parameter(torch.zeros(num_neurons))
```

**Changes needed:**
- Add `W_lateral` parameter (optional)
- Add `layer_type` for behavior differentiation
- Use during inference

**Option 2: Explicit Microcircuit Structure**

Create a microcircuit module that bundles layers:

```python
class CanonicalMicrocircuit(nn.Module):
    """
    3-layer canonical microcircuit.

    Bundles:
    - Layer 0 (Superficial): Input, error, lateral
    - Layer 1 (Middle): Integration
    - Layer 2 (Deep): Output, feedback
    """

    def __init__(
        self,
        input_size,
        layer0_size,
        layer1_size,
        layer2_size,
        is_granular=False  # Has direct sensory input
    ):
        super().__init__()

        # Layer 0: Superficial
        self.layer0 = PredictiveCodingLayer(
            layer_index=0,
            num_neurons=layer0_size,
            input_size_below=input_size,
            input_size_above=layer1_size,
            has_lateral=True,  # Superficial layers have lateral
            layer_type="superficial"
        )

        # Layer 1: Middle
        self.layer1 = PredictiveCodingLayer(
            layer_index=1,
            num_neurons=layer1_size,
            input_size_below=layer0_size,
            input_size_above=layer2_size,
            has_lateral=False,
            layer_type="middle"
        )

        # Layer 2: Deep
        self.layer2 = PredictiveCodingLayer(
            layer_index=2,
            num_neurons=layer2_size,
            input_size_below=layer1_size,
            input_size_above=0,  # Top layer
            has_lateral=False,
            layer_type="deep"
        )

        # For granular areas, layer 0 receives sensory
        # For agranular areas, layer 0 receives cortical
        self.is_granular = is_granular

    def forward(self, input_data, num_iterations=20):
        """
        Inference with microcircuit dynamics.
        """
        for _ in range(num_iterations):
            # Layer 0: Superficial
            # Bottom-up from input
            # Top-down from layer 1
            # Lateral within layer 0
            ...

            # Layer 1: Middle
            # Bottom-up from layer 0
            # Top-down from layer 2
            ...

            # Layer 2: Deep
            # Bottom-up from layer 1
            # Generates feedback to lower layers
            ...

        return self.layer2.state
```

**This bundles the microcircuit structure explicitly.**

---

## Most Efficient Implementation

### Recommendation: Hybrid Approach

**For convolution layers (vision, audio):**
- Conv operations already provide lateral connections (receptive field overlap)
- Don't need explicit `W_lateral`
- Just use standard Conv PC layers

**For dense layers (association, PFC):**
- Add `W_lateral` parameter to Layer 0 (superficial)
- Keep Layers 1-2 without lateral
- Minimal overhead

### Code Changes Needed

**1. Update `PredictiveCodingLayer` in `src/network/layer.py`:**

```python
class PredictiveCodingLayer(nn.Module):
    def __init__(
        self,
        layer_index: int,
        num_neurons: int,
        input_size_below: int,
        input_size_above: int,
        dtype: torch.dtype = torch.float32,
        has_lateral: bool = False,  # ← NEW
        layer_type: str = "middle"   # ← NEW: "superficial", "middle", "deep"
    ):
        super().__init__()

        self.layer_index = layer_index
        self.layer_type = layer_type
        self.num_neurons = num_neurons

        # Create neuron population
        self.neurons = NeuronPopulation(
            num_neurons=num_neurons,
            input_size=input_size_below,
            dtype=dtype
        )

        # Top-down connections (if not top layer)
        if input_size_above > 0:
            self.neurons.W_apical = nn.Parameter(
                torch.randn(num_neurons, input_size_above, dtype=dtype) * 0.01
            )
        else:
            self.neurons.W_apical = None

        # Lateral connections (optional, for superficial layers)
        if has_lateral:
            self.W_lateral = nn.Parameter(
                torch.randn(num_neurons, num_neurons, dtype=dtype) * 0.01
            )
        else:
            self.W_lateral = None

        # State
        self.state = nn.Parameter(torch.zeros(num_neurons, dtype=dtype))
```

**2. Update inference to use lateral connections:**

```python
def _inference_step_subnet(self, subnet, subnet_input):
    """Modified to include lateral connections."""

    for i, layer in enumerate(subnet.layers):
        # ... existing bottom-up and top-down ...

        # NEW: Lateral connections (if layer has them)
        if layer.W_lateral is not None:
            lateral_input = layer.W_lateral @ layer.get_state()
            lateral_contribution = torch.tanh(lateral_input) - layer.get_state()
            gradient += 0.1 * lateral_contribution  # Small weight

        # ... rest of inference ...
```

**3. Create subnets with layer types:**

```python
# Vision (conv - lateral is implicit in convolution)
vision_layer0 = ConvPCLayer(
    has_lateral=False,  # Conv already has receptive field overlap
    layer_type="superficial"
)

# Association (dense - need explicit lateral)
assoc_layer0 = DensePCLayer(
    has_lateral=True,   # Add lateral connections
    layer_type="superficial"
)

assoc_layer2 = DensePCLayer(
    has_lateral=False,  # Deep layers don't need lateral
    layer_type="deep"
)
```

---

## Parameter Overhead

**Without lateral connections:**
- Layer with N neurons, M inputs: N × M params

**With lateral connections:**
- Layer with N neurons, M inputs: N × M + N × N params

**Example: Association Layer 0 (512 neurons)**
- Without lateral: 512 × 1344 = 688K params
- With lateral: 688K + 512 × 512 = 950K params
- **Overhead: ~40%**

**Is it worth it?**
- Lateral connections enable contextual modulation
- Important for superficial layers (error computation)
- Skip for middle/deep layers

**Recommendation:**
- Add lateral to Layer 0 of association/PFC subnets
- Skip for vision (conv provides this)
- Skip for Layer 1-2 of all subnets

---

## Summary

### Proprioception

Add as 4th sensory stream:
```python
proprioception = DensePCSubnet(
    input_size=15,  # Cursor, gaze, window state
    layer_sizes=[64, 64, 64],
    is_granular=True  # Direct sensory input
)
```

Association receives: Vision (768) + Audio (512) + Proprio (64) = 1344 dims

### Neuron Changes

**Minimal additions:**
1. Add `W_lateral` parameter (optional)
2. Add `layer_type` field
3. Use lateral in inference (for superficial layers only)

**Efficient approach:**
- Lateral for dense Layer 0 (association, PFC)
- Skip for conv (already has receptive field overlap)
- Skip for Layer 1-2 (not needed)

**Overhead:** ~40% params for layers with lateral, but only apply selectively

### Implementation Priority

1. **First:** Get basic 3-layer structure working (no lateral)
2. **Second:** Add proprioception input
3. **Third:** Add lateral connections to association Layer 0
4. **Fourth:** Test if lateral connections actually help

Start simple, add complexity only when needed.

Does this clarify the implementation approach?
