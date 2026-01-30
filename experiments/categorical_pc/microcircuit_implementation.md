# Canonical Microcircuit: Implementation Guide

## The Laminar Structure (Simplified to 3 Layers)

### Full Biological Reality (6 layers):
```
Layer I:     Apical dendrites, feedback targets
Layer II/III: Superficial pyramidal (error computation, lateral)
Layer IV:    Granular (sensory input) - ONLY in sensory areas
Layer V:     Deep pyramidal (subcortical output)
Layer VI:    Corticothalamic (feedback to thalamus)
```

### Our Implementation (3 layers - practical):
```
Layer 0 (Superficial): Combines II/III + IV
  - Receives input (bottom-up)
  - Receives feedback (top-down to apical dendrites)
  - Computes prediction errors
  - Lateral connections within layer

Layer 1 (Middle): Combines aspects of II/III and V
  - Integration layer
  - Receives from Layer 0
  - Sends to Layer 2
  - Some lateral processing

Layer 2 (Deep): Combines V + VI
  - Output layer
  - Projects to next area (or motor/attention)
  - Sends feedback to lower layers
```

**Rationale:** 6 layers is biological detail. 3 layers captures the key computational structure.

---

## Granular vs Agranular Variants

### Granular (Vision, Audio - Sensory Areas)

**Biological:**
```
Layer I:     Feedback (from association cortex)
Layer II/III: Error computation
Layer IV:    ← SENSORY INPUT FROM THALAMUS
Layer V:     Output to superior colliculus / association
Layer VI:    Feedback to LGN / thalamus
```

**Our Implementation:**
```python
class GranularSubnet:
    """For vision, audio - direct sensory input"""

    def __init__(self):
        # Layer 0: INPUT LAYER (combines II/III + IV)
        # - Receives RAW sensory input
        # - Convolutional if spatial (vision)
        # - Dense if temporal (audio)
        self.layer0 = ConvPCLayer(  # or DensePCLayer for audio
            receives_sensory=True,  # ← Key difference
            has_lateral=True,       # Local/lateral connections
            receives_feedback=True  # From higher layers
        )

        # Layer 1: INTEGRATION
        self.layer1 = ConvPCLayer(
            receives_from_below=True,
            sends_to_above=True
        )

        # Layer 2: OUTPUT (deep)
        self.layer2 = DensePCLayer(  # Often fully connected
            projects_to_association=True,
            sends_feedback_to_below=True
        )
```

**Key property:** Has direct sensory input to Layer 0 (like biological Layer IV).

---

### Agranular (Association, Working Memory, Value - Non-Sensory)

**Biological:**
```
Layer I:     Feedback (from other PFC areas)
Layer II/III: Receives processed input from other cortex
[No Layer IV - no direct thalamic input]
Layer V:     Output to subcortical (motor, basal ganglia)
Layer VI:    Feedback to thalamus (mediodorsal)
```

**Our Implementation:**
```python
class AgranularSubnet:
    """For association, working memory, value - processed input"""

    def __init__(self):
        # Layer 0: RECEIVES PROCESSED INPUT (not raw sensory)
        # - Input is from OTHER cortical areas
        # - No convolution needed (already processed)
        # - Dense connections
        self.layer0 = DensePCLayer(
            receives_sensory=False,  # ← Key difference
            receives_from_other_areas=True,
            has_recurrence=True,     # PFC has strong recurrence
            receives_feedback=True
        )

        # Layer 1: MANIPULATION
        # In DLPFC: This is where working memory manipulation happens
        # (2024 study: superficial layers do manipulation)
        self.layer1 = DensePCLayer(
            working_memory_operations=True,
            receives_from_below=True
        )

        # Layer 2: OUTPUT (deep)
        # In DLPFC: This is motor output / action commands
        # (2024 study: deep layers send to motor/premotor)
        self.layer2 = DensePCLayer(
            projects_to_motor=True,
            projects_to_attention=True,
            sends_feedback_to_below=True
        )
```

**Key property:** No direct sensory input. Receives processed input from other cortical areas.

---

## Does "CNN" Still Apply?

**Short answer:** Yes for vision Layer 0, but not everywhere.

### Vision Subnet: Hybrid Architecture

```python
class VisionSubnet(GranularSubnet):
    """
    Vision uses CONVOLUTIONAL layers for Layer 0
    because it processes spatial sensory input.
    """

    def __init__(self):
        # LAYER 0: CONVOLUTIONAL (spatial processing)
        # Approximates V1 minicolumns via weight sharing
        self.layer0 = nn.ModuleDict({
            # Convolutional processing (horizontal)
            'conv': ConvPCLayer(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=2
                # Weight sharing = minicolumns
                # Local receptive fields = local connectivity
            ),

            # Within this layer, there's also laminar structure (vertical)
            # - Superficial sublayer: receives feedback
            # - Deep sublayer: sends output
        })

        # LAYER 1: STILL CONVOLUTIONAL
        # But larger receptive fields, more abstract features
        self.layer1 = ConvPCLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2
        )

        # LAYER 2: DENSE (flattened)
        # Output layer, no longer spatial
        self.layer2 = DensePCLayer(
            input_size=64 * 25 * 25,  # Flattened conv output
            output_size=768  # To association (ventral × dorsal)
        )
```

**Why CNN for vision?**
- Layer 0 processes RAW spatial input (pixels)
- Weight sharing = biological minicolumns (same feature everywhere)
- Local receptive fields = biological local connectivity
- Hierarchical = biological layer structure

**But:** This is LAYER 0 only. Higher layers can be dense.

---

### Audio Subnet: Temporal Processing (Not Convolutional)

```python
class AudioSubnet(GranularSubnet):
    """
    Audio uses TEMPORAL processing for Layer 0
    because it processes temporal sensory input (sound waves).

    NOT convolutional in spatial sense.
    Uses 1D convolution over time or recurrent structures.
    """

    def __init__(self):
        # LAYER 0: 1D TEMPORAL CONVOLUTION
        # Processes sound waveform
        self.layer0 = nn.ModuleDict({
            # 1D conv over time (not 2D spatial)
            'temporal_conv': Conv1DPCLayer(
                in_channels=1,      # Mono audio
                out_channels=64,    # Frequency-like features
                kernel_size=256,    # ~5ms at 48kHz
                stride=128
            ),

            # Or could use recurrent:
            # 'recurrent': RecurrentPCLayer(...)
        })

        # LAYER 1: TEMPORAL INTEGRATION
        # Longer timescales (phonemes, words)
        self.layer1 = Conv1DPCLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=64,  # Longer context
            stride=32
        )

        # LAYER 2: DENSE (semantic)
        # Output to association (word/phrase representations)
        self.layer2 = DensePCLayer(
            input_size=128 * temporal_length,
            output_size=512  # Semantic features
        )
```

**Why 1D conv (or recurrent) for audio?**
- Audio is temporal, not spatial
- Need to capture patterns over time (phonemes, prosody)
- 1D conv approximates temporal receptive fields
- Or recurrent (RNN/LSTM-like) for longer dependencies

**Term:** Could call it "temporal convolution" or just "temporal processing"

---

### Association Subnet: Dense Processing (Not Convolutional)

```python
class AssociationSubnet(AgranularSubnet):
    """
    Association uses DENSE layers
    because it receives already-processed input.

    No convolution - information is no longer spatial/temporal raw.
    """

    def __init__(self):
        # LAYER 0: DENSE
        # Receives vision (768) + audio (512) = 1280 dims
        self.layer0 = DensePCLayer(
            input_size=1280,  # Concatenated modalities
            output_size=512,
            has_recurrence=True,  # Integration over time
            receives_feedback=True
        )

        # LAYER 1: DENSE
        # Multimodal integration
        self.layer1 = DensePCLayer(
            input_size=512,
            output_size=256
        )

        # LAYER 2: DENSE OUTPUT
        # To working memory / value
        self.layer2 = DensePCLayer(
            input_size=256,
            output_size=256,
            projects_to_higher=True,
            sends_feedback=True
        )
```

**No convolution because:**
- Input is already processed (not raw pixels/audio)
- No spatial structure to preserve
- Integration, not feature extraction

---

## Summary: When to Use What

| Subnet | Variant | Layer 0 Type | Why |
|--------|---------|--------------|-----|
| Vision | Granular | **2D Conv** | Spatial sensory input (pixels) |
| Audio | Granular | **1D Conv** or RNN | Temporal sensory input (waveform) |
| Association | Agranular | **Dense** | Processed multimodal input |
| Working Memory | Agranular | **Dense** | Abstract representations |
| Value | Agranular | **Dense** | Decision evaluation |
| Motor | Mixed | **Dense** | Action outputs |

---

## Horizontal vs Vertical Organization

### Horizontal (within a layer):
- **Vision:** Weight sharing (conv) approximates minicolumns
- **Audio:** Temporal patterns, frequency bands
- **Association/PFC:** No special horizontal structure

### Vertical (across layers):
- **All areas:** Laminar structure (superficial vs deep)
- **All areas:** Feedback (deep → superficial of lower areas)
- **All areas:** Error computation (superficial layers)

**Both matter!** Vision needs both:
- Horizontal: Conv (minicolumns)
- Vertical: Layers (canonical microcircuit)

---

## Concrete Example: Vision Subnet Implementation

```python
class VisionSubnet(nn.Module):
    """
    Vision with both:
    - Horizontal: Convolutional (minicolumns via weight sharing)
    - Vertical: Laminar (canonical microcircuit)
    """

    def __init__(self):
        # === LAYER 0: SUPERFICIAL (Input + Error) ===
        # Granular: receives sensory input
        # Convolutional: spatial weight sharing

        self.layer0_conv = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # PC state for layer 0
        self.layer0_state = nn.Parameter(
            torch.zeros(32, H//2, W//2)  # Spatial feature map
        )

        # Receives feedback from layer 1 (to apical dendrites)
        self.layer0_feedback_weights = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2
        )

        # === LAYER 1: MIDDLE (Integration) ===

        self.layer1_conv = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1_state = nn.Parameter(
            torch.zeros(64, H//4, W//4)
        )

        self.layer1_feedback_weights = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2
        )

        # === LAYER 2: DEEP (Output) ===
        # Dense layer (spatial info collapsed)

        self.layer2_flatten = nn.Flatten()
        self.layer2_fc = nn.Linear(
            in_features=64 * (H//4) * (W//4),
            out_features=768  # Output size (ventral × dorsal)
        )

        self.layer2_state = nn.Parameter(
            torch.zeros(768)
        )

        # Feedback to layer 1 (projects back to spatial)
        self.layer2_feedback_weights = nn.Linear(
            in_features=768,
            out_features=64 * (H//4) * (W//4)
        )

    def forward(self, sensory_input, num_iterations=20):
        """
        Predictive coding inference with laminar structure.

        Bottom-up: Conv layers process spatial features
        Top-down: Transposed conv / linear send predictions down
        Iteration: States settle via prediction error minimization
        """

        for _ in range(num_iterations):
            # === LAYER 0: Superficial ===
            # Bottom-up from sensory input
            bottom_up_0 = torch.tanh(self.layer0_conv(sensory_input))

            # Top-down from layer 1
            top_down_0 = torch.tanh(
                self.layer0_feedback_weights(self.layer1_state)
            )

            # Prediction error
            error_0 = self.layer0_state - bottom_up_0
            feedback_error_0 = self.layer0_state - top_down_0

            # Update state (gradient descent on energy)
            self.layer0_state.data -= 0.1 * (error_0 + feedback_error_0)

            # === LAYER 1: Middle ===
            bottom_up_1 = torch.tanh(self.layer1_conv(self.layer0_state))
            top_down_1 = torch.tanh(
                self.layer1_feedback_weights(self.layer2_state).view_as(self.layer1_state)
            )

            error_1 = self.layer1_state - bottom_up_1
            feedback_error_1 = self.layer1_state - top_down_1

            self.layer1_state.data -= 0.1 * (error_1 + feedback_error_1)

            # === LAYER 2: Deep ===
            flattened = self.layer2_flatten(self.layer1_state)
            bottom_up_2 = torch.tanh(self.layer2_fc(flattened))

            # No top-down for layer 2 (highest layer in this subnet)
            error_2 = self.layer2_state - bottom_up_2

            self.layer2_state.data -= 0.1 * error_2

        return self.layer2_state
```

**This has:**
- ✓ Convolutional (horizontal structure)
- ✓ Laminar (vertical structure)
- ✓ Feedback (top-down predictions)
- ✓ Predictive coding (error minimization)

---

## Does "CNN" Term Still Apply?

**Yes and no:**

**"CNN" traditionally means:**
- Convolutional layers
- Feedforward
- Trained with backprop

**Our "Convolutional PC Subnet" means:**
- Convolutional layers (same)
- **Recurrent** (iterative inference)
- **Predictive coding** (local learning rules)

**Better terms:**
- "Convolutional Predictive Coding Network" (for vision)
- "Temporal Predictive Coding Network" (for audio)
- "Dense Predictive Coding Network" (for association/reasoning)

**But:** Can still use "conv layers" for the horizontal structure. It's accurate.

---

## Revised Architecture with Audio

```
Position 0: SENSORY (Granular - has Layer IV analog)
├─ Vision: Conv PC (2D spatial)
│   Layer 0: Conv (32 filters)
│   Layer 1: Conv (64 filters)
│   Layer 2: Dense (768 output)
│
├─ Audio: Conv1D / Recurrent PC (temporal)
│   Layer 0: 1D Conv or RNN
│   Layer 1: Temporal integration
│   Layer 2: Dense (512 output)
│
└─ Motor: Dense PC (coproduct)
    Layer 0: Action selection
    Layer 1: Motor primitives
    Layer 2: Output (keyboard/mouse/gaze)

Position 1: ASSOCIATION (Agranular - no Layer IV)
└─ Multimodal: Dense PC (product: vision × audio)
    Layer 0: Integration (1280 → 512)
    Layer 1: Abstraction (512 → 256)
    Layer 2: Output (256)

Position 2: ABSTRACT (Agranular)
├─ Working Memory: Dense PC (recurrent)
│   Layer 0: Maintenance
│   Layer 1: Manipulation
│   Layer 2: Output
│
└─ Value: Dense PC (vmPFC-like)
    Layer 0: Action structure
    Layer 1: Value estimation
    Layer 2: Output to motor

FEEDBACK:
├─ Attention → Vision (Layer 2 → Layer 0)
└─ Motor prep → Motor (Value → Motor)
```

**Total subnets:** 6 main + feedback = ~8 components
**All use canonical microcircuit (3-layer version)**
**Granular vs agranular depending on input type**
**Conv only where appropriate (vision Layer 0, audio temporal)**

Does this clarify how the canonical microcircuit maps to our implementation?
