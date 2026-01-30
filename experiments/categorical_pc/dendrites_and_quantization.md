# Basal vs Apical Dendrites: What They Actually Do

## The Biological Reality

### Pyramidal Neuron Structure

```
        ↑ (toward Layer I)
        |
    [Apical dendrites] ← Extend upward, receive FEEDBACK
        |
    [Cell body (soma)]
        |
    [Basal dendrites] ← Spread laterally, receive LOCAL inputs
        |
        ↓ (toward white matter)
      [Axon]
```

### What Each Dendrite Type Receives

**Apical Dendrites:**
- Receive **feedback from higher cortical areas** (top-down)
- Extend toward Layer I
- Modulate neuron activity based on context/predictions
- In our model: `W_apical` ✓ (correct)

**Basal Dendrites:**
- Receive inputs from **same layer** (lateral)
- Receive inputs from **lower layers** (bottom-up)
- Spread horizontally in same layer
- In our model: `W_basal` is ONLY bottom-up ✗ (incomplete!)

**We're missing the lateral component of basal dendrites!**

---

## Corrected Understanding

### Bottom-Up Connections
- From layer below to current layer
- Drive initial activity
- **Source:** Feedforward from previous processing stage

### Lateral Connections
- Within same layer
- Contextual modulation, normalization, competition
- **Source:** Neighboring neurons in same layer
- **Part of basal dendrites!**

### Top-Down Connections
- From layer above to current layer
- Predictions, context, attention
- **Source:** Feedback from higher processing stages
- **Target:** Apical dendrites

---

## What Our Implementation Should Be

### Current (Incorrect):

```python
class NeuronPopulation:
    def __init__(self):
        self.W_basal = ...   # ONLY bottom-up (wrong!)
        self.W_apical = ...  # Top-down (correct)
```

### Corrected:

```python
class NeuronPopulation:
    def __init__(self):
        # BASAL DENDRITES receive:
        self.W_basal_feedforward = ...  # Bottom-up (from layer below)
        self.W_basal_lateral = ...      # Lateral (from same layer)

        # APICAL DENDRITES receive:
        self.W_apical = ...             # Top-down (from layer above)
```

Or more simply:

```python
class NeuronPopulation:
    def __init__(self):
        self.W_feedforward = ...  # Bottom-up (clearer name)
        self.W_lateral = ...      # Lateral (explicit)
        self.W_feedback = ...     # Top-down (clearer name)
```

---

## Terminology Fix

| Connection Type | Biological | Our Current Name | Should Be |
|----------------|------------|------------------|-----------|
| Bottom-up | Part of basal dendrites | `W_basal` | `W_feedforward` |
| Lateral | Part of basal dendrites | (missing!) | `W_lateral` |
| Top-down | Apical dendrites | `W_apical` | `W_feedback` |

**"Basal" is confusing because it includes BOTH feedforward and lateral.**

Better names:
- `W_feedforward` - from layer below
- `W_lateral` - from same layer
- `W_feedback` - from layer above

---

## Updated Neuron Implementation

```python
class PredictiveCodingLayer(nn.Module):
    """
    Canonical microcircuit layer with proper dendrite types.
    """

    def __init__(
        self,
        layer_index: int,
        num_neurons: int,
        input_size_below: int,
        input_size_above: int,
        has_lateral: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.layer_index = layer_index
        self.num_neurons = num_neurons

        # FEEDFORWARD (bottom-up, from layer below)
        # Part of basal dendrites biologically
        self.W_feedforward = nn.Parameter(
            torch.randn(num_neurons, input_size_below, dtype=dtype) * 0.01
        )

        # LATERAL (within same layer)
        # Also part of basal dendrites biologically
        if has_lateral:
            self.W_lateral = nn.Parameter(
                torch.randn(num_neurons, num_neurons, dtype=dtype) * 0.01
            )
        else:
            self.W_lateral = None

        # FEEDBACK (top-down, from layer above)
        # Apical dendrites biologically
        if input_size_above > 0:
            self.W_feedback = nn.Parameter(
                torch.randn(num_neurons, input_size_above, dtype=dtype) * 0.01
            )
        else:
            self.W_feedback = None

        # State
        self.state = nn.Parameter(torch.zeros(num_neurons, dtype=dtype))

    def compute_prediction_from_below(self, input_below):
        """Bottom-up prediction (feedforward)."""
        return torch.tanh(self.W_feedforward @ input_below)

    def compute_lateral_influence(self):
        """Lateral influence from same layer."""
        if self.W_lateral is not None:
            return torch.tanh(self.W_lateral @ self.state)
        return torch.zeros_like(self.state)

    def compute_prediction_from_above(self, input_above):
        """Top-down prediction (feedback)."""
        if self.W_feedback is not None:
            return torch.tanh(self.W_feedback @ input_above)
        return torch.zeros_like(self.state)
```

**Now we have all three connection types explicitly!**

---

# 4-Bit Quantization for Rapid Prototyping

## The Biological Precedent

Human synapses have **~26 discrete strength levels** (Bartol et al., 2015).

26 states = log₂(26) ≈ **4.7 bits**

This suggests:
- **4-bit weights** are biologically plausible
- Not just an engineering hack - matches biological precision
- May be sufficient for learning

---

## Does CUDA Support 4-Bit?

**Yes!** Multiple options:

### 1. bitsandbytes Library (Easiest)

```python
import bitsandbytes as bnb

# 4-bit quantized linear layer
layer = bnb.nn.Linear4bit(
    input_features=512,
    output_features=256,
    bias=True,
    compute_dtype=torch.float16  # Compute in FP16, store in 4-bit
)
```

**Advantages:**
- Drop-in replacement for nn.Linear
- ~4× memory reduction
- 2-3× speed improvement on modern GPUs
- Training supported (with some caveats)

**Used by:** LLaMA, Mistral, many modern LLMs

---

### 2. PyTorch Native (torch.ao.quantization)

```python
import torch.ao.quantization as quant

# Post-training quantization
model_fp32 = MyModel()
model_int4 = quant.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint4x2  # 4-bit quantization
)
```

**Advantages:**
- Native PyTorch (no external dependencies)
- Well-integrated

**Disadvantages:**
- More complex API
- Quantization-aware training needs more setup

---

### 3. Custom CUDA Kernels (Advanced)

```python
# Use custom kernels for 4-bit matmul
# Libraries: CUTLASS, cuBLAS extensions
```

**Advantages:**
- Maximum performance
- Full control

**Disadvantages:**
- Complex to implement
- Not worth it for prototyping

---

## Recommended Approach for Our Use Case

### Phase 1: Prototype in FP32
- Get architecture working
- Verify learning dynamics
- Establish baseline

### Phase 2: Inference in 4-bit
- Use bitsandbytes for inference
- Test if performance degrades
- ~4× memory reduction → can fit larger networks

### Phase 3: Training in 4-bit (if needed)
- QLoRA-style training (4-bit + LoRA adapters)
- Much slower, but saves memory

---

## Memory Savings Calculation

### FP32 (Current):
```
Vision:        10M params × 4 bytes = 40 MB
Audio:         5M params × 4 bytes = 20 MB
Association:   3M params × 4 bytes = 12 MB
Working Mem:   2M params × 4 bytes = 8 MB
Value:         2M params × 4 bytes = 8 MB
Motor:         1M params × 4 bytes = 4 MB
------------------------------------------
TOTAL:         23M params × 4 bytes = 92 MB
```

### 4-bit (Quantized):
```
TOTAL:         23M params × 0.5 bytes = 11.5 MB
```

**Savings: 92 MB → 11.5 MB (8× reduction!)**

**Implications:**
- Can fit much larger models in same memory
- Faster inference (less memory bandwidth)
- Enables rapid experimentation

---

## Code Example: Drop-In Replacement

### Current Code:
```python
class PredictiveCodingLayer(nn.Module):
    def __init__(self, num_neurons, input_size):
        self.W_feedforward = nn.Parameter(
            torch.randn(num_neurons, input_size)
        )
```

### With 4-bit:
```python
import bitsandbytes as bnb

class PredictiveCodingLayer(nn.Module):
    def __init__(self, num_neurons, input_size, use_4bit=False):
        if use_4bit:
            # 4-bit quantized (for inference)
            self.W_feedforward = bnb.nn.Linear4bit(
                input_size, num_neurons, bias=False
            )
        else:
            # Standard FP32 (for training)
            self.W_feedforward = nn.Parameter(
                torch.randn(num_neurons, input_size)
            )
```

**Toggle with one flag!**

---

## Training Considerations

### Naive 4-bit Training: Doesn't Work Well
- Gradients need higher precision
- Weight updates get quantized away
- Learning stalls

### QLoRA Approach: Works
- Base model in 4-bit (frozen)
- Low-rank adapters in FP16 (trainable)
- Update only adapters, keep base quantized

```python
from peft import LoraConfig, get_peft_model

# Base model in 4-bit
base_model = PredictiveCodingNetwork(use_4bit=True)

# Add LoRA adapters (trainable)
lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=16,
    target_modules=["W_feedforward", "W_feedback"],
    lora_dropout=0.1
)

model = get_peft_model(base_model, lora_config)

# Now can train adapters while base is 4-bit
```

**Trainable params:** ~1-5% of full model
**Memory:** ~1/8 of FP32

---

## Recommended Strategy

### For Prototyping (Now):

1. **Train in FP32**
   - Get it working first
   - Don't optimize prematurely

2. **Convert to 4-bit for inference**
   - Test if performance drops
   - If acceptable, keep it

3. **If need more memory for training:**
   - Use QLoRA (4-bit base + FP16 adapters)
   - Or just train smaller models in FP32

### Code Setup:

```python
class PredictiveCodingNetwork:
    def __init__(self, use_4bit=False):
        self.use_4bit = use_4bit

        # All subnets can toggle 4-bit
        self.vision = VisionSubnet(use_4bit=use_4bit)
        self.audio = AudioSubnet(use_4bit=use_4bit)
        # ...

# During development
model = PredictiveCodingNetwork(use_4bit=False)  # FP32 for training

# For inference / deployment
model = PredictiveCodingNetwork(use_4bit=True)   # 4-bit for speed
```

---

## Summary

### Basal Dendrites Correction:
- **Basal dendrites** receive BOTH bottom-up AND lateral
- Our `W_basal` should be split into:
  - `W_feedforward` (bottom-up)
  - `W_lateral` (same layer)
- **Apical dendrites** (`W_feedback`) was correct

### 4-Bit Quantization:
- ✓ CUDA supports it (bitsandbytes, PyTorch native)
- ✓ Biologically plausible (26 synaptic states ≈ 4.7 bits)
- ✓ 8× memory reduction (92 MB → 11.5 MB)
- ✓ 2-3× speed improvement
- ✓ Enables rapid prototyping (larger models in same memory)

**Recommendation:**
1. Fix basal/apical terminology → feedforward/lateral/feedback
2. Train in FP32 first (get it working)
3. Add 4-bit inference option (easy win)
4. Use QLoRA if need memory during training

Does this clarify both the dendrite structure and quantization approach?
