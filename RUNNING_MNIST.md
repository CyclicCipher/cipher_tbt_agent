# Running the 7-Layer PC Network on MNIST

## Installation

```bash
# Install dependencies
pip install torch torchvision matplotlib tqdm numpy

# Or use conda
conda install pytorch torchvision matplotlib tqdm numpy -c pytorch
```

## Quick Test

Before running full training, verify the implementation works:

```bash
python test_pc_basic.py
```

This will test:
- Basic forward/backward passes
- Value node optimization
- Gradient flow
- Two-phase training algorithm

Expected output:
```
✓ Free energy decreased during inference
✓ Weights updated during training
✓ All tests passed!
```

## Full MNIST Training

Run the complete training:

```bash
python train_mnist_pc.py
```

### What to Expect

**Training time:** ~30-60 minutes (CPU) or ~5-10 minutes (GPU)

**Progress:** You'll see:
- Epoch-by-epoch progress bars
- Training/test accuracy and loss
- Free energy convergence per batch
- Per-layer energy diagnostics

**Outputs:**
- `best_pc_model.pt` - Best model checkpoint
- `diagnostics_epoch_N.png` - Diagnostics after each epoch
- `diagnostics_final.png` - Final comprehensive diagnostics

### Success Criteria

From NETWORK_PROPOSAL.md:

- ✓ Test accuracy >95% (comparable to backprop)
- ✓ No vanishing error signals in deep layers
- ✓ Inference converges within 35 iterations

### Diagnostics to Monitor

The diagnostic plots show:

1. **Accuracy over Training** - Should steadily increase
2. **Loss over Training** - Should decrease
3. **Per-Layer Prediction Errors** - Check for vanishing errors
4. **Inference Convergence** - Free energy should decrease each iteration
5. **Energy Ratio (Deep/Shallow)** - Should stay above 0.01
6. **Summary Statistics** - Best/final accuracy, warnings

## Troubleshooting

### Issue: Vanishing Errors

**Symptom:** Deep layer energies are 100x smaller than shallow
**Solution:** Add μPC residual scaling (see next section)

### Issue: Poor Accuracy (<90%)

**Possible causes:**
1. Too few inference iterations
2. Learning rates too high/low
3. Need μPC scaling for stability

**Try:**
```python
# Increase inference iterations
T_inference = 50  # instead of 35

# Adjust learning rates
inference_lr = 0.05  # reduce if unstable
weight_lr = 0.0005  # reduce if unstable
```

### Issue: NaN/Inf Values

**Cause:** Numerical instability
**Solution:**
1. Reduce learning rates
2. Add gradient clipping
3. Check weight initialization

## Adding μPC Scaling (If Needed)

If you see vanishing errors or instability, implement residual connections with μPC scaling:

```python
# In pc_layer.py, add residual scaling
class PCLayerWithResidual(PCLayer):
    def __init__(self, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale  # 1/√L

    def forward(self, mu):
        if not self.training:
            return mu

        # Standard PC
        if self._x is None or self._is_sample_x or mu.shape != self._x.shape:
            self._x = nn.Parameter(mu.detach().clone(), requires_grad=True)
            self._is_sample_x = False

        # Compute energy
        error = mu.detach() - self._x
        self._energy = 0.5 * (error ** 2).sum()

        # Add scaled residual connection
        return self._x + self.residual_scale * mu.detach()
```

Then create network with:
```python
# For 7 layers: residual_scale = 1/√7 ≈ 0.378
model = PCNetwork(layer_sizes=[784, 256, 256, 256, 256, 256, 128, 10],
                  activation='relu',
                  use_residual=True,
                  residual_scale=0.378)
```

## Comparing to Backpropagation

To verify PC performs comparably, train a standard MLP:

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Train with standard Adam
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Should achieve ~97-98% on MNIST
```

## Next Steps After MNIST

Once you achieve >95% accuracy:

1. **Document results** in MISTAKES.md
2. **Test on harder datasets** (FashionMNIST, CIFAR-10)
3. **Integrate with active inference** wrapper
4. **Add multimodal inputs** (vision + audio)
5. **Scale to game environment**

## Performance Notes

**Inference overhead:**
- PC requires 35 forward passes per batch vs 1 for backprop
- ~20-35x slower per batch
- Can be parallelized across batch dimension
- Trade-off: biological plausibility + local learning

**Memory usage:**
- Value nodes (x) add memory per layer
- ~1.5-2x memory vs standard MLP

**Optimization:**
- Use GPU (orders of magnitude faster)
- Reduce T_inference if convergence is fast
- Batch size affects inference stability
