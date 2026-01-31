"""Quick shape verification test for MNIST pipeline."""

import torch
from torchvision import datasets, transforms
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from categorical_network import PCConvVisionPreprocessor

# Load MNIST with same transform as training
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1ch→3ch
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=False,
    transform=transform
)

# Test single image
image, label = train_dataset[0]
print(f"✓ Single MNIST image: shape={image.shape}, dtype={image.dtype}, range=[{image.min():.3f}, {image.max():.3f}]")
assert image.shape == (3, 100, 100), f"Expected (3, 100, 100), got {image.shape}"
print(f"✓ Shape assertion passed: {image.shape}")

# Test model can accept it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preprocessor = PCConvVisionPreprocessor(dtype=torch.float32, precisions=[1.0, 10.0, 100.0])
preprocessor = preprocessor.to(device)

image = image.to(device)
print(f"✓ Image moved to device: {device}")

# Test forward pass
try:
    output = preprocessor.forward(image, num_iterations=2, inference_lr=0.1, use_lateral=True)
    print(f"✓ Forward pass successful: output shape={output.shape}")
    print(f"\nAll shape checks passed! Ready to train.")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    raise
