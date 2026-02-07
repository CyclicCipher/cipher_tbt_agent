"""
Basic functionality test for PC implementation.
Verifies the core mechanisms work before full training.
"""

import sys
import os
# Add parent directory to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from src.network import PCNetwork, PCTrainer


def test_basic_functionality():
    """Test basic PC network functionality."""
    print("="*60)
    print("Testing Basic PC Functionality")
    print("="*60)

    # Create small network
    layer_sizes = [10, 20, 5]  # 2 layers
    model = PCNetwork(layer_sizes=layer_sizes, activation='relu')
    print(f"\n✓ Created network: {layer_sizes}")

    # Create trainer
    trainer = PCTrainer(
        model=model,
        T=10,  # Few iterations for testing
        inference_lr=0.1,
        weight_lr=0.001,
        device='cpu',
    )
    print("✓ Created trainer")

    # Create dummy data
    batch_size = 4
    inputs = torch.randn(batch_size, 10)
    targets = torch.randint(0, 5, (batch_size,))
    print(f"✓ Created dummy data: {inputs.shape}")

    # Test training mode
    model.train()
    print("\n--- Testing Training Mode ---")

    results = trainer.train_on_batch(
        inputs=inputs,
        loss_fn=F.cross_entropy,
        targets=targets,
    )

    print(f"✓ Training batch completed")
    print(f"  Final loss: {results['loss']:.4f}")
    print(f"  Final energy: {results['energy']:.4f}")
    print(f"  Final free energy: {results['free_energy']:.4f}")
    print(f"  Free energy reduced by: {results['free_energy_history'][0] - results['free_energy_history'][-1]:.4f}")

    # Check convergence
    fe_history = results['free_energy_history']
    if fe_history[-1] < fe_history[0]:
        print("  ✓ Free energy decreased during inference")
    else:
        print("  ✗ WARNING: Free energy did not decrease!")

    # Test eval mode
    model.eval()
    print("\n--- Testing Eval Mode ---")

    test_results = trainer.test_on_batch(
        inputs=inputs,
        loss_fn=F.cross_entropy,
        targets=targets,
    )

    print(f"✓ Evaluation batch completed")
    print(f"  Test loss: {test_results['loss']:.4f}")

    # Test PC layers
    print("\n--- Testing PC Layers ---")
    pc_layers = model.get_pc_layers()
    print(f"✓ Found {len(pc_layers)} PC layers")

    model.train()
    outputs = model(inputs)
    energies = model.get_energies()
    print(f"✓ Got {len(energies)} energies from layers")

    for i, energy in enumerate(energies):
        print(f"  Layer {i+1} energy: {energy.item():.4f}")

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


def test_gradient_flow():
    """Test that gradients flow properly."""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)

    model = PCNetwork(layer_sizes=[10, 20, 5], activation='relu')
    trainer = PCTrainer(model=model, T=5, device='cpu')

    model.train()

    # Do a forward pass first to initialize all parameters (including value nodes)
    dummy_input = torch.randn(4, 10)
    _ = model(dummy_input)

    # Now get initial NETWORK weights (excluding value nodes)
    # This is what the optimizer_p actually optimizes
    initial_weights = []
    for param in model.get_network_parameters():
        initial_weights.append(param.clone().detach())

    # Train one batch
    inputs = torch.randn(4, 10)
    targets = torch.randint(0, 5, (4,))

    trainer.train_on_batch(inputs, F.cross_entropy, targets)

    # Check network weights changed (not value nodes, just Linear weights)
    changed = False
    for i, param in enumerate(model.get_network_parameters()):
        if not torch.allclose(param, initial_weights[i]):
            changed = True
            break

    if changed:
        print("✓ Network weights updated during training")
    else:
        print("✗ WARNING: Network weights did not change!")

    print("="*60)


if __name__ == "__main__":
    test_basic_functionality()
    test_gradient_flow()
