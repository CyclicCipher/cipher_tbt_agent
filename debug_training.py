"""
Debug script to diagnose why PC training isn't working.
Prints detailed information about what happens during one batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.network import PCNetwork, PCTrainer


def main():
    print("="*80)
    print("DEBUG: PC Training on Single Batch")
    print("="*80)

    # Load a single batch of MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)  # Small batch for debugging

    data, target = next(iter(loader))
    data = data.view(data.size(0), -1)  # Flatten

    print(f"\nData shape: {data.shape}")
    print(f"Targets: {target.tolist()}")

    # Create small network for debugging
    layer_sizes = [784, 128, 10]  # Just 2 layers
    model = PCNetwork(layer_sizes=layer_sizes, activation='relu')

    print(f"\nNetwork architecture: {layer_sizes}")
    print(f"Number of PC layers: {len(model.get_pc_layers())}")

    # Count parameters
    network_params = list(model.get_network_parameters())
    print(f"Number of network parameters (Linear weights): {len(network_params)}")
    total_network_params = sum(p.numel() for p in network_params)
    print(f"Total network parameter count: {total_network_params}")

    # Create trainer
    trainer = PCTrainer(
        model=model,
        T=10,  # Just 10 iterations for debugging
        inference_lr=0.1,
        weight_lr=0.001,
        device='cpu',
    )

    # Get initial predictions (before training)
    model.eval()
    with torch.no_grad():
        init_outputs = model(data)
        init_pred = init_outputs.argmax(dim=1)
        init_acc = (init_pred == target).float().mean().item()

    print(f"\n--- BEFORE TRAINING ---")
    print(f"Initial predictions: {init_pred.tolist()}")
    print(f"Initial accuracy: {init_acc:.2%}")

    # Train on this batch
    model.train()

    print(f"\n--- DURING TRAINING ---")
    print(f"Running {trainer.T} inference iterations...")

    results = trainer.train_on_batch(
        inputs=data,
        loss_fn=F.cross_entropy,
        targets=target,
    )

    print(f"\nInference complete:")
    print(f"  Initial free energy: {results['free_energy_history'][0]:.4f}")
    print(f"  Final free energy: {results['free_energy_history'][-1]:.4f}")
    print(f"  Reduction: {results['free_energy_history'][0] - results['free_energy_history'][-1]:.4f}")
    print(f"  Final loss: {results['loss']:.4f}")
    print(f"  Final energy: {results['energy']:.4f}")

    # Check if weights actually changed
    model.eval()
    with torch.no_grad():
        final_outputs = model(data)
        final_pred = final_outputs.argmax(dim=1)
        final_acc = (final_pred == target).float().mean().item()

    print(f"\n--- AFTER TRAINING ---")
    print(f"Final predictions: {final_pred.tolist()}")
    print(f"Final accuracy: {final_acc:.2%}")
    print(f"Accuracy change: {(final_acc - init_acc):.2%}")

    # Check if output distribution changed
    print(f"\n--- OUTPUT DISTRIBUTION ---")
    print("Initial output (first example):", init_outputs[0].tolist()[:5], "...")
    print("Final output (first example):", final_outputs[0].tolist()[:5], "...")

    # Check if any weights changed
    network_params_after = list(model.get_network_parameters())
    weights_changed = False
    max_change = 0.0
    for i, (p_before, p_after) in enumerate(zip(network_params, network_params_after)):
        change = (p_after - p_before).abs().max().item()
        max_change = max(max_change, change)
        if change > 1e-6:
            weights_changed = True

    print(f"\n--- WEIGHT UPDATES ---")
    print(f"Weights changed: {weights_changed}")
    print(f"Max weight change: {max_change:.6f}")

    if not weights_changed:
        print("\n⚠️  CRITICAL: Weights did not change at all!")
        print("This means the weight optimizer is not working.")

        # Debug: Check if optimizer_p has parameters
        print(f"\nOptimizer_p parameter groups: {len(trainer.optimizer_p.param_groups)}")
        if len(trainer.optimizer_p.param_groups) > 0:
            print(f"Number of parameters in optimizer_p: {len(trainer.optimizer_p.param_groups[0]['params'])}")

        # Debug: Check value nodes
        value_nodes = list(model.get_value_nodes())
        print(f"\nNumber of value nodes: {len(value_nodes)}")

    # Train for more batches to see if learning happens
    print("\n" + "="*80)
    print("Training for 10 more batches...")
    print("="*80)

    accuracies = []
    for batch_idx in range(10):
        data_batch, target_batch = next(iter(loader))
        data_batch = data_batch.view(data_batch.size(0), -1)

        model.train()
        results = trainer.train_on_batch(
            inputs=data_batch,
            loss_fn=F.cross_entropy,
            targets=target_batch,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(data_batch)
            pred = outputs.argmax(dim=1)
            acc = (pred == target_batch).float().mean().item()

        accuracies.append(acc)
        print(f"Batch {batch_idx+1}: acc={acc:.2%}, loss={results['loss']:.4f}")

    print(f"\nMean accuracy over 10 batches: {sum(accuracies)/len(accuracies):.2%}")
    print(f"Accuracy is improving: {accuracies[-1] > accuracies[0]}")


if __name__ == "__main__":
    main()
