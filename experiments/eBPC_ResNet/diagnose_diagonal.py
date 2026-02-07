"""Diagnose diagonal eBPC failure — trace NaN/explosion source."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from experiments.eBPC_ResNet.ebpc_diagonal_layer import DiagonaleBPCNetwork
from experiments.eBPC_ResNet.ebpc_diagonal_trainer import DiagonaleBPCTrainer


def check_tensor(name, t):
    """Print tensor stats and flag issues."""
    if t.numel() == 0:
        print(f"  {name}: EMPTY")
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    flag = ""
    if has_nan: flag += " *** NaN ***"
    if has_inf: flag += " *** Inf ***"
    if t.numel() <= 4:
        print(f"  {name}: {t.tolist()}{flag}")
    else:
        print(f"  {name}: min={t.min().item():.6e}, max={t.max().item():.6e}, "
              f"mean={t.mean().item():.6e}, std={t.std().item():.6e}{flag}")


def diagnose_layer(i, layer):
    """Print all parameters of a layer."""
    print(f"\n  --- Layer {i} ({layer.out_features}x{layer.in_features}) ---")
    check_tensor(f"η1 (V^-1 diag)", layer.eta1)
    check_tensor(f"η2 (MV^-1)", layer.eta2)
    check_tensor(f"η3", layer.eta3)
    check_tensor(f"η4", layer.eta4.unsqueeze(0))
    check_tensor(f"_min_phi", layer._min_phi)

    M, V_diag, Psi_diag, nu = layer.natural_to_standard()
    check_tensor(f"M (weight mean)", M)
    check_tensor(f"V_diag (col var)", V_diag)
    check_tensor(f"Psi_diag", Psi_diag)
    print(f"  nu: {nu}")

    # Raw Phi before clamping
    raw_Phi = layer.eta3 - (M * layer.eta2).sum(dim=1)
    check_tensor(f"raw Phi (before clamp)", raw_Phi)
    print(f"  Phi clamped? {(raw_Phi < layer._min_phi).any().item()}")

    prec = layer.get_expected_precision_diag()
    check_tensor(f"E[Σ^-1] (precision)", prec)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

    model = DiagonaleBPCNetwork([784, 128, 128, 128, 10])
    # Disable AMP for cleaner debugging
    trainer = DiagonaleBPCTrainer(model=model, T=5, e_lr=0.01, kappa=0.25,
                                  use_amp=False, adaptive_T=False, device=device)

    print("=" * 60)
    print("INITIAL STATE")
    print("=" * 60)
    for i, layer in enumerate(model.layers):
        diagnose_layer(i, layer)

    # Test eval forward
    model.eval()
    sample_x = torch.randn(4, 784).to(device)
    with torch.no_grad():
        out = model(sample_x)
    check_tensor("\nEval output", out)
    model.train()

    print("\n\n")
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= 5:
            break

        data = data.view(data.size(0), -1)
        print("=" * 60)
        print(f"BATCH {batch_idx}")
        print("=" * 60)

        # Manual forward to trace values
        inputs = data.to(device)
        targets = target.to(device)
        target_oh = F.one_hot(targets, 10).float()

        # Check errors before training
        errors = model.init_errors(inputs)
        for ei, e in enumerate(errors):
            check_tensor(f"  error[{ei}] init", e)

        # Forward with errors
        output, states, last_input = model.epc_forward(inputs, errors)
        check_tensor("  output (before optim)", output)
        for si, s in enumerate(states):
            check_tensor(f"  state[{si}]", s)
        check_tensor("  last_input (to output layer)", last_input)

        # Energy
        E = trainer._compute_energy(errors, output, target_oh)
        print(f"  Energy before optim: {E.item():.6e}")

        # Now run full training step
        results = trainer.train_on_batch(data, target)
        print(f"\n  Training result: loss={results['loss']}, energy={results['energy']:.6e}")
        print(f"  Layer energies: {[f'{e:.6e}' for e in results['layer_energies']]}")

        # Check state after update
        print(f"\n  --- After Hebbian update ---")
        for i, layer in enumerate(model.layers):
            diagnose_layer(i, layer)

        # Check eval forward
        model.eval()
        with torch.no_grad():
            out = model(inputs[:4])
        check_tensor("  Eval output (4 samples)", out)
        model.train()

        print()


if __name__ == "__main__":
    main()
