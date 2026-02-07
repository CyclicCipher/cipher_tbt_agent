"""
Diagnostics for low-rank eBPC NaN explosion.

Tests 6 hypotheses:
  H1: bfloat16 AMP corrupts Hebbian update inputs (states are bfloat16 after autocast)
  H2: Woodbury C matrix is ill-conditioned
  H3: η1 low-rank update diverges from full η1
  H4: M computation via Woodbury diverges from full M
  H5: Specific layer is the NaN origin (which blows up first?)
  H6: Full eBPC reference works fine on the same data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.eBPC_ResNet.ebpc_lowrank_layer import LowRankeBPCNetwork, LowRankeBPCLayer
from experiments.eBPC_ResNet.ebpc_lowrank_trainer import LowRankeBPCTrainer
from experiments.eBPC.ebpc_layer import eBPCNetwork
from experiments.eBPC.ebpc_trainer import eBPCTrainer


def get_mnist_batch(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data, target = next(iter(loader))
    return data.view(data.size(0), -1), target


def test_h1_bfloat16():
    """H1: Check if bfloat16 AMP contaminates the Hebbian update inputs."""
    print("\n" + "="*80)
    print("H1: bfloat16 AMP contamination check")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data, target = get_mnist_batch()
    data, target = data.to(device), target.to(device)
    target_oh = F.one_hot(target, 10).float()

    model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=128).to(device)

    # Run E-step WITH AMP (as in trainer)
    errors = model.init_errors(data)
    import torch.optim as optim
    error_optim = optim.Adam(errors, lr=0.01)

    print("\n--- With AMP (use_amp=True) ---")
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output, states, last_input = model.epc_forward(data, errors)
        E = 0.5 * ((target_oh - output) ** 2).sum(dim=1).mean()
    E.backward()
    error_optim.step()

    # Check dtypes of states
    print(f"  output dtype: {output.dtype}")
    for i, s in enumerate(states):
        print(f"  states[{i}] dtype: {s.dtype}, range: [{s.min().item():.4f}, {s.max().item():.4f}]")

    # Now trace what happens in Hebbian update
    z_star = [s.detach() for s in states] + [target_oh]
    h0 = model.activation(data)
    pre_activations = [model._augment_with_bias(h0)]
    for z in z_star[:-1]:
        h = model.activation(z)
        pre_activations.append(model._augment_with_bias(h))

    print("\n  Pre-activation dtypes (used in Hebbian update):")
    for i, f_pre in enumerate(pre_activations):
        print(f"    Layer {i+1} f_pre dtype: {f_pre.dtype}, "
              f"range: [{f_pre.min().item():.4f}, {f_pre.max().item():.4f}], "
              f"has_nan: {torch.isnan(f_pre).any().item()}")

    print(f"\n  z_star dtypes:")
    for i, z in enumerate(z_star):
        print(f"    z_star[{i}] dtype: {z.dtype}")

    # Now run WITHOUT AMP
    print("\n--- Without AMP (use_amp=False) ---")
    errors2 = model.init_errors(data)
    error_optim2 = optim.Adam(errors2, lr=0.01)
    output2, states2, _ = model.epc_forward(data, errors2)
    E2 = 0.5 * ((target_oh - output2) ** 2).sum(dim=1).mean()
    E2.backward()
    error_optim2.step()

    for i, s in enumerate(states2):
        print(f"  states2[{i}] dtype: {s.dtype}")

    z_star2 = [s.detach() for s in states2] + [target_oh]
    pre_activations2 = [model._augment_with_bias(model.activation(data))]
    for z in z_star2[:-1]:
        pre_activations2.append(model._augment_with_bias(model.activation(z)))

    print("\n  Pre-activation dtypes WITHOUT AMP:")
    for i, f_pre in enumerate(pre_activations2):
        print(f"    Layer {i+1} f_pre dtype: {f_pre.dtype}")

    # KEY: Compare ss2 precision
    print("\n  --- ss2 = z_post.T @ f_pre precision comparison ---")
    for i in range(len(model.layers)):
        f_amp = pre_activations[i]
        z_amp = z_star[i]
        f_no = pre_activations2[i]
        z_no = z_star2[i]

        ss2_amp = z_amp.T @ f_amp
        ss2_no = z_no.T @ f_no

        print(f"  Layer {i+1}:")
        print(f"    ss2 AMP dtype={ss2_amp.dtype}, max={ss2_amp.abs().max().item():.4e}, "
              f"has_nan={torch.isnan(ss2_amp).any().item()}")
        print(f"    ss2 no-AMP dtype={ss2_no.dtype}, max={ss2_no.abs().max().item():.4e}, "
              f"has_nan={torch.isnan(ss2_no).any().item()}")

    # CRITICAL: Check what happens to eigendecomposition in bfloat16
    print("\n  --- Eigendecomposition in bfloat16 vs float32 ---")
    f_pre_bf16 = pre_activations[1]  # Layer 2, bfloat16
    f_pre_f32 = f_pre_bf16.float()   # Convert to float32

    A_bf16 = f_pre_bf16.T  # [129, 128]
    A_f32 = f_pre_f32.T

    B_bf16 = A_bf16.T @ A_bf16  # [128, 128] in bfloat16
    B_f32 = A_f32.T @ A_f32     # [128, 128] in float32

    print(f"  B_bf16 dtype={B_bf16.dtype}, cond={torch.linalg.cond(B_bf16.float()).item():.2e}")
    print(f"  B_f32  dtype={B_f32.dtype}, cond={torch.linalg.cond(B_f32).item():.2e}")

    try:
        evals_bf16, evecs_bf16 = torch.linalg.eigh(B_bf16.float())
        evals_f32, evecs_f32 = torch.linalg.eigh(B_f32)
        print(f"  evals bf16 range: [{evals_bf16.min().item():.4e}, {evals_bf16.max().item():.4e}]")
        print(f"  evals f32  range: [{evals_f32.min().item():.4e}, {evals_f32.max().item():.4e}]")
        eval_diff = (evals_bf16 - evals_f32).abs().max().item()
        print(f"  Max eigenvalue difference: {eval_diff:.4e}")
    except Exception as e:
        print(f"  eigh failed: {e}")


def test_h2_woodbury_conditioning():
    """H2: Check Woodbury C matrix condition number after updates."""
    print("\n" + "="*80)
    print("H2: Woodbury C matrix conditioning")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data, target = get_mnist_batch()

    model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=128).to(device)
    trainer = LowRankeBPCTrainer(model=model, T=5, e_lr=0.01, kappa=0.25,
                                  adaptive_T=True, use_amp=True, device=device)

    # Run a few batches
    loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=128, shuffle=True, drop_last=True
    )

    for batch_idx, (batch_data, batch_target) in enumerate(loader):
        if batch_idx >= 15:
            break
        batch_data = batch_data.view(batch_data.size(0), -1)
        results = trainer.train_on_batch(batch_data, batch_target)

        # Check C conditioning per layer
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                d_inv = 1.0 / layer.eta1_d
                if layer.eta1_U.abs().max() < 1e-10:
                    print(f"  Batch {batch_idx}, Layer {i+1}: U=zero (diagonal-only)")
                    continue
                d_inv_U = d_inv.unsqueeze(1) * layer.eta1_U
                C = torch.eye(layer.rank_k, device=device) + layer.eta1_U.T @ d_inv_U
                cond = torch.linalg.cond(C).item()
                M = layer._M_cache
                print(f"  Batch {batch_idx}, Layer {i+1}: "
                      f"C_cond={cond:.2e}, "
                      f"|M|_max={M.abs().max().item():.2e}, "
                      f"d_range=[{layer.eta1_d.min().item():.2e}, {layer.eta1_d.max().item():.2e}], "
                      f"U_max={layer.eta1_U.abs().max().item():.2e}, "
                      f"psi_inv_min={layer.psi_inv_diag.min().item():.2e}")

        if torch.isnan(torch.tensor(results['loss'])):
            print(f"\n  NaN at batch {batch_idx}!")
            break


def test_h5_which_layer_first():
    """H5: Which layer diverges first? Detailed per-layer per-batch tracking."""
    print("\n" + "="*80)
    print("H5: Which layer diverges first?")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=128).to(device)
    trainer = LowRankeBPCTrainer(model=model, T=5, e_lr=0.01, kappa=0.25,
                                  adaptive_T=True, use_amp=True, device=device)

    loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=128, shuffle=True, drop_last=True
    )

    print(f"\n  {'Batch':>5} | {'Layer':>5} | {'|M|_max':>12} | {'|eta2|_max':>12} | "
          f"{'d_max':>12} | {'|U|_max':>12} | {'psi_inv_min':>12} | {'rank':>5}")
    print("  " + "-"*95)

    for batch_idx, (batch_data, batch_target) in enumerate(loader):
        if batch_idx >= 15:
            break
        batch_data = batch_data.view(batch_data.size(0), -1)
        results = trainer.train_on_batch(batch_data, batch_target)

        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                M = layer._M_cache
                U = layer.eta1_U
                eff_rank = 0
                if U.abs().max() > 1e-10:
                    sv = torch.linalg.svdvals(U)
                    eff_rank = (sv > sv[0] * 0.01).sum().item()

                print(f"  {batch_idx:>5} | {i+1:>5} | {M.abs().max().item():>12.2e} | "
                      f"{layer.eta2.abs().max().item():>12.2e} | "
                      f"{layer.eta1_d.max().item():>12.2e} | "
                      f"{U.abs().max().item():>12.2e} | "
                      f"{layer.psi_inv_diag.min().item():>12.2e} | "
                      f"{eff_rank:>5}")

        if torch.isnan(torch.tensor(results['loss'])):
            print(f"\n  NaN at batch {batch_idx}!")
            break


def test_h1_fix_cast_to_float32():
    """H1-fix: Run training with explicit float32 cast of z_star before Hebbian update."""
    print("\n" + "="*80)
    print("H1-fix: Training with float32 cast before Hebbian update")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=128).to(device)

    # Monkey-patch the trainer to cast z_star to float32
    trainer = LowRankeBPCTrainer(model=model, T=5, e_lr=0.01, kappa=0.25,
                                  adaptive_T=True, use_amp=True, device=device)

    original_update = trainer._bayesian_update_weights_lowrank

    def patched_update(inputs, z_star):
        # Cast all z_star to float32 before Hebbian update
        z_star_f32 = [z.float() for z in z_star]
        return original_update(inputs.float(), z_star_f32)

    trainer._bayesian_update_weights_lowrank = patched_update

    loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=128, shuffle=True, drop_last=True
    )

    print(f"\n  Running 50 batches with float32 Hebbian update...")
    for batch_idx, (batch_data, batch_target) in enumerate(loader):
        if batch_idx >= 50:
            break
        batch_data = batch_data.view(batch_data.size(0), -1)
        results = trainer.train_on_batch(batch_data, batch_target)

        if batch_idx % 10 == 0 or torch.isnan(torch.tensor(results['loss'])):
            model.eval()
            with torch.no_grad():
                outputs = model(batch_data.to(device))
                pred = outputs.argmax(dim=1)
                acc = (pred == batch_target.to(device)).float().mean().item()
            model.train()

            with torch.no_grad():
                for i, layer in enumerate(model.layers):
                    M = layer._M_cache
                    rank_info = ""
                    if layer.eta1_U.abs().max() > 1e-10:
                        sv = torch.linalg.svdvals(layer.eta1_U)
                        eff_rank = (sv > sv[0] * 0.01).sum().item()
                        rank_info = f"rank={eff_rank}"
                    else:
                        rank_info = "rank=0"

                    print(f"  Batch {batch_idx}, Layer {i+1}: "
                          f"|M|={M.abs().max().item():.2e}, "
                          f"psi_inv_min={layer.psi_inv_diag.min().item():.2e}, "
                          f"{rank_info}")

            print(f"  Batch {batch_idx}: loss={results['loss']:.4f}, acc={acc:.2%}, "
                  f"T={results['actual_T']}")

            if torch.isnan(torch.tensor(results['loss'])):
                print(f"\n  NaN at batch {batch_idx}!")
                break

    if not torch.isnan(torch.tensor(results['loss'])):
        print(f"\n  SUCCESS: No NaN after 50 batches!")
        print(f"  Final loss: {results['loss']:.4f}")


def test_h6_full_ebpc_reference():
    """H6: Run full eBPC on same data to confirm reference works."""
    print("\n" + "="*80)
    print("H6: Full eBPC reference comparison")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = eBPCNetwork([784, 128, 128, 128, 10]).to(device)
    trainer = eBPCTrainer(model=model, T=5, e_lr=0.01, kappa=0.25, device=device)

    loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=128, shuffle=True, drop_last=True
    )

    print(f"\n  Running 15 batches with full eBPC...")
    for batch_idx, (batch_data, batch_target) in enumerate(loader):
        if batch_idx >= 15:
            break
        batch_data = batch_data.view(batch_data.size(0), -1)
        results = trainer.train_on_batch(batch_data, batch_target)

        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                M, V, Psi, nu = layer.natural_to_standard()
                print(f"  Batch {batch_idx}, Layer {i+1}: "
                      f"|M|={M.abs().max().item():.2e}, "
                      f"|V|_diag_max={V.diag().max().item():.2e}, "
                      f"Psi_diag_min={Psi.diag().min().item():.2e}")

        if torch.isnan(torch.tensor(results['loss'])):
            print(f"\n  NaN at batch {batch_idx}! (UNEXPECTED)")
            break
        else:
            print(f"  Batch {batch_idx}: loss={results['loss']:.4f}")


def main():
    print("="*80)
    print("Low-Rank eBPC Diagnostics")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # H1: The most likely culprit — check bfloat16 contamination
    test_h1_bfloat16()

    # H5: Which layer diverges first?
    test_h5_which_layer_first()

    # H1-fix: Does casting to float32 fix it?
    test_h1_fix_cast_to_float32()

    # H6: Does full eBPC work on same data?
    test_h6_full_ebpc_reference()

    # H2: Woodbury conditioning (run last — slower)
    test_h2_woodbury_conditioning()

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
