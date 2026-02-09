"""
KRONOS Diagnostics Script.

Tests 6 hypotheses about why KRONOS underperforms Adam:
  H1: Gradient clipping fires too often, undoing preconditioning
  H2: SGD+momentum is a poor match for KFAC-preconditioned gradients
  H3: LRPD rank captures too little variance (especially layer 1)
  H4: Damping dominates factor eigenvalues (preconditioner ≈ scaled identity)
  H5: LRPD approximation error compounds over streaming updates
  H6: Factor staleness (10-step update lag) causes outdated preconditioning

Uses the best KRONOS config: Newton T=2, r=32, lr=3e-3, damping=0.01.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.ePC_ResNet.epc_model import PCE
from experiments.ePC_ResNet.architectures import get_mlp_mnist
from src.optimizers.kronos import KRONOS, _KronosState


def get_mnist_loaders(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            preds = model(data).argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
    return correct / total


def run_diagnostics(train_loader, test_loader, device, num_epochs=3):
    """Train with KRONOS and collect per-step diagnostics."""

    # Best config from benchmark
    architecture = get_mlp_mnist(hidden_size=128, num_hidden=3)
    model = PCE(
        architecture, iters=2, e_lr=0.01, output_loss='ce',
        error_optim='newton', damping=0.1,
    ).to(device)

    kronos = KRONOS(
        model, lr=0.003, damping=0.01, rank=32,
        ema_decay=0.95, update_freq=10, momentum=0.9, grad_clip=1.0,
    )

    # ---------------------------------------------------------------
    # Diagnostic accumulators
    # ---------------------------------------------------------------
    # H1: Gradient clipping
    clip_log = []  # (step, layer_idx, pre_clip_norm, clip_coef)

    # H2: Momentum alignment
    momentum_alignment_log = []  # (step, layer_idx, cosine_similarity)

    # H3: Eigenvalue spectra (sampled periodically)
    spectra_log = []  # (step, layer_idx, {d_a, U_a_norms, d_g, U_g_norms, trace_captured_a, trace_total_a, ...})

    # H4: Damping dominance
    damping_log = []  # (step, layer_idx, damping_a, damping_g, median_d_a, median_d_g, frac_dominated_a, frac_dominated_g)

    # H5: LRPD approximation quality (sampled)
    approx_quality_log = []  # (step, layer_idx, relative_error)

    # H6: Factor staleness
    staleness_log = []  # (step, layer_idx, factor_age_in_steps)

    # General
    raw_grad_norms = []  # (step, layer_idx, raw_norm)
    precond_grad_norms = []  # (step, layer_idx, precond_norm)
    weight_update_norms = []  # (step, layer_idx, update_norm)
    train_accs = []
    test_accs = []
    step_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_correct = epoch_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            batch_size = data.size(0)
            step_count += 1

            # Phase 1: Inference
            model(data, target)

            # Phase 2: Weight update
            kronos.zero_grad()
            loss = model.compute_weight_loss(data, target, batch_size)
            loss.backward()

            # === COLLECT PRE-STEP DIAGNOSTICS ===
            for layer_idx, (module, state) in enumerate(kronos._states.items()):
                if not state.initialized or module.weight.grad is None:
                    continue

                grad_w = module.weight.grad.clone()
                grad_b = module.bias.grad.clone() if module.bias is not None else None

                # Raw gradient norm
                raw_norm = grad_w.norm().item()
                if grad_b is not None:
                    raw_norm = (grad_w.norm() ** 2 + grad_b.norm() ** 2).sqrt().item()
                raw_grad_norms.append((step_count, layer_idx, raw_norm))

                # Preconditioned gradient (before clipping)
                precond_w, precond_b = state.precondition(grad_w, grad_b)
                precond_norm = precond_w.norm().item()
                if precond_b is not None:
                    precond_norm = (precond_w.norm() ** 2 + precond_b.norm() ** 2).sqrt().item()
                precond_grad_norms.append((step_count, layer_idx, precond_norm))

                # H1: Would clipping fire?
                clip_coef = 1.0 / (precond_norm + 1e-6)  # grad_clip=1.0
                clip_log.append((step_count, layer_idx, precond_norm, min(clip_coef, 1.0)))

                # H2: Momentum alignment
                p_state = kronos.state.get(module.weight, {})
                if 'momentum_w' in p_state:
                    mom_buf = p_state['momentum_w']
                    cos_sim = F.cosine_similarity(
                        precond_w.flatten().unsqueeze(0),
                        mom_buf.flatten().unsqueeze(0)
                    ).item()
                    momentum_alignment_log.append((step_count, layer_idx, cos_sim))

                # H4: Damping dominance
                trace_a = state._lrpd_trace(state.d_a, state.U_a).item()
                trace_g = state._lrpd_trace(state.d_g, state.U_g).item()
                pi = (trace_a * state.out_dim) / (trace_g * state.aug_in_dim + 1e-8)
                damping_a = (state.damping * pi) ** 0.5
                damping_g = (state.damping / pi) ** 0.5
                median_d_a = state.d_a.median().item()
                median_d_g = state.d_g.median().item()
                frac_dom_a = (state.d_a < damping_a).float().mean().item()
                frac_dom_g = (state.d_g < damping_g).float().mean().item()
                damping_log.append((step_count, layer_idx, damping_a, damping_g,
                                    median_d_a, median_d_g, frac_dom_a, frac_dom_g))

                # H6: Factor staleness
                steps_since_update = step_count - (state.steps * kronos._update_freq)
                staleness_log.append((step_count, layer_idx, state.steps, steps_since_update))

            # === Eigenvalue spectra & LRPD quality (every 50 steps) ===
            if step_count % 50 == 0:
                for layer_idx, (module, state) in enumerate(kronos._states.items()):
                    if not state.initialized:
                        continue

                    # H3: Captured variance
                    d_a = state.d_a.cpu().numpy()
                    U_a = state.U_a.cpu()
                    U_a_norms = (U_a ** 2).sum(dim=0).sqrt().numpy()
                    trace_captured_a = (U_a_norms ** 2).sum()
                    trace_total_a = d_a.sum() + trace_captured_a

                    d_g = state.d_g.cpu().numpy()
                    U_g = state.U_g.cpu()
                    U_g_norms = (U_g ** 2).sum(dim=0).sqrt().numpy()
                    trace_captured_g = (U_g_norms ** 2).sum()
                    trace_total_g = d_g.sum() + trace_captured_g

                    spectra_log.append((step_count, layer_idx, {
                        'd_a_stats': (d_a.min(), np.median(d_a), d_a.max(), d_a.mean()),
                        'd_g_stats': (d_g.min(), np.median(d_g), d_g.max(), d_g.mean()),
                        'U_a_col_norms': sorted(U_a_norms.tolist(), reverse=True),
                        'U_g_col_norms': sorted(U_g_norms.tolist(), reverse=True),
                        'variance_captured_a': trace_captured_a / (trace_total_a + 1e-8),
                        'variance_captured_g': trace_captured_g / (trace_total_g + 1e-8),
                    }))

                    # H5: LRPD approximation quality
                    # Reconstruct LRPD matrix and compare to truth from cached data
                    # We'll use the batch activations to compute a fresh A = a^T a / n
                    # and compare it to the LRPD reconstruction diag(d_a) + U_a @ U_a^T
                    if state._cached_a is not None:
                        a = state._cached_a
                        n = a.shape[0]
                        # True batch covariance
                        A_true = (a.T @ a) / n
                        # LRPD reconstruction
                        d_a_t = state.d_a
                        U_a_t = state.U_a
                        A_lrpd = torch.diag(d_a_t) + U_a_t @ U_a_t.T
                        # Relative Frobenius error
                        err = (A_true - A_lrpd).norm().item()
                        norm_true = A_true.norm().item()
                        rel_err = err / (norm_true + 1e-8)
                        approx_quality_log.append((step_count, layer_idx, rel_err, 'A'))

            # === Take the step ===
            # Record weight norms before step
            old_weights = {m: m.weight.data.clone() for m in kronos._states}
            kronos.step()

            # Weight update norms
            for layer_idx, (module, state) in enumerate(kronos._states.items()):
                delta = (module.weight.data - old_weights[module]).norm().item()
                weight_update_norms.append((step_count, layer_idx, delta))

            # Track accuracy every 20 batches
            if step_count % 20 == 0:
                with torch.no_grad():
                    outputs = model(data)
                    preds = outputs.argmax(dim=1)
                    acc = (preds == target).float().mean().item()
                    train_accs.append((step_count, acc))

        # End of epoch eval
        test_acc = evaluate(model, test_loader, device)
        test_accs.append((epoch + 1, test_acc))
        print(f"  Epoch {epoch+1}: Test {test_acc:.2%}")

    return {
        'clip_log': clip_log,
        'momentum_alignment_log': momentum_alignment_log,
        'spectra_log': spectra_log,
        'damping_log': damping_log,
        'approx_quality_log': approx_quality_log,
        'staleness_log': staleness_log,
        'raw_grad_norms': raw_grad_norms,
        'precond_grad_norms': precond_grad_norms,
        'weight_update_norms': weight_update_norms,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def plot_diagnostics(diag, save_path='diagnose_kronos.png'):
    """6-panel diagnostic plot, one per hypothesis."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    n_layers = 4  # MLP has 4 linear layers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    layer_names = ['L1 (785→128)', 'L2 (129→128)', 'L3 (129→128)', 'L4 (129→10)']

    # ---------------------------------------------------------------
    # H1: Gradient clipping [0,0]
    # ---------------------------------------------------------------
    ax = axes[0, 0]
    ax.set_title('H1: Gradient Clipping\n(precond norm > 1.0 triggers clip)', fontsize=10)
    for l in range(n_layers):
        entries = [(s, norm, coef) for s, li, norm, coef in diag['clip_log'] if li == l]
        if entries:
            steps, norms, coefs = zip(*entries)
            clipped = [1 if c < 1.0 else 0 for c in coefs]
            # Moving average of clip frequency
            window = 50
            if len(clipped) > window:
                clip_rate = np.convolve(clipped, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(clipped)+1), clip_rate,
                        color=colors[l], label=f'{layer_names[l]}', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Clip Frequency (rolling 50)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')

    # ---------------------------------------------------------------
    # H2: Momentum alignment [0,1]
    # ---------------------------------------------------------------
    ax = axes[0, 1]
    ax.set_title('H2: Momentum-Gradient Alignment\n(cos sim: precond grad vs momentum buf)', fontsize=10)
    for l in range(n_layers):
        entries = [(s, cos) for s, li, cos in diag['momentum_alignment_log'] if li == l]
        if entries:
            steps, cosines = zip(*entries)
            window = 50
            if len(cosines) > window:
                ma = np.convolve(cosines, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(cosines)+1), ma,
                        color=colors[l], label=layer_names[l], linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cosine Similarity (rolling 50)')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------------
    # H3: Variance captured by LRPD [0,2]
    # ---------------------------------------------------------------
    ax = axes[0, 2]
    ax.set_title('H3: Variance Captured by Low-Rank\n(trace(UU^T) / trace(A) per factor)', fontsize=10)
    for l in range(n_layers):
        entries = [(s, info) for s, li, info in diag['spectra_log'] if li == l]
        if entries:
            steps = [e[0] for e in entries]
            var_a = [e[1]['variance_captured_a'] for e in entries]
            var_g = [e[1]['variance_captured_g'] for e in entries]
            ax.plot(steps, var_a, '-', color=colors[l], label=f'{layer_names[l]} A', linewidth=1.5)
            ax.plot(steps, var_g, '--', color=colors[l], label=f'{layer_names[l]} G', linewidth=1.0)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction of Trace in UU^T')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------------
    # H4: Damping dominance [1,0]
    # ---------------------------------------------------------------
    ax = axes[1, 0]
    ax.set_title('H4: Damping Dominance\n(fraction of d_i < damping_factor)', fontsize=10)
    for l in range(n_layers):
        entries = [(s, fa, fg) for s, li, da, dg, mda, mdg, fa, fg
                   in diag['damping_log'] if li == l]
        if entries:
            steps, frac_a, frac_g = zip(*entries)
            window = 50
            if len(frac_a) > window:
                ma_a = np.convolve(frac_a, np.ones(window)/window, mode='valid')
                ma_g = np.convolve(frac_g, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(frac_a)+1), ma_a, '-',
                        color=colors[l], label=f'{layer_names[l]} A', linewidth=1.5)
                ax.plot(range(window, len(frac_g)+1), ma_g, '--',
                        color=colors[l], label=f'{layer_names[l]} G', linewidth=1.0)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction Dominated (rolling 50)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    # ---------------------------------------------------------------
    # H5: LRPD approximation quality [1,1]
    # ---------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_title('H5: LRPD Approximation Error\n(||A_true - A_lrpd||_F / ||A_true||_F)', fontsize=10)
    for l in range(n_layers):
        entries = [(s, err) for s, li, err, which in diag['approx_quality_log']
                   if li == l and which == 'A']
        if entries:
            steps, errs = zip(*entries)
            ax.plot(steps, errs, 'o-', color=colors[l], label=layer_names[l],
                    markersize=4, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Relative Frobenius Error')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------------
    # H6: Gradient norms — raw vs preconditioned [1,2]
    # Shows whether preconditioning is actually reshaping the gradient
    # or just scaling it (if all layers scale by same factor → useless)
    # ---------------------------------------------------------------
    ax = axes[1, 2]
    ax.set_title('H6: Preconditioning Effect\n(preconditioned / raw gradient norm ratio)', fontsize=10)
    for l in range(n_layers):
        raw = [(s, n) for s, li, n in diag['raw_grad_norms'] if li == l]
        pre = [(s, n) for s, li, n in diag['precond_grad_norms'] if li == l]
        if raw and pre:
            # Align by step
            ratios = []
            steps = []
            for (s1, r), (s2, p) in zip(raw, pre):
                if r > 1e-10:
                    ratios.append(p / r)
                    steps.append(s1)
            if len(ratios) > 50:
                window = 50
                ma = np.convolve(ratios, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(ratios)+1), ma,
                        color=colors[l], label=layer_names[l], linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('||precond_grad|| / ||raw_grad|| (rolling 50)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('KRONOS Diagnostics — Newton T=2 + KRONOS r=32 lr=3e-3 (MNIST)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic chart saved to {save_path}")
    plt.close()


def print_summary(diag):
    """Print a text summary of diagnostic findings."""
    print(f"\n{'='*70}")
    print("KRONOS DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")

    n_layers = 4
    layer_names = ['L1 (785→128)', 'L2 (129→128)', 'L3 (129→128)', 'L4 (129→10)']

    # Test accuracy
    print(f"\nTest accuracy by epoch:")
    for epoch, acc in diag['test_accs']:
        print(f"  Epoch {epoch}: {acc:.2%}")

    # H1: Clip frequency
    print(f"\n--- H1: Gradient Clipping ---")
    for l in range(n_layers):
        entries = [coef for _, li, _, coef in diag['clip_log'] if li == l]
        if entries:
            n_clipped = sum(1 for c in entries if c < 1.0)
            clip_rate = n_clipped / len(entries)
            avg_clip = np.mean([c for c in entries if c < 1.0]) if n_clipped > 0 else 1.0
            max_precond_norm = max(norm for _, li, norm, _ in diag['clip_log'] if li == l)
            print(f"  {layer_names[l]}: clip_rate={clip_rate:.1%}, "
                  f"avg_clip_coef={avg_clip:.4f}, max_precond_norm={max_precond_norm:.2f}")

    # H2: Momentum alignment
    print(f"\n--- H2: Momentum-Gradient Alignment ---")
    for l in range(n_layers):
        cosines = [cos for _, li, cos in diag['momentum_alignment_log'] if li == l]
        if cosines:
            # Early vs late
            n = len(cosines)
            early = cosines[:n//3]
            late = cosines[2*n//3:]
            print(f"  {layer_names[l]}: early_cos={np.mean(early):.3f}, "
                  f"late_cos={np.mean(late):.3f}, overall={np.mean(cosines):.3f}")

    # H3: Variance captured
    print(f"\n--- H3: Variance Captured by Low-Rank ---")
    for l in range(n_layers):
        entries = [(s, info) for s, li, info in diag['spectra_log'] if li == l]
        if entries:
            last = entries[-1][1]
            print(f"  {layer_names[l]}: var_captured_A={last['variance_captured_a']:.1%}, "
                  f"var_captured_G={last['variance_captured_g']:.1%}")
            print(f"    d_A stats: min={last['d_a_stats'][0]:.6f}, "
                  f"median={last['d_a_stats'][1]:.6f}, max={last['d_a_stats'][2]:.6f}")
            print(f"    d_G stats: min={last['d_g_stats'][0]:.6f}, "
                  f"median={last['d_g_stats'][1]:.6f}, max={last['d_g_stats'][2]:.6f}")

    # H4: Damping dominance
    print(f"\n--- H4: Damping Dominance ---")
    for l in range(n_layers):
        entries = [(da, dg, mda, mdg, fa, fg)
                   for _, li, da, dg, mda, mdg, fa, fg in diag['damping_log'] if li == l]
        if entries:
            # Last 100 steps average
            last_n = entries[-100:]
            avg_da = np.mean([e[0] for e in last_n])
            avg_dg = np.mean([e[1] for e in last_n])
            avg_mda = np.mean([e[2] for e in last_n])
            avg_mdg = np.mean([e[3] for e in last_n])
            avg_fa = np.mean([e[4] for e in last_n])
            avg_fg = np.mean([e[5] for e in last_n])
            print(f"  {layer_names[l]}:")
            print(f"    damping_a={avg_da:.6f}, median_d_a={avg_mda:.6f}, "
                  f"frac_dominated_a={avg_fa:.1%}")
            print(f"    damping_g={avg_dg:.6f}, median_d_g={avg_mdg:.6f}, "
                  f"frac_dominated_g={avg_fg:.1%}")

    # H5: Approximation error
    print(f"\n--- H5: LRPD Approximation Quality ---")
    for l in range(n_layers):
        entries = [err for _, li, err, _ in diag['approx_quality_log'] if li == l]
        if entries:
            print(f"  {layer_names[l]}: first={entries[0]:.4f}, "
                  f"last={entries[-1]:.4f}, trend={'↑ growing' if entries[-1] > entries[0] * 1.1 else '↓ shrinking' if entries[-1] < entries[0] * 0.9 else '→ stable'}")

    # H6: Preconditioning effect (raw vs precond ratio)
    print(f"\n--- H6: Preconditioning Effect (precond/raw norm ratio) ---")
    for l in range(n_layers):
        raw = [n for _, li, n in diag['raw_grad_norms'] if li == l]
        pre = [n for _, li, n in diag['precond_grad_norms'] if li == l]
        if raw and pre:
            ratios = [p/r for r, p in zip(raw, pre) if r > 1e-10]
            if ratios:
                print(f"  {layer_names[l]}: median_ratio={np.median(ratios):.2f}, "
                      f"mean_ratio={np.mean(ratios):.2f}, std={np.std(ratios):.2f}")

    print(f"\n{'='*70}")
    print("VERDICT (check chart for details)")
    print(f"{'='*70}")
    # Auto-detect strongest signals
    verdicts = []

    # H1 check
    for l in range(n_layers):
        entries = [coef for _, li, _, coef in diag['clip_log'] if li == l]
        if entries:
            clip_rate = sum(1 for c in entries if c < 1.0) / len(entries)
            if clip_rate > 0.5:
                verdicts.append(f"H1 STRONG: {layer_names[l]} clipped {clip_rate:.0%} of the time")

    # H3 check
    for l in range(n_layers):
        entries = [(s, info) for s, li, info in diag['spectra_log'] if li == l]
        if entries:
            last = entries[-1][1]
            if last['variance_captured_a'] < 0.5:
                verdicts.append(f"H3 STRONG: {layer_names[l]} A-factor captures only {last['variance_captured_a']:.0%} of variance")
            if last['variance_captured_g'] < 0.5:
                verdicts.append(f"H3 STRONG: {layer_names[l]} G-factor captures only {last['variance_captured_g']:.0%} of variance")

    # H4 check
    for l in range(n_layers):
        entries = [(fa, fg) for _, li, _, _, _, _, fa, fg in diag['damping_log'] if li == l]
        if entries:
            last_n = entries[-100:]
            avg_fa = np.mean([e[0] for e in last_n])
            avg_fg = np.mean([e[1] for e in last_n])
            if avg_fa > 0.8:
                verdicts.append(f"H4 STRONG: {layer_names[l]} A-factor {avg_fa:.0%} dominated by damping")
            if avg_fg > 0.8:
                verdicts.append(f"H4 STRONG: {layer_names[l]} G-factor {avg_fg:.0%} dominated by damping")

    if not verdicts:
        print("  No strong signals detected — issue may be subtle or multi-causal.")
    else:
        for v in verdicts:
            print(f"  >> {v}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("Running KRONOS diagnostics (Newton T=2, r=32, lr=3e-3, damping=0.01)")
    print("Training 3 epochs with detailed instrumentation...\n")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    diag = run_diagnostics(train_loader, test_loader, device, num_epochs=3)
    print_summary(diag)
    plot_diagnostics(diag)


if __name__ == "__main__":
    main()
