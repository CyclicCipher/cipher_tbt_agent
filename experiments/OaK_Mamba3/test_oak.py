"""
test_oak.py - Unit tests for the OaK (Options and Knowledge) mechanisms.

Tests whether the Options and GVF components work as designed,
using minimal synthetic tasks that can run in < 90s on GPU.

Test A - Option routing (structural, no training):
    Feed two identical token sequences with omega=0 vs omega=3.
    The OptionRegister should produce different residual streams.
    Pass: output L2 distance > threshold, AND embeddings receive non-zero gradients.

Test B - Option learning (functional, ~300 steps):
    Toy task: token sequences are ALL IDENTICAL (all SEP tokens).
    Only omega differs between episode types:
        Type 0: omega = 0 everywhere -> correct prediction = class 3
        Type 1: omega = 3 everywhere -> correct prediction = class 7
    The model CANNOT distinguish episode types without reading omega.
    Run with num_options=4 (can learn) and num_options=1 (cannot learn, control).
    Pass: accuracy(num_options=4) > 85% AND accuracy(num_options=1) < 65%.

Test C - GVF tracking (functional, ~300 steps):
    Two episode types distinguished by a 'regime token' at position 0:
        Regime token = 3 (easy): all other targets = 3, learnable -> CE -> 0
        Regime token = 7 (hard): all other targets = random 0-9 -> CE stays high
    Train task head + GVF head jointly. GVF-0 should learn that regime-7 episodes
    have higher prediction error than regime-3 episodes.
    Pass: GVF-0 mean(hard episodes) > GVF-0 mean(easy episodes) + 0.05
          AND actual CE gap confirms the task was learned.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'Mamba3'))

from oak_model import OaKConfig, OaKModel, SEP_TOKEN, NUM_COLORS

# ---------------------------------------------------------------------------
# Shared config: smallest viable model so tests run fast
# ---------------------------------------------------------------------------

def _make_model(num_options: int, device: torch.device) -> OaKModel:
    cfg = OaKConfig(
        d_model     = 64,
        d_state     = 32,
        expand      = 2,
        headdim     = 32,
        chunk_size  = 64,
        n_layer     = 2,
        mlp_expand  = 2,
        stable_ssm  = True,
        num_options = num_options,
        d_option    = 16,
        n_gvfs      = 5,
    )
    return OaKModel(cfg).to(device)


BATCH   = 64    # large batch so signal is clear
SEQ_LEN = 64    # exactly one chunk


# ---------------------------------------------------------------------------
# Test A: Option routing -- structural check (no training)
# ---------------------------------------------------------------------------

def test_a_option_routing(device: torch.device) -> bool:
    """Verify OptionRegister produces meaningfully different outputs for omega=0 vs 3."""
    print('\n' + '=' * 60)
    print('TEST A: Option routing (structural)')
    print('=' * 60)

    model = _make_model(num_options=4, device=device)
    model.eval()

    # Identical token sequence, batch=1
    tokens = torch.full((1, SEQ_LEN), SEP_TOKEN, dtype=torch.long, device=device)
    grid_segs = []

    with torch.no_grad():
        omega_0 = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
        omega_3 = torch.full((1, SEQ_LEN), 3, dtype=torch.long, device=device)

        out_0 = model(tokens, grid_segs, omega=omega_0)
        out_3 = model(tokens, grid_segs, omega=omega_3)

    logits_0 = out_0.task_logits[0]   # (T, C)
    logits_3 = out_3.task_logits[0]

    l2_dist = (logits_0 - logits_3).pow(2).mean().sqrt().item()
    max_diff = (logits_0 - logits_3).abs().max().item()

    print(f'  omega=0 vs omega=3 output divergence:')
    print(f'    RMS logit diff : {l2_dist:.4f}')
    print(f'    Max logit diff : {max_diff:.4f}')

    # Check gradient flow to option embeddings
    model.train()
    opt_embed_param = model.option_reg.embed.weight  # (num_options, d_option)
    omega_0 = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    out_grad = model(tokens, grid_segs, omega=omega_0)
    loss = out_grad.task_logits.sum()
    loss.backward()

    grad_norm = opt_embed_param.grad.norm().item() if opt_embed_param.grad is not None else 0.0
    print(f'    Embed grad norm: {grad_norm:.4f}')

    routing_ok  = l2_dist > 1e-4
    gradient_ok = grad_norm > 0.0
    verdict     = routing_ok and gradient_ok

    print(f'  Routing differs : {"PASS" if routing_ok  else "FAIL"} '
          f'(l2={l2_dist:.4f}, need > 1e-4)')
    print(f'  Gradient flows  : {"PASS" if gradient_ok else "FAIL"} '
          f'(norm={grad_norm:.4f}, need > 0)')
    print(f'  TEST A: {"PASS" if verdict else "FAIL"}')
    return verdict


# ---------------------------------------------------------------------------
# Test B: Option learning -- functional check
# ---------------------------------------------------------------------------

def _run_option_learning(
    num_options: int,
    omega_active: int,
    n_steps: int,
    device: torch.device,
    lr: float = 3e-3,
) -> float:
    """Train on the omega-conditioned task and return final accuracy.

    Task:
        Type-0 episodes: omega=0 everywhere -> target class = 3
        Type-1 episodes: omega=omega_active everywhere -> target class = 7

    Tokens are always all-SEP -- the only signal is omega.
    With num_options=1 omega is clamped to 0 so the model cannot distinguish types.
    """
    model = _make_model(num_options=num_options, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    grid_segs = []

    rng = np.random.default_rng(0)
    CLASS_A, CLASS_B = 3, 7
    acc_window = []

    for step in range(n_steps):
        types = torch.from_numpy(rng.integers(0, 2, size=BATCH).astype(np.int64)).to(device)
        tokens = torch.full((BATCH, SEQ_LEN), SEP_TOKEN, dtype=torch.long, device=device)

        omega = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long, device=device)
        omega[types == 1] = omega_active
        omega = omega.clamp(0, num_options - 1)   # clamp to valid range

        targets = torch.where(
            types.unsqueeze(1).expand(-1, SEQ_LEN) == 0,
            torch.tensor(CLASS_A, device=device),
            torch.tensor(CLASS_B, device=device),
        )

        opt.zero_grad()
        out = model(tokens, grid_segs, omega=omega)
        loss = F.cross_entropy(
            out.task_logits.reshape(-1, NUM_COLORS),
            targets.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            preds = out.task_logits.argmax(dim=-1)
            correct = (preds == targets).float().mean().item()
            acc_window.append(correct)
            if len(acc_window) > 20:
                acc_window.pop(0)

    return float(np.mean(acc_window[-20:]))


def test_b_option_learning(device: torch.device) -> bool:
    """Verify model learns omega-conditioned prediction, control cannot."""
    print('\n' + '=' * 60)
    print('TEST B: Option learning (functional)')
    print('=' * 60)
    print('  Task: identical token sequences, only omega differs.')
    print('  omega=0 -> predict class 3  |  omega=3 -> predict class 7')
    print()

    N_STEPS  = 300
    OMEGA_ON = 3

    t0 = time.time()
    print(f'  Training num_options=4 for {N_STEPS} steps...')
    acc_4 = _run_option_learning(4, OMEGA_ON, N_STEPS, device)
    print(f'    Accuracy: {acc_4*100:.1f}%  ({time.time()-t0:.1f}s)')

    t0 = time.time()
    print(f'  Training num_options=1 for {N_STEPS} steps (control)...')
    acc_1 = _run_option_learning(1, OMEGA_ON, N_STEPS, device)
    print(f'    Accuracy: {acc_1*100:.1f}%  ({time.time()-t0:.1f}s)')

    learns_ok  = acc_4 > 0.85
    control_ok = acc_1 < 0.65
    verdict    = learns_ok and control_ok

    print()
    print(f'  num_options=4 learns task: {"PASS" if learns_ok  else "FAIL"} '
          f'(acc={acc_4*100:.1f}%, need > 85%)')
    print(f'  num_options=1 cannot learn: {"PASS" if control_ok else "FAIL"} '
          f'(acc={acc_1*100:.1f}%, need < 65%)')
    print(f'  TEST B: {"PASS" if verdict else "FAIL"}')
    return verdict


# ---------------------------------------------------------------------------
# Test C: GVF tracking -- does GVF-0 predict uncertainty correctly?
# ---------------------------------------------------------------------------

def test_c_gvf_tracking(device: torch.device) -> bool:
    """Verify GVF-0 learns to distinguish high-error from low-error episodes.

    Why the BiMamba copy task fails:
        BiMamba sees ALL tokens, so any copy task (including random tokens) is
        trivially solved -- CE goes to zero everywhere, leaving nothing to predict.

    Fix -- use a 'regime token' as the distinguishing signal:
        Position 0: EASY_TOK (3) -> all other targets = 3 (learnable, CE -> 0)
                    HARD_TOK (7) -> all other targets = random 0-9 (unlearnable)
        Positions 1-63: always SEP (identical between episode types)

    The model must read position 0 to know the episode regime.
    After training:
        Easy (token=3) episodes: task head learns to predict 3 -> CE ~ 0
        Hard (token=7) episodes: targets are random -> CE ~ log(10) ~ 2.3
    GVF-0 should assign higher values to hard episodes.

    Pass criterion:
        mean GVF-0(hard) > mean GVF-0(easy) + 0.05   (GVF learned the pattern)
        AND  mean CE(hard) > mean CE(easy) + 0.1       (task was actually learned)
    """
    print('\n' + '=' * 60)
    print('TEST C: GVF-0 uncertainty tracking (functional)')
    print('=' * 60)
    EASY_TOK = 3
    HARD_TOK = 7
    N_STEPS  = 300
    LAMBDA   = 1.0
    print(f'  Regime token at pos 0: {EASY_TOK}=easy (target={EASY_TOK} always), '
          f'{HARD_TOK}=hard (targets random)')
    print(f'  Positions 1-{SEQ_LEN-1}: always SEP. Joint training, {N_STEPS} steps.')
    print()

    model     = _make_model(num_options=1, device=device)
    model.train()
    grid_segs = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng       = np.random.default_rng(2)

    for step in range(N_STEPS):
        # Random episode types in batch
        is_hard = torch.from_numpy(
            rng.integers(0, 2, size=BATCH).astype(np.int64)
        ).to(device)   # 0=easy, 1=hard

        # Tokens: regime token at pos 0, SEP everywhere else
        tokens = torch.full((BATCH, SEQ_LEN), SEP_TOKEN, dtype=torch.long, device=device)
        tokens[:, 0] = torch.where(
            is_hard.bool(),
            torch.tensor(HARD_TOK, device=device),
            torch.tensor(EASY_TOK, device=device),
        )

        # Targets: easy->EASY_TOK everywhere; hard->random 0-9
        rand_tgt = torch.from_numpy(
            rng.integers(0, NUM_COLORS, size=(BATCH, SEQ_LEN)).astype(np.int64)
        ).to(device)
        easy_tgt = torch.full((BATCH, SEQ_LEN), EASY_TOK, dtype=torch.long, device=device)
        targets = torch.where(
            is_hard.unsqueeze(1).expand(-1, SEQ_LEN).bool(),
            rand_tgt, easy_tgt,
        )

        omega = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(tokens, grid_segs, omega=omega)

        task_loss = F.cross_entropy(
            out.task_logits.reshape(-1, NUM_COLORS),
            targets.reshape(-1),
        )

        with torch.no_grad():
            per_tok_ce = F.cross_entropy(
                out.task_logits.detach().reshape(-1, NUM_COLORS),
                targets.reshape(-1), reduction='none',
            ).reshape(BATCH, SEQ_LEN)

        gvf0     = out.gvf_vals[..., 0]
        gvf_loss = F.mse_loss(gvf0, per_tok_ce)

        (task_loss + LAMBDA * gvf_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluate on fresh batch
    model.eval()
    with torch.no_grad():
        is_hard_e = torch.from_numpy(
            rng.integers(0, 2, size=BATCH).astype(np.int64)
        ).to(device)
        tok_e = torch.full((BATCH, SEQ_LEN), SEP_TOKEN, dtype=torch.long, device=device)
        tok_e[:, 0] = torch.where(
            is_hard_e.bool(),
            torch.tensor(HARD_TOK, device=device),
            torch.tensor(EASY_TOK, device=device),
        )
        rand_te = torch.from_numpy(
            rng.integers(0, NUM_COLORS, size=(BATCH, SEQ_LEN)).astype(np.int64)
        ).to(device)
        easy_te = torch.full((BATCH, SEQ_LEN), EASY_TOK, dtype=torch.long, device=device)
        tgt_e = torch.where(
            is_hard_e.unsqueeze(1).expand(-1, SEQ_LEN).bool(),
            rand_te, easy_te,
        )
        omega_e = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long, device=device)

        out_e    = model(tok_e, grid_segs, omega=omega_e)
        gvf0_e   = out_e.gvf_vals[..., 0]
        ce_e     = F.cross_entropy(
            out_e.task_logits.reshape(-1, NUM_COLORS),
            tgt_e.reshape(-1), reduction='none',
        ).reshape(BATCH, SEQ_LEN)

        easy_m = (is_hard_e == 0)
        hard_m = (is_hard_e == 1)
        mean_ce_easy  = ce_e[easy_m].mean().item()   if easy_m.any() else float('nan')
        mean_ce_hard  = ce_e[hard_m].mean().item()   if hard_m.any() else float('nan')
        mean_gvf_easy = gvf0_e[easy_m].mean().item() if easy_m.any() else float('nan')
        mean_gvf_hard = gvf0_e[hard_m].mean().item() if hard_m.any() else float('nan')

    ce_gap  = mean_ce_hard  - mean_ce_easy
    gvf_gap = mean_gvf_hard - mean_gvf_easy
    task_ok = ce_gap  > 0.1
    gvf_ok  = gvf_gap > 0.05
    verdict = task_ok and gvf_ok

    print(f'  Actual CE  -- easy : {mean_ce_easy:.3f} | hard : {mean_ce_hard:.3f} '
          f'| gap : {ce_gap:+.3f}')
    print(f'  GVF-0 pred -- easy : {mean_gvf_easy:.3f} | hard : {mean_gvf_hard:.3f} '
          f'| gap : {gvf_gap:+.3f}')
    print(f'  Task learned (CE gap > 0.1)  : {"YES" if task_ok else "NO"}')
    print(f'  GVF learned  (GVF gap > 0.05): {"YES" if gvf_ok  else "NO"}')
    print(f'  TEST C: {"PASS" if verdict else "FAIL"}')
    return verdict


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'OaK mechanism unit tests')
    print(f'Device: {device}')
    t_total = time.time()

    results = {}
    results['A'] = test_a_option_routing(device)
    results['B'] = test_b_option_learning(device)
    results['C'] = test_c_gvf_tracking(device)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    labels = {
        'A': 'Option routing (structural)',
        'B': 'Option learning (functional)',
        'C': 'GVF-0 tracking   (functional)',
    }
    all_pass = True
    for k, label in labels.items():
        ok = results[k]
        all_pass = all_pass and ok
        print(f'  Test {k} - {label}: {"PASS" if ok else "FAIL"}')

    print()
    print(f'Overall: {"ALL PASS" if all_pass else "SOME FAIL"}')
    print(f'Total time: {time.time() - t_total:.1f}s')

    if not results['B']:
        print()
        print('NOTE: Test B failure means omega is NOT reaching the output.')
        print('  Check: OptionRegister.proj dimensions, num_options arg wiring.')
    if not results['C']:
        print()
        print('NOTE: Test C failure means GVF-0 is not tracking prediction error.')
        if results['C'] is False:
            print('  If task_ok=NO: task head did not learn the regime -> '
                  'check regime token embedding, lr.')
            print('  If task_ok=YES but gvf_ok=NO: GVF head is not reading '
                  'the hidden state correctly -> check GVFHead.proj, lambda_gvf.')


if __name__ == '__main__':
    main()
