"""Minimal WY vs naive comparison — isolates the recurrence from the model.

Run on GPU: python experiments/Naja/test_wy_minimal.py
Prints per-timestep state diffs to find where divergence starts.

Tests 1-8: Phase 5a/5b (single Householder, per-channel decay)
Tests 9-11: Phase 5c (PoPE pair B₂ via virtual token expansion)
"""
import torch
import torch.nn.functional as F


def naive_recurrence_debug(x_write, B1, C, alpha, beta1,
                           B2=None, beta2=None, use_trapezoidal=False):
    """Naive sequential with per-step state tracking.

    DeltaProduct convention: erase₁+write₁ then erase₂+write₂.
    """
    batch, seqlen, nheads, headdim = x_write.shape
    d_state = B1.shape[-1]
    device, dtype = x_write.device, x_write.dtype

    h = torch.zeros(batch, nheads, headdim, d_state, device=device, dtype=dtype)
    states = []  # track state after each timestep
    outputs = []

    for t in range(seqlen):
        # Decay first (standard Gated DeltaNet convention)
        a_t = alpha[:, t]  # (batch, nheads, d_state)
        h = h * a_t.unsqueeze(2)

        # Householder 1: erase + write (DeltaProduct convention)
        b1_hat = F.normalize(B1[:, t], dim=-1)  # (batch, d_state)
        bt1 = beta1[:, t]  # (batch, nheads, 1)
        proj1 = torch.einsum('bnpd,bd->bnp', h, b1_hat)
        erase1 = bt1.unsqueeze(2) * torch.einsum('bnp,bd->bnpd', proj1, b1_hat)
        h = h - erase1

        write1 = torch.einsum('bnp,bd->bnpd', x_write[:, t], B1[:, t])
        h = h + bt1.unsqueeze(2) * write1

        # Householder 2: erase + write (PoPE pair)
        if B2 is not None and beta2 is not None:
            b2_hat = F.normalize(B2[:, t], dim=-1)
            bt2 = beta2[:, t]
            proj2 = torch.einsum('bnpd,bd->bnp', h, b2_hat)
            erase2 = bt2.unsqueeze(2) * torch.einsum('bnp,bd->bnpd', proj2, b2_hat)
            h = h - erase2

            write2 = torch.einsum('bnp,bd->bnpd', x_write[:, t], B2[:, t])
            h = h + bt2.unsqueeze(2) * write2

        # Readout
        y_t = torch.einsum('bnpd,bd->bnp', h, C[:, t])
        outputs.append(y_t)
        states.append(h.clone())

    return torch.stack(outputs, dim=1), states


def wy_recurrence_debug(x_write, B1, C, alpha, beta1,
                        B2=None, beta2=None,
                        use_trapezoidal=False, chunk_size=None):
    """WY chunkwise with per-channel decay and virtual token expansion.

    Phase 5b: per-channel decay via K_pos/K_neg.
    Phase 5c: B₂ via virtual token expansion (DeltaProduct n_h=2).
    """
    batch, seqlen, nheads, headdim = x_write.shape
    d_state = B1.shape[-1]
    device, dtype = x_write.device, x_write.dtype

    if chunk_size is None:
        chunk_size = seqlen

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad_len > 0:
        x_write = F.pad(x_write, (0, 0, 0, 0, 0, pad_len))
        B1 = F.pad(B1, (0, 0, 0, pad_len))
        if B2 is not None:
            B2 = F.pad(B2, (0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, 0, 0, pad_len), value=1.0)
        beta1 = F.pad(beta1, (0, 0, 0, 0, 0, pad_len))
        if beta2 is not None:
            beta2 = F.pad(beta2, (0, 0, 0, 0, 0, pad_len))
    L = seqlen + pad_len
    n_chunks = L // chunk_size

    # Map to DeltaNet (K, V, Q, beta)
    K1 = F.normalize(B1, dim=-1)  # (batch, L, d_state)
    Q_real = C  # (batch, L, d_state)
    beta1_flat = beta1[:, :, :, 0]  # (batch, L, nheads)

    # V1 = x_write * ||B1||
    b1_norm = B1.norm(dim=-1, keepdim=True).unsqueeze(2)  # (batch, L, 1, 1)
    V1 = x_write * b1_norm

    # Virtual token expansion for B₂
    use_virtual = B2 is not None and beta2 is not None

    if use_virtual:
        K2 = F.normalize(B2, dim=-1)
        b2_norm = B2.norm(dim=-1, keepdim=True).unsqueeze(2)
        V2 = x_write * b2_norm
        beta2_flat = beta2[:, :, :, 0]

        def interleave(a, b):
            stacked = torch.stack([a, b], dim=2)
            return stacked.reshape(a.shape[0], 2 * a.shape[1], *a.shape[2:])

        K = interleave(K1, K2)
        V = interleave(V1, V2)
        beta_flat = interleave(beta1_flat, beta2_flat)
        Q_zeros = torch.zeros_like(Q_real)
        Q = interleave(Q_zeros, Q_real)
        alpha_ones = torch.ones_like(alpha)
        alpha_eff = interleave(alpha, alpha_ones)

        L_eff = 2 * L
        Cs = 2 * chunk_size
    else:
        K = K1
        V = V1
        Q = Q_real
        beta_flat = beta1_flat
        alpha_eff = alpha
        L_eff = L
        Cs = chunk_size

    # Chunk
    def chunk(t):
        return t.reshape(batch, n_chunks, Cs, *t.shape[2:])

    K_c = chunk(K).unsqueeze(1).expand(-1, nheads, -1, -1, -1)
    Q_c = chunk(Q).unsqueeze(1).expand(-1, nheads, -1, -1, -1)
    V_chunked = chunk(V)  # (batch, n_chunks, Cs, nheads, headdim)
    V_c = V_chunked.permute(0, 3, 1, 2, 4)
    beta_c = chunk(beta_flat).permute(0, 3, 1, 2)
    alpha_c = chunk(alpha_eff).permute(0, 3, 1, 2, 4)  # (batch, nheads, n_chunks, Cs, d_state)

    # Per-channel decay quantities
    log_alpha_c = torch.log(alpha_c.clamp(min=1e-8))  # (batch, nheads, n_chunks, Cs, d_state)
    log_alpha_cumsum = torch.cumsum(log_alpha_c, dim=3)  # cumsum over Cs
    log_gamma_c = log_alpha_c.sum(dim=3)  # (batch, nheads, n_chunks, d_state)
    gamma_c = torch.exp(log_gamma_c)

    # Step 1: UT transform
    # A matrix with per-channel decay: K_pos @ K_neg^T
    K_pos = K_c * torch.exp(log_alpha_cumsum)
    K_neg = K_c * torch.exp(-log_alpha_cumsum)
    A = torch.einsum('bncid, bncjd -> bncij', K_pos, K_neg) * beta_c.unsqueeze(-1)
    mask = torch.tril(torch.ones(Cs, Cs, device=device, dtype=dtype), diagonal=-1).bool()
    A = A.masked_fill(~mask, 0.0)

    # T = (I + A)^{-1} diag(beta)
    IpA = torch.eye(Cs, device=device, dtype=dtype) + A
    eye_beta = torch.diag_embed(beta_c)
    T = torch.linalg.solve_triangular(IpA, eye_beta, upper=False, unitriangular=True)

    U = torch.einsum('bncij, bncjd -> bncid', T, V_c)

    # Decay-weighted pseudo-keys: W_state = T @ (K * exp(cumsum))
    cumsum_exp = torch.exp(log_alpha_cumsum)  # (batch, nheads, n_chunks, Cs, d_state)
    K_decay = K_c * cumsum_exp
    W_state = torch.einsum('bncij, bncjd -> bncid', T, K_decay)

    # Step 2: Forward-decayed keys for state update (per-channel)
    fwd_decay = torch.exp(
        log_gamma_c.unsqueeze(3) - log_alpha_cumsum
    )  # (batch, nheads, n_chunks, Cs, d_state)
    K_fwd = K_c * fwd_decay

    # Step 3: Inter-chunk scan (FLA-style)
    states_list = []
    S = torch.zeros(batch, nheads, d_state, headdim, device=device, dtype=dtype)
    for c in range(n_chunks):
        states_list.append(S.clone())
        WS_c = torch.einsum('bnid, bnde -> bnie', W_state[:, :, c], S)
        v_new = U[:, :, c] - WS_c
        g = gamma_c[:, :, c].unsqueeze(-1)  # (batch, nheads, d_state, 1)
        S = g * S + torch.einsum('bnid, bnie -> bnde', K_fwd[:, :, c], v_new)
    S_prev = torch.stack(states_list, dim=2)

    # Step 4: Output with per-channel decay
    decay_from_start = torch.exp(log_alpha_cumsum)
    Q_decay = Q_c * decay_from_start
    Y_off = torch.einsum('bncid, bncde -> bncie', Q_decay, S_prev)

    QKt = torch.einsum('bncid, bncjd -> bncij', Q_decay, K_neg)
    causal = torch.tril(torch.ones(Cs, Cs, device=device, dtype=dtype))
    QKt = QKt * causal

    WS = torch.einsum('bncid, bncde -> bncie', W_state, S_prev)
    intra = U - WS
    Y_diag = torch.einsum('bncij, bncje -> bncie', QKt, intra)

    Y = Y_diag + Y_off
    Y = Y.permute(0, 2, 3, 1, 4).reshape(batch, L_eff, nheads, headdim)

    # Extract odd positions if virtual expansion
    if use_virtual:
        Y = Y[:, 1::2]

    if pad_len > 0:
        Y = Y[:, :seqlen]

    return Y


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    batch, seqlen, nheads, headdim, d_state = 2, 8, 2, 4, 8

    # Simple inputs
    x_write = torch.randn(batch, seqlen, nheads, headdim, device=device)
    B1 = torch.randn(batch, seqlen, d_state, device=device)
    C = torch.randn(batch, seqlen, d_state, device=device)

    # Alpha ~ 0.95 (uniform, not per-channel for this test)
    alpha_val = 0.95
    alpha = torch.full((batch, seqlen, nheads, d_state), alpha_val, device=device)

    # Beta ~ 0.5
    beta = torch.full((batch, seqlen, nheads, 1), 0.5, device=device)

    print(f"\n=== Test 1: Constant alpha={alpha_val}, constant beta=0.5 ===")
    print(f"  Shapes: batch={batch}, seqlen={seqlen}, nheads={nheads}, "
          f"headdim={headdim}, d_state={d_state}")

    out_naive, states_naive = naive_recurrence_debug(x_write, B1, C, alpha, beta)
    out_wy = wy_recurrence_debug(x_write, B1, C, alpha, beta, chunk_size=seqlen)

    diff = (out_naive - out_wy).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")
    for t in range(seqlen):
        d = (out_naive[:, t] - out_wy[:, t]).abs().max().item()
        print(f"    t={t}: max_diff={d:.6e}")

    # Test 2: alpha=1.0 (no decay) — should isolate pure DeltaNet
    print(f"\n=== Test 2: alpha=1.0 (no decay) ===")
    alpha_nodecay = torch.ones_like(alpha)
    out_naive2, _ = naive_recurrence_debug(x_write, B1, C, alpha_nodecay, beta)
    out_wy2 = wy_recurrence_debug(x_write, B1, C, alpha_nodecay, beta, chunk_size=seqlen)
    diff2 = (out_naive2 - out_wy2).abs()
    print(f"  Max diff: {diff2.max().item():.6e}")
    for t in range(seqlen):
        d = (out_naive2[:, t] - out_wy2[:, t]).abs().max().item()
        print(f"    t={t}: max_diff={d:.6e}")

    # Test 3: beta=0 (no delta rule) — pure decay
    print(f"\n=== Test 3: beta=0 (no delta rule) ===")
    beta_zero = torch.zeros_like(beta)
    out_naive3, _ = naive_recurrence_debug(x_write, B1, C, alpha, beta_zero)
    out_wy3 = wy_recurrence_debug(x_write, B1, C, alpha, beta_zero, chunk_size=seqlen)
    diff3 = (out_naive3 - out_wy3).abs()
    print(f"  Max diff: {diff3.max().item():.6e}")

    # Test 4: Multi-chunk
    print(f"\n=== Test 4: Multi-chunk (chunk_size=4) ===")
    out_wy4 = wy_recurrence_debug(x_write, B1, C, alpha, beta, chunk_size=4)
    diff4 = (out_naive - out_wy4).abs()
    print(f"  Max diff: {diff4.max().item():.6e}")

    # Test 5: alpha=1, beta=1 (aggressive erase, no decay)
    print(f"\n=== Test 5: alpha=1.0, beta=1.0 ===")
    beta_one = torch.ones_like(beta)
    out_naive5, _ = naive_recurrence_debug(x_write, B1, C, alpha_nodecay, beta_one)
    out_wy5 = wy_recurrence_debug(x_write, B1, C, alpha_nodecay, beta_one, chunk_size=seqlen)
    diff5 = (out_naive5 - out_wy5).abs()
    print(f"  Max diff: {diff5.max().item():.6e}")

    # Test 6: Per-channel varying alpha (THE Phase 5b test)
    print(f"\n=== Test 6: Per-channel varying alpha (Phase 5b) ===")
    torch.manual_seed(123)
    # Each channel gets a different decay rate: some fast (0.5), some slow (0.99)
    alpha_perchannel = 0.5 + 0.49 * torch.rand(batch, seqlen, nheads, d_state, device=device)
    out_naive6, _ = naive_recurrence_debug(x_write, B1, C, alpha_perchannel, beta)
    out_wy6_single = wy_recurrence_debug(x_write, B1, C, alpha_perchannel, beta, chunk_size=seqlen)
    diff6s = (out_naive6 - out_wy6_single).abs()
    print(f"  Single-chunk max diff: {diff6s.max().item():.6e}")
    for t in range(seqlen):
        d = (out_naive6[:, t] - out_wy6_single[:, t]).abs().max().item()
        print(f"    t={t}: max_diff={d:.6e}")

    # Test 7: Per-channel alpha, multi-chunk (the hardest test)
    print(f"\n=== Test 7: Per-channel alpha, multi-chunk (chunk_size=4) ===")
    out_wy7 = wy_recurrence_debug(x_write, B1, C, alpha_perchannel, beta, chunk_size=4)
    diff7 = (out_naive6 - out_wy7).abs()
    print(f"  Max diff: {diff7.max().item():.6e}")
    for t in range(seqlen):
        d = (out_naive6[:, t] - out_wy7[:, t]).abs().max().item()
        print(f"    t={t}: max_diff={d:.6e}")

    # Test 8: Per-channel alpha, multi-chunk, beta=1 (aggressive)
    print(f"\n=== Test 8: Per-channel alpha, multi-chunk, beta=1.0 ===")
    out_naive8, _ = naive_recurrence_debug(x_write, B1, C, alpha_perchannel, beta_one)
    out_wy8 = wy_recurrence_debug(x_write, B1, C, alpha_perchannel, beta_one, chunk_size=4)
    diff8 = (out_naive8 - out_wy8).abs()
    print(f"  Max diff: {diff8.max().item():.6e}")

    # =====================================================================
    # Phase 5c tests: PoPE pair B₂ via virtual token expansion
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"Phase 5c: PoPE Pair (B₂) via Virtual Token Expansion")
    print(f"{'='*60}")

    torch.manual_seed(42)
    # B2 orthogonal to B1 (simulate PoPE pair)
    B2 = torch.randn(batch, seqlen, d_state, device=device)
    beta2 = torch.full((batch, seqlen, nheads, 1), 0.3, device=device)

    # Test 9: B₂ single chunk, constant alpha
    print(f"\n=== Test 9: B₂ pair, single chunk, alpha={alpha_val} ===")
    out_naive9, _ = naive_recurrence_debug(x_write, B1, C, alpha, beta,
                                           B2=B2, beta2=beta2)
    out_wy9 = wy_recurrence_debug(x_write, B1, C, alpha, beta,
                                  B2=B2, beta2=beta2, chunk_size=seqlen)
    diff9 = (out_naive9 - out_wy9).abs()
    print(f"  Max diff: {diff9.max().item():.6e}")
    for t in range(seqlen):
        d = (out_naive9[:, t] - out_wy9[:, t]).abs().max().item()
        print(f"    t={t}: max_diff={d:.6e}")

    # Test 10: B₂ multi-chunk, per-channel alpha (the hard test)
    print(f"\n=== Test 10: B₂ pair, multi-chunk, per-channel alpha ===")
    torch.manual_seed(123)
    alpha_pc = 0.5 + 0.49 * torch.rand(batch, seqlen, nheads, d_state, device=device)
    out_naive10, _ = naive_recurrence_debug(x_write, B1, C, alpha_pc, beta,
                                            B2=B2, beta2=beta2)
    out_wy10 = wy_recurrence_debug(x_write, B1, C, alpha_pc, beta,
                                   B2=B2, beta2=beta2, chunk_size=4)
    diff10 = (out_naive10 - out_wy10).abs()
    print(f"  Max diff: {diff10.max().item():.6e}")
    for t in range(seqlen):
        d = (out_naive10[:, t] - out_wy10[:, t]).abs().max().item()
        print(f"    t={t}: max_diff={d:.6e}")

    # Test 11: B₂ aggressive (beta1=1, beta2=1, per-channel alpha, multi-chunk)
    print(f"\n=== Test 11: B₂ aggressive (beta1=1, beta2=1, per-ch, multi-chunk) ===")
    beta2_one = torch.ones_like(beta2)
    out_naive11, _ = naive_recurrence_debug(x_write, B1, C, alpha_pc, beta_one,
                                            B2=B2, beta2=beta2_one)
    out_wy11 = wy_recurrence_debug(x_write, B1, C, alpha_pc, beta_one,
                                   B2=B2, beta2=beta2_one, chunk_size=4)
    diff11 = (out_naive11 - out_wy11).abs()
    print(f"  Max diff: {diff11.max().item():.6e}")

    # Summary
    print(f"\n=== Summary ===")
    all_pass = True
    for name, d in [("T1  const", diff), ("T2  nodecay", diff2),
                    ("T3  nobeta", diff3), ("T4  multi", diff4),
                    ("T5  aggr", diff5), ("T6  perchan", diff6s),
                    ("T7  perchan+multi", diff7), ("T8  perchan+multi+aggr", diff8),
                    ("T9  B2_single", diff9), ("T10 B2_perchan+multi", diff10),
                    ("T11 B2_aggressive", diff11)]:
        status = "PASS" if d.max().item() < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: max_diff={d.max().item():.6e}  [{status}]")
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")


if __name__ == '__main__':
    main()
