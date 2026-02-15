"""Minimal WY vs naive comparison — isolates the recurrence from the model.

Run on GPU: python experiments/Naja/test_wy_minimal.py
Prints per-timestep state diffs to find where divergence starts.
"""
import torch
import torch.nn.functional as F


def naive_recurrence_debug(x_write, B1, C, alpha, beta, use_trapezoidal=False):
    """Naive sequential with per-step state tracking."""
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

        # Erase from already-decayed state
        b1_hat = F.normalize(B1[:, t], dim=-1)  # (batch, d_state)
        bt = beta[:, t]  # (batch, nheads, 1)
        proj = torch.einsum('bnpd,bd->bnp', h, b1_hat)
        erase = bt.unsqueeze(2) * torch.einsum('bnp,bd->bnpd', proj, b1_hat)
        h = h - erase

        # Write
        write = torch.einsum('bnp,bd->bnpd', x_write[:, t], B1[:, t])
        h = h + bt.unsqueeze(2) * write

        # Readout
        y_t = torch.einsum('bnpd,bd->bnp', h, C[:, t])
        outputs.append(y_t)
        states.append(h.clone())

    return torch.stack(outputs, dim=1), states


def wy_recurrence_debug(x_write, B1, C, alpha, beta,
                        use_trapezoidal=False, chunk_size=None):
    """WY chunkwise with debug output, matching delta_recurrence_wy logic."""
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
        C = F.pad(C, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, 0, 0, pad_len), value=1.0)
        beta = F.pad(beta, (0, 0, 0, 0, 0, pad_len))
    L = seqlen + pad_len
    n_chunks = L // chunk_size
    Cs = chunk_size

    # Map to DeltaNet (K, V, Q, beta)
    K = F.normalize(B1, dim=-1)  # (batch, L, d_state)
    Q = C  # (batch, L, d_state)

    # Scalar decay
    alpha_scalar = alpha.mean(dim=-1)  # (batch, L, nheads)

    # Use original beta (decay handled by gamma in A matrix, not double-counted)
    beta_orig = beta[:, :, :, 0]  # (batch, L, nheads)

    # V = x_write * ||B1|| (write value, no alpha rescaling needed)
    b1_norm = B1.norm(dim=-1, keepdim=True).unsqueeze(2)  # (batch, L, 1, 1)
    V = x_write * b1_norm

    # Chunk
    def chunk(t):
        return t.reshape(batch, n_chunks, Cs, *t.shape[2:])

    K_c = chunk(K).unsqueeze(1).expand(-1, nheads, -1, -1, -1)
    Q_c = chunk(Q).unsqueeze(1).expand(-1, nheads, -1, -1, -1)
    V_c = chunk(V).permute(0, 3, 1, 2, 4) if V.dim() == 4 else None
    beta_c = chunk(beta_orig).permute(0, 3, 1, 2)
    alpha_c = chunk(alpha_scalar).permute(0, 3, 1, 2)

    # Need V_c in (batch, nheads, n_chunks, Cs, headdim)
    V_chunked = chunk(V)  # (batch, n_chunks, Cs, nheads, headdim)
    V_c = V_chunked.permute(0, 3, 1, 2, 4)

    # Decay quantities
    log_alpha_c = torch.log(alpha_c.clamp(min=1e-8))
    log_alpha_cumsum = torch.cumsum(log_alpha_c, dim=-1)
    log_gamma_c = log_alpha_c.sum(dim=-1)
    gamma_c = torch.exp(log_gamma_c)

    # Step 1: UT transform
    # A matrix with decay
    KKt = torch.einsum('bncid, bncjd -> bncij', K_c, K_c)
    A = KKt * beta_c.unsqueeze(-2)
    decay_matrix = torch.exp(
        log_alpha_cumsum.unsqueeze(-1) - log_alpha_cumsum.unsqueeze(-2)
    )
    A = A * decay_matrix
    mask = torch.tril(torch.ones(Cs, Cs, device=device, dtype=dtype), diagonal=-1).bool()
    A = A.masked_fill(~mask, 0.0)

    # T = (I + A)^{-1} diag(beta_orig)
    IpA = torch.eye(Cs, device=device, dtype=dtype) + A
    eye_beta = torch.diag_embed(beta_c)
    T = torch.linalg.solve_triangular(IpA, eye_beta, upper=False, unitriangular=True)

    W = torch.einsum('bncij, bncjd -> bncid', T, K_c)
    U = torch.einsum('bncij, bncjd -> bncid', T, V_c)

    # Step 2: Chunk state
    P = torch.eye(d_state, device=device, dtype=dtype) - \
        torch.einsum('bnctd, bncte -> bncde', K_c, W)
    H = torch.einsum('bnctd, bncte -> bncde', K_c, U)

    # Step 3: Inter-chunk scan
    states_list = []
    S = torch.zeros(batch, nheads, d_state, headdim, device=device, dtype=dtype)
    for c in range(n_chunks):
        states_list.append(S.clone())
        g = gamma_c[:, :, c].unsqueeze(-1).unsqueeze(-1)
        S = g * torch.einsum('bnij, bnjk -> bnik', P[:, :, c], S) + H[:, :, c]
    S_prev = torch.stack(states_list, dim=2)

    # Step 4: Output
    Y_off = torch.einsum('bncid, bncde -> bncie', Q_c, S_prev)
    QKt = torch.einsum('bncid, bncjd -> bncij', Q_c, K_c)
    causal = torch.tril(torch.ones(Cs, Cs, device=device, dtype=dtype))
    QKt = QKt * causal
    decay_out = torch.exp(
        log_alpha_cumsum.unsqueeze(-1) - log_alpha_cumsum.unsqueeze(-2)
    ) * causal
    QKt = QKt * decay_out

    WS = torch.einsum('bncid, bncde -> bncie', W, S_prev)
    intra = U - WS
    Y_diag = torch.einsum('bncij, bncje -> bncie', QKt, intra)

    decay_from_start = torch.exp(log_alpha_cumsum)
    Y_off = Y_off * decay_from_start.unsqueeze(-1)

    Y = Y_diag + Y_off
    Y = Y.permute(0, 2, 3, 1, 4).reshape(batch, L, nheads, headdim)

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


if __name__ == '__main__':
    main()
