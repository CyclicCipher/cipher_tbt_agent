"""Block-sparse Triton kernels for BDH-GPU.

Exploits the natural sparsity of BDH's ReLU activations.
After ReLU, ~50-95% of neurons are zero (depending on training).
These kernels skip zero blocks entirely.

Block size = 32 (NVIDIA warp size). A block is skipped if ALL
elements in it are zero.

Two kernels:
1. block_sparse_matmul: sparse (B, T, M) @ dense (M, K) -> (B, T, K)
   Used for: xy_sparse @ decoder
2. block_sparse_attention: sparse Q @ sparse K^T with causal mask
   Used for: x_sparse @ x_sparse^T in attention

Correctness: output must be bitwise identical to the dense version
(within floating point tolerance). Sparsity is an optimization, not
an approximation.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block-sparse matmul: sparse @ dense
# ---------------------------------------------------------------------------

BLOCK_M = 32  # block size along the sparse dimension


@triton.jit
def _block_sparse_matmul_kernel(
    # Pointers
    x_ptr, w_ptr, out_ptr,
    # Block mask: 1 = nonzero block, 0 = skip
    mask_ptr,
    # Strides for x: (batch_stride, m_stride)
    x_batch_stride, x_m_stride,
    # Strides for w: (m_stride, k_stride)
    w_m_stride, w_k_stride,
    # Strides for out: (batch_stride, k_stride)
    out_batch_stride, out_k_stride,
    # Mask stride: (batch_stride,)
    mask_batch_stride,
    # Dimensions
    M: tl.constexpr,         # sparse dimension (nh*N)
    K: tl.constexpr,         # output dimension (D)
    N_BLOCKS: tl.constexpr,  # M / BLOCK_M
    BLOCK_K: tl.constexpr,   # tile size for K dimension
    BLOCK_M_CST: tl.constexpr,  # block size (= 32)
):
    """Compute out[b, :K] = sum over nonzero blocks of x[b, block] @ w[block, :K].

    Each program handles one (batch, k_tile) pair.
    """
    pid_batch = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Output column indices for this tile.
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Accumulator.
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Iterate over blocks of the sparse dimension.
    for block_idx in range(N_BLOCKS):
        # Check if this block is nonzero.
        block_active = tl.load(mask_ptr + pid_batch * mask_batch_stride + block_idx)
        if block_active > 0:
            m_start = block_idx * BLOCK_M_CST

            # For each element in the block, accumulate x[m] * w[m, k_offsets].
            for m_local in range(BLOCK_M_CST):
                m_global = m_start + m_local
                if m_global < M:
                    x_val = tl.load(x_ptr + pid_batch * x_batch_stride + m_global * x_m_stride)
                    if x_val != 0.0:
                        w_row = tl.load(
                            w_ptr + m_global * w_m_stride + k_offsets * w_k_stride,
                            mask=k_mask,
                            other=0.0,
                        )
                        acc += x_val * w_row

    # Store result.
    tl.store(
        out_ptr + pid_batch * out_batch_stride + k_offsets * out_k_stride,
        acc,
        mask=k_mask,
    )


def block_sparse_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Sparse x @ dense w, skipping zero blocks in x.

    x: (batch, M) — sparse (many zeros from ReLU)
    w: (M, K) — dense
    Returns: (batch, K)
    """
    assert x.dim() == 2 and w.dim() == 2
    batch, M = x.shape
    K = w.shape[1]
    assert w.shape[0] == M

    # Compute block mask: which blocks of M have any nonzero element?
    N_BLOCKS = (M + BLOCK_M - 1) // BLOCK_M
    # Pad x to multiple of BLOCK_M for clean blocking.
    if M % BLOCK_M != 0:
        pad = BLOCK_M - (M % BLOCK_M)
        x = torch.nn.functional.pad(x, (0, pad))
        w = torch.nn.functional.pad(w, (0, 0, 0, pad))
        M_padded = M + pad
    else:
        M_padded = M

    # Block mask: (batch, N_BLOCKS). 1 if any element in block is nonzero.
    x_blocks = x.view(batch, N_BLOCKS, BLOCK_M)
    block_mask = (x_blocks.abs().sum(dim=-1) > 0).to(torch.int32)

    # Output.
    out = torch.zeros(batch, K, device=x.device, dtype=x.dtype)

    # Launch kernel.
    BLOCK_K = min(K, 64)
    grid = (batch, (K + BLOCK_K - 1) // BLOCK_K)

    _block_sparse_matmul_kernel[grid](
        x, w, out,
        block_mask,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        block_mask.stride(0),
        M=M_padded, K=K, N_BLOCKS=N_BLOCKS, BLOCK_K=BLOCK_K,
        BLOCK_M_CST=BLOCK_M,
    )

    return out


# ---------------------------------------------------------------------------
# PyTorch-native fallback (for correctness comparison)
# ---------------------------------------------------------------------------

def block_sparse_matmul_pytorch(x: torch.Tensor, w: torch.Tensor,
                                 block_size: int = 32) -> torch.Tensor:
    """Pure PyTorch block-sparse matmul using gather-scatter.

    Faster than the naive dense matmul when x is sufficiently sparse,
    without requiring Triton. Falls back gracefully on any device.
    """
    assert x.dim() == 2 and w.dim() == 2
    batch, M = x.shape
    K = w.shape[1]

    N_BLOCKS = (M + block_size - 1) // block_size
    if M % block_size != 0:
        pad = block_size - (M % block_size)
        x = torch.nn.functional.pad(x, (0, pad))
        w = torch.nn.functional.pad(w, (0, 0, 0, pad))
        M_padded = M + pad
    else:
        M_padded = M

    # Find nonzero blocks per batch element.
    x_blocks = x.view(batch, N_BLOCKS, block_size)
    block_active = x_blocks.abs().sum(dim=-1) > 0  # (batch, N_BLOCKS)

    # For each batch: gather active blocks, matmul, sum.
    out = torch.zeros(batch, K, device=x.device, dtype=x.dtype)
    for b in range(batch):
        active_idx = block_active[b].nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            continue
        # Gather active elements.
        m_indices = (active_idx.unsqueeze(1) * block_size +
                     torch.arange(block_size, device=x.device).unsqueeze(0)).reshape(-1)
        x_active = x[b, m_indices]  # (n_active * block_size,)
        w_active = w[m_indices, :]  # (n_active * block_size, K)
        out[b] = x_active @ w_active

    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    M, K, batch = 2048, 64, 64

    # Create sparse x (simulate post-ReLU BDH activations).
    x_dense = torch.randn(batch, M, device=device)
    x_dense[x_dense < 0.5] = 0  # ~70% sparse
    w = torch.randn(M, K, device=device) * 0.02

    sparsity = (x_dense == 0).float().mean().item()
    print(f"x shape: {x_dense.shape}, w shape: {w.shape}, sparsity: {sparsity:.1%}")

    # Correctness: dense vs pytorch block-sparse.
    ref = x_dense @ w
    out_pytorch = block_sparse_matmul_pytorch(x_dense, w)
    err_pytorch = (out_pytorch - ref).abs().max().item()
    print(f"PyTorch block-sparse max error: {err_pytorch:.2e}")

    # Correctness: dense vs triton block-sparse.
    if device == "cuda":
        out_triton = block_sparse_matmul(x_dense, w)
        err_triton = (out_triton - ref).abs().max().item()
        print(f"Triton block-sparse max error: {err_triton:.2e}")

    # Speed comparison.
    def bench(fn, n=100):
        # Warmup.
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    t_dense = bench(lambda: x_dense @ w)
    t_pytorch = bench(lambda: block_sparse_matmul_pytorch(x_dense, w))
    if device == "cuda":
        t_triton = bench(lambda: block_sparse_matmul(x_dense, w))
        print(f"\nDense:           {t_dense:.3f} ms")
        print(f"PyTorch sparse:  {t_pytorch:.3f} ms")
        print(f"Triton sparse:   {t_triton:.3f} ms")
        print(f"Triton speedup:  {t_dense / t_triton:.2f}x")
    else:
        print(f"\nDense:           {t_dense:.3f} ms")
        print(f"PyTorch sparse:  {t_pytorch:.3f} ms")
