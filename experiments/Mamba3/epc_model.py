"""
Error-based Predictive Coding wrapper for Mamba-3 (ePC-Mamba3).

Adapts the ePC framework for a stack of Mamba3 blocks. Each block
is a pre-norm Mixer + pre-norm MLP with block-level residual
connections. Errors are fp32 tensors placed after EVERY block,
including the last one (N errors for N blocks).

Standard architecture:
  Embedding → Mamba3Block 0 → + e_0 (fp32)
            → Mamba3Block 1 → + e_1 (fp32)
            → ...
            → Mamba3Block N-1 → + e_{N-1} (fp32)
            → RMSNorm → Output projection

With mHC (manifold-constrained hyperconnections, DeepSeek arXiv:2512.24880):
  Embedding → expand to n streams
            → mHCMamba3Block 0 → + e_0 (b, n, seq, d, fp32)
            → mHCMamba3Block 1 → + e_1
            → ...
            → mHCMamba3Block N-1 → + e_{N-1}
            → sum streams → RMSNorm → Output projection

  Each mHCMamba3Block replaces standard residuals with Sinkhorn-constrained
  stream mixing (H_res on the Birkhoff polytope), softmax aggregation (H_pre),
  and softmax distribution (H_post). At init, equivalent to standard residual.
  Errors are in the multi-stream space, giving the optimizer more degrees of freedom.

Empirical findings:
  - N-1 errors + no precision: 7% (random chance)
  - N-1 errors + geometric precision: 38% (learning but slow)
  - N errors + geometric precision + damping=0.1: 99.3% in 44 epochs!
  - iPC + N errors + geometric precision: 99.2% in 36 epochs

Reference: Goemaere et al. 2025, arXiv:2505.20137
Precision weighting: Salvatori et al. 2025, arXiv:2506.23800
Hyperconnections: DeepSeek arXiv:2512.24880
muPC (Depth-muP for PC): Innocenti et al. 2025, arXiv:2505.13124
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mamba3_block import Mamba3Config, Mamba3Block, Mamba3Mixer, SwiGLUMLP, RMSNorm


# ---------------------------------------------------------------------------
# Manifold-constrained Hyper-Connections (mHC)
# Based on DeepSeek arXiv:2512.24880
# ---------------------------------------------------------------------------

def sinkhorn_log(logits: Tensor, num_iters: int = 10, tau: float = 0.05) -> Tensor:
    """Project logits to doubly stochastic matrix via log-domain Sinkhorn-Knopp.

    The result lies on the Birkhoff polytope: all entries >= 0, every row
    and column sums to 1. Composing doubly stochastic matrices across
    layers keeps spectral radius <= 1, preventing signal blowup.

    Args:
        logits: (n, n) unconstrained parameter matrix.
        num_iters: Sinkhorn iterations (10 is sufficient for small n).
        tau: Temperature (lower = sharper, closer to permutation).

    Returns:
        (n, n) doubly stochastic matrix.
    """
    n = logits.shape[0]
    Z = logits / tau
    log_marginal = -math.log(n)

    u = torch.zeros(n, device=Z.device, dtype=Z.dtype)
    v = torch.zeros(n, device=Z.device, dtype=Z.dtype)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(1), dim=0)

    return torch.exp(Z + u.unsqueeze(1) + v.unsqueeze(0)) * n


class mHCModule(nn.Module):
    """One manifold-constrained hyperconnection module.

    Wraps a branch (Mixer or MLP) with three constrained operations:
      - H_res: doubly stochastic stream mixing (Birkhoff polytope via Sinkhorn)
      - H_pre: branch input aggregation (probability simplex via softmax)
      - H_post: branch output distribution (probability simplex via softmax)

    Per-layer flow:
        mixed = H_res @ streams           (doubly stochastic mixing)
        branch_in = H_pre @ streams       (aggregate to single stream)
        branch_out = F(branch_in)         (run Mixer or MLP)
        output = mixed + H_post * branch_out  (distribute back)

    At initialization:
      - H_res ≈ identity (streams don't mix)
      - H_pre selects one designated stream
      - H_post distributes uniformly to all streams
    This is equivalent to standard Pre-Norm residual connections.

    Reference: DeepSeek arXiv:2512.24880
    """

    def __init__(self, n_streams: int = 2, init_stream: int = 0,
                 sinkhorn_iters: int = 10, sinkhorn_tau: float = 0.05):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau

        # H_res: logits -> Sinkhorn -> doubly stochastic (near-identity at init)
        # Off-diagonal = -8.0, diagonal = 0.0
        # With tau=0.05: exp(-8/0.05) = exp(-160) ≈ 0, so off-diagonal ≈ 0
        H_res_init = torch.full((n_streams, n_streams), -8.0)
        H_res_init.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(H_res_init)

        # H_pre: logits -> softmax (selects init_stream at init)
        H_pre_init = torch.full((n_streams,), -8.0)
        H_pre_init[init_stream] = 0.0
        self.H_pre_logits = nn.Parameter(H_pre_init)

        # H_post: logits -> softmax (uniform at init)
        self.H_post_logits = nn.Parameter(torch.zeros(n_streams))

    def forward(self, streams: Tensor, branch_fn) -> Tensor:
        """Apply hyperconnection around a branch function.

        Args:
            streams: (batch, n_streams, seq, d_model)
            branch_fn: callable (batch, seq, d) -> (batch, seq, d)

        Returns:
            (batch, n_streams, seq, d_model)
        """
        # H_res: doubly stochastic stream mixing
        H_res = sinkhorn_log(
            self.H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)
        mixed = torch.einsum('st, bsld -> btld', H_res, streams)

        # H_pre: aggregate streams into single branch input
        H_pre = F.softmax(self.H_pre_logits, dim=0)
        branch_in = torch.einsum('s, bsld -> bld', H_pre, streams)

        # Run branch (Mixer or MLP)
        branch_out = branch_fn(branch_in)

        # H_post: distribute branch output back to streams
        H_post = F.softmax(self.H_post_logits, dim=0)
        distributed = branch_out.unsqueeze(1) * H_post.view(1, -1, 1, 1)

        return mixed + distributed


class mHCMamba3Block(nn.Module):
    """Mamba3 block with manifold-constrained hyperconnections.

    Replaces the standard ``x + Mixer(norm(x)); x + MLP(norm(x))``
    with mHC stream mixing around each sub-block. The block operates
    on multi-stream tensors (b, n, seq, d) and maintains n parallel
    residual streams.

    Each sub-block (mixer, MLP) has its own mHC module. The init_stream
    cycles through streams: mixer of layer i uses stream 2i % n,
    MLP uses stream (2i+1) % n.
    """

    def __init__(self, config: Mamba3Config, n_streams: int = 2,
                 layer_index: int = 0):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer = Mamba3Mixer(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)

        self.hc_mixer = mHCModule(
            n_streams, init_stream=(2 * layer_index) % n_streams)
        self.hc_mlp = mHCModule(
            n_streams, init_stream=(2 * layer_index + 1) % n_streams)

    def forward(self, streams: Tensor) -> Tensor:
        """Forward pass on multi-stream input.

        Args:
            streams: (batch, n_streams, seq, d_model)
        Returns:
            (batch, n_streams, seq, d_model)
        """
        streams = self.hc_mixer(
            streams, lambda x: self.mixer(self.mixer_norm(x)))
        streams = self.hc_mlp(
            streams, lambda x: self.mlp(self.mlp_norm(x)))
        return streams


# ---------------------------------------------------------------------------
# muPC: Depth-muP for Predictive Coding (Innocenti et al. 2025)
# Based on arXiv:2505.13124, adapting Bordelon et al. 2023 (arXiv:2309.16620)
# ---------------------------------------------------------------------------

class muPCMamba3Block(nn.Module):
    """Mamba3Block with Depth-muP scaling on non-residual contributions.

    Scales mixer and MLP outputs by alpha = 1/sqrt(d_model * L) where L
    is the total number of residual sub-layers across the network (2 per
    block: mixer + MLP). This prevents signal variance from growing with
    depth by shrinking each residual branch contribution.

    At alpha=1.0, equivalent to standard Mamba3Block.
    """

    def __init__(self, config: Mamba3Config, alpha: float = 1.0):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer = Mamba3Mixer(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.alpha * self.mixer(self.mixer_norm(x))
        x = x + self.alpha * self.mlp(self.mlp_norm(x))
        return x


def _sync_time():
    """Synchronize CUDA and return wall-clock time for profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class PCEMamba3(nn.Module):
    """Error-based Predictive Coding for Mamba-3 blocks.

    Each Mamba3Block (Mixer + MLP, WITH residual) is one ePC "layer".
    Errors are added between blocks during inference; weight updates
    use E_local for local learning.

    Args:
        config: Mamba3Config defining the block architecture.
        iters: Number of error optimization steps per batch (T).
        e_lr: Learning rate for error optimization (SGD/Adam).
        error_optim: 'sgd' or 'adam'.
    """

    def __init__(self, config: Mamba3Config, iters: int = 2,
                 e_lr: float = 0.02, error_optim: str = 'sgd',
                 precision_mode: str = 'none',
                 precision_base: float = 3.0,
                 use_mhc: bool = False, n_streams: int = 2,
                 use_mupc: bool = False,
                 # Legacy kwargs (ignored, kept for compat)
                 damping: float = 0.1, **_kwargs):
        super().__init__()
        self.config = config
        self.iters = iters
        self.e_lr = e_lr
        self.error_optim_mode = error_optim
        self.use_mhc = use_mhc
        self.use_mupc = use_mupc
        self.n_streams = n_streams if use_mhc else 1
        self._weight_phase_prediction = None
        self.profiling = False
        self._profile = {}

        # Per-layer precision weighting (Salvatori et al. 2025).
        # N errors for N layers (one after each block, including last).
        # Precisions normalized to mean=1 to preserve error-vs-output balance.
        n_errors = config.n_layer
        if precision_mode == 'none':
            self.precisions = [1.0] * n_errors
        elif precision_mode == 'linear':
            # Layer 0 (earliest) gets highest precision
            raw = [float(n_errors - i) for i in range(n_errors)]
            mean_p = sum(raw) / len(raw)
            self.precisions = [p / mean_p for p in raw]
        elif precision_mode == 'geometric':
            # Exponential scaling: base^(N-1-i) for layer i
            raw = [precision_base ** (n_errors - 1 - i) for i in range(n_errors)]
            mean_p = sum(raw) / len(raw)
            self.precisions = [p / mean_p for p in raw]
        else:
            raise ValueError(f"Unknown precision_mode: {precision_mode}")

        # Layers: standard, mHC, or muPC Mamba3Blocks
        if use_mhc:
            self.layers = nn.ModuleList([
                mHCMamba3Block(config, n_streams=n_streams, layer_index=i)
                for i in range(config.n_layer)
            ])
        elif use_mupc:
            # Depth-muP: alpha = 1/sqrt(d_model * L), L = 2*n_layer sub-layers
            n_sublayers = 2 * config.n_layer  # mixer + MLP per block
            self.mupc_alpha = 1.0 / math.sqrt(config.d_model * n_sublayers)
            self.layers = nn.ModuleList([
                muPCMamba3Block(config, alpha=self.mupc_alpha)
                for _ in range(config.n_layer)
            ])
        else:
            self.layers = nn.ModuleList([
                Mamba3Block(config) for _ in range(config.n_layer)
            ])

        # Output head: final norm
        self.out_norm = RMSNorm(config.d_model)

        # Errors (set during forward)
        self.errors = None

        # CE loss for sequences: y_pred (batch, seqlen, vocab), y (batch, seqlen)
        def _ce_loss(y_pred, y):
            b, l, v = y_pred.shape
            return F.cross_entropy(
                y_pred.reshape(b * l, v), y.reshape(b * l),
            )  # mean reduction (default)
        self._output_loss = _ce_loss

    def y_pred(self, x: Tensor) -> Tensor:
        """Forward pass with current errors.

        N errors for N layers: each error is added after its block's output.
        The last error sits right before RMSNorm, giving it direct influence
        on the output logits.

        With mHC: x is expanded to n streams before the first block, and
        reduced (summed) after the last block before RMSNorm.

        Args:
            x: (batch, seqlen, d_model) input embeddings.

        Returns:
            (batch, seqlen, d_model) pre-projection output.
        """
        s_i = x
        if self.use_mhc:
            # (b, seq, d) -> (b, n, seq, d)
            s_i = s_i.unsqueeze(1).expand(
                -1, self.n_streams, -1, -1).contiguous()
        for e_i, layer_i in zip(self.errors, self.layers):
            s_i = layer_i(s_i) + e_i
        if self.use_mhc:
            s_i = s_i.sum(dim=1)  # (b, n, seq, d) -> (b, seq, d)
        return self.out_norm(s_i)

    def E(self, x: Tensor, y: Tensor, output_proj: nn.Module) -> Tensor:
        """Energy using errors (global graph — for error optimization).

        E = 0.5 * sum(mean(e_l^2)) + output_loss.
        Mean-reduced error penalty matches mean-reduced output loss.

        DO NOT use for weight optimization (would be standard backprop).
        """
        E_errors = 0.5 * sum(e.pow(2).mean() for e in self.errors)
        logits = output_proj(self.y_pred(x))
        return E_errors + self._output_loss(logits, y)

    def E_local(self, x: Tensor, y: Tensor, output_proj: nn.Module) -> Tensor:
        """Energy using local interactions (detached — for weight optimization).

        All layers get uniform local MSE gradient (no precision weighting).
        With N errors for N layers, ALL blocks learn from local MSE terms.
        The CE output loss updates only output_proj (weight-tied embedding).

        With mHC: local MSE is computed in the stream space (b, n, seq, d).
        Stream reduction happens at the output before CE.
        """
        E = 0.0
        s_i = x
        if self.use_mhc:
            s_i = s_i.unsqueeze(1).expand(
                -1, self.n_streams, -1, -1).contiguous()
        for e_i, layer_i in zip(self.errors, self.layers):
            s_i_pred = layer_i(s_i)
            s_i = (s_i_pred + e_i).detach()
            E += 0.5 * F.mse_loss(s_i_pred, s_i)  # mean reduction
        # Output (s_i is detached — CE only updates output_proj/embedding)
        if self.use_mhc:
            s_i = s_i.sum(dim=1)
        s_out = self.out_norm(s_i)
        logits = output_proj(s_out)
        self._weight_phase_prediction = logits.detach()
        return E + self._output_loss(logits, y)

    @torch.no_grad()
    def init_zero_errors(self, x: Tensor):
        """Initialize zero errors with shape caching.

        Errors are ALWAYS fp32. fp16 rounds small corrections to zero,
        defeating early stopping (see MISTAKES.md).

        With mHC: errors are (b, n_streams, seq, d) to match the
        multi-stream representation between blocks.
        """
        input_shape = x.shape
        if (hasattr(self, '_cached_error_shapes')
                and self._cached_input_shape == input_shape):
            device = x.device
            self.errors = [
                torch.zeros(shape, dtype=torch.float32,
                            device=device, requires_grad=True)
                for shape in self._cached_error_shapes
            ]
            return

        # Forward pass to discover shapes (N errors for N layers)
        self.errors = []
        s_i = x
        if self.use_mhc:
            s_i = s_i.unsqueeze(1).expand(
                -1, self.n_streams, -1, -1).contiguous()
        for layer_i in self.layers:
            s_i = layer_i(s_i)
            self.errors.append(
                torch.zeros(s_i.shape, dtype=torch.float32,
                            device=s_i.device, requires_grad=True)
            )

        self._cached_input_shape = input_shape
        self._cached_error_shapes = [e.shape for e in self.errors]

    def minimize_error_energy(self, x: Tensor, y: Tensor,
                              output_proj: nn.Module,
                              early_stop_rtol: float = 1e-3,
                              min_iters: int = 2) -> float:
        """Inference phase: optimize errors to minimize energy.

        Args:
            x: Input embeddings.
            y: Target tokens.
            output_proj: Output projection module.
            early_stop_rtol: Stop early when relative energy reduction
                between consecutive iterations falls below this threshold.
                Set to 0 to disable. Default 1e-3.
            min_iters: Minimum iterations before early stopping is checked.

        Returns:
            Final energy value.
        """
        x = x.detach()

        prof = self.profiling

        if prof:
            _t = _sync_time()

        # Freeze weights during inference
        for p in self.layers.parameters():
            p.requires_grad_(False)
        for p in self.out_norm.parameters():
            p.requires_grad_(False)
        output_proj.requires_grad_(False)

        self.init_zero_errors(x)

        if prof:
            _t2 = _sync_time()
            prof_init = (_t2 - _t) * 1000
            prof_fwd = 0.0
            prof_bwd = 0.0
            prof_step = 0.0

        # Create first-order optimizer
        if self.error_optim_mode == 'adam':
            optim = torch.optim.Adam(self.errors, lr=self.e_lr)
        else:
            optim = torch.optim.SGD(self.errors, lr=self.e_lr)

        E_val = 0.0
        E_prev = float('inf')
        actual_iters = self.iters

        for t in range(self.iters):
            optim.zero_grad()

            if prof:
                _t = _sync_time()

            E = self.E(x, y, output_proj)
            E_val = E.item()

            if prof:
                _t2 = _sync_time()
                prof_fwd += (_t2 - _t) * 1000

            if t == 0:
                self._E_initial = E_val

            # Adaptive early stopping: skip remaining iterations when
            # energy reduction is negligible relative to current energy.
            # Only check after min_iters full steps have completed.
            if early_stop_rtol > 0 and t >= min_iters:
                rel_reduction = (E_prev - E_val) / (abs(E_prev) + 1e-10)
                if rel_reduction < early_stop_rtol:
                    actual_iters = t + 1
                    break

            E_prev = E_val

            if prof:
                _t = _sync_time()

            E.backward()

            if prof:
                _t2 = _sync_time()
                prof_bwd += (_t2 - _t) * 1000
                _t = _t2

            optim.step()

            if prof:
                prof_step += (_sync_time() - _t) * 1000

        self._E_final = E_val
        self._actual_iters = actual_iters

        # Unfreeze weights
        for p in self.layers.parameters():
            p.requires_grad_(True)
        for p in self.out_norm.parameters():
            p.requires_grad_(True)
        output_proj.requires_grad_(True)

        if prof:
            self._profile = {
                'init_ms': prof_init,
                'forward_ms': prof_fwd,
                'backward_ms': prof_bwd,
                'step_ms': prof_step,
            }

        return self._E_final

    def get_diagnostics(self) -> dict:
        """Collect per-layer diagnostics after inference."""
        diag = {
            'E_initial': getattr(self, '_E_initial', 0.0),
            'E_final': getattr(self, '_E_final', 0.0),
            'convergence': getattr(self, '_E_initial', 0.0) - getattr(self, '_E_final', 0.0),
            'error_norms': [],
            'layer_energies': [],
        }
        if self.errors is not None:
            for e in self.errors:
                if isinstance(e, Tensor):
                    norm = torch.linalg.vector_norm(e, ord=2, dim=None).item()
                    diag['error_norms'].append(norm)
                    diag['layer_energies'].append(0.5 * norm ** 2)

        diag['actual_iters'] = getattr(self, '_actual_iters', self.iters)
        diag['precisions'] = self.precisions

        return diag


class ePCMamba3LM(nn.Module):
    """Complete ePC-Mamba3 language model.

    Wraps PCEMamba3 with token embedding and output projection.

    Args:
        config: Mamba3Config.
        vocab_size: Number of tokens.
        iters: Error optimization steps per batch.
        e_lr: Error learning rate for SGD/Adam.
        error_optim: 'sgd' or 'adam'.
    """

    def __init__(self, config: Mamba3Config, vocab_size: int,
                 iters: int = 2, e_lr: float = 0.02,
                 error_optim: str = 'sgd',
                 precision_mode: str = 'none', precision_base: float = 3.0,
                 use_mhc: bool = False, n_streams: int = 2,
                 use_mupc: bool = False, **kwargs):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pce = PCEMamba3(config, iters=iters, e_lr=e_lr,
                             error_optim=error_optim,
                             precision_mode=precision_mode,
                             precision_base=precision_base,
                             use_mhc=use_mhc, n_streams=n_streams,
                             use_mupc=use_mupc)
        self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying
        self.out_proj.weight = self.embedding.weight

    def forward(self, input_ids: Tensor, targets: Tensor | None = None):
        """Forward pass.

        Args:
            input_ids: (batch, seqlen) token indices.
            targets: (batch, seqlen) target token indices. If None,
                returns logits without ePC inference.

        Returns:
            If targets: final energy (float).
            If no targets: logits (batch, seqlen, vocab_size).
        """
        x = self.embedding(input_ids)

        if targets is not None:
            return self.pce.minimize_error_energy(x, targets, self.out_proj)
        else:
            self.pce.errors = [0.0] * len(self.pce.layers)
            hidden = self.pce.y_pred(x)
            return self.out_proj(hidden)

    def compute_weight_loss(self, input_ids: Tensor, targets: Tensor) -> Tensor:
        """Compute E_local for weight optimizer (call after forward with targets)."""
        x = self.embedding(input_ids)
        return self.pce.E_local(x, targets, self.out_proj)

    def ipc_train_step(self, input_ids: Tensor, targets: Tensor,
                       weight_optimizer,
                       w_clip: float = 1.0) -> float:
        """Incremental PC: interleave error and weight updates.

        Each SGD step on errors is immediately followed by a weight
        update via E_local. This increases the rate of weight change by T×,
        helping break through the deadlock phase where the Jacobian dy/de
        is too small for errors to be informative.

        Standard ePC: T error steps → 1 weight step (per batch)
        iPC:          T × (1 error step → 1 weight step) (per batch)

        Returns:
            Final energy value.
        """
        x = self.embedding(input_ids).detach()
        pce = self.pce
        out_proj = self.out_proj

        # --- Setup: freeze weights, init errors ---
        for p in pce.layers.parameters():
            p.requires_grad_(False)
        for p in pce.out_norm.parameters():
            p.requires_grad_(False)
        out_proj.requires_grad_(False)

        pce.init_zero_errors(x)

        if pce.error_optim_mode == 'adam':
            e_optim = torch.optim.Adam(pce.errors, lr=pce.e_lr)
        else:
            e_optim = torch.optim.SGD(pce.errors, lr=pce.e_lr)

        E_val = 0.0

        for t in range(pce.iters):
            # --- Error step ---
            e_optim.zero_grad()

            E = pce.E(x, targets, out_proj)
            E_val = E.item()
            if t == 0:
                pce._E_initial = E_val

            E.backward()
            e_optim.step()

            # --- Weight step (iPC) ---
            for p in pce.layers.parameters():
                p.requires_grad_(True)
            for p in pce.out_norm.parameters():
                p.requires_grad_(True)
            out_proj.requires_grad_(True)

            weight_optimizer.zero_grad()
            w_loss = pce.E_local(x, targets, out_proj)
            w_loss.backward()
            if w_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=w_clip)
            weight_optimizer.step()

            # Re-freeze for next error step
            for p in pce.layers.parameters():
                p.requires_grad_(False)
            for p in pce.out_norm.parameters():
                p.requires_grad_(False)
            out_proj.requires_grad_(False)

        pce._E_final = E_val

        # Final unfreeze
        for p in pce.layers.parameters():
            p.requires_grad_(True)
        for p in pce.out_norm.parameters():
            p.requires_grad_(True)
        out_proj.requires_grad_(True)

        return E_val

    def get_diagnostics(self) -> dict:
        return self.pce.get_diagnostics()
