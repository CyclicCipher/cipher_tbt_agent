"""
Slot Workspace — biologically motivated sequence model.

Motivation
----------
Transformers treat the context window as both the memory store and the
substrate for attention (O(T^2) cost). SSMs compress history into a
fixed state but do no explicit cross-position reasoning. Neither maps
cleanly to the brain's architecture.

The brain separates:
  - Working memory (PFC, ~4-16 items): a small, goal-directed ACTIVE
    buffer. Attention operates OVER this buffer.
  - Sequential dynamics (thalamocortical loops): recurrent processing
    that tracks "where are we in the current computation."
  - Retrieval (hippocampus / directed attention): explicit lookup into
    long-term storage when the working memory lacks something needed.

This module implements that decomposition using Mamba3 as the recurrent
backbone and a slot-structured state as the working memory:

Architecture
------------
The K workspace slots ARE the nheads of the multi-head Mamba3 SSM.
After each Mamba step produces y: (batch, T, K, headdim), we run
IntraSlotAttention over the K dimension (not T):

    x_t -> [Mamba3 SSM] -> y: (batch, T, K, headdim)
         -> [IntraSlotAttention over K] -> refined_y
         -> [z-gate + out_proj] -> output: (batch, T, d_model)

IntraSlotAttention cost: O(K^2) per position, where K << T.
Total cost: O(K^2 * T) vs O(T^2) for attention over sequence.

No causal violation: attention is over K slots at time t, all of which
contain information only from x_1,...,x_t (from the causal Mamba step).

Latent-space reasoning
----------------------
IntraSlotAttention runs at every position without producing tokens.
The model can:
  - Merge slots that converged on the same entity
  - Create cross-slot bindings (subject x predicate, entity x property)
  - Suppress irrelevant slots
  - Refine representations across multiple layers
All of this is internal computation, not narrated through output tokens.
Chain-of-thought reasoning is not required; it happens in the slot space.

Optional scroll-back
--------------------
For exact retrieval from any document position (not just what survived
in the slots), ScrollBackAttention does cross-attention from the current
sequence representations to cached document representations.

Typical QA pipeline:
    doc_reps = model.encode_document(doc_tokens)    # process document
    logits = model.forward_with_scroll_back(        # process question
                 question_tokens, doc_reps)

Cost: O(K * doc_len) when invoked, not O(T^2).
The model is not constantly attending to the document; it "flips back"
explicitly at question-answering time.

Biological correspondences
--------------------------
  Mamba3 selective gates (A, B, C, Delta)  -> thalamocortical recurrence
                                              + basal ganglia write/forget
  K workspace slots (= nheads)             -> PFC working memory (~4-16 items)
  IntraSlotAttention (over K, not T)       -> PFC lateral connections +
                                              top-down gating (TRN searchlight)
  ScrollBackAttention (cross-attn to doc)  -> directed retrieval /
                                              "scroll back through the text"
  Stacked SlotBlocks                       -> cortical hierarchy

Relationship to existing architectures
---------------------------------------
  Transformer:  attention over T positions (O(T^2)), context window = memory
  Mamba3:       recurrent state, no cross-slot reasoning
  Jamba:        interleaved Mamba + attention-over-T (still O(window^2))
  SlotWorkspace: Mamba recurrence + attention-over-K (K << T, no context window)

The qualitative difference from Jamba: attention NEVER sees raw token
history. It only sees the K slot vectors — what Mamba decided to remember.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mamba3_block import (
    RMSNorm,
    SwiGLUMLP,
    apply_pope,
    apply_rope,
    ssd_trapz,
    stable_log_decay,
    segsum,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SlotConfig:
    """Configuration for the Slot Workspace model.

    Key constraint: n_slots == nheads.
    The K workspace slots ARE the multi-head dimension of the Mamba3 SSM.
    headdim is derived from d_model, expand, and n_slots — not set directly.

    Example: d_model=256, expand=2, n_slots=16 -> headdim = 512/16 = 32.
    Example: d_model=256, expand=2, n_slots=8  -> headdim = 512/8  = 64.
    """
    # Core model dimensions
    d_model: int = 256         # token embedding / residual stream dimension
    expand: int = 2            # SSM inner expansion factor (d_inner = expand * d_model)
    n_slots: int = 16          # K workspace slots (= nheads)
    d_state: int = 64          # SSM state dimension N
    chunk_size: int = 64       # SSD chunk size Q (must divide seqlen)
    n_layer: int = 4           # number of (SlotMixer + MLP) blocks

    # MLP
    mlp_expand: int = 4        # SwiGLU MLP hidden expansion

    # SSM options (inherited from Mamba3)
    use_pope: bool = True      # PoPE positional encoding (recommended)
    stable_ssm: bool = True    # StableSSM A-matrix reparameterization
    use_triton: bool = False   # Triton-accelerated SSD

    # Intra-slot attention (the core new component)
    intra_attn_heads: int = 4  # attention heads WITHIN the K-slot attention
                               # must divide headdim = d_inner // n_slots

    # Scroll-back cross-attention (optional, for long-range retrieval)
    use_scroll_back: bool = False
    scroll_back_heads: int = 4  # must divide d_model

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.d_inner % self.n_slots != 0:
            raise ValueError(
                f"d_inner = expand*d_model = {self.expand}*{self.d_model} = "
                f"{self.d_inner} must be divisible by n_slots={self.n_slots}. "
                f"Try adjusting expand or n_slots."
            )
        self.headdim = self.d_inner // self.n_slots   # each slot is headdim-dimensional
        self.nheads = self.n_slots
        if self.intra_attn_heads > 0 and self.headdim % self.intra_attn_heads != 0:
            raise ValueError(
                f"headdim={self.headdim} must be divisible by "
                f"intra_attn_heads={self.intra_attn_heads}"
            )
        if self.d_model % self.scroll_back_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"scroll_back_heads={self.scroll_back_heads}"
            )


# ---------------------------------------------------------------------------
# Intra-Slot Attention
# ---------------------------------------------------------------------------

class IntraSlotAttention(nn.Module):
    """Self-attention over K workspace slots.

    Operates on the SLOT dimension (K), not the sequence dimension (T).
    Cost: O(K^2) per position. For K=16 and T=10000: 256 vs 100M ops.

    No causal mask: all K slots at position t see each other. This is
    valid because all slots were produced by the causal Mamba step at t
    (they contain only information from x_1,...,x_t).

    This is where latent-space reasoning happens: slots can merge,
    bind, and suppress each other every step, without producing tokens.

    Args:
        headdim:  dimension of each slot vector
        n_slots:  number of slots (K)
        n_heads:  number of attention heads within the slot attention
    """

    def __init__(self, headdim: int, n_slots: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = headdim // n_heads
        self.n_slots = n_slots

        self.norm = RMSNorm(headdim)
        self.qkv = nn.Linear(headdim, 3 * headdim, bias=False)
        self.out_proj = nn.Linear(headdim, headdim, bias=False)

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, slots: Tensor) -> Tensor:
        """
        Args:
            slots: (batch, seqlen, K, headdim)
        Returns:
            (batch, seqlen, K, headdim) — residual + attention output
        """
        B, T, K, D = slots.shape
        residual = slots

        # Pre-norm
        x = self.norm(slots)   # (B, T, K, D)

        # Project to Q, K, V
        qkv = self.qkv(x)                           # (B, T, K, 3*D)
        Q, Kv, V = qkv.chunk(3, dim=-1)             # each (B, T, K, D)

        # Reshape for multi-head: (B, T, K, D) -> (B*T, n_heads, K, head_dim)
        H, dh = self.n_heads, self.head_dim

        def to_heads(t: Tensor) -> Tensor:
            return t.reshape(B * T, K, H, dh).transpose(1, 2)

        Q, Kv, V = map(to_heads, [Q, Kv, V])

        # Attention over K slots (no causal mask — same-timestep slots)
        out = F.scaled_dot_product_attention(Q, Kv, V, is_causal=False)
        # out: (B*T, n_heads, K, dh)

        # Reshape back: (B*T, n_heads, K, dh) -> (B, T, K, D)
        out = out.transpose(1, 2).reshape(B, T, K, D)
        out = self.out_proj(out)

        return residual + out


# ---------------------------------------------------------------------------
# Scroll-Back Cross-Attention
# ---------------------------------------------------------------------------

class ScrollBackAttention(nn.Module):
    """Cross-attention from current representations to document encodings.

    This implements the "scroll back" operation: when the working memory
    (slot state) doesn't contain the needed information, the model can
    look up exactly what was said at any document position.

    Usage pattern:
        1. Process full document: doc_reps = model.encode_document(doc_ids)
        2. Process question:      x = forward_layers(question_ids)
        3. Retrieve:              x = scroll_back(x, doc_reps)
        4. Generate answer:       logits = out_proj(norm(x))

    Cost: O(T_q * doc_len) per call (T_q = question length, typically small).
    The document representations are precomputed once and reused.

    The gate (initialized near zero) lets the model learn WHEN retrieval
    helps and suppresses it when the information is already in the slots.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learned gate: controls how much retrieval to inject.
        # Initialized to zero -> retrieval output starts at zero,
        # model learns to open gate where retrieval helps.
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        x: Tensor,                      # (batch, T_q, d_model) — query side
        doc_reps: Tensor,               # (batch, doc_len, d_model) — key/value
        doc_mask: Tensor | None = None, # (batch, doc_len) bool, True = valid
    ) -> Tensor:
        """Retrieve from doc_reps and add to x.

        Returns:
            (batch, T_q, d_model) — enriched representations
        """
        B, T_q, D = x.shape
        doc_len = doc_reps.shape[1]
        H, dh = self.n_heads, self.head_dim

        xn = self.norm_q(x)
        dn = self.norm_kv(doc_reps)

        Q = self.q_proj(xn)    # (B, T_q, D)
        K = self.k_proj(dn)    # (B, doc_len, D)
        V = self.v_proj(dn)    # (B, doc_len, D)

        # Multi-head reshape
        Q = Q.reshape(B, T_q, H, dh).transpose(1, 2)        # (B, H, T_q, dh)
        K = K.reshape(B, doc_len, H, dh).transpose(1, 2)    # (B, H, doc_len, dh)
        V = V.reshape(B, doc_len, H, dh).transpose(1, 2)    # (B, H, doc_len, dh)

        # Build additive mask from doc_mask (True=valid, False=padding)
        attn_mask = None
        if doc_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, doc_len, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(
                ~doc_mask[:, None, None, :], float('-inf')
            )

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T_q, D)
        out = self.out_proj(out)

        # Gated injection
        gate = torch.sigmoid(self.gate_proj(xn))
        return x + gate * out


# ---------------------------------------------------------------------------
# Slot Mixer
# ---------------------------------------------------------------------------

class SlotMixer(nn.Module):
    """Mamba3 SSM with slot-structured state and intra-slot attention.

    The nheads of the Mamba3 SSM are the K workspace slots.
    After the SSD step produces y: (batch, T, K, headdim), IntraSlotAttention
    refines the slot workspace before gating and output projection.

    Pipeline:
        u: (B, T, d_model)
        -> [in_proj] -> z, x, B, C, dt, theta, lam
        -> [SSD trapz with PoPE/StableSSM] -> y: (B, T, K, headdim)
        -> [D skip connection]
        -> [IntraSlotAttention over K]         <- latent workspace reasoning
        -> [flatten + z-gate + out_proj] -> (B, T, d_model)
    """

    def __init__(self, cfg: SlotConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_inner
        n = cfg.d_state

        d_bc = n // 2 if cfg.use_pope else n

        # Input projection: [z, x, B, C, dt, theta, lambda]
        d_in_proj = 2 * d + 2 * d_bc + cfg.nheads + n // 2 + 1
        self.in_proj = nn.Linear(cfg.d_model, d_in_proj, bias=False)

        self.B_bias = nn.Parameter(torch.ones(d_bc))
        self.C_bias = nn.Parameter(torch.ones(d_bc))
        self.B_norm = RMSNorm(d_bc)
        self.C_norm = RMSNorm(d_bc)

        if cfg.use_pope:
            self.pope_delta_B = nn.Parameter(torch.zeros(d_bc))
            self.pope_delta_C = nn.Parameter(torch.zeros(d_bc))

        self.dt_bias = nn.Parameter(torch.empty(cfg.nheads))
        if cfg.stable_ssm:
            self.A_raw = nn.Parameter(torch.empty(cfg.nheads))
        else:
            self.A_log = nn.Parameter(torch.empty(cfg.nheads))
        self.D = nn.Parameter(torch.empty(cfg.nheads))

        # The core new component: attention over K slots (not over T).
        # intra_attn_heads=0 disables it entirely (ablation / baseline mode).
        if cfg.intra_attn_heads > 0:
            self.slot_attn: IntraSlotAttention | None = IntraSlotAttention(
                headdim=cfg.headdim,
                n_slots=cfg.n_slots,
                n_heads=cfg.intra_attn_heads,
            )
        else:
            self.slot_attn = None

        self.out_proj = nn.Linear(d, cfg.d_model, bias=False)
        self._init_parameters()

    def _init_parameters(self):
        cfg = self.cfg
        if cfg.stable_ssm:
            nn.init.uniform_(self.A_raw, -3.0, -0.5)
        else:
            nn.init.uniform_(self.A_log, -5.0, -1.0)
        nn.init.ones_(self.D)
        dt_target = torch.exp(
            torch.empty(cfg.nheads).uniform_(math.log(0.001), math.log(0.1))
        )
        self.dt_bias.data.copy_(torch.log(torch.exp(dt_target) - 1))
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, u: Tensor) -> Tensor:
        cfg = self.cfg
        batch, seqlen, _ = u.shape

        if cfg.stable_ssm:
            A_raw = self.A_raw
        else:
            A_raw = -torch.exp(self.A_log)

        d_bc = cfg.d_state // 2 if cfg.use_pope else cfg.d_state
        proj = self.in_proj(u)
        z, x, B_raw, C_raw, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, d_bc, d_bc, cfg.nheads, cfg.d_state // 2, 1],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)
        lam = torch.sigmoid(lam_logit)

        B_raw = self.B_norm(B_raw) + self.B_bias
        C_raw = self.C_norm(C_raw) + self.C_bias

        theta_cumsum = torch.cumsum(theta, dim=1)
        if cfg.use_pope:
            B = apply_pope(B_raw, theta_cumsum, self.pope_delta_B)
            C = apply_pope(C_raw, theta_cumsum, self.pope_delta_C)
        else:
            B = apply_rope(B_raw, theta_cumsum)
            C = apply_rope(C_raw, theta_cumsum)

        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        dt_exp = dt.unsqueeze(-1)
        x_dt = x * dt_exp

        if cfg.stable_ssm:
            A_dt = stable_log_decay(A_raw * dt)
        else:
            A_dt = A_raw * dt

        _ssd_fn = ssd_trapz
        if cfg.use_triton:
            try:
                from triton_ssd import ssd_trapz_triton
                _ssd_fn = ssd_trapz_triton
            except ImportError:
                pass

        lam_expand = lam.unsqueeze(-1)
        x_raw_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev = x_raw_prev * dt_exp
        B_prev = F.pad(B[:, :-1].unsqueeze(2), (0, 0, 0, 0, 1, 0))
        B_curr = B.unsqueeze(2)
        C_curr = C.unsqueeze(2)

        y, _ = _ssd_fn(
            x_dt, x_prev, A_dt,
            B_curr, B_prev, C_curr,
            lam_expand, cfg.chunk_size,
        )
        # y: (batch, seqlen, K=nheads, headdim) — the slot workspace

        # D skip connection (standard Mamba skip)
        y = y + x * self.D.unsqueeze(-1)

        # ── INTRA-SLOT ATTENTION ──────────────────────────────────────────
        # Attend over K slots at each position. Cost: O(K^2 * T).
        # No causal violation: slots at position t only contain info from
        # x_1,...,x_t (produced by the causal Mamba step above).
        # This is where latent-space reasoning happens.
        # slot_attn is None when intra_attn_heads=0 (ablation baseline).
        if self.slot_attn is not None:
            y = self.slot_attn(y)   # (batch, seqlen, K, headdim)
        # ─────────────────────────────────────────────────────────────────

        # Flatten, gate, project
        y = y.reshape(batch, seqlen, cfg.d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Slot Block and Full Model
# ---------------------------------------------------------------------------

class SlotBlock(nn.Module):
    """One Slot Workspace layer: pre-norm SlotMixer + pre-norm SwiGLUMLP."""

    def __init__(self, cfg: SlotConfig):
        super().__init__()
        self.mixer_norm = RMSNorm(cfg.d_model)
        self.mixer = SlotMixer(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg.d_model, cfg.d_model * cfg.mlp_expand)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SlotLM(nn.Module):
    """Slot Workspace language model.

    Architecture:
        Embedding -> N x SlotBlock -> RMSNorm -> Linear(vocab)

    Each SlotBlock runs Mamba3 SSM followed by intra-slot attention over
    the K workspace slots, enabling latent-space reasoning at O(K^2)
    cost per position (not O(T^2)).

    For long-document QA, use:
        doc_reps = model.encode_document(doc_ids)
        logits = model.forward_with_scroll_back(question_ids, doc_reps)
    """

    def __init__(self, cfg: SlotConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([SlotBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embedding.weight   # weight tying

        if cfg.use_scroll_back:
            self.scroll_back = ScrollBackAttention(
                d_model=cfg.d_model,
                n_heads=cfg.scroll_back_heads,
            )

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for activation memory during training.

        Re-materialises each SlotBlock's forward pass during the backward
        pass instead of storing activations.  Reduces peak VRAM by roughly
        the number of layers at the cost of one extra forward per layer.
        Call once before the training loop; no effect during eval.
        """
        self._use_grad_ckpt = True

    def _run_layers(self, x: Tensor) -> Tensor:
        if getattr(self, '_use_grad_ckpt', False) and self.training:
            import torch.utils.checkpoint as ckpt
            for layer in self.layers:
                x = ckpt.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        return x

    def forward(self, input_ids: Tensor) -> Tensor:
        """Standard forward pass (language modelling, no scroll-back).

        Args:
            input_ids: (batch, seqlen)
        Returns:
            logits: (batch, seqlen, vocab_size)
        """
        x = self.embedding(input_ids)
        x = self._run_layers(x)
        return self.out_proj(self.norm(x))

    def encode_document(self, doc_ids: Tensor) -> Tensor:
        """Process a document and return contextual representations.

        Call this once per document and cache the result.  Pass it to
        forward_with_scroll_back for question answering.

        Args:
            doc_ids: (batch, doc_len) token indices
        Returns:
            doc_reps: (batch, doc_len, d_model) — contextual, before final norm
        """
        x = self.embedding(doc_ids)
        return self._run_layers(x)

    def forward_with_scroll_back(
        self,
        input_ids: Tensor,
        doc_reps: Tensor,
        doc_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with document scroll-back for long-range retrieval.

        Processes input_ids through the slot workspace layers, then
        cross-attends to doc_reps (precomputed by encode_document) before
        generating output logits.

        Typical use:
            1. doc_reps = model.encode_document(doc_ids)     # once per doc
            2. logits = model.forward_with_scroll_back(      # per question
                            question_ids, doc_reps)

        Args:
            input_ids: (batch, T_q) — question + answer prefix
            doc_reps:  (batch, doc_len, d_model) — from encode_document()
            doc_mask:  (batch, doc_len) bool, True=valid (for padded docs)
        Returns:
            logits: (batch, T_q, vocab_size)
        """
        if not self.cfg.use_scroll_back:
            raise RuntimeError(
                "ScrollBackAttention is not enabled. "
                "Set use_scroll_back=True in SlotConfig."
            )
        x = self.embedding(input_ids)
        x = self._run_layers(x)
        x = self.scroll_back(x, doc_reps, doc_mask)
        return self.out_proj(self.norm(x))

    def parameter_count(self) -> dict[str, int]:
        """Return parameter counts by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        scroll_count = count(self.scroll_back) if self.cfg.use_scroll_back else 0
        slot_attn_count = sum(
            count(layer.mixer.slot_attn)
            for layer in self.layers
            if layer.mixer.slot_attn is not None
        )
        total = count(self)
        return {
            'total':             total,
            'embedding':         count(self.embedding),
            'layers':            count(self.layers),
            'slot_attention':    slot_attn_count,
            'scroll_back':       scroll_count,
            'output_proj':       count(self.out_proj),
        }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check():
    """Quick shape/forward check. Run: python slot_workspace.py"""
    import sys

    print("Slot Workspace sanity check")
    print("=" * 50)

    cfg = SlotConfig(
        d_model=128,
        expand=2,
        n_slots=8,       # K=8 slots, headdim = 256/8 = 32
        d_state=32,
        chunk_size=32,
        n_layer=2,
        intra_attn_heads=4,
        use_scroll_back=True,
        scroll_back_heads=4,
    )
    print(f"Config: d_model={cfg.d_model}, n_slots={cfg.n_slots}, "
          f"headdim={cfg.headdim}, d_state={cfg.d_state}")

    vocab_size = 256
    model = SlotLM(cfg, vocab_size)

    counts = model.parameter_count()
    print(f"\nParameter counts:")
    for k, v in counts.items():
        print(f"  {k:20s}: {v:,}")

    # Test standard forward
    batch, seqlen = 2, 64
    ids = torch.randint(0, vocab_size, (batch, seqlen))
    logits = model(ids)
    assert logits.shape == (batch, seqlen, vocab_size), logits.shape
    print(f"\nforward()           : {tuple(ids.shape)} -> {tuple(logits.shape)}  OK")

    # Test encode_document
    doc_len = 128
    doc_ids = torch.randint(0, vocab_size, (batch, doc_len))
    doc_reps = model.encode_document(doc_ids)
    assert doc_reps.shape == (batch, doc_len, cfg.d_model), doc_reps.shape
    print(f"encode_document()   : {tuple(doc_ids.shape)} -> {tuple(doc_reps.shape)}  OK")

    # Test scroll-back forward (must be multiple of chunk_size=32)
    q_len = 32
    q_ids = torch.randint(0, vocab_size, (batch, q_len))
    logits_sb = model.forward_with_scroll_back(q_ids, doc_reps)
    assert logits_sb.shape == (batch, q_len, vocab_size), logits_sb.shape
    print(f"forward_with_scroll_back(): {tuple(q_ids.shape)} + doc -> "
          f"{tuple(logits_sb.shape)}  OK")

    # Test IntraSlotAttention is doing actual work (output != input)
    slots_in = torch.randn(1, 4, cfg.n_slots, cfg.headdim)
    attn_layer = model.layers[0].mixer.slot_attn
    slots_out = attn_layer(slots_in)
    assert not torch.allclose(slots_in, slots_out), "slot_attn is identity!"
    print(f"\nIntraSlotAttention output != input  OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    _sanity_check()
