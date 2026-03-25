"""
Compositor -- toy model, version 0.6

Architecture: categorical graph + pointer embeddings + Compositor blocks.

v0.6: Looped FFN-Composition sandwich (Phase 6 from ROADMAP.md).

The FFN WRAPS the composition layer. Each iteration:
1. Pre-read: FFN sees what graph edges are available from the current node
2. Navigation: FFN first half computes intent (what to query)
3. Relation selection: intent → K relation weights (which edge type to follow)
4. Composition: single graph hop using current node IDs + selected relations
5. State update: graph result feeds back into state for next iteration
6. Node update: argmax of graph_result similarity → next node for next hop

The loop replaces fixed n_compose_hops with dynamic iteration. Multi-hop
traversal emerges from repeated single-step hops with node identity updating.

Why this works when v0.1-v0.5 didn't:
- v0.1-v0.4: attention bypass let model ignore graph entirely
- v0.5: blocked bypass but FFN came AFTER composition — couldn't control traversal
- v0.6: FFN CONTROLS the traversal. It sees the graph before querying it,
  decides which relation to follow, and gets the result back for the next step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# PoPE -- Polar Positional Embedding
# --------------------------------------------------------------------------- #


class PoPE(nn.Module):
    """Polar Positional Embedding.

    Encodes position in the ANGLE and content in the MAGNITUDE of each
    dimension pair. Cleanly separates WHAT from WHERE in Q/K dot product.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for PoPE"
        self.d_model = d_model
        half = d_model // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half).float() / half))
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if x.dim() == 4:
            b, h, s, d = x.shape
            x = x.reshape(b * h, s, d)
            out = self._rotate(x, offset)
            return out.reshape(b, h, s, d)
        return self._rotate(x, offset)

    def _rotate(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        seq_len = x.size(1)
        half = self.d_model // 2
        pos = torch.arange(offset, offset + seq_len, device=x.device).float()
        angles = pos.unsqueeze(1) * self.freqs.unsqueeze(0)
        cos_a = angles.cos()
        sin_a = angles.sin()
        x1 = x[..., :half]
        x2 = x[..., half:]
        out1 = x1 * cos_a - x2 * sin_a
        out2 = x1 * sin_a + x2 * cos_a
        return torch.cat([out1, out2], dim=-1)


# --------------------------------------------------------------------------- #
# Categorical Graph
# --------------------------------------------------------------------------- #


class CategoricalGraph(nn.Module):
    """Persistent categorical knowledge graph with sigmoid edge probabilities.

    The graph has N nodes and K relation types. Morphism strengths are stored
    as a learned adjacency tensor A[K, N, N] -- the logit of relation k
    from node i to node j. Edge probabilities are sigmoid(A).

    v0.6: get_composed_adjacency() retained for backward compatibility with
    inspect_graph.py and structural_ops.py, but the Compositor model uses
    only get_edge_probs() (1-hop). Multi-hop composition is handled by the
    looped FFN-composition sandwich iterating single hops with node updates.
    """

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        n_relations: int,
        n_compose_hops: int = 2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.n_relations = n_relations
        self.n_compose_hops = n_compose_hops

        # Node embeddings -- addresses into the graph
        self.nodes = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)

        # Adjacency tensor -- logits for edge probabilities
        # Init at -4: sigmoid(-4) ~ 0.018 (sparse start)
        # Small noise for symmetry breaking
        self.A = nn.Parameter(torch.randn(n_relations, n_nodes, n_nodes) * 0.1 - 4.0)

        # Diagonal mask: True on diagonal, used to force no self-loops
        self.register_buffer(
            "diag_mask",
            torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0).expand(n_relations, -1, -1),
        )

    def get_edge_probs(self) -> torch.Tensor:
        """Get edge probabilities via sigmoid. Diagonal forced to 0.

        Returns: (K, N, N) -- independent edge probabilities in (0, 1).
        """
        A_masked = self.A.masked_fill(self.diag_mask, -20.0)
        return torch.sigmoid(A_masked)

    def get_composed_adjacency(self) -> torch.Tensor:
        """Compute composed adjacency: 1-hop + 2-hop + ... n-hop.

        Retained for backward compatibility with inspect_graph.py and
        structural_ops.py. The looped sandwich in CompositorBlock replaces
        this with dynamic iteration.
        """
        P = self.get_edge_probs()
        if self.n_compose_hops <= 1:
            return P
        result = P
        composed = P
        for _ in range(self.n_compose_hops - 1):
            composed = torch.bmm(composed, P)
            result = result + composed
        return result

    def get_node_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up node embeddings by index. (batch, seq) -> (batch, seq, d_model)"""
        return self.nodes[indices]


# --------------------------------------------------------------------------- #
# Composition Layer -- the novel component
# --------------------------------------------------------------------------- #


class SingleHopComposition(nn.Module):
    """Execute one graph hop: given node IDs and relation weights, traverse.

    This is a pure graph operation with no learned parameters of its own.
    The relation weights come from the FFN (which controls traversal),
    and node IDs come from either input_ids (first iteration) or the
    previous hop's result (subsequent iterations).

    Returns:
        graph_result: (B, S, d_model) -- weighted sum of arrived-at node embeddings
        reached_dist: (B, S, N) -- distribution over nodes reached (for node update)
    """

    def __init__(self, graph: CategoricalGraph):
        super().__init__()
        self.graph = graph

    def forward(
        self, P: torch.Tensor, node_ids: torch.Tensor, rel_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            P: (K, N, N) -- 1-hop edge probabilities
            node_ids: (B, S) -- current node indices (hard)
            rel_weights: (B, S, K) -- which relations to follow (from FFN)

        Returns:
            graph_result: (B, S, d_model) -- retrieved node embeddings
            reached_dist: (B, S, N) -- distribution over reached nodes
        """
        B, S = node_ids.shape
        K = self.graph.n_relations
        N = self.graph.n_nodes

        # 1. Hard node lookup: get adjacency rows for current nodes
        ids_flat = node_ids.reshape(B * S)  # (B*S,)
        reached = P[:, ids_flat, :]  # (K, B*S, N)
        reached = reached.permute(1, 0, 2)  # (B*S, K, N)

        # 2. Weighted combination of relations
        rw = rel_weights.reshape(B * S, K)  # (B*S, K)
        combined_reached = torch.einsum("bk,bkn->bn", rw, reached)  # (B*S, N)

        # 3. Retrieve node embeddings (values are node embeddings, weight-tied to output)
        nodes = self.graph.nodes  # (N, D)
        graph_result = combined_reached @ nodes  # (B*S, D)

        return graph_result.reshape(B, S, -1), combined_reached.reshape(B, S, N)


# --------------------------------------------------------------------------- #
# SwiGLU FFN
# --------------------------------------------------------------------------- #


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (standard, single-stream)."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.silu(self.w_gate(x)))


# --------------------------------------------------------------------------- #
# Multi-Head Attention
# --------------------------------------------------------------------------- #


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with PoPE."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.pope = PoPE(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape
        H = self.n_heads

        q = self.q_proj(x).view(B, S, H, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, self.d_head).transpose(1, 2)

        q = self.pope(q)
        k = self.pope(k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Compositor Block
# --------------------------------------------------------------------------- #


class CompositorBlock(nn.Module):
    """One Compositor block: Attention → Pre-read → FFN-Composition → Update.

    v0.6 (Ouro-style): Each block performs ONE graph hop. Multi-hop traversal
    comes from the model-level loop (Compositor.forward runs the full block
    stack T times with shared weights). This follows the Ouro/LoopLM design:
    loop the entire stack, not individual layers.

    Information flow per block:
      1. Attention: cross-position context (what tokens are nearby)
      2. Pre-read: mean edge strength per relation from current node → d_model
      3. FFN first half: [state, pre_read] → navigation intent
      4. Relation selection: navigation → K relation weights
      5. Composition: one graph hop using current node IDs + selected relations
      6. FFN second half: navigation × graph_result → residual update

    Node identity updating happens BETWEEN loop passes in Compositor.forward,
    not inside the block. This keeps the block a clean single-hop operator.
    """

    def __init__(
        self,
        graph: CategoricalGraph,
        d_model: int,
        n_heads: int,
        d_hidden: int,
    ):
        super().__init__()
        self.graph = graph
        K = graph.n_relations

        # Attention: cross-position context
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm_attn = nn.RMSNorm(d_model)

        # Graph hop operator (no learned params, just the traversal math)
        self.hop = SingleHopComposition(graph)

        # Pre-read summarizer: per-relation mean edge strength → (d_model,)
        self.pre_read_proj = nn.Linear(K, d_model, bias=False)

        # FFN first half: state + pre_read → navigation intent in d_hidden
        self.norm_state = nn.RMSNorm(d_model)
        self.w1 = nn.Linear(d_model * 2, d_hidden, bias=False)  # [state, pre_read]
        self.w_gate = nn.Linear(d_model * 2, d_hidden, bias=False)

        # Relation selector: navigation → K relation weights
        self.relation_selector = nn.Linear(d_hidden, K, bias=False)

        # Graph result integration:
        self.graph_proj = nn.Linear(d_model, d_hidden, bias=False)

        # FFN second half: combined → d_model residual update
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, P: torch.Tensor,
        node_ids: torch.Tensor = None, mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, S, d_model) -- input hidden states
            P: (K, N, N) -- 1-hop edge probabilities
            node_ids: (B, S) -- current graph position (updated between loop passes)
            mask: (B, S, S) -- causal attention mask

        Returns:
            x: (B, S, d_model) -- updated hidden states
            graph_result: (B, S, d_model) -- raw graph output (for node update in outer loop)
        """
        B, S, D = x.shape
        K = self.graph.n_relations

        # 1. Attention: cross-position context
        attn_out = x + self.attn(self.norm_attn(x), mask=mask)

        # 2. Pre-read: what edges are available from the current node?
        pre_read = self._compute_pre_read(P, node_ids)  # (B, S, D)

        # 3. FFN first half: what do I want from the graph?
        h = self.norm_state(attn_out)
        ffn_input = torch.cat([h, pre_read], dim=-1)  # (B, S, 2*D)
        navigation = self.w1(ffn_input) * F.silu(self.w_gate(ffn_input))  # (B, S, d_hidden)

        # 4. Relation selection
        rel_logits = self.relation_selector(navigation)  # (B, S, K)
        rel_weights = F.softmax(rel_logits, dim=-1)  # (B, S, K)

        # 5. Composition: one graph hop
        graph_result, reached_dist = self.hop(P, node_ids, rel_weights)

        # 6. FFN second half: combine navigation intent with graph result
        graph_hidden = self.graph_proj(graph_result)  # (B, S, d_hidden)
        combined = navigation * graph_hidden  # gated combination
        x = attn_out + self.w2(combined)  # residual from attention output

        return x, graph_result

    def _compute_pre_read(self, P: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        """Compute pre-read summary: mean edge strength per relation per position."""
        B, S = node_ids.shape
        K = P.shape[0]
        ids_flat = node_ids.reshape(B * S)
        adj_rows = P[:, ids_flat, :]  # (K, B*S, N)
        mean_strength = adj_rows.mean(dim=-1).permute(1, 0)  # (B*S, K)
        pre_read = self.pre_read_proj(mean_strength)  # (B*S, D)
        return pre_read.reshape(B, S, -1)


# --------------------------------------------------------------------------- #
# Compositor Model
# --------------------------------------------------------------------------- #


class Compositor(nn.Module):
    """The full Compositor model (v0.6 -- Ouro-style looped stack).

    Categorical graph + pointer embeddings + N Compositor blocks + output head.

    v0.6 (Ouro-style): The full block stack is applied T times (loop_depth).
    Each block does one graph hop per pass. Multi-hop traversal emerges from
    looping the entire stack: T passes × N blocks = T*N total hops.
    Weights are SHARED across loop passes (same blocks re-applied).

    After each full pass through the stack, node identities are updated via
    argmax of the last block's graph_result similarity to node embeddings.
    This is how the model walks along graph edges across loop passes.

    Output weight tying: logits = final_hidden @ graph.nodes[:vocab_size].T
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_hidden: int = 256,
        n_graph_nodes: int = 64,
        n_relations: int = 8,
        n_compose_hops: int = 2,
        max_steps: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_steps = max_steps  # loop depth (Ouro-style)

        self.graph = CategoricalGraph(
            n_nodes=n_graph_nodes,
            d_model=d_model,
            n_relations=n_relations,
            n_compose_hops=n_compose_hops,
        )

        self.blocks = nn.ModuleList([
            CompositorBlock(self.graph, d_model, n_heads, d_hidden)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.RMSNorm(d_model)
        self.output_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_model)))

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        N = self.graph.n_nodes
        x = self.graph.get_node_embeddings(input_ids)

        mask = self._make_causal_mask(S, x.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)

        # 1-hop edge probs — shared across all layers and loop passes
        P = self.graph.get_edge_probs()  # (K, N, N)

        # Current graph position — starts at input tokens, updated between passes
        node_ids = input_ids.clamp(max=N - 1)  # (B, S)

        # Ouro-style loop: run the full block stack T times
        for loop_pass in range(self.max_steps):
            graph_result = None
            for block in self.blocks:
                x, graph_result = block(x, P=P, node_ids=node_ids, mask=mask)

            # Node identity update between passes:
            # The last block's graph_result tells us where we arrived.
            # argmax similarity to node embeddings → new node IDs for next pass.
            if loop_pass < self.max_steps - 1 and graph_result is not None:
                with torch.no_grad():
                    similarity = graph_result.detach() @ self.graph.nodes.detach().T
                    node_ids = similarity.argmax(dim=-1)  # (B, S)

        x = self.final_norm(x)

        # Weight-tied output: dot with vocab node embeddings
        vocab_nodes = self.graph.nodes[:self.vocab_size]  # (V, D)
        logits = x @ vocab_nodes.T * self.output_scale  # (B, S, V)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_graph_params(self) -> list:
        """Return graph adjacency parameters (for zero weight decay)."""
        return [self.graph.A]

    def get_non_graph_params(self) -> list:
        """Return all parameters except graph adjacency."""
        graph_a_id = id(self.graph.A)
        return [p for p in self.parameters() if p.requires_grad and id(p) != graph_a_id]


# --------------------------------------------------------------------------- #
# Baseline transformer (for comparison)
# --------------------------------------------------------------------------- #


class BaselineTransformer(nn.Module):
    """Standard transformer with nn.Embedding, same size as Compositor."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            BaselineBlock(d_model, n_heads, d_hidden)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embedding(input_ids)
        mask = self._make_causal_mask(S, x.device).unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.final_norm(x)
        return self.output_proj(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineBlock(nn.Module):
    """Standard transformer block: Attention -> SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, d_hidden: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = SwiGLU(d_model, d_hidden)
        self.norm_attn = nn.RMSNorm(d_model)
        self.norm_ffn = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), mask=mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from data import VOCAB_SIZE

    model = Compositor(vocab_size=VOCAB_SIZE)
    baseline = BaselineTransformer(vocab_size=VOCAB_SIZE)

    print(f"Compositor params: {model.count_parameters():,}")
    print(f"Baseline params:   {baseline.count_parameters():,}")

    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    logits = model(x)
    print(f"Compositor output shape: {logits.shape}")

    logits_b = baseline(x)
    print(f"Baseline output shape: {logits_b.shape}")

    # Check edge probabilities at init
    P = model.graph.get_edge_probs()
    print(f"\nEdge probs at init:")
    print(f"  Range: [{P.min():.4f}, {P.max():.4f}]")
    print(f"  Mean:  {P.mean():.4f}")
    print(f"  Diagonal mean: {P.diagonal(dim1=-2, dim2=-1).mean():.6f}")

    # Verify Ouro-style loop
    print(f"\nOuro-style loop: loop_depth={model.max_steps}, n_layers={len(model.blocks)}")
    print(f"  {model.max_steps} passes × {len(model.blocks)} blocks = "
          f"{model.max_steps * len(model.blocks)} total hops per forward")
