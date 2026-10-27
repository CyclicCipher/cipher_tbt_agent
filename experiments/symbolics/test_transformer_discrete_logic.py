"""
test_transformer_discrete_logic.py
2-Adic Ultrametric Transformer with Attention Residuals (AttnRes).

Configured with 5 layers and regularized to prevent overconfidence/val loss explosion.
"""

import math
import random
import time
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =====================================================================
# MODEL & TRAINING HYPERPARAMETERS (EXPLICIT CONFIGURATION)
# =====================================================================
NUM_LAYERS     = 5         # Set to 5 layers as required
D_MODEL        = 128
N_HEADS        = 4
D_FF           = 512
MAX_TREE_DEPTH = 6         # Trees generated up to depth 5
NUM_SAMPLES    = 8000      # Expanded dataset to curb memorization
BATCH_SIZE     = 256
LEARNING_RATE  = 1.5e-3
WEIGHT_DECAY   = 0.01      # Regularization to prevent logit explosion
DROPOUT        = 0.1
EPOCHS         = 120

# =====================================================================
# 1. 2-ADIC TREE PARSER & TREE GEODESIC ENGINE
# =====================================================================
BINARY_OPS = ["AND", "OR", "XOR"]
UNARY_OPS = ["NOT"]
VALUES = ["0", "1"]

TOKEN_TO_ID = {
    "<PAD>": 0, "0": 1, "1": 2, "AND": 3, "OR": 4, "XOR": 5, "NOT": 6, "=": 7
}

def generate_deep_logic_tree(depth: int, max_depth: int = 5) -> list:
    """Recursively generates valid prefix Boolean expression trees."""
    if depth >= max_depth or (depth > 1 and random.random() < 0.18):
        return [random.choice(VALUES)]
    if random.random() < 0.2:
        return ["NOT"] + generate_deep_logic_tree(depth + 1, max_depth)
    op = random.choice(BINARY_OPS)
    return [op] + generate_deep_logic_tree(depth + 1, max_depth) + generate_deep_logic_tree(depth + 1, max_depth)

def parse_tree_to_2adic_paths(tokens: List[str]) -> Tuple[List[Tuple[int, ...]], List[int]]:
    """
    Computes exact Boolean valuations (bottom-up) and assigns each node
    its binary branching sequence sigma in {0, 1}* in Z_2.
    """
    stack_vals = []
    valuations = [0] * len(tokens)

    for idx in reversed(range(len(tokens))):
        tok = tokens[idx]
        if tok in VALUES:
            valuations[idx] = int(tok)
            stack_vals.append(valuations[idx])
        elif tok in UNARY_OPS:
            child_val = stack_vals.pop()
            valuations[idx] = 1 - child_val
            stack_vals.append(valuations[idx])
        elif tok in BINARY_OPS:
            v1 = stack_vals.pop()
            v2 = stack_vals.pop()
            if tok == "AND": valuations[idx] = v1 & v2
            elif tok == "OR": valuations[idx] = v1 | v2
            elif tok == "XOR": valuations[idx] = v1 ^ v2
            stack_vals.append(valuations[idx])

    full_paths = [()] * len(tokens)

    def subtree_size(start: int) -> int:
        needed = 1
        curr = start
        while needed > 0 and curr < len(tokens):
            t = tokens[curr]
            if t in BINARY_OPS: needed += 1
            elif t in VALUES: needed -= 1
            curr += 1
        return curr - start

    def assign_prefixes(curr_idx: int, curr_prefix: Tuple[int, ...]):
        full_paths[curr_idx] = curr_prefix
        tok = tokens[curr_idx]
        if tok in UNARY_OPS:
            assign_prefixes(curr_idx + 1, curr_prefix + (0,))
        elif tok in BINARY_OPS:
            left_start = curr_idx + 1
            left_len = subtree_size(left_start)
            right_start = left_start + left_len
            assign_prefixes(left_start, curr_prefix + (0,))
            assign_prefixes(right_start, curr_prefix + (1,))

    assign_prefixes(0, ())
    return full_paths, valuations

def compute_2adic_tree_distance_matrix(paths: List[Tuple[int, ...]], max_seq_len: int) -> torch.Tensor:
    """
    Computes the 2-adic tree geodesic metric:
        d_T(u, v) = depth(u) + depth(v) - 2 * LCA_depth(u, v)
        D2(u, v)  = 1.0 - 2^(-d_T(u, v))
    """
    N = len(paths)
    D2 = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            p1, p2 = paths[i], paths[j]
            lca_len = 0
            for b1, b2 in zip(p1, p2):
                if b1 == b2: lca_len += 1
                else: break
            d_tree = len(p1) + len(p2) - 2 * lca_len
            D2[i, j] = 1.0 - (2.0 ** (-float(d_tree)))
    return D2


# =====================================================================
# 2. ROTARY POSITION EMBEDDING & 2-ADIC ULTRAMETRIC ATTENTION
# =====================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[2]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class AdelicUltrametricAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.beta = nn.Parameter(torch.tensor(1.2))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, d2_matrix: torch.Tensor):
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q), self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        ultrametric_bias = -torch.abs(self.beta) * d2_matrix.unsqueeze(1)
        attn = F.softmax(scores + ultrametric_bias, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


# =====================================================================
# 3. ATTENTION RESIDUALS (AttnRes) DEPTH ROUTER
# =====================================================================
class DepthAttentionResidualRouter(nn.Module):
    def __init__(self, d_model: int, max_layers: int = 16):
        super().__init__()
        self.layer_queries = nn.Parameter(torch.randn(max_layers, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, history: list, layer_idx: int):
        if len(history) == 1:
            return history[0]
        stacked = torch.stack(history, dim=1)
        B, K, S, D = stacked.shape
        keys = self.k_proj(stacked)
        q = self.layer_queries[layer_idx].view(1, 1, 1, D)
        logits = (keys * q).sum(dim=-1) / math.sqrt(D)
        alpha = F.softmax(logits, dim=1).unsqueeze(-1)
        return (alpha * stacked).sum(dim=1)


# =====================================================================
# 4. 2-ADIC COORDINATE EMBEDDING + TRANSFORMER (5 LAYERS)
# =====================================================================
class AdelicLogicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        n_layers: int = NUM_LAYERS,
        max_depth: int = MAX_TREE_DEPTH,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        # 2-Adic Branch Coordinate Embedding in Z_2
        self.branch_embed = nn.Parameter(torch.randn(max_depth, 2, d_model) * 0.02)

        self.attn_blocks = nn.ModuleList([
            AdelicUltrametricAttention(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(2 * n_layers + 1)])
        self.router = DepthAttentionResidualRouter(d_model, max_layers=2 * n_layers + 2)
        self.unembed = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor, branch_paths: torch.Tensor, d2_matrix: torch.Tensor):
        B, S, max_d = branch_paths.shape

        mask_0 = (branch_paths == 0).float().unsqueeze(-1)
        mask_1 = (branch_paths == 1).float().unsqueeze(-1)
        b_emb_0 = self.branch_embed[:, 0, :].view(1, 1, max_d, -1)
        b_emb_1 = self.branch_embed[:, 1, :].view(1, 1, max_d, -1)
        p2_emb = (mask_0 * b_emb_0 + mask_1 * b_emb_1).sum(dim=2)

        v0 = self.drop(self.embed(x) + p2_emb)
        history = [v0]

        sublayer_idx = 1
        for i in range(self.n_layers):
            # Attention with AttnRes
            h_attn = self.router(history, sublayer_idx)
            v_attn = self.attn_blocks[i](self.norms[2 * i](h_attn), d2_matrix)
            history.append(v_attn)
            sublayer_idx += 1

            # MLP with AttnRes
            h_mlp = self.router(history, sublayer_idx)
            v_mlp = self.mlp_blocks[i](self.norms[2 * i + 1](h_mlp))
            history.append(v_mlp)
            sublayer_idx += 1

        final_h = self.router(history, sublayer_idx)
        final_h = self.norms[-1](final_h)
        return self.unembed(final_h)


# =====================================================================
# 5. CALIBRATED 2-ADIC FILTRATION LOSS
# =====================================================================
class AdelicFiltrationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, depths: torch.Tensor, mask: torch.Tensor):
        B, S, C = logits.shape
        flat_logits = logits.view(B * S, C)
        flat_targets = targets.view(B * S)
        flat_mask = mask.view(B * S)
        flat_depths = depths.view(B * S).float()

        v2_weights = torch.pow(2.0, -flat_depths) * flat_mask
        raw_loss = self.ce(flat_logits, flat_targets)
        loss = (raw_loss * v2_weights).sum() / (v2_weights.sum() + 1e-8)
        return loss


# =====================================================================
# 6. DATASET & 120-EPOCH BENCHMARK
# =====================================================================
def build_dataset(num_samples: int = NUM_SAMPLES, max_seq_len: int = 25):
    random.seed(42)
    samples = []
    seen = set()

    while len(samples) < num_samples:
        expr = generate_deep_logic_tree(depth=1, max_depth=MAX_TREE_DEPTH - 1)
        expr_str = " ".join(expr)
        if expr_str in seen or len(expr) + 1 > max_seq_len:
            continue
        seen.add(expr_str)

        paths, valuations = parse_tree_to_2adic_paths(expr)
        tokens = expr + ["="]
        seq_len = len(tokens)

        tok_ids = [TOKEN_TO_ID[t] for t in tokens] + [0] * (max_seq_len - seq_len)
        targets = valuations + [valuations[0]] + [0] * (max_seq_len - seq_len)
        depths = [len(p) for p in paths] + [0] + [0] * (max_seq_len - seq_len)
        mask = [1.0] * len(expr) + [1.0] + [0.0] * (max_seq_len - seq_len)

        branch_paths = torch.full((max_seq_len, MAX_TREE_DEPTH), fill_value=-1, dtype=torch.long)
        for i, p in enumerate(paths):
            for d_idx, branch in enumerate(p[:MAX_TREE_DEPTH]):
                branch_paths[i, d_idx] = branch

        d2_matrix = compute_2adic_tree_distance_matrix(paths + [()], max_seq_len)
        eq_pos = len(expr)

        samples.append((tok_ids, targets, depths, mask, branch_paths, d2_matrix, eq_pos, valuations[0], expr_str))

    return samples


def run_logic_experiment(epochs: int = EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 82)
    print(f"2-ADIC ULTRAMETRIC TRANSFORMER (Layers={NUM_LAYERS}, AttnRes + Z_2 Coordinates + RoPE)")
    print(f"Device: {device} | Samples: {NUM_SAMPLES} | Weight Decay: {WEIGHT_DECAY}")
    print("=" * 82)

    data = build_dataset(num_samples=NUM_SAMPLES, max_seq_len=25)
    split = int(0.75 * len(data))
    train, val = data[:split], data[split:]

    def to_tensors(batch):
        return (
            torch.tensor([b[0] for b in batch], dtype=torch.long),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
            torch.tensor([b[2] for b in batch], dtype=torch.long),
            torch.tensor([b[3] for b in batch], dtype=torch.float32),
            torch.stack([b[4] for b in batch]),
            torch.stack([b[5] for b in batch]),
            torch.tensor([b[6] for b in batch], dtype=torch.long),
            torch.tensor([b[7] for b in batch], dtype=torch.long)
        )

    tr_x, tr_y, tr_d, tr_m, tr_bp, tr_d2, tr_pos, tr_root = to_tensors(train)
    val_x, val_y, val_d, val_m, val_bp, val_d2, val_pos, val_root = to_tensors(val)

    train_loader = DataLoader(TensorDataset(tr_x, tr_y, tr_d, tr_m, tr_bp, tr_d2, tr_pos, tr_root), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y, val_d, val_m, val_bp, val_d2, val_pos, val_root), batch_size=BATCH_SIZE * 2, shuffle=False)

    model = AdelicLogicTransformer(
        vocab_size=len(TOKEN_TO_ID),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=NUM_LAYERS,
        max_depth=MAX_TREE_DEPTH,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = AdelicFiltrationLoss()

    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<9} | {'Train Acc':<10} | {'Val Acc':<9} | {'Status'}")
    print("-" * 75)

    start_time = time.time()
    solved_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for bx, by, bd, bm, bbp, bd2, bpos, broot in train_loader:
            bx, by, bd, bm, bbp, bd2 = bx.to(device), by.to(device), bd.to(device), bm.to(device), bbp.to(device), bd2.to(device)
            bpos, broot = bpos.to(device), broot.to(device)

            optimizer.zero_grad()
            logits = model(bx, bbp, bd2)
            loss = criterion(logits, by, bd, bm)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)
            preds = logits[torch.arange(len(broot)), bpos, :].argmax(dim=-1)
            train_correct += (preds == broot).sum().item()
            train_total += len(broot)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for vx, vy, vd, vm, vbp, vd2, vpos, vroot in val_loader:
                    vx, vy, vd, vm, vbp, vd2 = vx.to(device), vy.to(device), vd.to(device), vm.to(device), vbp.to(device), vd2.to(device)
                    vpos, vroot = vpos.to(device), vroot.to(device)

                    logits = model(vx, vbp, vd2)
                    loss = criterion(logits, vy, vd, vm)
                    val_loss += loss.item() * len(vx)

                    preds = logits[torch.arange(len(vroot)), vpos, :].argmax(dim=-1)
                    val_correct += (preds == vroot).sum().item()
                    val_total += len(vroot)

            t_loss = train_loss / train_total
            v_loss = val_loss / val_total
            t_acc = train_correct / train_total
            v_acc = val_correct / val_total

            status = ""
            if v_acc >= 0.95 and solved_epoch is None:
                solved_epoch = epoch
                status = ">>> 95%+ SOLVED! <<<"

            print(f"{epoch:<6} | {t_loss:<11.4f} | {v_loss:<9.4f} | {t_acc*100:<9.1f}% | {v_acc*100:<8.1f}% | {status}")

            if v_acc >= 0.995:
                break

    print(f"\n[+] Completed in {time.time()-start_time:.2f}s | 95%+ Solved Epoch: {solved_epoch}")


if __name__ == "__main__":
    run_logic_experiment(epochs=EPOCHS)