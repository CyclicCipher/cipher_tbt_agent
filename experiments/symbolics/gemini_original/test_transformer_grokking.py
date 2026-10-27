"""
test_transformer_grokking.py
Genuine Adelic Modular Arithmetic via the Hasse Principle on A_Q with AttnRes.

Modulus: N = 105 = 3 * 5 * 7
Places S = {infty, 3, 5, 7}

Hyperparameters aligned with the discrete logic experiment:
  - WEIGHT_DECAY = 0.01
  - DROPOUT      = 0.1
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =====================================================================
# MODEL & TRAINING HYPERPARAMETERS (EXPLICIT CONFIGURATION)
# =====================================================================
MODULUS        = 105       # N = 3 * 5 * 7
D_MODEL        = 128
N_HEADS        = 4
D_FF           = 256
BATCH_SIZE     = 512
LEARNING_RATE  = 2e-3
WEIGHT_DECAY   = 0.01      # Matched to discrete logic experiment
DROPOUT        = 0.1       # Matched to discrete logic experiment
EPOCHS         = 120
LAMBDA_TATE    = 0.5

# =====================================================================
# 1. ROTARY POSITION EMBEDDING & MULTI-HEAD ATTENTION
# =====================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8):
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


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = DROPOUT):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q), self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =====================================================================
# 2. ATTENTION RESIDUALS (AttnRes) DEPTH ROUTER
# =====================================================================
class DepthAttentionResidualRouter(nn.Module):
    def __init__(self, d_model: int, max_layers: int = 8):
        super().__init__()
        self.layer_queries = nn.Parameter(torch.randn(max_layers, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, layer_history: list, current_layer_idx: int):
        if len(layer_history) == 1:
            return layer_history[0]
        stacked = torch.stack(layer_history, dim=1)
        B, K, S, D = stacked.shape
        keys = self.k_proj(stacked)
        q = self.layer_queries[current_layer_idx].view(1, 1, 1, D)
        logits = (keys * q).sum(dim=-1) / math.sqrt(D)
        alpha = F.softmax(logits, dim=1).unsqueeze(-1)
        return (alpha * stacked).sum(dim=1)


# =====================================================================
# 3. ADELIC LOCAL-GLOBAL EMBEDDING & TRANSFORMER
# =====================================================================
class AdelicEmbedding(nn.Module):
    """
    Decomposes residue classes in Z/105Z into the Adelic product space:
        A_{105} = R/105Z x F_3 x F_5 x F_7
    """
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        d_p = d_model // 4  # 32 dimensions per place

        # Place infty: Archimedean circle projection
        self.proj_infty = nn.Linear(2, d_p)

        # Non-Archimedean local places (residue fields F_p)
        self.embed_3 = nn.Embedding(3, d_p)
        self.embed_5 = nn.Embedding(5, d_p)
        self.embed_7 = nn.Embedding(7, d_p)

    def forward(self, x: torch.Tensor):
        is_eq = (x == MODULUS)
        safe_x = torch.where(is_eq, torch.zeros_like(x), x)

        # Archimedean S^1 coordinates
        theta = 2.0 * math.pi * safe_x.float() / float(MODULUS)
        circle_coords = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        h_infty = self.proj_infty(circle_coords)

        # Non-Archimedean local projections
        h_3 = self.embed_3(safe_x % 3)
        h_5 = self.embed_5(safe_x % 5)
        h_7 = self.embed_7(safe_x % 7)

        h_A = torch.cat([h_infty, h_3, h_5, h_7], dim=-1)
        h_A = torch.where(is_eq.unsqueeze(-1), torch.zeros_like(h_A), h_A)
        return h_A


class AdelicModularTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.adelic_embed = AdelicEmbedding(d_model)
        self.drop = nn.Dropout(dropout)

        self.norm1 = nn.RMSNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.router = DepthAttentionResidualRouter(d_model, max_layers=4)
        self.final_norm = nn.RMSNorm(d_model)

        # Local-Global Readout Heads
        self.head_infty = nn.Linear(d_model, MODULUS)
        self.head_3 = nn.Linear(d_model, 3)
        self.head_5 = nn.Linear(d_model, 5)
        self.head_7 = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor):
        v0 = self.drop(self.adelic_embed(x))
        history = [v0]

        # Layer 1: Attn with AttnRes depth routing
        h1 = self.router(history, current_layer_idx=1)
        v1 = self.attn(self.norm1(h1))
        history.append(v1)

        # Layer 2: MLP with AttnRes depth routing
        h2 = self.router(history, current_layer_idx=2)
        v2 = self.mlp(self.norm2(h2))
        history.append(v2)

        # Final readout aggregation
        h3 = self.router(history, current_layer_idx=3)
        out = self.final_norm(h3)[:, 2, :]  # Readout at '=' position

        return {
            "infty": self.head_infty(out),
            "p3": self.head_3(out),
            "p5": self.head_5(out),
            "p7": self.head_7(out)
        }


# =====================================================================
# 4. HASSE LOCAL-GLOBAL & TATE CHARACTER LOSS
# =====================================================================
class AdelicHasseLoss(nn.Module):
    def __init__(self, lambda_tate: float = LAMBDA_TATE):
        super().__init__()
        self.lambda_tate = lambda_tate
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: dict, targets: torch.Tensor):
        loss_3 = self.ce(preds["p3"], targets % 3)
        loss_5 = self.ce(preds["p5"], targets % 5)
        loss_7 = self.ce(preds["p7"], targets % 7)
        loss_infty = self.ce(preds["infty"], targets)

        # Tate Global Additive Character Penalty
        pred_c = preds["infty"].argmax(dim=-1).float()
        theta_infty = 2.0 * math.pi * pred_c / float(MODULUS)
        theta_local = (
            2.0 * math.pi * (preds["p3"].argmax(dim=-1).float()) / 3.0 +
            2.0 * math.pi * (preds["p5"].argmax(dim=-1).float()) / 5.0 +
            2.0 * math.pi * (preds["p7"].argmax(dim=-1).float()) / 7.0
        )
        phase_error = 1.0 - torch.cos(theta_infty - theta_local)
        loss_tate = phase_error.mean()

        return loss_infty + (loss_3 + loss_5 + loss_7) + self.lambda_tate * loss_tate


# =====================================================================
# 5. DATASET & 120-EPOCH BENCHMARK
# =====================================================================
def run_grokking_experiment(epochs: int = EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 82)
    print(f"GENUINE ADELIC GROKKING (AttnRes + RoPE + A_Q Hasse Loss, N={MODULUS})")
    print(f"Device: {device} | Weight Decay: {WEIGHT_DECAY} | Dropout: {DROPOUT}")
    print("=" * 82)

    a = torch.arange(MODULUS).repeat_interleave(MODULUS)
    b = torch.arange(MODULUS).repeat(MODULUS)
    eq = torch.full_like(a, fill_value=MODULUS)
    targets = (a + b) % MODULUS
    inputs = torch.stack([a, b, eq], dim=1)

    torch.manual_seed(42)
    indices = torch.randperm(len(inputs))
    split = int(0.5 * len(inputs))

    train_x, train_y = inputs[indices[:split]], targets[indices[:split]]
    val_x, val_y = inputs[indices[split:]], targets[indices[split:]]

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE * 2, shuffle=False)

    model = AdelicModularTransformer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = AdelicHasseLoss(lambda_tate=LAMBDA_TATE)

    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<9} | {'Train Acc':<10} | {'Val Acc':<9} | {'Status'}")
    print("-" * 75)

    grok_epoch = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            global_pred = preds["infty"].argmax(dim=-1)
            train_correct += (global_pred == by).sum().item()
            train_total += len(by)

        # Evaluate epochs 1-10 individually, then every 10 epochs
        if epoch <= 10 or epoch % 10 == 0:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    preds = model(vx)
                    loss = criterion(preds, vy)
                    val_loss += loss.item() * len(vy)

                    global_pred = preds["infty"].argmax(dim=-1)
                    val_correct += (global_pred == vy).sum().item()
                    val_total += len(vy)

            t_loss = train_loss / train_total
            v_loss = val_loss / val_total
            t_acc = train_correct / train_total
            v_acc = val_correct / val_total

            status = ""
            if v_acc >= 0.99 and grok_epoch is None:
                grok_epoch = epoch
                status = ">>> GROKKED! <<<"

            print(f"{epoch:<6} | {t_loss:<11.4f} | {v_loss:<9.4f} | {t_acc*100:<9.1f}% | {v_acc*100:<8.1f}% | {status}")

            if v_acc >= 0.995:
                break

    print(f"\n[+] Completed in {time.time()-start:.2f}s | Grokking Point: Epoch {grok_epoch}")


if __name__ == "__main__":
    run_grokking_experiment(epochs=EPOCHS)