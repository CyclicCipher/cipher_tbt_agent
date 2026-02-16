"""
Continual learning methods for compositional curriculum training.

Three approaches to prevent catastrophic forgetting while enabling
compositional reuse of learned skills:

1. Differential LR by Layer — lower learning rates for early layers
2. EWC (Elastic Weight Consolidation) — penalize changes to important weights
3. DER++ (Dark Experience Replay++) — match stored logits during replay

See docs/research/continual_learning/README.md for research context.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Differential Learning Rates by Layer
# ---------------------------------------------------------------------------

def build_layer_lr_groups(model, base_lr, lr_decay=0.5, weight_decay=0.01):
    """Create optimizer parameter groups with per-layer learning rates.

    Early layers (embedding, first blocks) get lower LR for stability.
    Later layers get higher LR for plasticity.

    LR for depth i: base_lr * lr_decay^(n_layers - i)

    For n_layer=4, lr_decay=0.5:
      embedding:  base_lr * 0.0625  (most protected)
      layer[0]:   base_lr * 0.125
      layer[1]:   base_lr * 0.25
      layer[2]:   base_lr * 0.5
      layer[3]:   base_lr * 1.0     (most plastic)
      norm:       base_lr * 1.0
    Note: out_proj.weight is tied to embedding.weight (counted once).
    """
    groups = []
    n_layers = len(model.layers)

    # Embedding (most protected; includes out_proj.weight via weight tying)
    emb_lr = base_lr * (lr_decay ** n_layers)
    groups.append({
        'params': list(model.embedding.parameters()),
        'lr': emb_lr,
        'weight_decay': weight_decay,
        'name': 'embedding',
    })

    # Mamba3 blocks (progressive LR)
    for i, layer in enumerate(model.layers):
        layer_lr = base_lr * (lr_decay ** (n_layers - 1 - i))
        groups.append({
            'params': list(layer.parameters()),
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'name': f'layer_{i}',
        })

    # Final norm (most plastic; out_proj excluded — tied to embedding)
    groups.append({
        'params': list(model.norm.parameters()),
        'lr': base_lr,
        'weight_decay': weight_decay,
        'name': 'norm',
    })

    return groups


# ---------------------------------------------------------------------------
# 2. EWC (Elastic Weight Consolidation)
# ---------------------------------------------------------------------------

class EWC:
    """Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    After each stage, computes diagonal Fisher Information and stores
    parameter snapshots. During subsequent stages, adds a quadratic
    penalty pulling important weights back toward post-convergence values.

    Uses online EWC: Fisher is a running mean across stages, so memory
    cost is constant (2x model size) regardless of number of stages.
    """

    def __init__(self, lambda_ewc: float = 400.0):
        self.lambda_ewc = lambda_ewc
        self.fisher: Dict[str, Tensor] = {}
        self.optpar: Dict[str, Tensor] = {}
        self.n_stages = 0

    def register_stage(self, model, train_seqs, device, amp_ctx,
                       n_samples: int = 200):
        """Compute Fisher Information after a stage passes.

        Uses the log-likelihood of correct next tokens as the objective
        for Fisher computation (as in the original EWC paper).
        """
        loader = DataLoader(TensorDataset(train_seqs),
                            batch_size=64, shuffle=True)
        model.eval()
        fisher_acc = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                      if p.requires_grad}

        total = 0
        for (seqs,) in loader:
            if total >= n_samples:
                break
            seqs = seqs.to(device)
            model.zero_grad()
            with amp_ctx():
                logits = model(seqs)
            # Log-likelihood of correct next tokens (ignore PAD=0)
            log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
            targets = seqs[:, 1:]
            mask = targets != 0
            ll = (log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask).sum()
            ll.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_acc[n] += p.grad.detach().pow(2) * seqs.shape[0]
            total += seqs.shape[0]

        # Average and accumulate (online EWC: running mean)
        for n in fisher_acc:
            fisher_acc[n] /= max(total, 1)
            if n in self.fisher:
                self.fisher[n] = (self.fisher[n] * self.n_stages + fisher_acc[n]) / (self.n_stages + 1)
            else:
                self.fisher[n] = fisher_acc[n]

        self.optpar = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.n_stages += 1
        model.zero_grad()

    def penalty(self, model) -> Tensor:
        """Compute EWC penalty: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2."""
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.optpar[n]).pow(2)).sum()
        return (self.lambda_ewc / 2.0) * loss

    def state_dict(self):
        return {'fisher': self.fisher, 'optpar': self.optpar,
                'n_stages': self.n_stages, 'lambda_ewc': self.lambda_ewc}

    def load_state_dict(self, d):
        self.fisher = d['fisher']
        self.optpar = d['optpar']
        self.n_stages = d['n_stages']
        self.lambda_ewc = d['lambda_ewc']


# ---------------------------------------------------------------------------
# 3. DER++ (Dark Experience Replay++)
# ---------------------------------------------------------------------------

class DERPlusPlus:
    """Dark Experience Replay++ (Buzzega et al. 2020).

    After each stage, snapshots model logits at answer positions on
    training data. During subsequent stages, adds MSE loss between
    current and stored logits. This preserves the model's BEHAVIOR
    (circuit output), not just its accuracy.

    Storage: ~(vocab_size * n_result_tokens * 4 bytes) per sample.
    For vocab=25, n_result=2: ~200 bytes/sample = 100KB for 500 samples.
    """

    def __init__(self, alpha: float = 0.5, n_snapshot: int = 500):
        self.alpha = alpha
        self.n_snapshot = n_snapshot
        self.snapshots: List[Tuple[Tensor, Tensor, int]] = []
        # Each: (seqs_cpu, logits_cpu, n_result_tokens)

    @torch.no_grad()
    def register_stage(self, model, train_seqs, device, amp_ctx,
                       n_result_tokens):
        """Snapshot logits at answer positions after a stage passes."""
        model.eval()
        n = min(self.n_snapshot, len(train_seqs))
        perm = torch.randperm(len(train_seqs))[:n]
        seqs = train_seqs[perm]

        chunk_size = 128
        all_logits = []
        for i in range(0, n, chunk_size):
            batch = seqs[i:i + chunk_size].to(device)
            with amp_ctx():
                logits = model(batch)
            # Logits at answer-prediction positions
            answer_logits = []
            for k in range(n_result_tokens):
                pos = -(n_result_tokens - k) - 1
                answer_logits.append(logits[:, pos].float().cpu())
            all_logits.append(torch.stack(answer_logits, dim=1))

        stored_logits = torch.cat(all_logits, dim=0)  # (n, n_result, vocab)
        self.snapshots.append((seqs.cpu(), stored_logits, n_result_tokens))

    def loss(self, model, device, amp_ctx, batch_size=32) -> Tensor:
        """Compute DER++ MSE loss: match current logits to stored snapshots."""
        if not self.snapshots:
            return torch.tensor(0.0, device=device)

        total_mse = torch.tensor(0.0, device=device)
        n_terms = 0

        per_stage = max(1, batch_size // len(self.snapshots))
        for stored_seqs, stored_logits, nr in self.snapshots:
            n = min(per_stage, len(stored_seqs))
            perm = torch.randperm(len(stored_seqs))[:n]
            seqs = stored_seqs[perm].to(device)
            target = stored_logits[perm].to(device)

            with amp_ctx():
                logits = model(seqs)
            current = []
            for k in range(nr):
                pos = -(nr - k) - 1
                current.append(logits[:, pos].float())
            current = torch.stack(current, dim=1)

            total_mse = total_mse + F.mse_loss(current, target)
            n_terms += 1

        return self.alpha * total_mse / max(n_terms, 1)

    def state_dict(self):
        return {'snapshots': self.snapshots, 'alpha': self.alpha,
                'n_snapshot': self.n_snapshot}

    def load_state_dict(self, d):
        self.snapshots = d['snapshots']
        self.alpha = d['alpha']
        self.n_snapshot = d['n_snapshot']
