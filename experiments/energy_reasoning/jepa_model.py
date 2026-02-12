"""
JEPA model with Langevin dynamics for energy-based reasoning.

Components:
  - Mamba3Encoder: Mamba3 backbone producing latent representations.
  - Mamba3Predictor: Narrow Mamba3 conditioned on z, predicts target reps.
  - JEPAModel: Full system (context encoder, EMA target encoder, predictor,
    decoder, VICReg, Langevin inference).
  - LangevinDynamics: Searches for z* minimizing prediction energy.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3Block, RMSNorm


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Mamba3Encoder(nn.Module):
    """Mamba3 backbone encoder for JEPA.

    Embeds tokens, optionally replaces masked positions with a learnable
    mask embedding, then runs through N Mamba3 blocks + RMSNorm.

    Output: (batch, seq_len, d_model) latent representations.
    """

    def __init__(self, config: Mamba3Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.randn(config.d_model) * 0.02)
        self.layers = nn.ModuleList(
            [Mamba3Block(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices.
            mask: (batch, seq_len) boolean. True = replace with mask embedding.

        Returns:
            (batch, seq_len, d_model) representations.
        """
        x = self.embedding(input_ids)
        if mask is not None:
            x = x.clone()
            x[mask] = self.mask_emb
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Mamba3Predictor(nn.Module):
    """Narrow Mamba3 predictor with z conditioning.

    Projects encoder output to a narrow bottleneck (d_pred), adds z
    conditioning, runs through predictor Mamba3 blocks, then projects
    back to encoder dimension for comparison with target.

    z conditioning is additive: projected z is broadcast over all positions
    and added to the predictor input. This is the simplest conditioning
    mechanism; if ablations show z is ignored, FiLM is the next step.
    """

    def __init__(self, d_enc: int, pred_config: Mamba3Config, d_z: int):
        """
        Args:
            d_enc: Encoder output dimension (e.g., 128).
            pred_config: Mamba3Config for predictor (d_model = d_pred, e.g., 64).
            d_z: Latent reasoning variable dimension.
        """
        super().__init__()
        d_pred = pred_config.d_model
        self.input_proj = nn.Linear(d_enc, d_pred)
        self.z_proj = nn.Linear(d_z, d_pred)
        self.layers = nn.ModuleList(
            [Mamba3Block(pred_config) for _ in range(pred_config.n_layer)]
        )
        self.norm = RMSNorm(d_pred)
        self.output_proj = nn.Linear(d_pred, d_enc)

    def forward(self, s_context: Tensor, z: Tensor = None) -> Tensor:
        """
        Args:
            s_context: (batch, seq_len, d_enc) encoder representations.
            z: (batch, d_z) latent variable, or None (Stage 1, no z).

        Returns:
            s_pred: (batch, seq_len, d_enc) predicted representations.
        """
        x = self.input_proj(s_context)  # (batch, seq_len, d_pred)
        if z is not None:
            z_cond = self.z_proj(z).unsqueeze(1)  # (batch, 1, d_pred)
            x = x + z_cond  # broadcast over seq_len
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output_proj(x)  # (batch, seq_len, d_enc)


# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------

def vicreg_loss(s: Tensor) -> tuple[Tensor, Tensor]:
    """Compute VICReg variance and covariance regularization.

    Applied to encoder output to prevent representation collapse.

    Args:
        s: (batch, seq_len, d) representations.

    Returns:
        L_var: Variance loss (hinge: penalizes std < 1 per dimension).
        L_cov: Covariance loss (penalizes off-diagonal correlations).
    """
    s_flat = s.reshape(-1, s.shape[-1])  # (N, d)
    # Variance: encourage std >= 1 along each dimension
    std = torch.sqrt(s_flat.var(dim=0) + 1e-4)
    L_var = F.relu(1.0 - std).mean()
    # Covariance: decorrelate dimensions
    s_c = s_flat - s_flat.mean(dim=0)
    n = max(s_c.shape[0] - 1, 1)
    cov = (s_c.T @ s_c) / n
    d = cov.shape[0]
    off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
    L_cov = off_diag / d
    return L_var, L_cov


# ---------------------------------------------------------------------------
# Langevin dynamics
# ---------------------------------------------------------------------------

class LangevinDynamics:
    """Langevin MCMC sampler for finding z* that minimizes prediction energy.

    Uses cyclical noise annealing (cSGLD-inspired):
        sigma_t = sigma_max * 0.5 * (1 + cos(pi * t / T))

    High noise early (explore), low noise late (exploit).
    Adaptive stopping: halt if relative energy change < threshold.
    """

    def __init__(self, d_z: int, eta: float = 0.01, sigma_max: float = 0.1,
                 T: int = 5, adaptive_threshold: float = 1e-3):
        self.d_z = d_z
        self.eta = eta
        self.sigma_max = sigma_max
        self.T = T
        self.adaptive_threshold = adaptive_threshold

    def sample(self, predictor: Mamba3Predictor, s_context: Tensor,
               s_target: Tensor, mask: Tensor,
               device: torch.device) -> tuple[Tensor, list]:
        """Run Langevin dynamics to find z* minimizing E_pred.

        Encoder outputs (s_context, s_target) are pre-computed and frozen.
        Only the predictor runs per step (it's small: 2-layer, d=64).

        Args:
            predictor: The predictor network.
            s_context: (batch, seq_len, d_enc) context encoder output (detached).
            s_target: (batch, seq_len, d_enc) target encoder output (detached).
            mask: (batch, seq_len) boolean, True = masked positions.
            device: Torch device.

        Returns:
            z_star: (batch, d_z) optimized latent variable.
            energies: List of energy values per step.
        """
        batch_size = s_context.shape[0]
        d_enc = s_context.shape[-1]

        z = torch.randn(batch_size, self.d_z, device=device)
        z.requires_grad_(True)

        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        n_masked = mask.sum().clamp(min=1).float()

        energies = []
        for t in range(self.T):
            s_pred = predictor(s_context, z)
            diff = (s_pred - s_target) * mask_expanded
            E = diff.pow(2).sum() / (n_masked * d_enc)
            energies.append(E.item())

            grad_z = torch.autograd.grad(E, z)[0]

            # Cyclical annealing noise
            sigma_t = self.sigma_max * 0.5 * (1 + math.cos(math.pi * t / self.T))
            noise = torch.randn_like(z) * sigma_t

            z = (z - self.eta * grad_z + noise).detach().requires_grad_(True)

            # Adaptive stopping
            if len(energies) >= 2:
                rel_change = abs(energies[-1] - energies[-2]) / (abs(energies[-2]) + 1e-8)
                if rel_change < self.adaptive_threshold:
                    break

        return z.detach(), energies


# ---------------------------------------------------------------------------
# Full JEPA model
# ---------------------------------------------------------------------------

class JEPAModel(nn.Module):
    """Joint Embedding Predictive Architecture with Langevin reasoning.

    Architecture:
        Context Encoder (Mamba3, 4-layer, d=128)
            - Processes input with masked positions replaced by mask embedding
        Target Encoder (EMA of context encoder)
            - Processes full input (no masking), parameters updated via EMA
        Predictor (Mamba3, 2-layer, d=64, with z conditioning)
            - Takes context representations + z, predicts target representations
        Decoder (Linear, d_enc -> vocab_size)
            - For evaluation: maps predicted representations to token logits

    Training:
        L = L_jepa + lambda_decode * L_decode + lambda_var * L_var + lambda_cov * L_cov

        L_jepa:   ||s_pred[mask] - s_target[mask]||^2  (representation space)
        L_decode: CE(decoder(s_pred[mask]), tokens[mask])  (token space, trains decoder)
        L_var:    VICReg variance hinge loss
        L_cov:    VICReg covariance loss
    """

    def __init__(
        self,
        enc_config: Mamba3Config,
        pred_config: Mamba3Config,
        vocab_size: int,
        d_z: int = 64,
        ema_tau: float = 0.996,
    ):
        super().__init__()
        self.enc_config = enc_config
        self.pred_config = pred_config
        self.vocab_size = vocab_size
        self.d_z = d_z
        self.ema_tau = ema_tau

        # Context encoder (trained via gradient descent)
        self.context_encoder = Mamba3Encoder(enc_config, vocab_size)
        # Target encoder (updated via EMA, no gradients)
        self.target_encoder = Mamba3Encoder(enc_config, vocab_size)
        # Predictor (narrow bottleneck with z conditioning)
        self.predictor = Mamba3Predictor(enc_config.d_model, pred_config, d_z)
        # Decoder (linear probe for token-level evaluation)
        self.decoder = nn.Linear(enc_config.d_model, vocab_size)

        # Initialize target as copy of context, freeze it
        self._init_target_encoder()

    def _init_target_encoder(self):
        """Copy context encoder params to target encoder and freeze."""
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.data.copy_(p_c.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: target = tau * target + (1 - tau) * context."""
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.data.mul_(self.ema_tau).add_(p_c.data, alpha=1 - self.ema_tau)

    def encode(self, input_ids: Tensor, mask: Tensor = None) -> Tensor:
        """Encode with context encoder (mask applied)."""
        return self.context_encoder(input_ids, mask=mask)

    @torch.no_grad()
    def encode_target(self, input_ids: Tensor) -> Tensor:
        """Encode with target encoder (no mask, no grad)."""
        return self.target_encoder(input_ids, mask=None)

    def predict(self, s_context: Tensor, z: Tensor = None) -> Tensor:
        """Run predictor on context representations."""
        return self.predictor(s_context, z)

    def decode(self, s: Tensor) -> Tensor:
        """Decode representations to token logits."""
        return self.decoder(s)

    def compute_jepa_loss(self, s_pred: Tensor, s_target: Tensor,
                          mask: Tensor) -> Tensor:
        """L2 prediction loss on masked positions.

        Args:
            s_pred: (batch, seq_len, d_enc) predicted representations.
            s_target: (batch, seq_len, d_enc) target representations (detached).
            mask: (batch, seq_len) boolean.

        Returns:
            Scalar loss: mean squared error over masked positions.
        """
        diff = s_pred - s_target.detach()
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        diff_masked = diff * mask_expanded
        n_masked = mask.sum().clamp(min=1).float()
        return diff_masked.pow(2).sum() / (n_masked * s_pred.shape[-1])

    def compute_decode_loss(self, s_pred: Tensor, tokens: Tensor,
                            mask: Tensor) -> Tensor:
        """Cross-entropy decoding loss on masked positions.

        Args:
            s_pred: (batch, seq_len, d_enc) predicted representations.
            tokens: (batch, seq_len) ground truth token indices.
            mask: (batch, seq_len) boolean.

        Returns:
            Scalar CE loss on masked positions.
        """
        logits = self.decode(s_pred)  # (batch, seq_len, vocab_size)
        # Select masked positions
        logits_masked = logits[mask]    # (n_masked, vocab_size)
        tokens_masked = tokens[mask]    # (n_masked,)
        if logits_masked.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device)
        return F.cross_entropy(logits_masked, tokens_masked)

    def forward_train(self, input_ids: Tensor, mask: Tensor,
                      z: Tensor = None) -> dict:
        """Full training forward pass.

        Args:
            input_ids: (batch, seq_len) tokens.
            mask: (batch, seq_len) boolean mask.
            z: (batch, d_z) latent variable, or None.

        Returns:
            dict with losses and representations for diagnostics.
        """
        s_context = self.encode(input_ids, mask=mask)
        s_target = self.encode_target(input_ids)
        s_pred = self.predict(s_context, z)

        L_jepa = self.compute_jepa_loss(s_pred, s_target, mask)
        L_decode = self.compute_decode_loss(s_pred, input_ids, mask)
        L_var, L_cov = vicreg_loss(s_context)

        return dict(
            L_jepa=L_jepa,
            L_decode=L_decode,
            L_var=L_var,
            L_cov=L_cov,
            s_context=s_context,
            s_target=s_target,
            s_pred=s_pred,
        )

    @torch.no_grad()
    def forward_eval(self, input_ids: Tensor, mask: Tensor,
                     z: Tensor = None) -> Tensor:
        """Evaluation forward pass. Returns logits."""
        s_context = self.encode(input_ids, mask=mask)
        s_pred = self.predict(s_context, z)
        return self.decode(s_pred)

    def get_trainable_params(self):
        """Parameters updated by gradient descent (everything except target encoder)."""
        params = []
        params.extend(self.context_encoder.parameters())
        params.extend(self.predictor.parameters())
        params.extend(self.decoder.parameters())
        return params
