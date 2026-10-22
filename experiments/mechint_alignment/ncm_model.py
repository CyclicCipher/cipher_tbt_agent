"""Next-Concept Modeling (NCM; Kumbong et al. 2026, arXiv 2610.14162), built from the authors' code (HazyResearch/ncm,
MIT licence, cloned into third_party/ncm, commit c702742), for predicting a world's state H steps ahead (E11).

The repo provides the architecture (models/vqvae.py, models/quant.py, models/next_concept.py) but no training loop;
the two training stages here follow the paper (§3.3–3.4):
1. TOKENIZER: a multiscale residual VQ-VAE over the target state. Loss = reconstruction cross-entropy + the VQ
   terms; codebooks learn by EMA; "fine drop" (nested dropout) truncates the hierarchy at a random level half of the
   time, so coarse levels carry the most important information.
2. GENERATOR: a transformer that, given a prefix (start state + actions, as raw token embeddings), predicts every code
   of level k in parallel from the cumulative reconstruction of levels < k. Level 0 from learned queries; block-
   diagonal attention; loss = per-level-weighted cross-entropy (alpha = 0.5); conditioning corruption (a fraction of
   conditioning codes replaced by random codes of the same level) for robustness to its own errors.
Generation: K forward passes (one per level), then the tokenizer's decoder maps the reconstruction to tokens.
The configuration mirrors the authors' Sudoku setup (a structured grid; config/eval/sudoku-hard.yaml), scaled down.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "ncm"))
from models.next_concept import NextConceptTransformer   # noqa: E402  (the authors' code)
from models.vqvae import VQVAE                           # noqa: E402

LEVELS = [1, 2, 4, 7, 14, 28, 49, 98, 147, 196]         # coarse to fine; the last = the 196 cells
CODEBOOK = 16


def tokenizer_config(in_vocab, length, levels=LEVELS, embed_dim=128, restart_threshold=0.0):
    return OmegaConf.create(dict(
        n_layers=4, context_length=length, embed_dim=embed_dim, in_vocab_size=in_vocab,
        vqvae_vocab_size=[CODEBOOK] * len(levels), compression_factor=1, dropout=0.1, dec_layers=1,
        pre_quant_dropout=0.0, pre_quant_groupnorm=8, pre_quant_norm_type="group", use_qk_norm=True,
        vector_quantizer_config=dict(
            clss="multiscale_residual_vector_quantizer", decay=0.99, epsilon=1e-5, commitment_cost=0.25,
            restart_threshold=restart_threshold,           # > 0: dead codes are restarted on current inputs (E11b)
            learned_l1_sampling=True, learned_all_sampling=False,
            quant_resi=dict(enabled=True, ratio=0.5, share_mode=0, learnable_ratio=False),
            levels=dict(use_manual_levels=True, manual_levels=list(levels)),
            learned_l1_cascade=True, learned_l1_cascade_base=4, learned_l1_cascade_norm=False,
            learned_l1_cascade_norm_final=False,
            aux=dict(fine_drop=dict(prob=0.5, min_keep=1)))))


def generator_config(prefix_len, n_levels, n_embd=256, n_layer=6, n_head=8):
    return OmegaConf.create(dict(
        n_layer=n_layer, n_head=n_head, bias=True, dropout=0.1, n_embd=n_embd, vocab_size=[CODEBOOK] * n_levels,
        attn_pattern="block_diagonal", use_level_encoding=True, prefix_len=prefix_len, use_rope=True,
        rope_base=10000, use_qk_norm=True, raw_token_prefix=True, query_level0=True, tie_heads_to_codebook=False))


def make_tokenizer(in_vocab, length, **cfg):
    return VQVAE(**tokenizer_config(in_vocab, length, **cfg))


def make_generator(vqvae, prefix_len, **size):
    return NextConceptTransformer(vqvae_local=vqvae, config=generator_config(prefix_len, len(vqvae.vq.levels), **size))


def tokenizer_loss(vqvae, tokens, weights=None):
    """Stage 1: reconstruction cross-entropy (optionally weighted per cell, E11b) + VQ terms. tokens [B, L]."""
    logits, vq_loss, *_ = vqvae(tokens)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tokens.reshape(-1), reduction="none")
    rec = ce.mean() if weights is None else (ce * weights.reshape(-1)).sum() / weights.sum()
    return rec + vq_loss, rec


def level_weights(levels, alpha=0.5, device=None):
    """Paper eq. 10: w_k = l_k^-alpha / sum_j l_j^(1-alpha), per code position."""
    z = sum(l ** (1 - alpha) for l in levels)
    return torch.cat([torch.full((l,), l ** -alpha / z) for l in levels]).to(device)


def generator_loss(net, vqvae, prefix_tokens, target_tokens, corrupt=0.1, w=None):
    """Stage 2 (the tokenizer frozen): predict every level's codes in one pass from the corrupted reconstructions."""
    vq = vqvae.vq
    with torch.no_grad():
        idx = vqvae.encode(target_tokens)                                   # [B, sum(levels)] ground-truth codes
        noisy = idx.clone()
        flip = torch.rand(idx.shape, device=idx.device) < corrupt
        noisy[flip] = torch.randint(0, CODEBOOK, (int(flip.sum()),), device=idx.device)
        cond = vq.idx_to_ncm_input(noisy)                                   # inputs for levels 1..K-1
        prefix = vqvae.encoder.wte(prefix_tokens)
    x = torch.cat([prefix, net.build_l0_query_input(idx.shape[0], prefix.dtype), cond], 1)
    logits = net(x)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), idx.reshape(-1), reduction="none")
    ce = ce.view(idx.shape)
    w = level_weights(vq.levels, device=idx.device) if w is None else w
    return (ce * w).sum(1).mean(), (logits.argmax(-1) == idx).float().mean()


@torch.no_grad()
def generate(net, vqvae, prefix_tokens):
    """Greedy coarse-to-fine generation: K generator passes, then decode -> tokens [B, L]."""
    codes = net.generate(None, prefix_tokens, temperature=0.0, top_k=0, top_p=1.0)
    return vqvae.decode(codes).argmax(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
