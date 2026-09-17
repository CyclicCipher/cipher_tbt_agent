"""The objective wraps the model; the model is untouched.

Phase-0 recon of the repo's baseline (`experiments/transformers/h1_lid.py::Model`, the class every recent
transformer experiment trains):
  * forward: `tok (B,T) long -> logits (B,T,V)`; no aux loss, no multi-exit, no halting, no loop-count knob
  * causal: `F.scaled_dot_product_attention(..., is_causal=True)` in every block
  * positions: swappable; the recent baseline uses `pos="rope"` (no learned position table, so any max_len works)
  * embeddings: UNTIED (`emb` and `head` are separate modules)
  * KV cache: none — generation recomputes the prefix each step, sliced to the live length
  * optimiser in the repo: AdamW with warmup + cosine; `torch.compile` is used successfully on CUDA there
  * no prequential/epiplexity utility at the token level exists; `probe_bits` below is it

Sizes (vocab 73): ~4M at d=256 / 5 layers / 4 heads; ~1M at d=128 / 5 layers; ~10M at d=384 / 6 layers.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_T = Path(__file__).resolve().parent.parent / "transformers"
if str(_T) not in sys.path:
    sys.path.insert(0, str(_T))
from h1_lid import Model                      # noqa: E402  (the repo baseline, imported not copied)

from env.tokens import M0, PAD, V             # noqa: E402

SIZES = {"1M": dict(d_model=128, n_layer=5, n_head=4),
         "4M": dict(d_model=256, n_layer=5, n_head=4),
         "10M": dict(d_model=384, n_layer=6, n_head=4)}


class ModelAdapter(nn.Module):
    def __init__(self, size="4M", max_len=192, amp=True):
        super().__init__()
        cfg = SIZES[size]
        self.model = Model(cfg["d_model"], cfg["n_layer"], cfg["n_head"], max_len=max_len, pos="rope", n_vocab=V)
        self.d = cfg["d_model"]
        self.amp = amp
        self.compiled = None
        # Measured at 4M: a 1024-chunk peaks at 2.0 GB during prefill (transient activations, not the cache), and after
        # E1/E2 the allocator already holds ~2.5 GB — on a 4 GB card that spills to shared memory and decode falls from
        # 20 ms to 260 ms per step. 512 has the same throughput at half the peak; 10M gets 256.
        self.gen_chunk = 256 if size == "10M" else 512

    # ── core ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _ac(self):
        dev = next(self.parameters()).device.type
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.amp and dev == "cuda"))

    def forward(self, ids):
        """logits [B,T,V], aux_loss (None: the baseline has no auxiliary loss)."""
        with self._ac():
            net = self.compiled if (self.compiled is not None and self.training) else self.model
            return net(ids).float(), None

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def set_loop_depth(self, n):
        return False                          # the architecture has no such knob

    # ── generation, with a KV cache built in the ADAPTER from the model's own weights ──────────────────────────────
    # The baseline has no cache, and recomputing the prefix costs Σ_t (P+t) ≈ 2.7k token-positions per 29-token
    # program instead of 108 — measured at 0.44 s per (1024, ~95) forward, i.e. 13 s per 1024 sequences, 52 s per wake.
    # `_prefill` / `_decode` re-run exactly the arithmetic of `h1_lid.Attn`/`Block` (rope on interleaved pairs, SDPA at
    # scale 1/sqrt(hd)) against the same parameters, caching K and V per layer. The core model is untouched;
    # `cache_matches_model` asserts the cached logits equal the model's own to 1e-4 in fp32.
    def _rope(self, z, start, theta):
        T = z.shape[2]
        t = torch.arange(start, start + T, device=z.device, dtype=torch.float32)[:, None]
        ang = t * theta
        cos, sin = ang.cos()[None, None], ang.sin()[None, None]
        a, b = z[..., 0::2], z[..., 1::2]
        return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)

    def _attn_qkv(self, blk, x, start):
        B, T, C = x.shape
        H, hd = blk.attn.h, blk.attn.hd
        q, k, v = blk.attn.qkv(x).chunk(3, dim=-1)
        q, k, v = (z.view(B, T, H, hd).transpose(1, 2) for z in (q, k, v))
        theta = blk.attn.theta
        return self._rope(q, start, theta), self._rope(k, start, theta), v

    def _alloc_caches(self, B, T_max, device, dtype):
        """One set of K/V buffers per `generate` call, reused by every chunk (sliced for a short last chunk): no
        per-chunk allocation churn, which is what fragments a 4 GB card."""
        m = self.model
        return [[torch.empty(B, blk.attn.h, T_max, blk.attn.hd, dtype=dtype, device=device) for _ in range(2)]
                for blk in m.blocks]

    def _prefill(self, ids, caches):
        m = self.model
        h = m.emb(ids)
        B, P, C = h.shape
        for blk, (K, Vc) in zip(m.blocks, caches):
            q, k, v = self._attn_qkv(blk, blk.n1(h), 0)
            K[:B, :, :P], Vc[:B, :, :P] = k, v
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(blk.attn.hd))
            del q, k, v
            h = h + blk.attn.proj(y.transpose(1, 2).reshape(B, P, C))
            del y
            h = h + blk.mlp(blk.n2(h))
        return m.head(m.norm(h[:, -1:]))[:, 0]

    def _decode(self, tok, caches, pos):
        m = self.model
        h = m.emb(tok[:, None])
        B, _, C = h.shape
        for blk, (K, Vc) in zip(m.blocks, caches):
            q, k, v = self._attn_qkv(blk, blk.n1(h), pos)
            K[:B, :, pos:pos + 1], Vc[:B, :, pos:pos + 1] = k, v
            y = F.scaled_dot_product_attention(q, K[:B, :, :pos + 1], Vc[:B, :, :pos + 1],
                                               scale=1.0 / math.sqrt(blk.attn.hd))
            h = h + blk.attn.proj(y.transpose(1, 2).reshape(B, 1, C))
            h = h + blk.mlp(blk.n2(h))
        return m.head(m.norm(h))[:, 0]

    @torch.no_grad()
    def generate(self, prompts, max_new, temperature, grammar, mode="prog", chunk=None, rng=None):
        """Equal-length prompts (B,P) -> (B, P+max_new). Grammar-masked, then temperature; T=0 is greedy. Finished
        sequences emit PAD. KV-cached: one prefill over the prompt, then one single-token step per generated token."""
        was_training = self.training
        self.eval()
        outs = []
        P = prompts.shape[1]
        chunk = chunk or self.gen_chunk
        dev = prompts.device
        dtype = torch.bfloat16 if (self.amp and dev.type == "cuda") else torch.float32
        caches = self._alloc_caches(min(chunk, prompts.shape[0]), P + max_new, dev, dtype)
        for s in range(0, prompts.shape[0], chunk):
            ids = prompts[s:s + chunk]
            b = ids.shape[0]
            states = [grammar.start(mode)] * b
            cols = []
            with self._ac():
                logits = self._prefill(ids, caches)
            for step in range(max_new):
                if all(st[1] == "pad" for st in states):
                    cols.append(torch.full((b,), PAD, dtype=torch.long, device=ids.device))
                    continue
                logits = logits.float().masked_fill(~grammar.masks(states, ids.device), float("-inf"))
                if temperature > 0:
                    nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=rng).squeeze(1)
                else:
                    nxt = logits.argmax(-1)
                cols.append(nxt)
                states = [grammar.advance(st, t) for st, t in zip(states, nxt.tolist())]
                if step + 1 < max_new:
                    with self._ac():
                        logits = self._decode(nxt, caches, P + step)
            outs.append(torch.cat([prompts[s:s + chunk], torch.stack(cols, dim=1)], dim=1))
        if was_training:
            self.train()
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def cache_matches_model(self, T=40, new=12):
        """Max abs diff between cached decoding's logits and the model's full forward, fp32. Must be < 1e-4."""
        amp, self.amp = self.amp, False
        dev = next(self.parameters()).device
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, V, (3, T + new), generator=g).to(dev)
        ref = self.model(ids)
        caches = self._alloc_caches(3, T + new, dev, torch.float32)
        logits = self._prefill(ids[:, :T], caches)
        got = [logits]
        for j in range(new - 1):
            got.append(self._decode(ids[:, T + j], caches, T + j))
        got = torch.stack(got, dim=1)                                 # predictions at positions T-1 .. T+new-2
        self.amp = amp
        return float((got - ref[:, T - 1:T + new - 1]).abs().max())

    def seq_logprob(self, ids, gen_mask, grammar, mode, prompt_len):
        """Σ log π over generated tokens, with grad, from the SAME grammar-masked distribution used to sample."""
        logits, _ = self.forward(ids)
        logits = logits[:, :-1]
        tgt = ids[:, 1:]
        B, T = tgt.shape
        masks = torch.stack([grammar.replay(ids[b, prompt_len:].tolist(), mode) for b in range(B)]).to(ids.device)
        full = torch.ones(B, T, V, dtype=torch.bool, device=ids.device)
        full[:, prompt_len - 1:prompt_len - 1 + masks.shape[1]] = masks[:, :T - prompt_len + 1]
        logp = torch.log_softmax(logits.masked_fill(~full, float("-inf")), dim=-1)
        lp = logp.gather(2, tgt[:, :, None]).squeeze(2)
        return (lp * gen_mask[:, 1:].float()).sum(1)

    @torch.no_grad()
    def probe_bits(self, ids, positions):
        """Per-sequence mean bits/token over `positions` (target positions), and the per-position bits."""
        logits, _ = self.forward(ids)
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        pos = torch.tensor(positions, device=ids.device) - 1                 # target at p is predicted at p-1
        bits = -logp[:, pos].gather(2, tgt[:, pos, None]).squeeze(2) / math.log(2)
        return bits.mean(1), bits

    # ── library slots ────────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def init_slot(self, slot_id, from_ids):
        """Slot row <- mean of its expansion's op embeddings; same for the untied output row (weight and bias)."""
        row = M0 + slot_id
        src = torch.tensor(from_ids, device=self.model.emb.weight.device)
        self.model.emb.weight[row] = self.model.emb.weight[src].mean(0)
        self.model.head.weight[row] = self.model.head.weight[src].mean(0)
        self.model.head.bias[row] = self.model.head.bias[src].mean()

    @torch.no_grad()
    def reset_slot(self, slot_id):
        """On pruning: re-draw the rows from the modules' own init distributions."""
        row = M0 + slot_id
        self.model.emb.weight[row].normal_()                                  # nn.Embedding default N(0, 1)
        bound = 1.0 / math.sqrt(self.d)                                       # nn.Linear default U(-1/sqrt(in), ..)
        self.model.head.weight[row].uniform_(-bound, bound)
        self.model.head.bias[row].uniform_(-bound, bound)

    def maybe_compile(self, enabled):
        if enabled and next(self.parameters()).device.type == "cuda":
            self.compiled = torch.compile(self.model)


def causality_test(adapter, T=64, trials=3):
    """Perturb tokens at positions > t; logits at positions <= t must be unchanged (fp32, max abs diff < 1e-5)."""
    adapter.eval()
    amp, adapter.amp = adapter.amp, False
    dev = next(adapter.parameters()).device
    worst = 0.0
    with torch.no_grad():
        for k in range(trials):
            g = torch.Generator().manual_seed(k)
            ids = torch.randint(0, V, (2, T), generator=g).to(dev)
            t = 10 + k * 15
            ids2 = ids.clone()
            ids2[:, t + 1:] = torch.randint(0, V, (2, T - t - 1), generator=g).to(dev)
            a = adapter.model(ids)[:, :t + 1]
            b = adapter.model(ids2)[:, :t + 1]
            worst = max(worst, float((a - b).abs().max()))
    adapter.amp = amp
    return worst
