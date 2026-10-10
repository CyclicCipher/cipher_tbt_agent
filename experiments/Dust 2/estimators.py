"""Dust's gradient estimator (node perturbation, one population member per token) and the variants under test.

`estimate(model, idx, tgt, cfg, state)` returns {parameter: gradient of the mean per-token loss} without any backward
pass, plus the cost in forward-pass equivalents. `backprop(model, idx, tgt)` returns the true gradients and the true
per-token errors at every site (the measuring stick for the cosine diagnostics).

The baseline (cfg defaults) follows Dust (DUST.md §1–2): every rerun site (embedding, attention hub `o`, attention
projection, MLP hidden, MLP output) is perturbed in its own passes and scored by the change of each token's loss;
q, k, v are scored LOCALLY by alignment with the hub's estimated error, k and v crediting later tokens with γ_loc; the head
is perturbed on its logits. Variants (cfg.noise, cfg.ent_frac, cfg.local) change only how directions are drawn, which
tokens are perturbed, or which loss scores a draw.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch

SIG = dict(emb=0.2, o=0.2, proj=0.2, fc=0.2, out=0.2, qkv=0.2, loc=0.05, head=0.05, aux=0.05)


@dataclass
class Cfg:
    K: int = 32                 # draws per rerun site
    chunk: int = 16             # draws stacked in one pass
    sig: dict = field(default_factory=lambda: dict(SIG))
    gamma: float = 0.0          # credit decay over later tokens for rerun sites
    gamma_loc: float = 0.98     # credit decay for k, v (scored through the hub)
    qkv_mode: str = "hub"       # "hub" (Dust) or "direct" (rerun from qkv like any site)
    head_mult: int = 4          # head draws = head_mult * K (Dust: 4 x population)
    noise: str = "gauss"        # gauss | anti | orth | sobol | lr_rand | lr_pca | guided | oracle (diagnostic)
    rank: int = 8               # subspace dimension for lr_* and guided
    beta: float = 0.5           # guided: share of the perturbation's variance inside the subspace
    ema: float = 0.1            # guided: EMA rate of the error covariance
    ent_frac: float = 1.0       # entropy gating: fraction of tokens perturbed (the highest-entropy ones)
    local: bool = False         # score each site by the next auxiliary head's loss (model built with aux_every > 0)
    exact_head: bool = False    # (own idea O2) head error in closed form: softmax - one-hot, no draws
    amp: str = ""               # "" (fp32), "bf16" or "fp16": precision of the clean pass and every rerun
    # ---- own ideas (Experiment 2) ----
    sparse_c: float = 0.0       # O1: at the MLP hidden, perturb only units within sparse_c*σ of being active (0 = off)
    top_guide: float = 0.0      # O3: share of each writer draw along the exact top-of-stream error (needs exact_head)
    hub_T: bool = False         # O4: hub and hidden errors from their own block's writer estimate through the local
                                #     linear map (W_proj^T, W_out^T); the hidden's activation slope by per-unit draws
    act_K: int = 4              # O4: per-unit draws for the activation slope
    # ---- thermodynamics (Experiment 3) ----
    shaping: str = ""           # T2: "" (linear, Dust), "boltz" (weights softmax(z/lam)), "rank" (centred ranks)
    lam: float = 1.0            # T2: Boltzmann temperature, in units of each token's reward spread
    simul: bool = False         # T3: every site perturbed in every pass, no clean pass, no cache (simul.py)
    whiten: bool = False        # E2b: estimate g (Σ^-1 times the average) rather than Σg, for guided/oracle/top_guide


def discount(T, gamma, dev):
    """Gam[t, s] = gamma^(s - t) for s >= t, else 0, so r = c @ Gam.T gives r_t = sum_{s>=t} gamma^(s-t) c_s."""
    i = torch.arange(T, device=dev)
    lag = (i[None, :] - i[:, None]).float()
    return torch.where(lag >= 0, gamma ** lag.clamp(min=0), torch.zeros((), device=dev))


# ---- drawing directions ---------------------------------------------------------------------------------------------
def _orthonormal(D, k, g, dev):
    q, _ = torch.linalg.qr(torch.randn(D, k, generator=g, device=dev))
    return q                                                  # (D, k)


def draws(key, K, B, T, D, cfg, state, g, dev, v0=None, dt=torch.float32, guide=None):
    """Yields chunks (Kc, B, T, D) of directions, dtype dt, with E[a a^T] = I (or Σ for `guided`). With a per-token unit
    vector `guide` (B, T, D) (O3), a share cfg.top_guide of each draw's variance goes along it."""
    if guide is not None and cfg.top_guide > 0:
        bt = cfg.top_guide
        for a in draws(key, K, B, T, D, cfg, state, g, dev, v0, dt):
            xi = torch.randn(a.shape[0], B, T, 1, generator=g, device=dev, dtype=dt)
            yield math.sqrt(1 - bt) * a + math.sqrt(bt * D) * xi * guide.to(dt)[None]
        return
    n, Kc = cfg.noise, cfg.chunk
    if n in ("orth", "sobol"):
        if n == "orth":       # per token, the K directions are orthogonal (blocks of D if K > D), each of norm sqrt(D)
            blocks = []
            for j in range(0, K, D):
                kk = min(D, K - j)
                z = torch.randn(B * T, D, kk, generator=g, device=dev)
                Lc = torch.linalg.cholesky(z.transpose(1, 2) @ z)            # Cholesky-QR: Q = Z L^-T
                q = torch.linalg.solve_triangular(Lc, z.transpose(1, 2), upper=False).transpose(1, 2)
                blocks.append(q.permute(2, 0, 1) * math.sqrt(D))
            allk = torch.cat(blocks, 0).view(K, B, T, D)
        else:                 # scrambled Sobol points -> Gaussian; per token a random sign per coordinate and own order
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g, device=dev))
            eng = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed)
            u = eng.draw(K).to(dev).clamp(1e-6, 1 - 1e-6)
            z = torch.special.ndtri(u)                                  # (K, D)
            perm = torch.rand(B, T, K, generator=g, device=dev).argsort(-1)  # (B, T, K)
            sgn = torch.randint(0, 2, (B, T, D), generator=g, device=dev).float() * 2 - 1
            allk = z[perm].permute(2, 0, 1, 3) * sgn[None]
        allk = allk.to(dt)
        for j in range(0, K, Kc):
            yield allk[j:j + Kc]
        return
    if n in ("lr_rand", "lr_pca", "guided", "oracle"):
        if n == "lr_rand":
            U = state.setdefault(("U", key), _orthonormal(D, cfg.rank, g, dev))
        elif n == "lr_pca":   # top principal directions of the site's clean outputs in this batch
            x = v0.reshape(-1, D).float()
            x = x - x.mean(0)
            _, vec = torch.linalg.eigh(x.T @ x)
            U = vec[:, -cfg.rank:]
        elif n == "oracle":   # diagnostic only: the top error directions from BACKPROP (measure.py puts them in state)
            U = state[("O", key)][:, -cfg.rank:]
        else:
            C = state.get(("C", key))
            U = None if C is None else torch.linalg.eigh(C)[1][:, -cfg.rank:]
        U = None if U is None else U.to(dt)
    for j in range(0, K, Kc):
        kc = min(Kc, K - j)
        if n == "anti":
            h = torch.randn((kc + 1) // 2, B, T, D, generator=g, device=dev, dtype=dt)
            yield torch.cat([h, -h], 0)[:kc]
        elif n in ("lr_rand", "lr_pca"):
            e = torch.randn(kc, B, T, cfg.rank, generator=g, device=dev, dtype=dt)
            yield (e @ U.T) * math.sqrt(D / cfg.rank)
        elif n in ("guided", "oracle") and U is not None:
            e = torch.randn(kc, B, T, D, generator=g, device=dev, dtype=dt)
            s = torch.randn(kc, B, T, cfg.rank, generator=g, device=dev, dtype=dt)
            yield math.sqrt(1 - cfg.beta) * e + math.sqrt(cfg.beta * D / cfg.rank) * (s @ U.T)
        else:
            yield torch.randn(kc, B, T, D, generator=g, device=dev, dtype=dt)


def subspace(key, cfg, state):
    """The shared subspace U (D, rank) the draws of `key` use this step (guided / oracle), or None."""
    if cfg.noise == "oracle":
        return state[("O", key)][:, -cfg.rank:].float()
    if cfg.noise == "guided":
        C = state.get(("C", key))
        return None if C is None else torch.linalg.eigh(C)[1][:, -cfg.rank:].float()
    return None


def whiten(gh, key, cfg, state, guide=None):
    """Σ^-1 gh per token, with Σ = α I + V diag(c) Vᵀ (V = [U, the token's guide]), by the Woodbury identity."""
    B, T, D = gh.shape
    U = subspace(key, cfg, state)
    has_g = guide is not None and cfg.top_guide > 0
    if U is None and not has_g:
        return gh
    bt = cfg.top_guide if has_g else 0.0
    alpha = (1 - bt) * ((1 - cfg.beta) if U is not None else 1.0)
    cols, cs = [], []
    if U is not None:
        cols.append(U[None, None].expand(B, T, D, U.shape[1]))
        cs += [(1 - bt) * cfg.beta * D / cfg.rank] * U.shape[1]
    if has_g:
        cols.append(guide.float()[..., None])
        cs.append(bt * D)
    V = torch.cat(cols, -1)
    Vt = V.transpose(-1, -2)
    M = alpha * torch.diag(1.0 / torch.tensor(cs, device=gh.device)) + Vt @ V
    y = torch.linalg.solve(M, Vt @ gh[..., None])
    return (gh - (V @ y)[..., 0]) / alpha


def update_guided(key, ghat, cfg, state):
    if cfg.noise != "guided":
        return
    x = ghat.reshape(-1, ghat.shape[-1])
    C = x.T @ x
    C = C / C.diagonal().sum().clamp(min=1e-30)
    old = state.get(("C", key))
    state[("C", key)] = C if old is None else (1 - cfg.ema) * old + cfg.ema * C


# ---- the two estimators ----------------------------------------------------------------------------------------------
def rerun_site(model, c, site, K, sig, gamma, cfg, state, g, mask, stop=None, guide=None, unit_mask=None):
    """Perturb `site`, rerun, score each token's loss change. Returns ghat (B, T, D) ≈ d(sum of losses)/d(site output)."""
    v0 = c[site]
    B, T, D = v0.shape
    dev = v0.device
    clean = c["loss"] if stop is None else c["loss_aux", stop]
    if site[0] == "aux":
        clean = c["loss_aux", site[1]]
    Gam = discount(T, gamma, dev) if gamma > 0 else None
    Sra = torch.zeros(B, T, D, device=dev)
    Sa = torch.zeros(B, T, D, device=dev)
    rsum = torch.zeros(B, T, device=dev)
    shaped = cfg.shaping != "" and site[0] not in ("head", "aux")
    Rs, As = [], []
    for a in draws(site, K, B, T, D, cfg, state, g, dev, v0, v0.dtype, guide):
        if mask is not None:
            a = a * mask[None, :, :, None].to(a.dtype)
        if unit_mask is not None:
            a = a * unit_mask[None].to(a.dtype)
        kc = a.shape[0]
        loss = model.resume(site, (v0[None] + sig * a).reshape(kc * B, T, D), c, stop_aux=stop).view(kc, B, T)
        r = clean[None] - loss                                  # > 0: the draw lowered that token's loss (fp32)
        if Gam is not None:
            r = r @ Gam.T
        if shaped:
            Rs.append(r)
            As.append(a)
            continue
        Sra += torch.einsum("kbt,kbtd->btd", r.to(a.dtype), a).float()
        Sa += a.sum(0).float()
        rsum += r.sum(0)
    if shaped:
        R, A = torch.cat(Rs, 0), torch.cat(As, 0)                  # (K, B, T), (K, B, T, D)
        mu, sd = R.mean(0), R.std(0).clamp(min=1e-8)
        if cfg.shaping == "boltz":                                 # lam -> infinity gives back (R - mu): Dust's estimator
            w = torch.softmax((R - mu) / sd / cfg.lam, 0)
            coef = cfg.lam * sd * (w - 1.0 / K) * K
        else:                                                      # centred ranks, rescaled to the reward spread
            u = R.argsort(0).argsort(0).float() / (K - 1) - 0.5
            coef = sd * u / u.std(0).clamp(min=1e-8)
        ghat = -torch.einsum("kbt,kbtd->btd", coef.to(A.dtype), A).float() / (K * sig)
    else:
        ghat = -(Sra - (rsum / K)[..., None] * Sa) / (K * sig)
    if cfg.whiten and site[0] not in ("head", "aux"):
        ghat = whiten(ghat, site, cfg, state, guide)
    update_guided(site, ghat, cfg, state)
    return ghat, K * model.resume_cost(site, stop)


def hub_local(model, c, b, g_o, cfg, state, g, mask):
    """q, k, v of block b: perturb, recompute only the attention, score each head by -<ĝ_o, Δo> (Dust's local scoring);
    k and v credit the scores of later query tokens with γ_loc^lag."""
    qkv0, o0 = c[("qkv", b)], c[("o", b)]
    B, T, _ = qkv0.shape
    d, H = model.d, model.H
    Dh = d // H
    dev = qkv0.device
    sig = cfg.sig["loc"]
    Gam = discount(T, cfg.gamma_loc, dev)
    out = torch.zeros(qkv0.shape, device=dev)
    for j, kind in enumerate("qkv"):
        sl = slice(j * d, (j + 1) * d)
        Sra = torch.zeros(B, T, H, Dh, device=dev)
        Sa = torch.zeros(B, T, H, Dh, device=dev)
        rsum = torch.zeros(B, T, H, device=dev)
        for a in draws((kind, b), cfg.K, B, T, d, cfg, state, g, dev, qkv0[..., sl], qkv0.dtype):
            if mask is not None:
                a = a * mask[None, :, :, None].to(a.dtype)
            kc = a.shape[0]
            p = qkv0[None].repeat(kc, 1, 1, 1)
            p[..., sl] = p[..., sl] + sig * a
            dO = model.attend(p.view(kc * B, T, 3 * d)).view(kc, B, T, d) - o0[None]
            score = -(g_o[None] * dO.float()).view(kc, B, T, H, Dh).sum(-1)  # predicted loss decrease, per head
            r = score if kind == "q" else torch.einsum("kbsh,ts->kbth", score, Gam)
            ah = a.view(kc, B, T, H, Dh)
            Sra += torch.einsum("kbth,kbthe->bthe", r.to(a.dtype), ah).float()
            Sa += ah.sum(0).float()
            rsum += r.sum(0)
        gh = -(Sra - (rsum / cfg.K)[..., None] * Sa) / (cfg.K * sig)
        if cfg.whiten:
            gh = whiten(gh.reshape(B, T, d), (kind, b), cfg, state).reshape(gh.shape)
        out[..., sl] = gh.reshape(B, T, d)
        update_guided((kind, b), gh.reshape(B, T, d), cfg, state)
    f = model.flops_tok()
    return out, 3 * cfg.K * (T * d) / f["fwd"]


def act_local(model, h0, ez, K, sig, g):
    """O4: the error at the MLP hidden pre-activation, per unit, from the error ez at its output z = act(h): perturb every
    unit at once, score unit j by -ez_j * Δz_j (its own output only, so each unit is a 1-dimensional problem). Works for
    any activation, differentiable or not; estimates ez_j times the noise-smoothed slope."""
    h = h0.float()
    z0 = model.act(h)
    Sra = torch.zeros_like(h)
    Sa = torch.zeros_like(h)
    rs = torch.zeros_like(h)
    for _ in range(K):
        a = torch.randn(h.shape, generator=g, device=h.device)
        r = -ez * (model.act(h + sig * a) - z0)
        Sra += r * a
        Sa += a
        rs += r
    return -(Sra - rs / K * Sa) / (K * sig)


def top_error(model, c, tgt):
    """O3: the exact error at the top of the residual stream (before the final norm), from the head's closed-form error:
    no draws, no pass through the network."""
    z = c[("head",)].float()
    p = torch.softmax(15 * torch.tanh(z / 15), -1)
    valid = (tgt >= 0).float()[..., None]
    ez = (p - torch.nn.functional.one_hot(tgt.clamp(min=0), model.V).float()) * valid * (1 - torch.tanh(z / 15) ** 2)
    eu = ez @ model.Whead.float()
    x = c["xL"].float()
    rr = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    y = x / rr
    return (eu - y * (y * eu).sum(-1, keepdim=True) / x.shape[-1]) / rr


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def entropy_mask(c, frac):
    z = c[("head",)].float()
    p = torch.softmax(15 * torch.tanh(z / 15), -1)
    H = -(p * p.clamp(min=1e-12).log()).sum(-1)            # (B, T)
    thr = torch.quantile(H.flatten(), 1 - frac)
    return (H >= thr).float()


def shadow(model, cfg, state):
    """The network every pass runs on: the model itself (fp32), or a 16-bit copy of its weights refreshed each step.
    Losses are computed in fp32 (TinyGPT.tok_loss); rewards, accumulators and gradients stay fp32."""
    if not cfg.amp:
        return model
    sh = state.get("shadow")
    if sh is None:
        sh = state["shadow"] = copy.deepcopy(model).to(dict(bf16=torch.bfloat16, fp16=torch.float16)[cfg.amp])
    for p16, p in zip(sh.parameters(), model.parameters()):
        p16.copy_(p)
    return sh


@torch.no_grad()
def estimate(model_fp32, idx, tgt, cfg, state, g):
    model = shadow(model_fp32, cfg, state)
    if cfg.simul:
        from simul import estimate_simul
        grads, cost = estimate_simul(model_fp32, idx, tgt, cfg.K, cfg.sig["proj"], cfg.sig["head"], cfg.chunk, g, model)
        return grads, dict(loss=float("nan"), cost=cost, err={})
    c = model.forward_cache(idx, tgt)
    B, T = idx.shape
    N = B * T
    mask = entropy_mask(c, cfg.ent_frac) if cfg.ent_frac < 1 else None
    err, cost = {}, 1.0                                    # the clean forward
    aux = model.aux_blocks

    def stop_for(b):
        if not cfg.local:
            return None
        nxt = [e for e in aux if e >= b]
        return nxt[0] if nxt else None

    sites = [("emb",)]
    for b in range(model.L):
        if cfg.hub_T:
            sites += [("proj", b), ("out", b)]
        else:
            sites += ([("o", b)] if cfg.qkv_mode == "hub" else [("qkv", b)]) + [("proj", b), ("fc", b), ("out", b)]
    gtop = unit(top_error(model, c, tgt)) if cfg.top_guide > 0 else None
    for s in sites:
        b = 0 if s[0] == "emb" else s[1]
        guide = gtop if s[0] in ("proj", "out", "emb") else None
        um = None
        if cfg.sparse_c > 0 and s[0] == "fc":
            um = model.margin(c[s].float()) < cfg.sparse_c * cfg.sig["fc"]
        err[s], k = rerun_site(model, c, s, cfg.K, cfg.sig[s[0]], cfg.gamma, cfg, state, g, mask, stop_for(b), guide, um)
        cost += k
    if cfg.hub_T:
        f = model.flops_tok()
        for b in range(model.L):
            err[("o", b)] = err[("proj", b)] @ model.Wproj[b].float()
            err[("fc", b)] = act_local(model, c[("fc", b)], err[("out", b)] @ model.Wout[b].float(), cfg.act_K,
                                       cfg.sig["fc"], g)
            cost += cfg.act_K * model.m / f["fwd"] * 2
    if cfg.qkv_mode == "hub":
        for b in range(model.L):
            err[("qkv", b)], k = hub_local(model, c, b, err[("o", b)], cfg, state, g, mask)
            cost += k
    heads = [("head",)] + [("aux", b) for b in aux]
    for s in heads:
        if cfg.exact_head:
            z = c[s].float()
            p = torch.softmax(15 * torch.tanh(z / 15), -1)
            valid = (tgt >= 0).float()[..., None]
            p = (p - torch.nn.functional.one_hot(tgt.clamp(min=0), model.V).float()) * valid
            err[s] = p * (1 - torch.tanh(z / 15) ** 2)           # through the soft-cap
        else:
            hc = Cfg(**{**cfg.__dict__, "noise": "gauss"})
            err[s], k = rerun_site(model, c, s, cfg.head_mult * cfg.K, cfg.sig["head"], 0.0, hc, state, g, None)
            cost += k
    grads = {}
    m = model_fp32
    for s in m.weight_sites():
        if s[0] == "emb":
            ge = err[s].reshape(-1, m.d)
            grads[m.wte] = torch.zeros_like(m.wte).index_add_(0, idx.reshape(-1), ge) / N
            grads[m.wpe] = torch.zeros_like(m.wpe).index_add_(0, torch.arange(T, device=idx.device).repeat(B), ge) / N
        else:
            grads[m.weight_of(s)] = torch.einsum("btd,bti->di", err[s], c["in", s].float()) / N
    return grads, dict(loss=c["loss"].mean().item(), cost=cost, err=err)


def backprop(model, idx, tgt, aux_weight=0.0):
    """True gradients of the mean per-token loss (+ aux_weight × the mean aux losses), and the true per-token errors
    d(sum of losses)/d(site output) at every site."""
    model.zero_grad(set_to_none=True)
    c = model.forward_cache(idx, tgt, retain=True)
    loss = c["loss"].mean()
    for b in model.aux_blocks:
        loss = loss + aux_weight * c["loss_aux", b].mean()
    loss.backward()
    N = idx.numel()
    grads = {p: p.grad.detach().clone() for p in model.parameters() if p.grad is not None}
    errs = {s: c[s].grad.detach() * N for s in list(c) if isinstance(s, tuple) and len(s) <= 2 and isinstance(s[0], str)
            and s[0] in ("emb", "qkv", "o", "proj", "fc", "out", "head", "aux") and torch.is_tensor(c[s])
            and c[s].grad is not None}
    model.zero_grad(set_to_none=True)
    return grads, errs, loss.item()
