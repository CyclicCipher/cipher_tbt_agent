"""Phase 0 -- discover the ring Z_m (FEW_SHOT_ARITHMETIC.md idea #2: "teach the number line first").

Trains the shared PoPE transformer on succ / pred / circular-distance comparison, then *probes whether
a ring actually emerged* in the value embeddings -- the point is that the metric space is DISCOVERED
from experience, not hardcoded (idea #1, rejected). Two checks:
  * held-out comparison accuracy -- the metric generalises (a real line), not memorised pairs;
  * ring_var / dist_corr -- the VAL(v) embedding rows organise into an ordered circle.

The discovered model (its value rows especially) is the init for phase-1 few-shot addition.
Tiny + CPU => seconds per run; NOT a long training loop (Mistake #36).
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from baselines.sigreg import SIGReg  # noqa: E402
from tasks.number_line import NumberLineDiscovery  # noqa: E402
from tasks.vocab import VAL0  # noqa: E402


def build_model(task, dim=64, n_heads=4, n_layers=2, seed=0) -> FixedDepthTransformer:
    torch.manual_seed(seed)
    cfg = FixedDepthConfig(
        vocab_size=task.vocab_size, dim=dim, n_heads=n_heads, n_layers=n_layers,
        max_seq=6, pos_mode="pope", n_axes=2,
    )
    return FixedDepthTransformer(cfg)


def ce_masked(logits, batch) -> torch.Tensor:
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, v), batch.targets.reshape(-1), reduction="none")
    ce = ce.reshape(batch.targets.shape)
    return (ce * batch.loss_mask).sum() / batch.loss_mask.sum().clamp_min(1.0)


@torch.no_grad()
def score_acc(model, batch, device) -> float:
    b = batch.to(device)
    pred = model(b.input_ids, coord=b.coord)[:, b.score_pos].argmax(-1)
    return (pred == b.targets[:, b.score_pos]).float().mean().item()


def _ring_metrics(E, task) -> tuple[float, float]:
    """ring_var = fraction of variance captured by the fundamental ring basis [cos,sin](2pi v/m)
    (rotation-invariant in that plane); dist_corr = Pearson corr of pairwise row-distance with true
    circular distance. Both ~1 => the m rows form a clean ordered ring ordered by value."""
    m = task.P
    E = E - E.mean(0)
    ang = 2 * math.pi * torch.arange(m, dtype=torch.float) / m
    basis = torch.stack([ang.cos(), ang.sin()], 1)
    basis = basis - basis.mean(0)
    proj = basis @ torch.linalg.pinv(basis) @ E
    ring_var = (proj.pow(2).sum() / E.pow(2).sum().clamp_min(1e-9)).item()
    emb_d, cir_d = [], []
    for a in range(m):
        for b in range(a + 1, m):
            emb_d.append((E[a] - E[b]).norm().item())
            cir_d.append(task.dist(a, b))
    ed = torch.tensor(emb_d); cd = torch.tensor(cir_d, dtype=torch.float)
    ed = ed - ed.mean(); cd = cd - cd.mean()
    dist_corr = (ed @ cd / (ed.norm() * cd.norm()).clamp_min(1e-9)).item()
    return ring_var, dist_corr


@torch.no_grad()
def ring_probe(model, task) -> tuple[float, float]:
    """Ring structure in the input embedding rows (== unembedding, tied head)."""
    E = model.embed.weight[VAL0:VAL0 + task.P].detach().float()
    return _ring_metrics(E, task)


@torch.no_grad()
def spectrum_probe(model, task) -> tuple[float, float]:
    """Frequency content of the discovered value code (BLUEPRINT §C probe). DFT the hidden reps z(a)
    over the value index a; n_eff = participation ratio of the power spectrum = effective number of
    frequencies (1 = single-frequency collapse / pure circle; higher = multi-scale grid code); top =
    fraction of power in the dominant frequency. Target: n_eff > 1 (multi-frequency)."""
    m = task.P
    b = task.successor_batch()
    Z = model.encode(b.input_ids, b.coord)[:, 0].detach().float()      # (m, dim)
    Z = Z - Z.mean(0)
    a = torch.arange(m, dtype=torch.float)
    k = torch.arange(1, m // 2 + 1, dtype=torch.float)
    ang = 2 * math.pi * torch.outer(k, a) / m                          # (K, m)
    power = (ang.cos() @ Z).pow(2).sum(-1) + (ang.sin() @ Z).pow(2).sum(-1)   # (K,)
    p = power / power.sum().clamp_min(1e-9)
    n_eff = 1.0 / p.pow(2).sum().clamp_min(1e-9)
    return n_eff.item(), p.max().item()


@torch.no_grad()
def ring_probe_hidden(model, task) -> tuple[float, float]:
    """Ring structure in the model's *hidden* representation of each value -- the contextualised rep of
    the VAL(a) token (position 0 of the successor batch). Catches a ring that lives in activations
    rather than the input embeddings."""
    b = task.successor_batch()
    H = model.encode(b.input_ids, b.coord)[:, 0].detach().float()   # (m, dim), row a = rep of VAL(a)
    return _ring_metrics(H, task)


def _spectral_concentration(z, m, device):
    """Herfindahl concentration of the value code's power spectrum (DFT over the value index). Low =
    power spread across many frequencies (multi-scale grid code); high = single-frequency collapse.
    Minimising this is frequency-domain anti-collapse -- SIGReg's analogue in the Fourier domain
    (BLUEPRINT §C, the multi-scale pillar). Lets the number/values of scales emerge, nothing hand-set."""
    zc = z - z.mean(0)
    a = torch.arange(m, dtype=torch.float, device=device)
    k = torch.arange(1, m // 2 + 1, dtype=torch.float, device=device)
    ang = 2 * math.pi * torch.outer(k, a) / m
    power = (ang.cos() @ zc).pow(2).sum(-1) + (ang.sin() @ zc).pow(2).sum(-1)
    p = power / power.sum().clamp_min(1e-9)
    return p.pow(2).sum()


def train_discovery(task, steps=600, lr=3e-3, weight_decay=0.01, dim=64, seed=0, device="cpu",
                    n_compare_train=200, w_sig=0.1, w_equiv=1.0, w_spread=0.0, use_compare=True,
                    log_every=0):
    """Discovery objective: succ/pred(/compare) CE + SIGReg + (w_equiv>0) the ring-forcing equivariance
    term (one rotation R=exp(G-G^T) with z(a+1)=R z(a), orbit closing) + (w_spread>0) the multi-scale
    pillar -- frequency-domain anti-collapse so the code uses *multiple* frequencies, not a single
    pure circle (BLUEPRINT §C). Emergent-spectrum design: scales emerge, none hand-set.
    use_compare=False drops the comparison task (e.g. MultiplicativeRing, which has no natural metric
    experience) -- the ring is then forced by succ/pred + equivariance alone."""
    torch.manual_seed(seed)
    m = task.P
    model = build_model(task, dim=dim, seed=seed).to(device)
    # the shared +1 movement, parametrised as a true rotation R = exp(G - G^T) (PURE_MATH §10): a Lie
    # generator, so the discrete ring extends to fractional/continuous movements via R^t = exp(t·G).
    G = torch.nn.Parameter(0.01 * torch.randn(dim, dim, device=device))
    succ = task.successor_batch().to(device)                           # rows in orbit order
    pred = task.predecessor_batch().to(device)
    if use_compare:
        cmp_tr_pairs, cmp_ho_pairs = task.split_compare(n_compare_train, seed=seed)
        cmp_tr = task.compare_batch(cmp_tr_pairs).to(device)
        cmp_ho = task.compare_batch(cmp_ho_pairs).to(device)
    sig = SIGReg(n_slices=256).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + [G], lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for step in range(steps):
        l_succ = ce_masked(model(succ.input_ids, coord=succ.coord), succ)
        l_pred = ce_masked(model(pred.input_ids, coord=pred.coord), pred)
        l_cmp = ce_masked(model(cmp_tr.input_ids, coord=cmp_tr.coord), cmp_tr) if use_compare else succ.input_ids.new_zeros((), dtype=torch.float)
        z = model.encode(succ.input_ids, succ.coord)[:, 0]             # (m, dim): row a = rep of orbit[a]
        l_sig = sig(z, generator=g)
        # equivariance + closure: z[(a+1)%m] == R z[a] for all a (roll up by 1 == successor on the ring)
        Rmat = torch.matrix_exp(G - G.t())                            # orthogonal (a rotation)
        l_equiv = (torch.roll(z, -1, 0) - z @ Rmat.t()).pow(2).sum(-1).mean() if w_equiv else z.new_zeros(())
        l_spread = _spectral_concentration(z, m, device) if w_spread else z.new_zeros(())
        loss = l_succ + l_pred + l_cmp + w_sig * l_sig + w_equiv * l_equiv + w_spread * l_spread
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            model.eval()
            ho = score_acc(model, cmp_ho, device) if use_compare else float("nan")
            hrv, hdc = ring_probe_hidden(model, task)
            neff, top = spectrum_probe(model, task)
            model.train()
            print(f"    step {step:>4}  succ {l_succ.item():.3f}  cmp {l_cmp.item():.3f}  "
                  f"equiv {l_equiv.item():.3f}  spread {l_spread.item():.3f}  | cmp_ho {ho:.3f}  "
                  f"hid_dc {hdc:.3f}  n_eff {neff:.2f}  top {top:.2f}", flush=True)
    model.eval()
    model._R = torch.matrix_exp(G - G.t()).detach()       # stash the learned generator for phase-1
    rv, dc = ring_probe(model, task)
    hrv, hdc = ring_probe_hidden(model, task)
    neff, top = spectrum_probe(model, task)
    return model, {
        "succ_acc": score_acc(model, succ, device),
        "pred_acc": score_acc(model, pred, device),
        "cmp_train_acc": score_acc(model, cmp_tr, device) if use_compare else float("nan"),
        "cmp_heldout_acc": score_acc(model, cmp_ho, device) if use_compare else float("nan"),
        "emb_dist_corr": dc, "hid_dist_corr": hdc,
        "n_eff_freq": neff, "dominant_power": top,
    }


def train_torus2d(task, steps=1000, lr=3e-3, weight_decay=0.01, dim=64, seed=0, device="cpu",
                  w_equiv=1.0, w_comm=1.0, w_sig=0.1, log_every=0):
    """2-torus discovery (BLUEPRINT §A/§H): two generators Rx,Ry from x-/y-step experience, with
    z(x+1,y)=Rx z, z(x,y+1)=Ry z (both orbits closing via the rolls) and Rx,Ry **commuting** (path
    independence). Stashes model._Rx/_Ry for the zero-shot 2-D navigation test."""
    torch.manual_seed(seed)
    model = build_model(task, dim=dim, seed=seed).to(device)
    Gx = torch.nn.Parameter(0.01 * torch.randn(dim, dim, device=device))
    Gy = torch.nn.Parameter(0.01 * torch.randn(dim, dim, device=device))
    bx = task.step_batch(0).to(device)                   # x-step; rows in cell-index order
    by = task.step_batch(1).to(device)                   # y-step
    sig = SIGReg(n_slices=256).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + [Gx, Gy], lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    mx, my = task.mx, task.my
    model.train()
    for step in range(steps):
        l_x = ce_masked(model(bx.input_ids, coord=bx.coord), bx)
        l_y = ce_masked(model(by.input_ids, coord=by.coord), by)
        z = model.encode(bx.input_ids, bx.coord)[:, 0]   # (N, dim): row index = rep of cell (x,y)
        zgrid = z.view(mx, my, -1)
        Rx = torch.matrix_exp(Gx - Gx.t())
        Ry = torch.matrix_exp(Gy - Gy.t())
        tx = torch.roll(zgrid, -1, dims=0).reshape(z.shape)   # z(x+1,y)
        ty = torch.roll(zgrid, -1, dims=1).reshape(z.shape)   # z(x,y+1)
        l_eqx = (tx - z @ Rx.t()).pow(2).sum(-1).mean()
        l_eqy = (ty - z @ Ry.t()).pow(2).sum(-1).mean()
        l_comm = (Rx @ Ry - Ry @ Rx).pow(2).sum()             # path independence
        l_sig = sig(z, generator=g)
        loss = l_x + l_y + w_equiv * (l_eqx + l_eqy) + w_comm * l_comm + w_sig * l_sig
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:>4}  sx {l_x.item():.3f}  sy {l_y.item():.3f}  "
                  f"eqx {l_eqx.item():.3f}  eqy {l_eqy.item():.3f}  comm {l_comm.item():.4f}  "
                  f"sig {l_sig.item():.3f}", flush=True)
    model.eval()
    model._Rx = torch.matrix_exp(Gx - Gx.t()).detach()
    model._Ry = torch.matrix_exp(Gy - Gy.t()).detach()
    return model


def eff_rank(z) -> float:
    """Effective rank (participation ratio) of cov(z): (Σλ)²/Σλ². The collapse invariant -- the geometry
    needs eff_rank >= 2*(#generators). Method-agnostic, so it's the fair yardstick for anti-collapse."""
    zc = z - z.mean(0)
    cov = (zc.t() @ zc) / max(z.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    return (ev.sum().pow(2) / ev.pow(2).sum().clamp_min(1e-12)).item()


def vicreg(z, gamma=1.0, w_var=25.0, w_cov=1.0):
    """Hard anti-collapse (VICReg-style): a *hinge* variance floor (push each dim's std up to gamma, then
    inactive -- a floor, not a distributional push) + covariance decorrelation. Internal weights fixed."""
    std = (z.var(dim=0) + 1e-4).sqrt()
    var = torch.relu(gamma - std).mean()
    zc = z - z.mean(0)
    cov = (zc.t() @ zc) / max(z.shape[0] - 1, 1)
    cov_off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return w_var * var + w_cov * cov_off / z.shape[1]


def train_torusnd(task, steps=1500, lr=3e-3, weight_decay=0.01, dim=64, seed=0, device="cpu",
                  w_equiv=1.0, w_comm=1.0, anti_collapse="sigreg", w_ac=5.0, log_every=0):
    """n-torus discovery (BLUEPRINT §A): n generators (one per axis) from per-axis step experience, each
    z(...,c_a+1,...)=R_a z, all pairwise commuting. w_sig defaults high (Finding 6: anti-collapse must
    scale with #generators). Stashes model._Rs = [R_0..R_{n-1}] for the zero-shot n-D navigation test."""
    torch.manual_seed(seed)
    model = build_model(task, dim=dim, seed=seed).to(device)
    n = task.ndim
    Gs = [torch.nn.Parameter(0.01 * torch.randn(dim, dim, device=device)) for _ in range(n)]
    batches = [task.step_batch(a).to(device) for a in range(n)]
    sig = SIGReg(n_slices=256).to(device) if anti_collapse == "sigreg" else None
    opt = torch.optim.AdamW(list(model.parameters()) + Gs, lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for step in range(steps):
        l_pred = sum(ce_masked(model(b.input_ids, coord=b.coord), b) for b in batches)
        z = model.encode(batches[0].input_ids, batches[0].coord)[:, 0]    # (N, dim) row-major
        zgrid = z.view(*task.shape, -1)
        Rs = [torch.matrix_exp(G - G.t()) for G in Gs]
        l_eq = sum((torch.roll(zgrid, -1, dims=a).reshape(z.shape) - z @ Rs[a].t()).pow(2).sum(-1).mean()
                   for a in range(n))
        l_comm = sum((Rs[a] @ Rs[b] - Rs[b] @ Rs[a]).pow(2).sum()
                     for a in range(n) for b in range(a + 1, n))
        l_ac = sig(z, generator=g) if anti_collapse == "sigreg" else vicreg(z)
        loss = l_pred + w_equiv * l_eq + w_comm * l_comm + w_ac * l_ac
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:>4}  pred {l_pred.item():.3f}  eq {l_eq.item():.3f}  "
                  f"comm {l_comm.item():.4f}  ac {l_ac.item():.3f}  eff_rank {eff_rank(z.detach()):.1f}",
                  flush=True)
    model.eval()
    model._Rs = [torch.matrix_exp(G - G.t()).detach() for G in Gs]
    with torch.no_grad():
        model._eff_rank = eff_rank(model.encode(batches[0].input_ids, batches[0].coord)[:, 0])
    return model


def train_hex(task, steps=1500, lr=3e-3, weight_decay=0.01, dim=64, seed=0, device="cpu",
              w_equiv=1.0, w_comm=1.0, w_sig=5.0, w_iso=0.0, log_every=0):
    """Hexagonal / conformal-isometry discovery (BLUEPRINT §A). 3 generators for the 3 hex directions,
    with closure R0·R1·R2=I and (w_iso>0) an isotropy term forcing equal step magnitude in every
    direction = conformal isometry. Stashes model._Rs and ._step_mags."""
    torch.manual_seed(seed)
    m = task.m
    model = build_model(task, dim=dim, seed=seed).to(device)
    Gs = [torch.nn.Parameter(0.01 * torch.randn(dim, dim, device=device)) for _ in range(3)]
    batches = [task.step_batch(k).to(device) for k in range(3)]
    sig = SIGReg(n_slices=256).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + Gs, lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    eye = torch.eye(dim, device=device)
    model.train()
    for step in range(steps):
        l_pred = sum(ce_masked(model(b.input_ids, coord=b.coord), b) for b in batches)
        z = model.encode(batches[0].input_ids, batches[0].coord)[:, 0]
        zgrid = z.view(m, m, -1)
        Rs = [torch.matrix_exp(G - G.t()) for G in Gs]
        l_eq = 0.0
        mags = []
        for k, d in enumerate(task.DIRS):
            tgt = torch.roll(zgrid, shifts=(-d[0], -d[1]), dims=(0, 1)).reshape(z.shape)
            disp = z @ Rs[k].t()
            l_eq = l_eq + (tgt - disp).pow(2).sum(-1).mean()
            mags.append((disp - z).norm(dim=1).mean())
        l_comm = sum((Rs[a] @ Rs[b] - Rs[b] @ Rs[a]).pow(2).sum() for a in range(3) for b in range(a + 1, 3))
        l_close = (Rs[0] @ Rs[1] @ Rs[2] - eye).pow(2).sum()              # hexagonal closure
        l_iso = torch.stack(mags).var()                                  # conformal isometry
        l_sig = sig(z, generator=g)
        loss = (l_pred + w_equiv * l_eq + w_comm * (l_comm + l_close)
                + w_sig * l_sig + w_iso * l_iso)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:>4}  pred {l_pred.item():.3f}  eq {l_eq.item():.3f}  "
                  f"close {l_close.item():.4f}  iso {l_iso.item():.4f}  sig {l_sig.item():.3f}", flush=True)
    model.eval()
    model._Rs = [torch.matrix_exp(G - G.t()).detach() for G in Gs]
    model._step_mags = [mm.item() for mm in mags]
    return model


if __name__ == "__main__":
    task = NumberLineDiscovery(17, seed=0)
    for w_equiv in (0.0, 1.0):
        print(f"=== w_equiv={w_equiv} ===")
        _, stats = train_discovery(task, steps=600, w_equiv=w_equiv, log_every=200)
        print("  stats:", {k: round(v, 3) for k, v in stats.items()})
