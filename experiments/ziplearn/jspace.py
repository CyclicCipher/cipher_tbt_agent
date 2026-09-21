"""E8b — the J-space of the gradient arm, standard residual vs attention residuals (OPEN-9).

After Gurnee et al. (Anthropic, July 2026): the Jacobian lens is the average Jacobian of the final residual with
respect to a layer's state, J_l = E[ d h_final,t' / d h_l,t ] over source positions t, later positions t' >= t and
prompts; the J-lens vector for token k at layer l is the direction in h_l-space that raises token k later (row k of
the averaged Jacobian of the LOGITS, which folds in the final norm and unembedding); J-space is the sparse
non-negative span of those vectors.

What "the state after layer l that later computation reads" is: the residual stream after block l for the standard
model; block l's summed output -- its SOURCE -- for the attention-residual model (that is what later mixers read).
Layer 0 is the embedding in both. h_final is the state entering the final norm.

Measured per layer and model, on the two E8b models (same seed, same task, same steps):
  identity share  trace(J)/d for the SAME-position Jacobian (t' = t) -- how much of a perturbation passes straight
                  through (the standard model's identity highway gives ~1; attention residuals give a route weight);
  future share    ||J_future|| / (||J_same|| + ||J_future||): the part of a perturbation's effect that lands on LATER
                  positions, i.e. is carried by attention to other tokens -- broadcast;
  gain, rank      ||J||_F/sqrt(d) and the effective rank (sum s)^2 / sum s^2 of J's singular values;
  J-space share   fraction of activation variance at output positions inside span{J-lens vectors} (5 tokens);
  persistence     mean cosine between token k's J-lens vector at layer l and at layer l+1;
  swap rate       add a scaled J-lens vector for a target digit at the position that predicts the last output
                  digit; fraction of cases where the prediction flips to that digit (the paper's report swap).

    python experiments/ziplearn/jspace.py        (~1 min on the GPU) -> runs/e8b/jspace.json, jspace.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformers"))
import h1_lid as H   # noqa: E402

L, V = H.L, H.V


def load(path, dev):
    ck = torch.load(path, weights_only=False)
    m = H.Model(d_model=ck["d_model"], n_layer=ck["n_layer"], n_head=ck["n_head"], max_len=ck["k"] * 2 * L + 2,
                pos=ck["pos"], res=ck["res"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    for q in m.parameters():                                                         # no reverse-mode graphs: the
        q.requires_grad_(False)                                                      # Jacobians are forward-mode
    return m, ck


def forward_states(model, tok, deltas=None, want="final"):
    """The model's forward with an optional perturbation added to the state after block l (l = 0: the embedding).
    Returns the state entering the final norm ("final") or the logits."""
    deltas = deltas or {}
    h = model.emb(tok)
    if model.pos is not None:
        h = h + model.pos[:, :tok.shape[1]]
    if 0 in deltas:
        h = h + deltas[0]
    if model.res == "std":
        for i, b in enumerate(model.blocks):
            h = b(h)
            if (i + 1) in deltas:
                h = h + deltas[i + 1]
        final = h
    else:
        blocks, partial = [], h
        for i, b in enumerate(model.blocks):
            x = b.res_attn(blocks + [partial])
            blocks.append(partial)
            partial = b.attn(b.n1(x))
            x = b.res_mlp(blocks + [partial])
            partial = partial + b.mlp(b.n2(x))
            if (i + 1) in deltas:
                partial = partial + deltas[i + 1]
        final = model.res_final(blocks + [partial])
    return final if want == "final" else model.head(model.norm(final))


def jac_fwd(f, x, chunk=24):
    """Forward-mode Jacobian of f at x, one chunk of basis directions at a time: (out..., d)."""
    d = x.numel()
    eye = torch.eye(d, device=x.device)
    cols = []
    for i in range(0, d, chunk):
        E = eye[i:i + chunk]
        cols.append(vmap(lambda e: jvp(f, (x,), (e,))[1])(E))                       # (chunk, out...)
    J = torch.cat(cols, 0)                                                           # (d, out...)
    return J.movedim(0, -1)                                                          # (out..., d)


def jacobians(model, tok, layer, src_positions, dev):
    """Averaged Jacobians of h_final and of the logits at positions t' >= t with respect to the state at layer
    `layer`, position t, for t in src_positions; averaged over prompts in `tok`."""
    B, T = tok.shape
    d = model.emb.weight.shape[1]
    J_final = torch.zeros(d, d, device=dev)                                          # same position, t' = t
    J_future = torch.zeros(d, d, device=dev)                                         # later positions, t' > t
    J_logit = torch.zeros(V, d, device=dev)
    count = 0
    for b in range(B):
        tb = tok[b:b + 1]
        for t in src_positions:
            def f_final(delta_t):
                delta = torch.zeros(1, T, d, device=dev)
                delta = delta.index_add(1, torch.tensor([t], device=dev), delta_t[None, None, :])
                return forward_states(model, tb, {layer: delta}, "final")[0, t:, :]      # (T - t, d)
            def f_logit(delta_t):
                delta = torch.zeros(1, T, d, device=dev)
                delta = delta.index_add(1, torch.tensor([t], device=dev), delta_t[None, None, :])
                return forward_states(model, tb, {layer: delta}, "logits")[0, t:, :]     # (T - t, V)
            z = torch.zeros(d, device=dev)
            with sdpa_kernel([SDPBackend.MATH]):
                Jf = jac_fwd(f_final, z)                                             # (T - t, d, d)
                Jl = jac_fwd(f_logit, z)                                             # (T - t, V, d)
            J_final += Jf[0].detach()
            J_future += Jf[1:].mean(0).detach()
            J_logit += Jl[0].detach()
            del Jf, Jl
            count += 1
    return J_final / count, J_future / count, J_logit / count


@torch.no_grad()
def states_at(model, tok, layer):
    """The state after block `layer` for every token (no perturbation), by re-running the forward and capturing."""
    h = model.emb(tok)
    if model.pos is not None:
        h = h + model.pos[:, :tok.shape[1]]
    if layer == 0:
        return h
    if model.res == "std":
        for i, b in enumerate(model.blocks):
            h = b(h)
            if i + 1 == layer:
                return h
    blocks, partial = [], h
    for i, b in enumerate(model.blocks):
        x = b.res_attn(blocks + [partial])
        blocks.append(partial)
        partial = b.attn(b.n1(x))
        x = b.res_mlp(blocks + [partial])
        partial = partial + b.mlp(b.n2(x))
        if i + 1 == layer:
            return partial
    raise ValueError(layer)


def variance_share(states, vectors):
    """Fraction of the (centred) variance of `states` (N, d) lying in span(vectors) (V, d)."""
    X = states - states.mean(0, keepdim=True)
    Q, _ = torch.linalg.qr(vectors.T)                                                # (d, r) orthonormal basis
    proj = X @ Q
    return float((proj ** 2).sum() / (X ** 2).sum())


@torch.no_grad()
def swap_rate(model, tok, layer, lens_vectors, t_pred, dev, alpha=None):
    """Add alpha * (J-lens vector of a target digit) to the state at layer `layer`, position t_pred, and count how
    often the prediction at t_pred flips to that digit. alpha defaults to the state's typical norm."""
    base = forward_states(model, tok, None, "logits")[:, t_pred].argmax(-1)          # (B,)
    st = states_at(model, tok, layer)[:, t_pred]
    scale = float(st.norm(dim=-1).mean()) if alpha is None else alpha
    flips, total = 0, 0
    for k in range(V):
        g = lens_vectors[k]
        g = g / (g.norm() + 1e-9) * scale
        delta = torch.zeros_like(states_at(model, tok, layer))
        delta[:, t_pred] = g
        pred = forward_states(model, tok, {layer: delta}, "logits")[:, t_pred].argmax(-1)
        sel = base != k
        flips += int((pred[sel] == k).sum())
        total += int(sel.sum())
    return flips / max(1, total)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = HERE / "runs" / "e8b"
    torch.manual_seed(0)
    train, _test, probs, _w = H.build_tasks(0)
    g = torch.Generator(device=dev).manual_seed(5)
    K = 8
    tok, _ = H.make_batch(64, K, train, probs, dev, g)                               # (64, 96)
    T = tok.shape[1]
    src_positions = [k * 2 * L + L for k in range(1, K)]                             # first output digit of demos 2..8
    t_pred = T - 2                                                                   # predicts the last output digit
    out_positions = [p for p in range(T) if H.out_mask(K, dev)[p]]
    results = {}
    print(f"{'model':<13}{'layer':>6}{'identity':>10}{'gain':>7}{'future':>8}{'rank':>6}{'J-space share':>15}{'persist':>9}{'swap':>7}")
    for name in ("rope_std", "rope_attnres"):
        model, ck = load(out / f"{name}.pt", dev)
        res = {}
        lens = {}
        for layer in (0, 1, 2, 3):
            Jf, Ju, Jl = jacobians(model, tok[:6], layer, src_positions, dev)
            d = Jf.shape[0]
            s = torch.linalg.svdvals(Jf)
            st = states_at(model, tok, layer)[:, out_positions].reshape(-1, d).float()
            lens[layer] = Jl
            res[layer] = dict(identity_share=float(torch.trace(Jf) / d), gain=float(Jf.norm() / d ** 0.5),
                              future_share=float(Ju.norm() / (Jf.norm() + Ju.norm())),
                              top_singular=float(s[0]), effective_rank=float(s.sum() ** 2 / (s ** 2).sum()),
                              jspace_share=variance_share(st, Jl), swap=swap_rate(model, tok, layer, Jl, t_pred, dev))
        for layer in (0, 1, 2):
            a, b = lens[layer], lens[layer + 1]
            res[layer]["persistence"] = float(F.cosine_similarity(a, b, dim=1).mean())
        res[3]["persistence"] = float("nan")
        for layer, r in res.items():
            print(f"{name:<13}{layer:>6}{r['identity_share']:>10.3f}{r['gain']:>7.2f}{r['future_share']:>8.2f}{r['effective_rank']:>6.1f}"
                  f"{r['jspace_share']:>15.3f}{r['persistence']:>9.3f}{r['swap']:>7.2f}")
        results[name] = {str(k): v for k, v in res.items()}
        if model.res != "std":
            results[name]["routes"] = model.routes(tok)
    json.dump(results, open(out / "jspace.json", "w"), indent=1)
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, key, title in zip(axes, ("identity_share", "effective_rank", "jspace_share", "swap"),
                              ("identity share  trace(J)/d", "effective rank of J", "J-space share of variance", "swap rate")):
        for name, mk in (("rope_std", "o-"), ("rope_attnres", "s--")):
            ax.plot([0, 1, 2, 3], [results[name][str(l)][key] for l in range(4)], mk, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("state after block (0 = embedding)")
        ax.set_xticks([0, 1, 2, 3])
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "jspace.png", dpi=130)
    print(f"\nfigure: {out / 'jspace.png'}")


if __name__ == "__main__":
    main()
