"""Read the weights: participation matrix, regions, ablation, wiring, formation times -- plan section 7.

    python experiments/inner_objective/inspect_weights.py --runs runs/A runs/B runs/Ap runs/Bp --out runs/results

Everything here is computed from what `train.py` recorded: the participation sufficient statistics, the snapshots,
the mask/trust telemetry and the final weights. "Region" is used only where a correlation-defined cluster is ALSO
functionally selective under ablation; the two definitions are reported side by side so that the word is earned.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transformers"))
from h1_lid import Model, per_demo_exact                                            # noqa: E402
import tasks as TASKS                                                                # noqa: E402
from train import Flat, env_batch, trials_to_criterion                               # noqa: E402

L = TASKS.L

K_REGIONS = 9


# ── the participation matrix ────────────────────────────────────────────────────────────────────────────────────────
def participation(tr):
    """corr over training steps between environment e's gradient on coordinate i and the update i actually took."""
    n = tr["n"]
    Sg, Sgd, Sgg, Sd, Sdd = (tr[k].double() for k in ("S_g", "S_gd", "S_gg", "S_d", "S_dd"))
    cov = Sgd - Sg * Sd[None, :] / n
    vg = (Sgg - Sg ** 2 / n).clamp(min=0)
    vd = (Sdd - Sd ** 2 / n).clamp(min=0)
    corr = cov / (vg.sqrt() * vd.sqrt()[None, :] + 1e-30)
    return corr.float().numpy()                                    # (E, P)


# ── regions by k-means on participation rows ────────────────────────────────────────────────────────────────────────
def kmeans(X, k, rng, iters=25):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab, C


def auc(score, label):
    """Mann-Whitney AUC of `score` for the positive class `label`."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def module_of(name):
    if name.startswith("emb"):
        return "embed"
    if name.startswith("head") or name.startswith("norm"):
        return "head"
    parts = name.split(".")
    layer = int(parts[1])
    if "attn" in name:
        return f"L{layer}.attn.{parts[3]}"
    if "mlp" in name:
        return f"L{layer}.mlp.{parts[3]}"
    return f"L{layer}.{parts[2]}"


def analyse_run(run_dir, rng):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    tr = torch.load(run_dir / "trace.pt", weights_only=False)
    train_pairs, E_real = cfg["train_pairs"], len(cfg["train_pairs"])
    NAMES = cfg["primitives"]
    task = TASKS.make(cfg.get("task", "compose"), cfg["seed"])
    P = participation(tr)                                          # (E, P)
    real = P[:E_real].T                                            # (P, 17)
    real = np.nan_to_num(real)
    layout = cfg["layout"]
    mod = np.empty(real.shape[0], dtype=object)
    for it in layout:
        mod[it["start"]:it["end"]] = module_of(it["name"])

    # dead / masked coordinates: never meaningfully updated
    total_move = tr["S_dd"].numpy()
    dead = total_move < np.quantile(total_move, 0.02) * 1e-3 + 1e-20
    mask_count = tr["mask_count"].numpy() / tr["n"]
    live = ~dead

    # clusters
    X = real[live]
    lab_live, C = kmeans(X, K_REGIONS, rng)
    lab = np.full(real.shape[0], -1)
    lab[live] = lab_live
    prims = np.array(cfg["prims"], dtype=bool)                                                # (primitives, envs)
    regions = []
    for j in range(K_REGIONS):
        m = lab == j
        prof = real[m].mean(0)
        aucs = [auc(prof, prims[p]) for p in range(len(NAMES))]
        best = int(np.nanargmax(aucs))
        anat = {}
        for name in np.unique(mod[m]):
            anat[str(name)] = float((mod[m] == name).mean())
        regions.append(dict(id=j, size=int(m.sum()), selectivity=float(aucs[best]), primitive=NAMES[best],
                            aucs={NAMES[p]: round(float(a), 3) for p, a in enumerate(aucs)},
                            shared=float(prof.mean()), masked_frac=float(mask_count[m].mean()),
                            anatomy={k: round(v, 3) for k, v in sorted(anat.items(), key=lambda kv: -kv[1])[:4]}))
    n_prim_regions = sum(1 for r in regions if r["selectivity"] >= 0.9)

    # noise-environment participation (Ap / Bp): how much do the noise environments drive weights, vs real ones?
    noise_part = float(np.abs(np.nan_to_num(P[E_real:])).mean()) if P.shape[0] > E_real else None
    real_part = float(np.abs(real).mean())

    # ── ablation: functional selectivity ──────────────────────────────────────────────────────────────────────────
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(d_model=cfg["d"], n_layer=cfg["layers"], n_head=cfg["heads"], max_len=cfg["k"] * 2 * L + 2,
                  pos="rope", n_vocab=cfg["V"]).to(dev)
    model.load_state_dict(tr["model"])
    flat = Flat(model)
    base_vec = flat.get().clone()
    g = torch.Generator(device=dev).manual_seed(7)
    test_pairs = [tuple(p) for p in cfg["test_pairs"]]

    @torch.no_grad()
    def per_primitive_acc():
        acc_env = []
        for pair in train_pairs:
            tok = env_batch(task, [tuple(pair)], 0, 64, cfg["k"], dev, g)[0]
            acc_env.append(float(per_demo_exact(model, tok, cfg["k"])[-1]))
        acc_env = np.array(acc_env)
        per_p = {NAMES[p]: float(acc_env[prims[p]].mean()) for p in range(len(NAMES))}
        held = float(np.mean([a for _p, _h, a in trials_to_criterion(model, task, test_pairs, cfg["k"], dev, g, n=128)]))
        return per_p, held

    def set_zero(mask):
        v = base_vec.clone()
        v[torch.as_tensor(mask, device=dev)] = 0.0
        i = 0
        with torch.no_grad():
            for p, s in zip(flat.params, flat.sizes):
                p.copy_(v[i:i + s].view_as(p))
                i += s

    set_zero(np.zeros(real.shape[0], dtype=bool))
    base_p, base_held = per_primitive_acc()
    ablation = []
    for r in regions:
        set_zero(lab == r["id"])
        pp, held = per_primitive_acc()
        drop = {p: base_p[p] - pp[p] for p in base_p}
        vals = np.array(list(drop.values()))
        top = int(np.argmax(vals))
        r["ablation_drop"] = {p: round(v, 3) for p, v in drop.items()}
        r["functional_selectivity"] = float(vals[top] - np.delete(vals, top).mean())
        r["ablation_primitive"] = list(drop)[top]
        r["ablation_held_drop"] = float(base_held - held)
        ablation.append(vals.tolist())
    set_zero(dead)
    _pp, held_dead = per_primitive_acc()
    set_zero(np.zeros(real.shape[0], dtype=bool))
    # random-membership control: a region-sized random set of live coordinates
    ctrl = []
    for r in regions[:3]:
        idx = rng.choice(np.flatnonzero(live), r["size"], replace=False)
        m = np.zeros(real.shape[0], dtype=bool)
        m[idx] = True
        set_zero(m)
        pp, _h = per_primitive_acc()
        vals = np.array([base_p[p] - pp[p] for p in base_p])
        ctrl.append(float(vals.max() - np.delete(vals, int(np.argmax(vals))).mean()))
    set_zero(np.zeros(real.shape[0], dtype=bool))

    # ── wiring ────────────────────────────────────────────────────────────────────────────────────────────────────
    snaps = tr["snaps"].float().numpy()                            # (S, P)
    steps = tr["snap_steps"]
    dW = np.abs(np.diff(snaps, axis=0))                            # (S-1, P)
    upd = np.stack([dW[:, lab == r["id"]].mean(1) for r in regions], 1)   # (S-1, R)
    co_update = np.corrcoef(upd.T) if upd.shape[0] > 2 else np.eye(len(regions))
    cum = np.cumsum(upd, 0) / (upd.sum(0)[None, :] + 1e-12)
    for j, r in enumerate(regions):
        r["formed_50pct_step"] = int(steps[1 + int(np.argmax(cum[:, j] >= 0.5))])
    # reads-from: attention heads' key-read subspaces vs earlier regions' write subspaces (top-3 singular vectors)
    reads = []
    W = {k: v.detach().cpu() for k, v in tr["model"].items()}
    d, H = cfg["d"], cfg["heads"]
    hd = d // H
    idx_of = {it["name"]: (it["start"], it["end"]) for it in layout}

    def write_dirs(region_mask, name, shape):
        a, b = idx_of[name]
        w = np.zeros(shape[0] * shape[1])
        sel = region_mask[a:b]
        w[sel] = W[name].numpy().reshape(-1)[sel]
        w = w.reshape(shape)
        if not np.any(w):
            return None
        u, s, _ = np.linalg.svd(w, full_matrices=False)          # columns of w are residual directions -> u
        return u[:, :3], float(s[:3].sum() / (s.sum() + 1e-12))
    for l in range(cfg["layers"]):
        qkv = W[f"blocks.{l}.attn.qkv.weight"].numpy()           # (3d, d)
        for h in range(H):
            k_rows = qkv[d + h * hd: d + (h + 1) * hd]              # (hd, d): reads residual directions = right sv
            _u, _s, vt = np.linalg.svd(k_rows, full_matrices=False)
            read = vt[:3].T                                        # (d, 3)
            for r in regions:
                m = lab == r["id"]
                best = 0.0
                for l2 in range(l):                                 # writers strictly earlier
                    for name, shape in ((f"blocks.{l2}.attn.proj.weight", (d, d)), (f"blocks.{l2}.mlp.2.weight", (d, 4 * d))):
                        wd = write_dirs(m, name, shape)
                        if wd is None:
                            continue
                        ov = float(np.linalg.norm(wd[0].T @ read) ** 2 / 3)
                        best = max(best, ov)
                if best > 0:
                    reads.append(dict(reader=f"L{l}.h{h}", region=r["id"], primitive=r["primitive"], overlap=round(best, 3)))
    reads.sort(key=lambda x: -x["overlap"])

    # ── telemetry over time ───────────────────────────────────────────────────────────────────────────────────────
    logs = [json.loads(l) for l in open(run_dir / "train.jsonl")]
    evals = [json.loads(l) for l in open(run_dir / "eval.jsonl")]
    trust_real = [float(np.mean(r["trust"][:E_real])) for r in logs]
    trust_noise = [float(np.mean(r["trust"][E_real:])) for r in logs] if P.shape[0] > E_real else None
    masked = [r.get("masked_frac") for r in logs]
    near_zero = float((np.abs(snaps[-1]) < 1e-3).mean())

    return dict(arm=cfg["arm"], params=cfg["params"], regions=regions, n_primitive_regions=n_prim_regions,
                mean_selectivity=float(np.mean([r["selectivity"] for r in regions])),
                mean_functional_selectivity=float(np.mean([r["functional_selectivity"] for r in regions])),
                control_functional_selectivity=float(np.mean(ctrl)), dead_frac=float(dead.mean()),
                near_zero_frac=near_zero, base_per_primitive=base_p, base_held=base_held, held_after_dead_ablation=held_dead,
                noise_participation=noise_part, real_participation=real_part, co_update=co_update.tolist(),
                reads_from=reads[:12], trust_real=trust_real, trust_noise=trust_noise, masked_frac=masked,
                evals=evals, steps=[r["step"] for r in logs], P=real, lab=lab)


def figures(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for res in results:
        arm = res["arm"]
        P, lab = res["P"], res["lab"]
        order = np.argsort(lab + (lab < 0) * 100)
        sub = order[:: max(1, len(order) // 4000)]
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.imshow(P[sub], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        bounds = np.cumsum([np.sum(lab[sub] == j) for j in range(K_REGIONS)])
        for b in bounds[:-1]:
            ax.axhline(b, color="k", lw=0.5)
        ax.set_xlabel("training composition (environment)")
        ax.set_ylabel("weight (sorted by region)")
        ax.set_title(f"{arm}: participation matrix, regions outlined")
        fig.tight_layout()
        fig.savefig(out / f"participation_{arm}.png", dpi=120)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        NAMES = list(res["regions"][0]["ablation_drop"])
        M = np.array([[r["ablation_drop"][p] for p in NAMES] for r in res["regions"]])
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(NAMES)))
        ax.set_xticklabels(NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(res["regions"])))
        ax.set_yticklabels([f"R{r['id']} ({r['primitive']})" for r in res["regions"]])
        ax.set_title(f"{arm}: accuracy drop when a region is removed")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out / f"ablation_{arm}.png", dpi=120)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for res in results:
        ev = res["evals"]
        axes[0].plot([e["step"] for e in ev], [e["held_last_acc"] for e in ev], marker="o", label=res["arm"])
        axes[1].plot([e["step"] for e in ev], [e["held_ttc_mean"] for e in ev], marker="o", label=res["arm"])
        if res["masked_frac"] and res["masked_frac"][0] is not None:
            axes[2].plot(res["steps"], res["masked_frac"], label=f"{res['arm']} masked")
        if res["trust_noise"]:
            axes[2].plot(res["steps"], res["trust_noise"], ls="--", label=f"{res['arm']} noise trust")
    axes[0].set_title("held-out accuracy (novel compositions)")
    axes[1].set_title("demonstrations to criterion (lower = better)")
    axes[2].set_title("inner objective: masked fraction / noise trust")
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = [analyse_run(r, rng) for r in args.runs]
    figures(results, out)
    slim = [{k: v for k, v in r.items() if k not in ("P", "lab")} for r in results]
    json.dump(slim, open(out / "inspect.json", "w"), indent=1)
    for r in slim:
        print(f"{r['arm']}: primitive regions {r['n_primitive_regions']}/{K_REGIONS} | mean selectivity "
              f"{r['mean_selectivity']:.3f} | functional selectivity {r['mean_functional_selectivity']:.3f} "
              f"(random-set control {r['control_functional_selectivity']:.3f}) | dead {100*r['dead_frac']:.1f}% "
              f"near-zero {100*r['near_zero_frac']:.1f}% | noise participation {r['noise_participation']} vs real "
              f"{r['real_participation']:.3f}")


if __name__ == "__main__":
    main()
