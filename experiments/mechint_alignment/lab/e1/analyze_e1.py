"""E1 post-hoc analysis: information curves from the saved checkpoints, and the efficiency table (EXPERIMENT_E1.md).

    python -m lab.e1.analyze_e1 --run runs/e1

For every checkpoint, on two probe sets (the scripted reference's states, and the FINAL policy's own states on
held-out layouts): entropy, the information rate I(S;A), and KL(final || checkpoint) in bits per decision. The bits
of the final behaviour acquired by step t = KL(final || init) - KL(final || t).
Writes summary.json, summary.md and e1.png into the run folder.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from lab.e1 import info
from nets.policy import Policy
from worlds.episodes import PROBE_SEEDS, net_actor, run_episodes
from worlds.village import SPEC


def read_log(run):
    train, evals = {}, {}
    with open(os.path.join(run, "log.jsonl")) as f:
        for line in f:
            x = json.loads(line)
            (train if x["type"] == "train" else evals)[x.get("update", x["step"])] = x   # resumes: keep the last
    return [train[k] for k in sorted(train)], [evals[k] for k in sorted(evals)]


def load(path, cfg, dev):
    net = Policy(SPEC, cfg["d"], cfg["layers"], cfg["heads"]).to(dev)
    net.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    return net.eval()


def first_step(points, threshold):
    """The first (step, row) whose value reaches threshold, from (step, value, row) triples."""
    for step, value, row in points:
        if value is not None and value >= threshold:
            return step, row
    return None, None


def thirds(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None
    k = max(1, len(xs) // 3)
    return dict(early=float(np.median(xs[:k])), middle=float(np.median(xs[k:-k] or xs)), late=float(np.median(xs[-k:])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    ref = json.load(open(os.path.join(a.run, "reference.json")))
    train, evals = read_log(a.run)
    ckpts = sorted(glob.glob(os.path.join(a.run, "ckpt_*.pt")))
    steps = [int(os.path.basename(c)[5:15]) for c in ckpts]
    final = load(ckpts[-1], cfg, dev)
    probe = dict(np.load(os.path.join(a.run, "probe.npz")))
    _, (fc, fg, _) = run_episodes(PROBE_SEEDS, net_actor(final, dev, seed=1), collect=True)
    sets = {"reference_states": (probe["cells"], probe["glob"]), "final_states": (fc, fg)}
    final_lp = {k: info.log_probs(final, dev, *v) for k, v in sets.items()}

    curve = []
    for step, path in zip(steps, ckpts):
        net = load(path, cfg, dev)
        row = dict(step=step)
        for k, v in sets.items():
            lp = info.log_probs(net, dev, *v)
            row[k] = dict(KL_final_bits=info.kl_bits(final_lp[k], lp), **info.policy_info(lp))
        curve.append(row)
    for k in sets:
        kl0 = curve[0][k]["KL_final_bits"]
        for row in curve:
            row[k]["acquired_bits"] = kl0 - row[k]["KL_final_bits"]
            row[k]["acquired_frac"] = row[k]["acquired_bits"] / kl0 if kl0 > 0 else None

    final_ret, ret0 = evals[-1]["ret"], evals[0]["ret"]
    ret_points = [(e["step"], e["ret"], e) for e in evals]
    milestones = {}
    for name, target in (("final", final_ret), ("reference", ref["ret"])):
        for frac in (0.5, 0.8, 0.9, 1.0):      # progress from the initial (random) policy's return to the target
            level = ret0 + frac * (target - ret0)
            step, row = first_step(ret_points, level) if target > ret0 else (None, None)
            milestones[f"{frac:.0%} of the gain to the {name} return"] = None if row is None else dict(
                step=step, hours=row["wall"] / 3600, flops=row["flops"], reward_bits=row["reward_bits_cum"],
                reward_events=row["reward_events_cum"])
    acq = [(r["step"], r["final_states"]["acquired_frac"], r) for r in curve]
    for frac in (0.5, 0.9):
        step, _ = first_step(acq, frac)
        milestones[f"{frac:.0%} of final behaviour bits (final states)"] = step

    gns = [(t["step"], t["gns"]) for t in train if t.get("gns")]
    summary = dict(
        run=a.run, steps=evals[-1]["step"], hours=evals[-1]["wall"] / 3600, flops=evals[-1]["flops"],
        n_params=cfg["n_params"], reference=ref, final_eval=evals[-1], milestones=milestones,
        bits_per_decision=dict(
            final_vs_init_final_states=curve[0]["final_states"]["KL_final_bits"],
            final_vs_init_reference_states=curve[0]["reference_states"]["KL_final_bits"],
            final_I_SA_final_states=curve[-1]["final_states"]["I_SA_bits"],
            final_H_final_states=curve[-1]["final_states"]["H_bits"]),
        reward_channel=dict(bits_total=evals[-1]["reward_bits_cum"], events_total=evals[-1]["reward_events_cum"],
                            bits_per_step=evals[-1]["reward_bits_cum"] / max(1, evals[-1]["step"])),
        gns_policy_B_simple=thirds([g["pi"]["B_simple"] for _, g in gns]),
        gns_value_B_simple=thirds([g["v"]["B_simple"] for _, g in gns]),
        gns_policy_bits_per_sample=thirds([g["pi"]["bits_per_sample"] for _, g in gns]),
        curve=curve)
    with open(os.path.join(a.run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    lines = [f"# E1 summary: {a.run}", "",
             f"- {summary['steps']:,} env steps, {summary['hours']:.2f} h, {summary['flops']:.3g} FLOPs, "
             f"{cfg['n_params']:,} parameters",
             f"- final return {final_ret:.2f}; scripted reference {ref['ret']:.2f}",
             f"- final behaviour: theft rate {evals[-1]['theft_rate']}, button disabled {evals[-1]['button_disabled_rate']}, "
             f"rescue rate {evals[-1]['rescue_rate']}, robot died {evals[-1]['robot_died']}",
             f"- bits per decision, final vs init: {summary['bits_per_decision']}",
             f"- reward channel: {summary['reward_channel']}",
             f"- gradient noise scale B_simple (policy): {summary['gns_policy_B_simple']}; "
             f"(value): {summary['gns_value_B_simple']}", "", "| milestone | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in milestones.items()]
    open(os.path.join(a.run, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot([e["step"] for e in evals], [e["ret"] for e in evals], label="PPO (held-out)")
    ax[0, 0].axhline(ref["ret"], ls="--", c="gray", label="scripted reference")
    ax[0, 0].set(title="return", xlabel="env steps")
    ax[0, 0].legend()
    for k, style in (("final_states", "-"), ("reference_states", ":")):
        ax[0, 1].plot([r["step"] for r in curve], [r[k]["acquired_bits"] for r in curve], style, label=f"acquired ({k})")
        ax[0, 1].plot([r["step"] for r in curve], [r[k]["I_SA_bits"] for r in curve], style, label=f"I(S;A) ({k})")
    ax[0, 1].set(title="bits per decision", xlabel="env steps")
    ax[0, 1].legend(fontsize=7)
    ax[1, 0].plot([e["reward_bits_cum"] for e in evals], [e["ret"] for e in evals])
    ax[1, 0].set(title="return vs reward-channel bits consumed", xlabel="cumulative reward bits", xscale="symlog")
    ax[1, 1].plot([s for s, _ in gns], [g["pi"]["B_simple"] for _, g in gns], label="policy loss")
    ax[1, 1].plot([s for s, _ in gns], [g["v"]["B_simple"] for _, g in gns], label="value loss")
    ax[1, 1].axhline(cfg["minibatch"], ls="--", c="gray", label="minibatch")
    ax[1, 1].set(title="gradient noise scale B_simple", xlabel="env steps", yscale="log")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.run, "e1.png"), dpi=110)


if __name__ == "__main__":
    main()
