"""
diagnostics.py - Analysis and diagnostics for GrokkingMamba3 experiments.

Usage:
    # Compare two saved runs
    python diagnostics.py --runs baseline/ growable/ --labels Baseline GrowableLinear

    # Analyze a single run in detail
    python diagnostics.py --runs run1/ --detail

    # Run all five ablation conditions sequentially and compare
    python diagnostics.py --ablation

Input: a directory containing log.json (and optionally model.pt) from train.py.

All outputs are printed to stdout (no matplotlib dependency required).
For plotting, set --plot to write SVG files (requires matplotlib).
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_log(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "log.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Grokking analysis
# ---------------------------------------------------------------------------

def grokking_step(log: Dict, threshold: float = 0.95) -> Optional[int]:
    """Return first step at which test accuracy exceeds threshold."""
    for step, acc in zip(log["step"], log["test_acc"]):
        if acc >= threshold:
            return step
    return None


def generalization_gap_at(log: Dict, step: int) -> float:
    """Train accuracy - test accuracy at a given step."""
    try:
        idx = log["step"].index(step)
        return log["train_acc"][idx] - log["test_acc"][idx]
    except ValueError:
        return float("nan")


def area_under_gap(log: Dict) -> float:
    """Area under the (train_acc - test_acc) curve -- lower is better."""
    gaps  = [tr - te for tr, te in zip(log["train_acc"], log["test_acc"])]
    steps = log["step"]
    if len(steps) < 2:
        return float("nan")
    return float(np.trapz(gaps, steps))


def memorization_onset(log: Dict, threshold: float = 0.95) -> Optional[int]:
    """First step at which train accuracy exceeds threshold."""
    for step, acc in zip(log["step"], log["train_acc"]):
        if acc >= threshold:
            return step
    return None


# ---------------------------------------------------------------------------
# Rank analysis (for GrowableLinear runs)
# ---------------------------------------------------------------------------

def rank_timeline(log: Dict) -> List[Dict]:
    """Return list of (step, {name: rank}) for all eval steps."""
    return list(zip(log["step"], log["ranks"]))


def nuclear_norm_timeline(log: Dict) -> List[Dict]:
    """Return list of (step, {name: norm}) for all eval steps."""
    return list(zip(log["step"], log["nuclear_norms"]))


def growth_events(log: Dict) -> List[Dict]:
    return log.get("growth_events", [])


# ---------------------------------------------------------------------------
# Singular value analysis (requires model.pt)
# ---------------------------------------------------------------------------

def load_model_singular_values(run_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Load model.pt and extract singular values of GrowableLinear modules.
    Returns {layer_name: sorted_singular_values} or None if not available."""
    ckpt_path = os.path.join(run_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        return None

    try:
        import torch
        sys.path.insert(0, os.path.join(_HERE, '..', '..', 'Mamba3'))
        from grokking_mamba3 import GrokkingConfig, GrokkingMamba3LM

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        config = GrokkingConfig(**{k: v for k, v in cfg_dict.items()
                                   if k not in ('d_inner', 'nheads', 'd_bc')})
        model = GrokkingMamba3LM(config)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.singular_values()
    except Exception as e:
        print(f"  (could not load singular values: {e})")
        return None


# ---------------------------------------------------------------------------
# Alignment loss analysis
# ---------------------------------------------------------------------------

def alignment_loss_timeline(log: Dict) -> List:
    return list(zip(log["step"], log["align_loss"]))


# ---------------------------------------------------------------------------
# Text reporting
# ---------------------------------------------------------------------------

def report_single(log: Dict, label: str = "", run_dir: str = "") -> None:
    """Print a detailed diagnostic report for one run."""
    steps = log["step"]
    if not steps:
        print(f"  {label}: no data")
        return

    grok   = grokking_step(log)
    mem    = memorization_onset(log)
    auc    = area_under_gap(log)
    final_tr = log["train_acc"][-1]
    final_te = log["test_acc"][-1]
    max_te   = max(log["test_acc"])

    print(f"\n{'='*60}")
    print(f"Run: {label or run_dir}")
    print(f"{'='*60}")
    print(f"  Total steps          : {steps[-1]}")
    print(f"  Grokking step        : {grok if grok else 'NOT REACHED'}")
    print(f"  Memorization step    : {mem if mem else 'NOT REACHED'}")
    if grok and mem:
        print(f"  Grok lag (grok-mem)  : {grok - mem} steps")
    print(f"  Final train acc      : {final_tr*100:.1f}%")
    print(f"  Final test  acc      : {final_te*100:.1f}%")
    print(f"  Peak  test  acc      : {max_te*100:.1f}%")
    print(f"  Area under gap       : {auc:.1f} (lower = less memorization)")

    # Growth events
    gevs = growth_events(log)
    if gevs:
        print(f"\n  Growth events ({len(gevs)} total):")
        for ev in gevs:
            print(f"    step {ev['step']}: {ev['n_grew']} module(s) grew")

    # Alignment loss
    al_timeline = alignment_loss_timeline(log)
    if any(v > 0 for _, v in al_timeline):
        init_al  = al_timeline[0][1]  if al_timeline else 0.0
        final_al = al_timeline[-1][1] if al_timeline else 0.0
        print(f"\n  Alignment loss: init={init_al:.4f}  final={final_al:.4f}")

    # Nuclear norms
    nn_timeline = nuclear_norm_timeline(log)
    if nn_timeline and nn_timeline[0][1]:
        print(f"\n  Nuclear norms (per layer):")
        # Report first, middle, last checkpoint
        for desc, entry in [("init", nn_timeline[0]),
                             ("mid",  nn_timeline[len(nn_timeline)//2]),
                             ("final",nn_timeline[-1])]:
            step, norms = entry
            norm_str = "  ".join(f"{n:.3f}" for n in norms.values())
            print(f"    {desc:>5} (step {step:>6}): {norm_str}")

    # Singular values from model checkpoint
    if run_dir:
        svs = load_model_singular_values(run_dir)
        if svs:
            print(f"\n  Final singular values (GrowableLinear):")
            for name, sv in svs.items():
                sorted_sv = sorted(sv, reverse=True)
                top5 = "  ".join(f"{v:.4f}" for v in sorted_sv[:5])
                print(f"    {name}: rank={len(sv)}  top5=[{top5}]")

    # Loss curves (ASCII plot)
    print(f"\n  Accuracy curves (every ~10 eval steps):")
    _ascii_dual_curve(log["step"], log["train_acc"], log["test_acc"],
                      label_a="train", label_b="test", width=50, height=8)


def _ascii_dual_curve(
    steps: List[int],
    y_a:   List[float],
    y_b:   List[float],
    label_a: str = "A",
    label_b: str = "B",
    width:   int = 60,
    height:  int = 8,
) -> None:
    """Print a simple ASCII dual-line chart."""
    if not steps:
        return
    # Subsample to width points
    n = len(steps)
    idx = [int(i * (n-1) / max(width-1, 1)) for i in range(width)]
    xs = [steps[i] for i in idx]
    ya = [y_a[i]   for i in idx]
    yb = [y_b[i]   for i in idx]

    ymin, ymax = 0.0, 1.0
    rows = []
    for row in range(height - 1, -1, -1):
        threshold = ymin + (ymax - ymin) * row / (height - 1)
        line = f"  {threshold:.1f} |"
        for a, b in zip(ya, yb):
            if abs(a - threshold) < (ymax - ymin) / (2 * height) and \
               abs(b - threshold) < (ymax - ymin) / (2 * height):
                line += "*"
            elif abs(a - threshold) < (ymax - ymin) / (2 * height):
                line += "T"   # train
            elif abs(b - threshold) < (ymax - ymin) / (2 * height):
                line += "t"   # test
            else:
                line += " "
        rows.append(line)

    for r in rows:
        print(r)
    print(f"       +{''.join('-' for _ in range(width))}")
    print(f"        step {xs[0]}{'':>{width-10}}step {xs[-1]}")
    print(f"        T={label_a}  t={label_b}  *=both")


def report_comparison(
    logs:    List[Dict],
    labels:  List[str],
    run_dirs: List[str],
) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    header = f"{'Condition':<28}  {'Grok step':>10}  {'Mem step':>9}  {'Grok lag':>9}  {'Final tr':>9}  {'Final te':>9}  {'Peak te':>8}"
    print(header)
    print("-" * len(header))

    for log, label in zip(logs, labels):
        grok  = grokking_step(log)
        mem   = memorization_onset(log)
        lag   = (grok - mem) if (grok and mem) else None
        final_tr = log["train_acc"][-1] * 100 if log["train_acc"] else float("nan")
        final_te = log["test_acc"][-1]  * 100 if log["test_acc"]  else float("nan")
        peak_te  = max(log["test_acc"])  * 100 if log["test_acc"]  else float("nan")

        grok_str = str(grok)  if grok  else "---"
        mem_str  = str(mem)   if mem   else "---"
        lag_str  = str(lag)   if lag   else "---"
        print(f"  {label:<26}  {grok_str:>10}  {mem_str:>9}  {lag_str:>9}  "
              f"{final_tr:>8.1f}%  {final_te:>8.1f}%  {peak_te:>7.1f}%")

    print()
    print("Interpretation:")
    print("  Grok step  : first step where test acc > 95% (lower is better)")
    print("  Grok lag   : grok_step - mem_step (lower = less memorization phase)")
    print("  Final te   : test accuracy at end of training")


# ---------------------------------------------------------------------------
# Ablation runner (calls train.py with different configs)
# ---------------------------------------------------------------------------

def run_ablation(base_args: List[str], save_root: str = "ablation_results") -> None:
    """Run all five ablation conditions and save results."""
    import subprocess

    conditions = [
        ("baseline",          []),
        ("nuclear_only",      ["--nuclear", "0.001"]),
        ("growable_only",     ["--growable"]),
        ("alignment_only",    ["--alignment", "0.1"]),
        ("all_combined",      ["--growable", "--nuclear", "0.001", "--alignment", "0.1"]),
    ]

    python = sys.executable
    script = os.path.join(_HERE, "train.py")

    for name, extra_args in conditions:
        save_dir = os.path.join(save_root, name)
        cmd = [python, script] + base_args + extra_args + ["--save", save_dir]
        print(f"\n{'='*60}")
        print(f"Running condition: {name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)

    print("\n\nAll conditions done. Loading and comparing...")
    logs   = []
    labels = []
    dirs   = []
    for name, _ in conditions:
        d = os.path.join(save_root, name)
        if os.path.exists(os.path.join(d, "log.json")):
            logs.append(load_log(d))
            labels.append(name)
            dirs.append(d)

    if logs:
        report_comparison(logs, labels, dirs)
        for log, label, d in zip(logs, labels, dirs):
            report_single(log, label=label, run_dir=d)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GrokkingMamba3 diagnostics and comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs",   nargs="+", default=[],
                        help="One or more run directories to analyze")
    parser.add_argument("--labels", nargs="+", default=[],
                        help="Labels for each run (same order as --runs)")
    parser.add_argument("--detail", action="store_true",
                        help="Print detailed per-run report (else just comparison)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run all five ablation conditions sequentially")
    parser.add_argument("--ablation_args", nargs="*", default=[],
                        help="Extra args passed to train.py for ablation runs "
                             "(e.g. --steps 10000 --p 97)")
    parser.add_argument("--save_root", type=str, default="ablation_results",
                        help="Root directory for ablation outputs")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(args.ablation_args, save_root=args.save_root)
        return

    if not args.runs:
        print("Provide at least one --runs DIR.  Use --ablation to run all conditions.")
        parser.print_help()
        return

    labels = args.labels if args.labels else [os.path.basename(r.rstrip("/\\")) or r
                                               for r in args.runs]
    logs   = [load_log(r) for r in args.runs]

    if len(logs) > 1:
        report_comparison(logs, labels, args.runs)

    if args.detail or len(logs) == 1:
        for log, label, run_dir in zip(logs, labels, args.runs):
            report_single(log, label=label, run_dir=run_dir)


if __name__ == "__main__":
    main()
