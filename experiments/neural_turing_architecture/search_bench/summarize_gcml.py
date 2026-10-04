"""Markdown tables from the GCML tests (planning doc §12–§13).

    python experiments/neural_turing_architecture/search_bench/summarize_gcml.py
"""
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
FILES = ["gcml_V0_V1raw.json", "gcml_V1sr_V2coord.json", "gcml_V2sr.json"]
VARS = ["V0", "V1raw", "V1sr", "V2coord", "V2sr"]
CODE = {"V0": "raw", "V1raw": "raw", "V1sr": "sr", "V2coord": "coord", "V2sr": "sr"}


def load():
    W = []
    for f in FILES:
        W += json.load(open(RUNS / f))["worlds"]
    return W


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def g0(W):
    print("| variant | its code | invariance of its code | raw | sr | coord |\n|---|---|---|---|---|---|")
    for v in VARS:
        ws = [w for w in W if w["variant"] == v]
        inv = {c: mean([w["invariance"][c] for w in ws if c in w["invariance"]]) for c in ("raw", "sr", "coord")}
        f = lambda c: "—" if inv[c] != inv[c] else f"{inv[c]:.2f}"
        print(f"| {v} | {CODE[v]} | **{f(CODE[v])}** | {f('raw')} | {f('sr')} | {f('coord')} |")


def g1(W):
    """Signal strength (probe_gcml.py): mean ‖Δc‖ of a transition relative to a demonstration's full move."""
    rows = json.load(open(RUNS / "gcml_probe.json"))
    print("| d | random thoughts | search transitions |\n|---|---|---|")
    for d in (64, 256):
        rs = [r for r in rows if r["d"] == d]
        lo = lambda k: min(r[k] for r in rs)  # noqa: E731
        hi = lambda k: max(r[k] for r in rs)  # noqa: E731
        print(f"| {d} | {lo('random_signal'):.2f}–{hi('random_signal'):.2f} | {lo('search_signal'):.2f}–{hi('search_signal'):.2f} |")


def probes():
    rows = json.load(open(RUNS / "gcml_probe.json"))
    ks = json.load(open(RUNS / "gcml_probe_kstep.json"))
    print("| variant | d | G0 invariance (its code) | 1-step predictability on search transitions | k-step probe k=1 | k=2 | k=3 |")
    print("|---|---|---|---|---|---|---|")
    W = load()
    for v in VARS:
        for d in (64, 256):
            inv = mean([w["invariance"][CODE[v]] for w in W if w["variant"] == v and w["d"] == d])
            pr = mean([r["predictability"] for r in rows if r["variant"] == v and r["d"] == d])
            kk = [mean([r[f"k{k}"] for r in ks if r["variant"] == v and r["d"] == d]) for k in (1, 2, 3)]
            print(f"| {v} | {d} | {inv:.2f} | {pr:.2f} | " + " | ".join(f"{x:.2f}" for x in kk) + " |")


ROWS = [("random thought", "random thought"),
        ("value gradient ∇_z V (cost 3)", "value gradient ∇_z V — cost 3"),
        ("exact inverse J⁺·(goal−state) (reference)", "exact local inverse J⁺·(goal − state) — reference"),
        ("W[random,ridge]·(goal−state)", "W from random thoughts (ridge) · (goal − state)"),
        ("W[random,hebb]·(goal−state)", "W from random thoughts (Hebbian, GCML's rule) · (goal − state)"),
        ("W[search,hebb]·(goal−state)", "W from search experience (Hebbian) · (goal − state)"),
        ("W[search,ridge]·(goal−state)", "W from search experience (ridge) · (goal − state)"),
        ("W[search,ridge]·∇V", "W from search experience (ridge) · ∇_c V — no goal"),
        ("W[search,mlp]·(goal−state)", "W(c) network from search experience · (goal − state)"),
        ("W[search,mlp]·∇V", "W(c) network from search experience · ∇_c V — no goal"),
        ("W[demo,ridge]·(goal−state)", "W from demonstrations (ridge) · (goal − state)"),
        ("W[demo,mlp]·(goal−state)", "W(c) network from demonstrations · (goal − state)")]


def g2(W, metric="progress"):
    head = "| proposal | " + " | ".join(f"{v} d={d}" for v in VARS for d in (64, 256)) + " |"
    print(head + "\n|" + "---|" * (1 + 2 * len(VARS)))
    for key, label in ROWS:
        cells = []
        for v in VARS:
            for d in (64, 256):
                ws = [w for w in W if w["variant"] == v and w["d"] == d]
                cells.append(f"{mean([w['direction'][key + '|' + metric] for w in ws]):.2f}")
        print(f"| {label} | " + " | ".join(cells) + " |")


def g1_curve(W):
    print("| variant d | random N=256 | random N=1024 | random all (4096) | search N=256 | search N=1024 | search all |")
    print("|" + "---|" * 7)
    for v in VARS:
        for d in (64, 256):
            ws = [w for w in W if w["variant"] == v and w["d"] == d]
            cells = []
            for s in ("random", "search"):
                for n in (256, 1024, "all"):
                    key = f"W[{s},ridge]·(goal−state)|progress" if n == "all" else f"W[{s},ridge@{n}]·(goal−state)|progress"
                    vals = [w["direction"].get(key) for w in ws]
                    vals = [x for x in vals if x is not None]
                    cells.append(f"{mean(vals):.2f}" if vals else "—")
            print(f"| {v} d={d} | " + " | ".join(cells) + " |")


ALG = ["grad_greedy", "mcts_hybrid", "mcts_guided", "gcml_greedy", "gcml_greedy_v", "invnet_greedy", "invnet_greedy_v",
       "mcts_gcml", "mcts_gcml_v", "mcts_invnet", "mcts_invnet_v", "mcts_hybrid_gcml"]


def g3(W, d, budgets=(64, 256, 1024)):
    print("| algorithm | " + " | ".join(f"{v}" for v in VARS) + " |")
    print("|" + "---|" * (1 + len(VARS)))
    for a in ALG:
        cells = []
        for v in VARS:
            rs = [r for w in W if w["variant"] == v and w["d"] == d for r in w["search"] if r["algo"] == a]
            cells.append(" / ".join(f"{mean([r['solved'] for r in rs if r['budget'] == b]):.2f}" for b in budgets))
        print(f"| `{a}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    W = load()
    print(f"{len(W)} worlds\n\n## G0 action-effect invariance (mean over worlds and d)\n")
    g0(W)
    print("\n## G1 signal strength: mean ‖Δc‖ relative to a full move (range over variants and worlds)\n")
    g1(W)
    print("\n## preconditions: which probe predicts usefulness?\n")
    probes()
    print("\n## G1 ridge W vs amount of data: one-step progress\n")
    g1_curve(W)
    print("\n## G2 direction quality: one-step progress from every node of 5 held-out problems per world\n")
    g2(W)
    for d in (64, 256):
        print(f"\n## G3/G4 search success, d={d}, budget 64 / 256 / 1024 (40 held-out problems per cell)\n")
        g3(W, d)
