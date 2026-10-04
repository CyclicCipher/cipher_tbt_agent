"""Markdown tables from the benchmark's JSON results (planning doc §11).

    python experiments/neural_turing_architecture/search_bench/summarize.py
"""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def load(*names):
    res = []
    for n in names:
        res += json.load(open(RUNS / n))["results"]
    return res


def rate(res, **kw):
    r = [x["solved"] for x in res if all((x["cfg"].get(k) if k in x["cfg"] else x.get(k)) == v for k, v in kw.items())]
    return (sum(r) / len(r), len(r)) if r else (None, 0)


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def dimension_table():
    res = load("gridA_loose.json", "gridA_tight.json", "gridA_hybrid.json")
    algos = ["random_shooting", "cem", "smc", "look_iso", "mcts_iso", "look_randsub", "look_oracle", "grad_greedy",
             "smc_grad", "grad_trajopt", "look_langevin", "mcts_grad", "mcts_langevin", "mcts_guided", "mcts_hybrid"]
    ds = [16, 64, 256, 1024]
    head = "| method | " + " | ".join(f"loose d={d}" for d in ds) + " | " + " | ".join(f"tight d={d}" for d in ds) + " |"
    print(head)
    print("|" + "---|" * (1 + 2 * len(ds)))
    for a in algos:
        cells = []
        for beta in (16.0, 8.0):
            for d in ds:
                lo, _ = rate(res, algo=a, d=d, beta=beta, budget=512)
                hi, _ = rate(res, algo=a, d=d, beta=beta, budget=4096)
                cells.append(f"{fmt(lo)} / {fmt(hi)}")
        print(f"| `{a}` | " + " | ".join(cells) + " |")


def robustness_table():
    res = load("gridB_loose.json", "gridB_tight.json")
    algos = ["look_iso", "mcts_iso", "grad_greedy", "smc_grad", "grad_trajopt", "mcts_langevin", "mcts_guided", "mcts_hybrid"]
    conds = [(0.1, 0.0, 0.0, "baseline"), (0.1, 0.0, 0.5, "exploitable V"), (0.1, 0.3, 0.0, "prior cos 0.3"),
             (0.3, 0.0, 0.0, "V noise 0.3"), (0.3, 0.3, 0.0, "V noise 0.3 + prior")]
    print("| method | " + " | ".join(f"loose: {c[3]}" for c in conds) + " | " + " | ".join(f"tight: {c[3]}" for c in conds) + " |")
    print("|" + "---|" * (1 + 2 * len(conds)))
    for a in algos:
        cells = []
        for beta in (16.0, 8.0):
            for nz, pc, hk, _ in conds:
                lo, _ = rate(res, algo=a, beta=beta, value_noise=nz, prior_cos=pc, value_hack=hk, budget=512)
                hi, _ = rate(res, algo=a, beta=beta, value_noise=nz, prior_cos=pc, value_hack=hk, budget=4096)
                cells.append(f"{fmt(lo)} / {fmt(hi)}")
        print(f"| `{a}` | " + " | ".join(cells) + " |")


def budget_table():
    res = load("gridC_budget.json")
    algos = ["grad_greedy", "smc_grad", "grad_trajopt", "mcts_langevin", "mcts_guided", "mcts_hybrid"]
    bs = [128, 256, 512, 1024, 2048]
    print("| method | " + " | ".join(f"loose B={b}" for b in bs) + " | " + " | ".join(f"tight B={b}" for b in bs) + " |")
    print("|" + "---|" * (1 + 2 * len(bs)))
    for a in algos:
        cells = [fmt(rate(res, algo=a, beta=beta, budget=b)[0]) for beta in (16.0, 8.0) for b in bs]
        print(f"| `{a}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    print("## dimension (value noise 0.1, uninformed prior; B=512 / B=4096; 60 instances per cell)\n")
    dimension_table()
    print("\n## robustness at d=256 (B=512 / B=4096; 40 instances per cell)\n")
    robustness_table()
    print("\n## budget at d=256 (value noise 0.1, uninformed prior; 40 instances per cell)\n")
    budget_table()
