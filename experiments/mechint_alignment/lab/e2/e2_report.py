"""E2 report: per-rule 'seen to 90%' and final accuracy for one or more runs (python -m lab.e2.e2_report runs/e2/A ...)."""
import json
import sys

from worlds.village_events import EVENTS, UNCHECKABLE


def rows(run):
    return [json.loads(x) for x in open(f"{run}/log.jsonl")]


def seen_to(rs, e, thr=0.9):
    for r in rs:
        if e in r and r[e]["acc"] >= thr:
            return r[e]["seen"]
    return None


def main(runs):
    data = {run: rows(run) for run in runs}
    names = [r.split("/")[-1] for r in runs]
    print("rule".ljust(16) + "".join(n[:22].rjust(24) for n in names))
    print(" " * 16 + "".join("seen->90% | final acc".rjust(24) for _ in names))
    for e in EVENTS:
        if e in UNCHECKABLE:
            continue
        line = e.ljust(16)
        for run in runs:
            rs = data[run]
            s = seen_to(rs, e)
            line += f"{'-' if s is None else s:>12} | {rs[-1][e]['acc']:.2f} ({rs[-1][e]['seen']})".rjust(24)
        print(line)
    for key in ("bits_per_transition", "changed_acc", "unchanged_acc"):
        print(key.ljust(16) + "".join(f"{data[r][-1]['_overall'][key]:.3f}".rjust(24) for r in runs))
    print("train seconds".ljust(16) + "".join(f"{data[r][-1]['train_seconds']:.0f}".rjust(24) for r in runs))


if __name__ == "__main__":
    main(sys.argv[1:])


def ratios(base, other, rules=None):
    """Median over rules learned (>= 90%) by both of seen_to(base) / seen_to(other); and the rules only `other` learned."""
    b, o = rows(base), rows(other)
    rs = rules or [e for e in EVENTS if e not in UNCHECKABLE]
    both = [(e, seen_to(b, e), seen_to(o, e)) for e in rs]
    shared = [(e, x, y) for e, x, y in both if x and y]
    only_o = [e for e, x, y in both if y and not x]
    only_b = [e for e, x, y in both if x and not y]
    import numpy as np
    med = float(np.median([x / y for _, x, y in shared])) if shared else None
    return dict(median_ratio=med, shared=shared, only_other=only_o, only_base=only_b)
