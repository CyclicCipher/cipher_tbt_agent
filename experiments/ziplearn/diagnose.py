"""E25 — what would hindsight have to learn? A diagnostic over the player's step traces on the replica games.

Reads the traces written by `play_games.py --trace 1` and, per game and level, decomposes the actions spent into what
they were spent on, classifies every wrong prediction by what went wrong in it, and prints what the player was
thinking at a few of its mistakes: the frame, the plan it was following and why, what it expected, what it got.

Accounting per level (every action falls into exactly one class):
  discovery      actions before any goal-directed plan existed (exploring for the win)
  goal-directed  actions inside a plan the goal model produced
  exploratory    actions inside an exploration plan after a goal was known
  blind          no plan at all
Prediction errors, classified by cell:
  missed change    a cell changed and was predicted unchanged   -- with the cell's window unknown, or known
  spurious change  a cell was predicted to change and did not
  wrong colour     a cell changed to a colour other than the predicted one
Also: no-effect actions (the frame did not change), revisits (the action led to a frame seen before), goal keys
refuted (a goal-directed plan's last action did not score).

    python experiments/ziplearn/diagnose.py --runs runs/e25            -> runs/e25/diagnosis.json, printed report
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GLYPH = {0: ".", 1: "#", 2: "A", 3: "G", 4: "k", 5: "D", 6: "B", 7: "P", 8: "S", 9: "X", 10: "*", 11: "+", 12: "o", 13: "x", 14: "%", 15: "@"}


def show(grid):
    return "\n".join("".join(GLYPH.get(int(v), "?") for v in row) for row in grid)


def diff_cells(a, b):
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        return None
    return list(zip(*np.nonzero(a != b)))


def analyse(trace):
    by_level = {}
    for r in trace:
        by_level.setdefault(r["level"], []).append(r)
    out = {}
    for level, recs in sorted(by_level.items()):
        spent = Counter()
        seen_goal = False
        errors = Counter()
        wrong_windows = Counter()
        no_effect = revisits = refuted = 0
        examples = []
        for r in recs:
            kind = (r["choice"] or {}).get("kind", "blind")
            if kind == "goal":
                seen_goal = True
            if kind in ("goal", "continuing") and seen_goal and r.get("fired_win", False):
                spent["goal-directed"] += 1
            elif not seen_goal:
                spent["discovery"] += 1
            elif kind == "blind":
                spent["blind"] += 1
            else:
                spent["exploratory"] += 1
            if "observed" not in r:                                  # a win: the after-frame is another level
                continue
            frame, obs = np.array(r["frame"]), np.array(r["observed"])
            if obs.shape == frame.shape and np.array_equal(frame, obs):
                no_effect += 1
            if r.get("visited_before"):
                revisits += 1
            if r["predicted"] is not None:
                pred = np.array(r["predicted"])
                if pred.shape == obs.shape and not np.array_equal(pred, obs):
                    changed = set(diff_cells(frame, obs) or [])
                    pchanged = set(diff_cells(frame, pred) or [])
                    missed = changed - pchanged
                    spurious = pchanged - changed
                    both = changed & pchanged
                    wrong_col = {c for c in both if pred[c] != obs[c]}
                    errors["missed change"] += len(missed)
                    errors["spurious change"] += len(spurious)
                    errors["wrong colour"] += len(wrong_col)
                    if r.get("unknown"):
                        wrong_windows["unknown windows in the frame"] += 1
                    else:
                        wrong_windows["all windows known"] += 1
                    if len(examples) < 2 and (missed or wrong_col):
                        examples.append(dict(step=r["step"], action=r["action"], choice=r["choice"], unknown=r.get("unknown"),
                                             frame=show(frame), predicted=show(pred), observed=show(obs),
                                             missed=[tuple(int(x) for x in c) for c in sorted(missed)][:6],
                                             spurious=[tuple(int(x) for x in c) for c in sorted(spurious)][:6]))
            if r.get("fired_win") and r["outcome"] != "win":
                refuted += 1
        n = len(recs)
        distinct = len({str(r["frame"]) for r in recs})                  # coverage: frames actually stood in
        wrong_preds = sum(wrong_windows.values())
        with_pred = sum(1 for r in recs if r["predicted"] is not None and "observed" in r)
        out[level] = dict(actions=n, distinct_frames=distinct, spent=dict(spent), prediction_errors=dict(errors), wrong_predictions=wrong_preds,
                          predictions=with_pred, wrong_by_knowledge=dict(wrong_windows), no_effect=no_effect,
                          revisits=revisits, goal_keys_refuted=refuted, won=any(r["outcome"] == "win" for r in recs),
                          examples=examples)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs" / "e25"))
    ap.add_argument("--examples", type=int, default=1, help="how many 'what was it thinking' examples to print per failed level")
    args = ap.parse_args()
    runs = Path(args.runs)
    report = {}
    for path in sorted(runs.glob("trace_*.json")):
        game = path.stem[len("trace_"):]
        trace = json.load(open(path))
        report[game] = analyse(trace)
    print(f"{'game':<11}{'lvl':>4}{'won':>5}{'acts':>6}{'frames':>7} | {'discovery':>9}{'goal-dir':>9}{'explore':>8}{'blind':>6} | "
          f"{'wrong pred':>10}{'/preds':>7} | {'missed':>7}{'spur.':>6}{'colour':>7} | {'no-eff':>7}{'revisit':>8}{'refuted':>8}")
    for game, levels in report.items():
        for lvl, d in levels.items():
            s, e = d["spent"], d["prediction_errors"]
            print(f"{game:<11}{lvl:>4}{str(d['won'])[0]:>5}{d['actions']:>6}{d['distinct_frames']:>7} | {s.get('discovery', 0):>9}{s.get('goal-directed', 0):>9}"
                  f"{s.get('exploratory', 0):>8}{s.get('blind', 0):>6} | {d['wrong_predictions']:>10}{d['predictions']:>7} | "
                  f"{e.get('missed change', 0):>7}{e.get('spurious change', 0):>6}{e.get('wrong colour', 0):>7} | "
                  f"{d['no_effect']:>7}{d['revisits']:>8}{d['goal_keys_refuted']:>8}")
    print("\nWHAT WAS IT THINKING — examples from levels it did not win (frame / what it predicted / what happened):")
    for game, levels in report.items():
        for lvl, d in levels.items():
            if d["won"]:
                continue
            for ex in d["examples"][:args.examples]:
                print(f"\n{game} level {lvl}, step {ex['step']}: action {ex['action']}, plan {ex['choice']}, unknown windows for this action: {ex['unknown']}")
                f, p, o = ex["frame"].split("\n"), ex["predicted"].split("\n"), ex["observed"].split("\n")
                w = max(len(x) for x in f)
                print(f"   {'frame':<{w}}   {'predicted':<{w}}   observed")
                for a, b, c in zip(f, p, o):
                    print(f"   {a:<{w}}   {b:<{w}}   {c}")
                print(f"   missed changes at {ex['missed']}; spurious at {ex['spurious']}")
    json.dump(report, open(runs / "diagnosis.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
