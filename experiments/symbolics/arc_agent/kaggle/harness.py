"""Local scoring of an agent on the public ARC-AGI-3 games, the way the competition scores it.

- Each game runs locally from `environment_files/` (the arc_agi toolkit, OFFLINE mode; Python 3.12, `venv312`), one
  process per game, all games in parallel: a run takes as long as its slowest game.
- A level scores min(1, (human baseline actions / agent actions)^2) if completed, else 0 (docs.arcprize.org/methodology;
  the doc caps a level's score near 1). Actions are counted from the level's start and include RESETs.
- A game's score is the average of its level scores weighted by the level number (level 1 weighs 1, level 2 weighs 2,
  ...); the overall score is the mean over games.
- The docs set no action cap. To bound compute, a level is abandoned (and the game ends) at CAP x its baseline.
  Recorded evaluations use CAP 10 (a level would score 0.01 there). `--fast` (for development) uses CAP 3 (0.11) and
  a 0.05 s search budget per step.

    venv312/Scripts/python.exe experiments/symbolics/arc_agent/kaggle/harness.py --fast            (all 25, ~1 min)
    venv312/Scripts/python.exe experiments/symbolics/arc_agent/kaggle/harness.py --arm control ls20 ft09
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
os.chdir(ROOT)
logging.disable(logging.WARNING)


def play(game_id, arm, cap, budget, draws=None, time_limit=None, hunger=None):
    """One game in its own process. draws: the goal-type prior's draws (None: whatever is shipped)."""
    logging.disable(logging.WARNING)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState
    import arc_agent.executive as executive
    from arc_agent.agent import ArcAgent
    from explorer import Explorer
    executive.BUDGET = budget
    executive.FIXED_HUNGER = hunger
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    info = next(e for e in arc.get_environments() if e.game_id == game_id)
    env = arc.make(game_id)
    base = info.baseline_actions
    fd = env.reset()
    agent = ArcAgent(game_id, type_draws=draws) if arm == "brain" else Explorer(game_id)
    level, used, done, t0 = 0, 0, [], time.time()
    while level < len(base):
        a, data = agent.act(fd)
        fd = env.step(GameAction.from_id(a), data=data or {})
        used += 1
        if fd.levels_completed > level:             # level completed
            done.append(used)
            level, used = fd.levels_completed, 0
            if fd.state == GameState.WIN:
                break
        elif used >= cap * base[level]:             # abandoned
            break
        if time_limit and time.time() - t0 > time_limit:
            break
    scores = [min(1.0, (b / a) ** 2) for b, a in zip(base, done)] + [0.0] * (len(base) - len(done))
    return dict(game=game_id[:4], levels=len(base), solved=len(done), actions=done, baseline=base[:len(done) + 1],
                score=sum((i + 1) * s for i, s in enumerate(scores)) / sum(range(1, len(base) + 1)),
                seconds=round(time.time() - t0, 1), notes=dict(getattr(agent, "notes", {})),
                win_types=list(getattr(getattr(agent, "executive", None), "goals", None).win_types)
                if arm == "brain" else [])


def priors_for(ids, mode):
    """mode "even": no draws; "cv": 5 folds, each game's draws from the warm-up of the other folds' games;
    "shipped": the agent's priors.json."""
    if mode == "shipped":
        return [None] * len(ids)
    if mode == "even":
        return [[] for _ in ids]
    warm = json.load(open(HERE.parents[1] / "runs" / "warmup.json"))["games"]
    allids = sorted(warm)
    fold = {g: i % 5 for i, g in enumerate(allids)}
    return [[t for h in allids if fold[h] != fold.get(g, -1) for t in warm[h]] for g in ids]


def main(games=None, arm="brain", out=None, fast=False, cap=None, budget=None, priors="shipped", time_limit=None,
         hunger=None):
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    ids = sorted(e.game_id for e in arc.get_environments())
    if games:
        ids = [g for g in ids if g[:4] in games]
    c0, b0 = (3, 0.05) if fast else (10, 0.3)
    cap, budget = cap or c0, budget or b0
    draws = priors_for(ids, priors)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=min(len(ids), os.cpu_count() or 4)) as pool:
        futs = [pool.submit(play, g, arm, cap, budget, d, time_limit, hunger) for g, d in zip(ids, draws)]
        for f in as_completed(futs):                # each game reported as it finishes
            res.append(f.result())
            print(res[-1], flush=True)
    res.sort(key=lambda r: r["game"])
    total = sum(r["score"] for r in res) / len(res)
    print(f"\n{arm}{' (fast)' if fast else ''}: mean score {total:.4f} over {len(res)} games | levels solved "
          f"{sum(r['solved'] for r in res)} of {sum(r['levels'] for r in res)} | {time.time() - t0:.0f} s wall", flush=True)
    if out:
        json.dump(dict(results=res, mean=total, arm=arm, fast=fast), open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {"arm": "brain", "out": None, "priors": "shipped", "hunger": None}
    for k in ("arm", "out", "priors", "hunger"):
        if f"--{k}" in args:
            i = args.index(f"--{k}")
            opts[k] = args[i + 1]
            args = args[:i] + args[i + 2:]
    fast = "--fast" in args
    args = [a for a in args if a != "--fast"]
    main(args or None, opts["arm"], opts["out"], fast, priors=opts["priors"],
         hunger=float(opts["hunger"]) if opts["hunger"] else None)
