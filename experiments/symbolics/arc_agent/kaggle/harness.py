"""Local scoring of an agent on the public ARC-AGI-3 games, the way the competition scores it.

- Each game runs locally from `environment_files/` (the arc_agi toolkit, OFFLINE mode; Python 3.12, `venv312`).
- A level scores min(1, (human baseline actions / agent actions)^2) if completed, else 0 (docs.arcprize.org/methodology;
  the doc caps a level's score near 1). Actions are counted from the level's start and include RESETs.
- A game's score is the average of its level scores weighted by the level number (level 1 weighs 1, level 2 weighs 2,
  ...); the overall score is the mean over games.
- The docs set no action cap. To bound compute, a level is abandoned (and the game ends) at CAP x its baseline; at
  10 x the level would score 0.01 anyway.

    venv312/Scripts/python.exe experiments/symbolics/arc_agent/kaggle/harness.py            (all 25 games)
    venv312/Scripts/python.exe experiments/symbolics/arc_agent/kaggle/harness.py ls20 ft09  (some)
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)
logging.disable(logging.WARNING)

from arc_agi import Arcade, OperationMode          # noqa: E402
from arcengine import GameAction, GameState         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from explorer import Explorer                       # noqa: E402  (the control arm)
from arc_agent.agent import ArcAgent                # noqa: E402  (the brain arm)

ARMS = {"control": Explorer, "brain": ArcAgent}

CAP = 10                                            # a level is abandoned at CAP x its human baseline


def play(arc, info, make_agent, verbose=False):
    env = arc.make(info.game_id)
    base = info.baseline_actions
    fd = env.reset()
    agent = make_agent(info.game_id)
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
        elif used >= CAP * base[level]:             # abandoned
            break
    scores = [min(1.0, (b / a) ** 2) for b, a in zip(base, done)] + [0.0] * (len(base) - len(done))
    out = dict(game=info.game_id[:4], levels=len(base), solved=len(done), actions=done, baseline=base[:len(done) + 1],
               score=sum((i + 1) * s for i, s in enumerate(scores)) / sum(range(1, len(base) + 1)), seconds=round(time.time() - t0, 1), notes=getattr(agent, "notes", {}))
    if verbose:
        print(out, flush=True)
    return out


def main(games=None, make_agent=Explorer, out=None):
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    infos = sorted(arc.get_environments(), key=lambda e: e.game_id)
    if games:
        infos = [e for e in infos if e.game_id[:4] in games]
    res = [play(arc, e, make_agent, verbose=True) for e in infos]
    total = sum(r["score"] for r in res) / len(res)
    print(f"\nmean score {total:.4f} over {len(res)} games | levels solved {sum(r['solved'] for r in res)} of "
          f"{sum(r['levels'] for r in res)} | {sum(r['seconds'] for r in res):.0f} s", flush=True)
    if out:
        json.dump(dict(results=res, mean=total), open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    arm = "brain"
    if "--arm" in args:
        i = args.index("--arm")
        arm = args[i + 1]
        args = args[:i] + args[i + 2:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(args or None, ARMS[arm], out)
