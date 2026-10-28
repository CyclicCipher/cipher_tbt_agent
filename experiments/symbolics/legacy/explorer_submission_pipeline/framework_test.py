"""Run the built `my_agent.py` through the official ARC-AGI-3-Agents framework loop (`Agent.main()`), on local games,
the way the Kaggle gateway drives it. Checks the submission works end to end; scoring is harness.py's job.

    venv312/Scripts/python.exe experiments/symbolics/arc_agent/kaggle/framework_test.py <framework clone> [games] [max actions]

<framework clone>: a checkout of github.com/arcprize/ARC-AGI-3-Agents. Its agents/__init__.py is replaced by the
same slim one the Kaggle notebook writes (so LLM-agent dependencies aren't needed).
"""
import importlib.util
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FW = Path(sys.argv[1]).resolve()
GAMES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ls20", "vc33"]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300

(FW / "agents" / "__init__.py").write_text(
    "from typing import Type\nfrom .agent import Agent, Playback\nfrom .swarm import Swarm\n"
    "from .templates.random_agent import Random\nAVAILABLE_AGENTS: dict = {'random': Random}\n")
sys.path.insert(0, str(FW))
os.chdir(ROOT)
logging.basicConfig(level=logging.WARNING)

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

spec = importlib.util.spec_from_file_location("my_agent", HERE / "my_agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.MyAgent.MAX_ACTIONS = STEPS

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
for g in GAMES:
    env = arc.make(g)
    agent = mod.MyAgent(card_id="local", game_id=g, agent_name=f"MyAgent.local.{g}", ROOT_URL="http://localhost",
                        record=False, arc_env=env, tags=["local"])
    agent.main()
    f = agent.frames[-1]
    print(f"{g}: state {f.state.name}, levels completed {f.levels_completed}, actions {agent.action_counter}, "
          f"notes {dict(agent.core.notes)}", flush=True)
