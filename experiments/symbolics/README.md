# experiments/symbolics

**`arc_agent/` is the only active project here: the arc agent, aimed at ARC-AGI-3.** Its design is
`arc_agent/ARCHITECTURE.md`; its lab book is `arc_agent/docs/AGENT.md`.

`legacy/` holds everything else that was in this folder (the function-finder experiments, the curriculum, the structure
and coordinate experiments, the experiments that led to the arc agent, and Gemini's TBT agent). It is kept for
reference only: nothing in it is imported, run or worked on, and nothing in `arc_agent/` depends on it (2026-10-28).

Run everything from this folder with `venv312`, e.g. `../../venv312/Scripts/python.exe -m arc_agent.test_conditions`.
