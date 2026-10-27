# Symbolics

Symbolic function finding, combinatorial search, small transformer experiments and a symbolic Thousand Brains agent,
from a session of the user with Gemini (2026-10-25/26). Claude added controls and the real versions of the adelic
math (2026-10-26). Start with `docs/ANALYSIS.md`.

Each subject folder separates origin: `gemini/` holds the scripts from the Gemini session, unchanged; `claude/` holds
Claude's controls and rewrites.

| folder | subject | gemini/ | claude/ |
|---|---|---|---|
| `regression/` | function finding (symbolic regression) | symbolic ResNet (greedy residual fitting), the attention-residual regressor, fuzzy AST search, target propagation | `fem2d.py` (a 2D finite-element simulator), `symreg.py` (the symbolic ResNet done properly), `surrogates.py` (experiment A: formulas from simulations), `diagnostics.py` (what limits it), `explore_fixes.py` (exploratory fixes) |
| `search/` | combinatorial search (subset sum) | greedy residual search, the "P vs NP" tests, the AttnRes variants, the "adele distance" | `controls_adelic_search.py` (random planted instances), `local_global_search.py` (the real version: local-global counting) |
| `neural/` | transformers: grokking mod N, Boolean expression trees | the adelic grokking and 2-adic logic transformers | `controls_transformers.py` (ablations), `grokking_adelic.py` and `logic_2adic.py` (the real versions) |
| `continual/` | continual learning with columns | the 100-column symbolic Thousand Brains net | `controls_continual.py` (tasks sharing an input) |
| `arc_agent/` | the ARC-AGI-3 replica agent | engine, worlds, column, bus, brain, hippocampus, test suite, the sleep plan | — |
| `toolkit/` | the shared adelic math | — | `adelic.py`, `check_adelic_toolkit.py` (brute-force checks), `check_gemini_math.py` (checks of Gemini's math) |
| `docs/` | write-ups | `cipher's written summary.txt` (the user's summary of the session) | `ANALYSIS.md` (verdicts), `CONTROLS.md` (controls of Gemini's versions), `REAL_ADELIC.md` (the real versions), `SURROGATES.md` (experiment A), `DREAMCODER_NOTES.md` (DreamCoder, and what it offers here) |

**Running.** Claude's scripts run as modules from this folder, with the repo venv:

    ../../venv/Scripts/python.exe -m search.claude.local_global_search

Gemini's scripts run as plain scripts from their own folder, as written:

    cd arc_agent/gemini; ../../../../venv/Scripts/python.exe brain.py

The agent scripts print Greek letters (Δ). If their output is redirected on Windows, set
`PYTHONIOENCODING=utf-8` first.

Outputs go to `runs/` (git-ignored).
