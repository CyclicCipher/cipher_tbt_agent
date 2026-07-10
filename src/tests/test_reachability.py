"""test_reachability.py — enforces RULES.md #2: "unwired is a bug, not a state."

Every module in src/tbt/ must be reachable from the live entry point (agent.py) through the IMPORT GRAPH, or be listed in
STANDALONE with a written reason. The instant a module is un-wired — as the basal ganglia silently was in the `7c09cec`
collapse — this test goes RED and forces a choice: re-wire it or delete it. It is STATIC (parses the AST, does not execute
the code), so it also catches lazy (inside-function) imports. Run this file directly to print the live wired map (STATUS.md):
    python src/tests/test_reachability.py
"""

from __future__ import annotations

import ast
import os

_TBT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tbt"))
_ENTRY = "agent"

# Modules legitimately NOT reachable from the live loop. Each MUST carry a written reason. The goal is that this is empty.
STANDALONE: dict = {
    "htm": "the ONE cortical-layer mechanism (HTM sequence memory; was htm+sequence, merged); validated via the canonical pipeline; wired when the column dynamics are built.",
    "encoders": "salvaged SDR transduction library (data → overlap-bearing SDR); wired at the peripheral when a column is built.",
    "column": "the cortical-column composition — research + whole-column PLAN + wiring scaffold (PART A/B in its docstring); step() dynamics NOT built yet, so not wired into agent.py. Under construction, not 'done' (RULES #3).",
}


def _modules() -> set:
    """Every top-level .py module in src/tbt/ — NOT the Legacy archive (a subdir), NOT __init__, NOT __pycache__."""
    return {f[:-3] for f in os.listdir(_TBT)
            if f.endswith(".py") and f != "__init__.py" and os.path.isfile(os.path.join(_TBT, f))}


def _imports(module: str, known: set) -> set:
    """The tbt modules `module` imports (`from .X`, `from . import X`, `from tbt.X`, `import tbt.X`), restricted to `known`."""
    with open(os.path.join(_TBT, module + ".py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:                       # from .X import ...
                out.add(node.module.split(".")[0])
            elif node.level == 1 and node.module is None:             # from . import X, Y
                out |= {a.name for a in node.names}
            elif node.module and node.module.startswith("tbt."):      # from tbt.X import ...
                out.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("tbt."):                         # import tbt.X
                    out.add(a.name.split(".")[1])
    return out & known


def _reachable(entry: str, graph: dict) -> set:
    seen, stack = set(), [entry]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(graph.get(m, ()))
    return seen


def _graph():
    mods = _modules()
    return mods, {m: _imports(m, mods) for m in mods}


def test_entry_point_exists():
    assert os.path.isfile(os.path.join(_TBT, _ENTRY + ".py")), f"the live entry point src/tbt/{_ENTRY}.py is missing"


def test_every_module_is_wired_or_standalone():
    mods, graph = _graph()
    orphaned = mods - _reachable(_ENTRY, graph) - set(STANDALONE)
    assert not orphaned, (
        "ORPHANED modules (RULES.md #2 — wire them into the loop from agent.py, or add to STANDALONE with a reason): "
        f"{sorted(orphaned)}")


def test_standalone_allowlist_is_honest():
    """No stale / rubber-stamp allowlisting: every STANDALONE entry is a real module that is genuinely NOT reachable."""
    mods, graph = _graph()
    reachable = _reachable(_ENTRY, graph)
    for m in STANDALONE:
        assert m in mods, f"STANDALONE names a non-existent module: {m}"
        assert m not in reachable, f"STANDALONE lists {m}, but it IS reachable from agent — drop it from the allowlist"


def _print_map():
    mods, graph = _graph()
    reachable = _reachable(_ENTRY, graph)
    print(f"src/tbt/  entry={_ENTRY}.py  modules={len(mods)}")
    for m in sorted(mods):
        tag = "WIRED" if m in reachable else ("STANDALONE" if m in STANDALONE else "*** ORPHANED ***")
        print(f"  [{tag:^16}] {m}")


if __name__ == "__main__":
    _print_map()
