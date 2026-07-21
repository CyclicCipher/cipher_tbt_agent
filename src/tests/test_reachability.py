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
    # 2026-07-10: the first slice wired everything — agent → column → {htm, encoders}. Add an entry ONLY with a written reason
    # if a module is legitimately built-but-not-yet-wired again.
    #
    # 2026-07-21 — the TOUCH modality, built + validated in isolation, wiring PENDING (the active next slice). Both go reachable
    # when `Agent` construction moves to `modalities=[vision(), touch()]` and the object model is conditioned on FELT contact;
    # these entries are DELETED then. Design: notes/touch_and_body_design.md.
    "touch": "the SKIN peripheral + BODY surface (test_touch, green). Wiring = the contact-dynamics slice.",
    "modality": "the sensory-modality factory + vision/touch specs (test_modality, green). Wiring = the same slice.",
}


def _modules() -> set:
    """Every .py module under src/tbt/ as a dotted name — top-level files AND one level of SUBPACKAGE ('hippocampus' for its
    __init__, 'hippocampus.map' for a submodule). NOT the Legacy archive, NOT __pycache__, NOT the top-level __init__."""
    mods: set = set()
    for f in os.listdir(_TBT):
        p = os.path.join(_TBT, f)
        if f.endswith(".py") and f != "__init__.py" and os.path.isfile(p):
            mods.add(f[:-3])
        elif os.path.isdir(p) and "Legacy" not in f and os.path.isfile(os.path.join(p, "__init__.py")):
            mods.add(f)                                               # the subpackage itself (its __init__)
            for g in os.listdir(p):
                if g.endswith(".py") and g != "__init__.py" and os.path.isfile(os.path.join(p, g)):
                    mods.add(f + "." + g[:-3])                        # a submodule, dotted
    return mods


def _path(module: str) -> str:
    """The file backing a (possibly dotted) module: 'agent' → agent.py; 'hippocampus' → hippocampus/__init__.py;
    'hippocampus.map' → hippocampus/map.py."""
    if "." in module:
        pkg, sub = module.split(".", 1)
        return os.path.join(_TBT, pkg, sub + ".py")
    d = os.path.join(_TBT, module)
    return os.path.join(d, "__init__.py") if os.path.isdir(d) else os.path.join(_TBT, module + ".py")


def _pkg_parts(module: str) -> list:
    """The subpackage path (under tbt) that `module`'s relative imports are relative to: a top-level file 'agent' → [];
    a submodule 'hippocampus.map' → ['hippocampus']; a package init 'hippocampus' → ['hippocampus'] (the __init__ IS the
    package). Level-1 `from .X` resolves inside this path; each extra level pops one off."""
    if "." in module:
        return module.split(".")[:-1]
    return [module] if os.path.isdir(os.path.join(_TBT, module)) else []


def _imports(module: str, known: set) -> set:
    """The tbt modules `module` imports — relative (`from .X`, `from . import X`, `from ..X`) or absolute (`from tbt.X`,
    `import tbt.X`), resolved to dotted `known` names. A dotted target pulls in its PACKAGE too (Python loads the __init__)."""
    with open(_path(module), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    parts = _pkg_parts(module)
    out: set = set()

    def _emit(target: list) -> None:
        if not target:
            return
        name = ".".join(target)
        if name in known:
            out.add(name)
        if len(target) > 1 and target[0] in known:                    # the package __init__ loads with a submodule
            out.add(target[0])

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:                                            # relative: from (dots)module import names
                base = parts[: len(parts) - (node.level - 1)]
                if node.module:                                       # from .pkg.sub import ...
                    _emit(base + node.module.split("."))
                else:                                                 # from . import X, Y  → each name a submodule
                    for a in node.names:
                        _emit(base + [a.name])
            elif node.module and node.module.startswith("tbt."):      # from tbt.X.Y import ...
                _emit(node.module.split(".")[1:])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("tbt."):                         # import tbt.X.Y
                    _emit(a.name.split(".")[1:])
    return out


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
