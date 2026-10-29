"""The hard rule (ARCHITECTURE.md section 0), checked on the brain's own source: no module of the brain imports a
world, or reads a label, a value or a written symbol. The worlds and the tests may; the brain may not.
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/test_rule.py -q
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BRAIN = ("brain", "column", "l4", "l5", "l6", "l23", "hippocampus", "frameLearning", "config")
WORLDS = ("arithmetic_world",)
LABEL_NAMES = {"symbols", "written", "answer", "decode", "labels", "OPS", "sdr_of", "held_out", "train"}


def test_brain_never_sees_a_world_or_a_label():
    for name in BRAIN:
        tree = ast.parse(open(os.path.join(HERE, name + ".py"), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                assert not any(w in (m or "") for m in mods for w in WORLDS), (name, mods)
            if isinstance(node, ast.Attribute):
                assert node.attr not in LABEL_NAMES, (name, node.attr)
