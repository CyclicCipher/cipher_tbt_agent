"""The two task families, behind one interface, so `train.py` does not care which it is running.

compose  -- `h1_lid`'s domain: 7 primitive bijections on 6-digit (mod 5) sequences, tasks = ordered pairs,
            17 trained / 8 held-out. Its held-out compositions have never been solved at this scale (NOTES.md), so the
            exact-match measure is censored there; kept because the plan pre-registered it.
affine   -- modular arithmetic: x -> (a*x + b) mod p, elementwise on 6-digit sequences, p = 11. Succession, addition,
            subtraction, multiplication and division are all members (a=1,b=1; a=1; a=1,b=-k; b=0; a=k^-1,b=0).
            Environments are (a, b) pairs drawn from 6 values of each; 24 are trained, 12 held out, and every held-out
            pair's a and b each appear in training with OTHER partners -- so solving one is recombination, never recall.
            The shared structure is exactly the "multiply by a" and "add b" circuits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transformers"))
import h1_lid  # noqa: E402

L = 6


class Compose:
    name = "compose"
    V = h1_lid.V

    def __init__(self, seed):
        self.train, self.test, _p, _w = h1_lid.build_tasks(seed)
        self.primitives = list(h1_lid.NAMES)

    def apply(self, x, task):
        return h1_lid.apply_pair(x, task)

    def contains(self, task, primitive_index):
        return primitive_index in task

    def describe(self, task):
        return f"{h1_lid.NAMES[task[0]]}∘{h1_lid.NAMES[task[1]]}"


class Affine:
    name = "affine"
    P = 11
    V = 11
    A_SET = [1, 2, 3, 5, 7, 9]
    B_SET = [0, 1, 4, 6, 8, 10]

    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        pairs = [(a, b) for a in self.A_SET for b in self.B_SET]
        order = torch.randperm(len(pairs), generator=g).tolist()
        # hold out 12 pairs such that every a and every b still appears in training
        for _ in range(500):
            test = [pairs[i] for i in order[:12]]
            train = [pairs[i] for i in order[12:]]
            if {a for a, _b in train} == set(self.A_SET) and {b for _a, b in train} == set(self.B_SET):
                break
            order = torch.randperm(len(pairs), generator=g).tolist()
        self.train, self.test = train, test
        # "primitives" for region analysis: the 6 multipliers and the 6 offsets
        self.primitives = [f"a={a}" for a in self.A_SET] + [f"b={b}" for b in self.B_SET]

    def apply(self, x, task):
        a, b = task
        return (a * x + b) % self.P

    def contains(self, task, primitive_index):
        a, b = task
        if primitive_index < len(self.A_SET):
            return a == self.A_SET[primitive_index]
        return b == self.B_SET[primitive_index - len(self.A_SET)]

    def describe(self, task):
        return f"{task[0]}x+{task[1]} mod {self.P}"


def make(name, seed):
    return {"compose": Compose, "affine": Affine}[name](seed)
