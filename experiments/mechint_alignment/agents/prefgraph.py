"""The preference graph (PREFERENCE_GRAPH.md), first instance (E10a): two nodes joined by a content edge, combined
either by a fixed exchange rate or by a veto. Implements the planner's preference protocol (mpc.py: start, fork,
step), so the same brute-force planner serves it.

- Appreciation: the compression progress of the agent's own 5 x 5 view (E9's appreciator, in summed bits). It also
  reports the progress it felt on each context: the message of the CONTENT EDGE.
- Preservation: each context's value is the progress felt on it (per view, the mean over its instances, summed over
  views; the content edge). A step's loss is sum_k value_k * max(0, log2(1 + n_k before) - log2(1 + n_k after)),
  with n_k the number of paintable cells of the canvas with context k: destroying the last instances of an
  appreciated structure costs most, one of many copies costs little, and making copies earns nothing.
- Graph: drive nodes are summed with fixed weights; a veto node's violation beyond a tolerance dominates everything.
Not built yet: internal states and satiation, gain edges, selection with commitment, value heads in a network.
"""
from collections import deque

import numpy as np

from worlds import canvas as cv
from nets.context_coder import ContextCoder

S = cv.K + 1                      # the coder's symbols: K colours + WALL
N_CTX = (S + 1) ** 4
TAU = 16
CONTEXTS = ContextCoder(S)        # for its contexts() only (stateless)


def world_contexts(world):
    """The context of every cell the agent's actions alone determine, read as in a view (WALL beyond the canvas), as a
    flat array. A cell is left out if it or any of its four causal neighbours is on the noisy screen: those contexts
    change every step whatever the agent does (E10a's apparatus defect)."""
    padded = np.pad(world.colour, ((1, 0), (1, 1)), constant_values=cv.WALL)
    scr = np.pad(world.screen, ((1, 0), (1, 1)))
    touched = scr[1:, 1:-1] | scr[:-1, :-2] | scr[:-1, 1:-1] | scr[:-1, 2:] | scr[1:, :-2]
    return CONTEXTS.contexts(padded)[1:, 1:-1][~touched]


class Appreciation:
    name = "appreciation"

    def start(self, world):
        coder = ContextCoder(S)
        return {"coder": coder, "past": deque([coder.counts.copy()], maxlen=TAU + 2)}

    def fork(self, st):
        return {"coder": st["coder"].copy(), "past": st["past"]}

    def step(self, world, st, h, real, msg):
        view, coder = world.view(), st["coder"]
        ctx = coder.contexts(view)
        past = st["past"]
        j = len(past) - 1 - TAU + (h - 1)                 # the model of TAU steps ago (empty before that)
        lag = ContextCoder(S, counts=past[j]) if j >= 0 else ContextCoder(S)
        prog = lag.bits(view, ctx) - coder.bits(view, ctx)
        inner = ctx[1:, 1:-1].ravel()                     # cells whose four causal neighbours are in view
        n = np.bincount(inner, minlength=N_CTX)
        felt = np.bincount(inner, weights=prog[1:, 1:-1].ravel(), minlength=N_CTX)
        msg["felt"] = np.divide(felt, n, out=np.zeros(N_CTX), where=n > 0)       # mean progress per context seen
        coder.update(view, ctx)
        if real:
            past.append(coder.counts.copy())
        return float(prog.sum())


class Preservation:
    name = "preservation"

    def start(self, world):
        return {"value": np.zeros(N_CTX), "n": np.bincount(world_contexts(world), minlength=N_CTX)}

    def fork(self, st):
        return {"value": st["value"].copy(), "n": st["n"]}

    def step(self, world, st, h, real, msg):
        st["value"] += msg["felt"]                        # the content edge: what was appreciated is what is valued
        n = np.bincount(world_contexts(world), minlength=N_CTX)
        drop = np.maximum(0.0, np.log2(1.0 + st["n"]) - np.log2(1.0 + n))
        st["n"] = n
        return -float((np.maximum(st["value"], 0.0) * drop).sum())


class Graph:
    def __init__(self, nodes, weights, veto=(), tol=0.5, big=1e4):
        self.nodes, self.weights, self.veto, self.tol, self.big = nodes, weights, set(veto), tol, big

    def start(self, world):
        return {"nodes": [n.start(world) for n in self.nodes], "real": True, "log": {n.name: 0.0 for n in self.nodes}}

    def fork(self, st):
        return {"nodes": [n.fork(s) for n, s in zip(self.nodes, st["nodes"])], "real": False, "log": None}

    def step(self, world, st, h):
        msg, total = {}, 0.0
        for node, s in zip(self.nodes, st["nodes"]):
            sig = node.step(world, s, h, st["real"], msg)
            if st["real"]:
                st["log"][node.name] += sig
            if node.name in self.veto:
                total -= self.big * max(0.0, -sig - self.tol)
            else:
                total += self.weights.get(node.name, 0.0) * sig
        return total
