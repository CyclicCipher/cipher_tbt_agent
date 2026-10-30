"""Trained arithmetic brains, shared by the harnesses within one process (performance: the user, 2026-10-29).

**Why sharing is safe:**
- training is deterministic: the same world, seed and flags give the same brain;
- the harnesses query it without learning.

So each (world, flags) is trained once per run, not once per harness: A1 at p = 97 is trained by parts 3, 5 and 9
alike.

**The key:** the world's arguments and the brain's flags, each with its defaults filled in, so that callers that name
a default and callers that don't share one brain.

**The bound:** the most recent `KEEP` brains are kept, to bound memory (a p = 97 brain is about 100 MB).
"""
import inspect
import time
from collections import OrderedDict

import numpy as np

from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import ARITHMETIC, BrainConfig
from arc_agent_v2.worlds.arithmetic_world import ArithmeticWorld

KEEP = 8
_cache = OrderedDict()


def _defaults(f):
    return {k: v.default for k, v in inspect.signature(f).parameters.items() if v.default is not inspect.Parameter.empty}


WORLD, BRAIN = _defaults(ArithmeticWorld.__init__), _defaults(Brain.__init__)


def trained(p, seed, world=None, brain=None):
    """(world, brain, seconds of training and sleep) for ArithmeticWorld(p, seed, **world) and Brain(**brain)."""
    world, brain = dict(world or {}), dict(brain or {})
    w = {**WORLD, **world, "p": p, "seed": seed}
    b = {**BRAIN, **brain}
    w["ops"] = tuple(w["ops"])
    b.pop("config", None)
    key = (tuple(sorted(w.items())), tuple(sorted(b.items())))
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    wd = ArithmeticWorld(**{k: v for k, v in w.items()})
    br = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed), **b)
    t = time.time()
    for ep in wd.epoch(np.random.default_rng(seed)):
        br.episode(ep)
    br.sleep()
    _cache[key] = (wd, br, time.time() - t)
    while len(_cache) > KEEP:
        _cache.popitem(last=False)
    return _cache[key]
