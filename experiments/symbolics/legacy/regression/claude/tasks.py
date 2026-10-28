"""The tasks of experiment B (docs/CURRICULUM.md): 48 curriculum tasks and 6 held-out tasks, all from fem2d.py.

A task: input names, a box, a simulator call, constraints (slack functions, >= 0 when feasible; their values are the
task's margins), and for held-out tasks an extrapolation box and a far-extrapolation box. All inputs are positive.
"""
from functools import partial

import numpy as np

from regression.claude import fem2d

NU = fem2d.NU


class Task:
    def __init__(self, name, family, names, lo, hi, fn, constraints=(), extrap=None, far=None, extrap_ok=None,
                 limit=None):
        self.name, self.family, self.names = name, family, list(names)
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        self.fn, self.constraints = fn, list(constraints)
        self.extrap, self.far, self.extrap_ok, self.limit = extrap, far, extrap_ok, limit

    @property
    def mnames(self):
        return [f"s{j + 1}" for j in range(len(self.constraints))]

    def margins(self, X):
        X = np.atleast_2d(X)
        return np.stack([c(X) for c in self.constraints], 1) if self.constraints else np.zeros((len(X), 0))

    def feasible(self, X):
        return np.all(self.margins(X) >= 0, 1) if self.constraints else np.ones(len(np.atleast_2d(X)), bool)

    def sample(self, rng, n, lo=None, hi=None, extra_ok=None):
        lo, hi = (self.lo if lo is None else np.array(lo, float)), (self.hi if hi is None else np.array(hi, float))
        out = []
        while len(out) < n:
            X = rng.uniform(lo, hi, (4 * n, len(lo)))
            keep = self.feasible(X)
            if extra_ok is not None:
                keep &= np.array([extra_ok(x) for x in X])
            out += list(X[keep])
        return np.array(out[:n])

    def domain(self, rng, n=2000):
        """Feasible designs for the safety check: 80% uniform, 20% on the box faces or near a constraint boundary."""
        inner = self.sample(rng, int(0.8 * n))
        edge = []
        while len(edge) < n - len(inner):
            x = self.sample(rng, 1)[0]
            if self.constraints and rng.random() < 0.5:
                X = self.sample(rng, 200)
                x = X[np.argmin(self.margins(X).min(1))]
            else:
                k = rng.integers(len(x))
                x[k] = self.lo[k] if rng.random() < 0.5 else self.hi[k]
                if not self.feasible(x[None])[0]:
                    continue
            edge.append(x)
        X = np.vstack([inner, np.array(edge)])
        return {"X": X, "M": self.margins(X)}

    def __call__(self, x):
        return float(self.fn(*x))


# ---- simulator calls --------------------------------------------------------------------------------------------------
def beam(names, fixed, support, load):
    """A uniform beam: names among L, h, nu."""
    def f(*x):
        v = dict(fixed, **dict(zip(names, x)))
        return fem2d.beam_general(v["L"], (v["h"],), v["nu"], support, load)
    return f


def stepped(nseg, L, support, load, with_L=False):
    def f(*x):
        heights = tuple(x[:nseg])
        return fem2d.beam_general(x[nseg] if with_L else L, heights, NU, support, load)
    return f


def plate(names, fixed):
    def f(*x):
        v = dict(fixed, **dict(zip(names, x)))
        return fem2d.plate_hole_ratio(v["r"], v.get("xc"), v.get("yc"), v.get("lx", 1.0), 1.0, v.get("nu", NU))
    return f


def beam_hole(names, fixed, load="tip"):
    def f(*x):
        v = dict(fixed, **dict(zip(names, x)))
        return fem2d.beam_hole_ratio(v["xc"], v["yc"], v["r"], 4.0, NU, "cantilever", load)
    return f


def col(names, k):
    return lambda X: X[:, names.index(k)]


def hole_constraints(names, fixed, lx, ly, margin):
    """The hole stays inside the plate or beam with a margin: four slacks."""
    get = lambda X, k: col(names, k)(X) if k in names else np.full(len(X), fixed[k])
    return [lambda X: get(X, "xc") - get(X, "r") - margin, lambda X: lx - margin - get(X, "xc") - get(X, "r"),
            lambda X: get(X, "yc") - get(X, "r") - margin, lambda X: ly - margin - get(X, "yc") - get(X, "r")]


# ---- the curriculum (48 tasks) ----------------------------------------------------------------------------------------
BOX = {"L": (2.0, 8.0), "h": (0.3, 1.0), "nu": (0.1, 0.45)}
BOX2 = {"L": (3.0, 6.0), "h": (0.5, 1.0), "nu": (0.1, 0.45)}
DEFAULT = {"L": 4.0, "h": 1.0, "nu": NU}


def curriculum():
    tasks = []
    sets = [["L"], ["h"], ["L", "h"], ["L", "nu"], ["h", "nu"], ["L", "h", "nu"]]
    for support, load, family, picks in (("cantilever", "tip", "cantilever_tip", sets + [["L"], ["L", "h"]]),
                                         ("cantilever", "uniform", "cantilever_uniform", sets + [["L"], ["L", "h"]]),
                                         ("clamped", "mid", "clamped", sets[:4]), ("clamped", "uniform", "clamped", sets[:4])):
        for k, names in enumerate(picks):
            box = BOX2 if k >= len(sets) else BOX
            tasks.append(Task(f"{support}_{load}_{'_'.join(names)}_{k}", family, names, [box[n][0] for n in names],
                              [box[n][1] for n in names], beam(names, DEFAULT, support, load)))
    for nseg, L, lo, hi, with_L, tag in ((2, 4.0, 0.4, 1.0, False, "a"), (3, 4.0, 0.4, 1.0, False, "a"),
                                         (2, 4.0, 0.4, 1.0, True, "L"), (3, 3.0, 0.4, 1.0, False, "short"),
                                         (2, 4.0, 0.5, 1.0, False, "b"), (3, 4.0, 0.5, 1.0, False, "b")):
        names = [f"h{i}" for i in range(1, nseg + 1)] + (["L"] if with_L else [])
        tasks.append(Task(f"stepped_{nseg}_{tag}", "stepped", names, [lo] * nseg + ([3.0] if with_L else []),
                          [hi] * nseg + ([6.0] if with_L else []), stepped(nseg, L, "cantilever", "tip", with_L)))
    plates = [(["r"], {"xc": 0.5, "yc": 0.5}, [0.05], [0.3]), (["r"], {"xc": 0.5, "yc": 0.65}, [0.05], [0.2]),
              (["yc"], {"xc": 0.5, "r": 0.15}, [0.3], [0.7]), (["r", "xc"], {"yc": 0.5}, [0.05, 0.3], [0.25, 0.7]),
              (["r"], {"xc": 1.0, "yc": 0.5, "lx": 2.0}, [0.05], [0.35]), (["r"], {"xc": 0.5, "yc": 0.5, "nu": 0.2}, [0.05], [0.3]),
              (["xc"], {"yc": 0.5, "r": 0.2}, [0.32], [0.68]), (["r", "nu"], {"xc": 0.5, "yc": 0.5}, [0.05, 0.1], [0.3, 0.45])]
    for k, (names, fixed, lo, hi) in enumerate(plates):
        lx = fixed.get("lx", 1.0)
        tasks.append(Task(f"plate_{'_'.join(names)}_{k}", "plate", names, lo, hi, plate(names, fixed),
                          hole_constraints(names, dict(fixed, nu=NU), lx, 1.0, 0.1)))
    holes = [(["r"], {"xc": 1.0, "yc": 0.5}, [0.05], [0.3]), (["r"], {"xc": 2.0, "yc": 0.4}, [0.05], [0.3]),
             (["r"], {"xc": 3.0, "yc": 0.6}, [0.05], [0.3]), (["xc"], {"yc": 0.5, "r": 0.15}, [0.4], [3.6]),
             (["yc"], {"xc": 1.0, "r": 0.15}, [0.25], [0.75]), (["xc", "r"], {"yc": 0.5}, [0.5, 0.05], [3.5, 0.3]),
             (["yc", "r"], {"xc": 2.0}, [0.3, 0.05], [0.7, 0.2]), (["xc", "yc"], {"r": 0.1}, [0.4, 0.2], [3.6, 0.8]),
             (["r"], {"xc": 0.5, "yc": 0.5}, [0.05], [0.25]), (["xc"], {"yc": 0.25, "r": 0.15}, [0.4], [3.6])]
    for k, (names, fixed, lo, hi) in enumerate(holes):
        tasks.append(Task(f"beamhole_{'_'.join(names)}_{k}", "beam_hole", names, lo, hi, beam_hole(names, fixed),
                          hole_constraints(names, fixed, 4.0, 1.0, 0.05)))
    assert len(tasks) == 48, len(tasks)
    return tasks


# ---- held-out tasks ---------------------------------------------------------------------------------------------------
def euler_bernoulli(nseg, L, support, load):
    """The textbook law for a stepped beam (bending only), as coefficients on 1/h_i^3."""
    xs = np.linspace(0, L, nseg + 1)
    if support == "cantilever" and load == "tip":
        return np.array([4 * ((L - xs[i]) ** 3 - (L - xs[i + 1]) ** 3) for i in range(nseg)])
    if support == "cantilever" and load == "uniform":
        return np.array([1.5 * ((L - xs[i]) ** 4 - (L - xs[i + 1]) ** 4) for i in range(nseg)])
    return None


def heldout():
    hs = lambda k: [f"h{i}" for i in range(1, k + 1)]
    below = lambda lo: (lambda x: x.min() < lo)
    out = [
        Task("H1_stepped6_tip", "stepped", hs(6), [0.4] * 6, [1.0] * 6, stepped(6, 4.0, "cantilever", "tip"),
             extrap=([0.3] * 6, [1.0] * 6), far=([0.3] * 6, [1.3] * 6), extrap_ok=below(0.4), limit="slender"),
        Task("H2_beamhole_xyr", "beam_hole", ["xc", "yc", "r"], [0.6, 0.25, 0.05], [3.4, 0.65, 0.2],
             beam_hole(["xc", "yc", "r"], {}), hole_constraints(["xc", "yc", "r"], {}, 4.0, 1.0, 0.05),
             extrap=([0.6, 0.25, 0.2], [3.4, 0.65, 0.26]), far=([0.3, 0.2, 0.2], [3.7, 0.8, 0.275]), limit="no_hole"),
        Task("H3_plate_r_yc", "plate", ["r", "yc"], [0.05, 0.5], [0.3, 0.7], plate(["r", "yc"], {"xc": 0.5}),
             [lambda X: 0.9 - X[:, 0] - X[:, 1]] + hole_constraints(["r", "yc"], {"xc": 0.5}, 1.0, 1.0, 0.0),
             extrap=([0.3, 0.5], [0.38, 0.52]), far=([0.3, 0.5], [0.4, 0.5]), limit="small_hole"),
        Task("H4_stepped4_uniform", "stepped", hs(4), [0.4] * 4, [1.0] * 4, stepped(4, 4.0, "cantilever", "uniform"),
             extrap=([0.3] * 4, [1.0] * 4), far=([0.3] * 4, [1.3] * 4), extrap_ok=below(0.4), limit="slender"),
        Task("H5_clamped3_mid", "stepped", hs(3), [0.4] * 3, [1.0] * 3, stepped(3, 4.0, "clamped", "mid"),
             extrap=([0.3] * 3, [1.0] * 3), far=([0.3] * 3, [1.3] * 3), extrap_ok=below(0.4)),
        Task("H6_beamhole_xr_uniform", "beam_hole", ["xc", "r"], [0.5, 0.05], [3.5, 0.3],
             beam_hole(["xc", "r"], {"yc": 0.5}, "uniform"), hole_constraints(["xc", "r"], {"yc": 0.5}, 4.0, 1.0, 0.05),
             extrap=([0.5, 0.3], [3.5, 0.36]), far=([0.35, 0.3], [3.65, 0.375]), limit="no_hole"),
    ]
    return out
