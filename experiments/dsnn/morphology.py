"""Neuron shapes: trees of sections, and their cut into compartments (LAB.md E0b, type A).

One concept: geometry.
- **A section** is an unbranched stretch of dendrite: 3-D points (x, y, z, diameter), and the point of its parent it
  starts from.
- **`read_hoc`** reads a reconstruction exported by NEURON's CellBuilder (`topol()` and `pt3dadd` blocks) as data.
- **`main_path`** follows, from a section, the child whose subtree has the most membrane.
- **`Tree.scaled`** multiplies every length and diameter, as a model's `geom()` may (Gidon et al.'s: × 1.25, for the
  imaging's shrinkage; LAB.md E0e). **`Tree.with_cylinder`** adds a section that has no 3-D points (their axon).
- **`compartments`** cuts the chosen sections into pieces: at most `max_len` µm each, or `nseg_of(section)` equal
  segments as NEURON does. Every other subtree becomes an equivalent cylinder attached where it attaches (LAB.md E0b,
  amendment 2). Each piece gets its membrane area and the axial resistance to its parent piece.

Units: µm, µm², MΩ (axial resistance), Ω·cm (Ra).
"""
from dataclasses import dataclass, field
import math
import re

import numpy as np


@dataclass
class Tree:
    points: dict                      # section -> (n, 4) array of x, y, z, diameter
    parent: dict                      # section -> (parent section, x on the parent); the soma has none
    children: dict = field(default_factory=dict)

    def __post_init__(self):
        for c, (p, _) in self.parent.items():
            self.children.setdefault(p, []).append(c)

    def length(self, s):
        return float(_arc(self.points[s])[-1])

    def area(self, s):
        return _area(self.points[s], 0.0, self.length(s))

    def subtree(self, s):
        out = [s]
        for c in self.children.get(s, []):
            out += self.subtree(c)
        return out

    def subtree_area(self, s):
        return sum(self.area(x) for x in self.subtree(s))

    def dist(self, s, x=1.0):
        """Path distance (µm) from the soma to point x of section s. The soma counts as a point."""
        if s == "soma":
            return 0.0
        p, px = self.parent[s]
        return self.dist(p, px) + x * self.length(s)

    def scaled(self, f):
        """Every length and diameter × f (NEURON's `L = L*f` with `diam3d*f`)."""
        return Tree({s: P * f for s, P in self.points.items()}, dict(self.parent))

    def with_cylinder(self, name, parent, x, L, d):
        """A section with no 3-D points, as a straight cylinder of length L and diameter d at point x of parent."""
        pts = dict(self.points)
        pts[name] = np.array([[0.0, 0.0, 0.0, d], [L, 0.0, 0.0, d]])
        par = dict(self.parent)
        par[name] = (parent, x)
        return Tree(pts, par)


def read_hoc(path):
    """A CellBuilder export: its `topol()` (connect statements, also in `for i = a, b` loops) and its 3-D points."""
    txt = open(path).read()
    parent = {}

    def name(expr, i):
        m = re.match(r"(\w+)(?:\[([^\]]+)\])?$", expr)
        return m.group(1) if m.group(2) is None else "%s[%d]" % (m.group(1), eval(m.group(2), {"i": i}))

    topo = txt[txt.index("proc topol()"):txt.index("proc shape3d_1()")]
    for line in topo.splitlines():
        line = line.strip()
        m = re.match(r"for i = (\d+), (\d+) connect (\S+?)\(0\), (\S+?)\(([\d.]+)\)", line)
        if m:
            for i in range(int(m.group(1)), int(m.group(2)) + 1):
                parent[name(m.group(3), i)] = (name(m.group(4), i), float(m.group(5)))
            continue
        m = re.match(r"connect (\S+?)\(0\), (\w+(?:\[\d+\])?)(?:\(([\d.]+)\))?", line)
        if m:
            parent[m.group(1)] = (m.group(2), float(m.group(3) or 1.0))
    points, cur = {}, None
    for line in txt[txt.index("proc shape3d_1()"):txt.index("proc basic_shape()")].splitlines():
        m = re.search(r"(\w+(?:\[\d+\])?) \{", line)
        if m:
            cur = m.group(1)
            if "pt3dclear" in line:
                points[cur] = []
            points.setdefault(cur, [])
        for a in re.findall(r"pt3dadd\(([^)]*)\)", line):
            points[cur].append([float(v) for v in a.split(",")])
    points = {s: np.array(p) for s, p in points.items()}
    parent = {s: v for s, v in parent.items() if s in points}       # the axon has no points: not part of the cell
    return Tree(points, parent)


def main_path(tree, s):
    out = [s]
    while tree.children.get(s):
        s = max(tree.children[s], key=tree.subtree_area)
        out.append(s)
    return out


# ---- geometry of a stretch of one section ----------------------------------------------------------------------
def _arc(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1))])


def _stretch(P, a, b):
    """The arc positions and diameters of section P between arc lengths a and b (its points, plus the two ends)."""
    s, d = _arc(P), P[:, 3]
    inside = (s > a) & (s < b)
    ss = np.concatenate([[a], s[inside], [b]])
    return ss, np.interp(ss, s, d)


def _area(P, a, b):
    """Membrane area (µm²) between arc lengths a and b, as NEURON computes it: π·(d1 + d2)/2 per piece."""
    s, d = _stretch(P, a, b)
    return float(np.sum(math.pi * (d[:-1] + d[1:]) / 2 * np.diff(s)))


def _resistance(P, a, b, Ra):
    """Axial resistance (MΩ) between arc lengths a and b: Σ 4·Ra·Δl / (π·d1·d2), for a linear taper."""
    lo, hi = min(a, b), max(a, b)
    s, d = _stretch(P, lo, hi)
    return float(np.sum(4.0 * Ra * np.diff(s) / (math.pi * d[:-1] * d[1:]))) * 0.01


# ---- the cut into compartments ---------------------------------------------------------------------------------
@dataclass
class Geometry:
    names: list            # compartment names; 0 is the soma
    parent: np.ndarray     # parent compartment, -1 for the soma
    R: np.ndarray          # axial resistance to the parent (MΩ)
    area: np.ndarray       # membrane area (µm²)
    dist: np.ndarray       # path distance from the soma to the compartment's centre (µm)
    length: np.ndarray     # the compartment's own length (µm); 0 for the soma
    where: dict            # section -> list of (compartment, arc start, arc end)

    def at(self, s, x):
        """The compartment that holds point x of section s."""
        if s == "soma":
            return 0
        L = self.where[s][-1][2]
        for c, a, b in self.where[s]:
            if x * L <= b + 1e-9:
                return c
        return self.where[s][-1][0]


def compartments(tree, modelled, Ra, Rm, Cm, max_len=30.0, nseg_of=None):
    """Cut the `modelled` sections into pieces of at most max_len µm, or into nseg_of(section) equal segments. Every
    other subtree (an unmodelled section whose parent is modelled, and everything beyond it) becomes an equivalent
    cylinder (`equivalent_cylinder`) attached where the subtree attaches, cut by NEURON's d_lambda rule at
    d_lambda = 0.1 · max_len / 30. The soma is one isopotential compartment (its points give its area)."""
    modelled = set(modelled)
    names, parent, R, area, dist, length = ["soma"], [-1], [0.0], [tree.area("soma")], [0.0], [0.0]
    where = {}

    def attach(p, px):
        """(the compartment that holds point px of section p, the resistance from its centre to that point)."""
        if p == "soma":
            return 0, 0.0                                          # the soma is isopotential
        pc = geo_at(p, px)
        _, a, b = next(x for x in where[p] if x[0] == pc)
        return pc, _resistance(tree.points[p], (a + b) / 2, px * tree.length(p), Ra)

    def add(s):
        P = tree.points[s]
        L = tree.length(s)
        n = nseg_of(s) if nseg_of else max(1, math.ceil(L / max_len - 1e-9))
        p, px = tree.parent[s]
        pc, r_up = attach(p, px)
        where[s] = []
        for j in range(n):
            a, b = j * L / n, (j + 1) * L / n
            c = len(names)
            names.append("%s.%d" % (s, j))
            if j == 0:
                parent.append(pc)
                R.append(r_up + _resistance(P, 0.0, (a + b) / 2, Ra))
            else:
                parent.append(c - 1)
                R.append(_resistance(P, (a - L / n + a) / 2, (a + b) / 2, Ra))
            area.append(_area(P, a, b))
            length.append(b - a)
            dist.append(tree.dist(p, px) + (a + b) / 2)
            where[s].append((c, a, b))
        for ch in tree.children.get(s, []):
            if ch in modelled:
                add(ch)

    def geo_at(s, x):
        L = tree.length(s)
        for c, a, b in where[s]:
            if x * L <= b + 1e-9:
                return c
        return where[s][-1][0]

    for ch in tree.children.get("soma", []):
        if ch in modelled:
            add(ch)
    d_lambda = 0.1 * max_len / 30.0
    for u in tree.points:                                         # the unmodelled subtrees
        if u == "soma" or u in modelled or not (tree.parent[u][0] == "soma" or tree.parent[u][0] in modelled):
            continue
        p, px = tree.parent[u]
        pc, r_up = attach(p, px)
        _, d, L = equivalent_cylinder(tree, u, Ra, Rm)
        lam_f = 1e5 * math.sqrt(d / (4 * math.pi * 100.0 * Ra * Cm))          # µm, at 100 Hz (NEURON's lambda_f)
        n = int((L / (d_lambda * lam_f) + 0.9) / 2) * 2 + 1
        half = 4.0 * Ra * (L / n / 2) / (math.pi * d * d) * 0.01               # MΩ
        where[u + "~eq"] = []
        for j in range(n):
            c = len(names)
            names.append("%s~eq.%d" % (u, j))
            parent.append(pc if j == 0 else c - 1)
            R.append(r_up + half if j == 0 else 2 * half)
            area.append(math.pi * d * L / n)
            length.append(L / n)
            dist.append(tree.dist(p, px) + (j + 0.5) * L / n)
            where[u + "~eq"].append((c, j * L / n, (j + 1) * L / n))
    return Geometry(names, np.array(parent), np.array(R), np.array(area, float), np.array(dist), np.array(length),
                    where)


def equivalent_cylinder(tree, u, Ra, Rm):
    """(area, diameter, length) of the cylinder that stands for the subtree from section u (Rall's equivalent cylinder,
    for a tree that need not obey the 3/2 rule):
    - the same membrane area A;
    - electrotonic length X = 2·x̄, with x̄ the area-weighted mean electrotonic distance of the subtree's membrane
      from u's start (each piece's Δl / λ(d), λ = sqrt(d·Rm / (4·Ra))), so the cylinder's membrane has the same mean
      electrotonic distance;
    - so π·d·X·λ(d) = A fixes d, and L = X·λ(d)."""
    lam = lambda d: 100.0 * math.sqrt(d * Rm / (4.0 * Ra))       # µm, d in µm
    tot = acc = 0.0
    stack = [(u, 0.0)]
    while stack:
        s, x = stack.pop()
        P = tree.points[s]
        arc, d = _arc(P), P[:, 3]
        for j in range(len(arc) - 1):
            dl, dm = arc[j + 1] - arc[j], (d[j] + d[j + 1]) / 2
            if dl <= 0:
                continue
            dx = dl / lam(dm)
            a = math.pi * dm * dl
            tot += a
            acc += a * (x + dx / 2)
            x += dx
        stack += [(c, x) for c in tree.children.get(s, [])]
    X = 2.0 * acc / tot
    d = (tot / (math.pi * X * 100.0 * math.sqrt(Rm / (4.0 * Ra)))) ** (2.0 / 3.0)
    return tot, d, X * lam(d)
