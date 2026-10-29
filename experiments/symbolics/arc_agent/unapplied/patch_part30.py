"""Part 30's splice: the Permute frame, its recruitment and growth, Column.grow, world W17 and its scoring."""
S = r"C:/Users/julia/AppData/Local/Temp/claude/C--Users-julia-GitHub-predictive-coding-agent/f1514020-295b-4f78-8b2d-297f4e9e4687/scratchpad/permute_frame.py"


def patch(p, reps):
    s = open(p, encoding="utf-8").read()
    for a, b in reps:
        assert s.count(a) == 1, (p, a[:80])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)


s = open("arc_agent/frames.py", encoding="utf-8").read()
assert s.count("class Recolour:") == 1
s = s.replace("class Recolour:", open(S, encoding="utf-8").read() + "class Recolour:")
if "linear_sum_assignment" not in s.split("class ")[0]:
    s = s.replace("import numpy as np\n", "import numpy as np\nfrom scipy.optimize import linear_sum_assignment\n", 1)
open("arc_agent/frames.py", "w", encoding="utf-8").write(s)

patch("arc_agent/column.py", [
    ("""    def harvest(self):""",
     """    def grow(self, grow):
        \"\"\"The frame's support grew (part 30, Permute): every operator learned so far, extended to the new cells.\"\"\"
        for b in list(self.keys.values()) + [self.clock]:
            b.identity = self.frame.identity
            b.n0 = Counter({grow(op): n for op, n in b.n0.items()})
            if hasattr(b, "ops"):
                b.ops = [grow(op) for op in b.ops]
                b.opindex = {op: i for i, op in enumerate(b.ops)}
                b.cache = {}

    def harvest(self):"""),
])

patch("arc_agent/brain.py", [
    ("from arc_agent.frames import H, W, Bar, Recolour, Translation",
     "from arc_agent.frames import H, W, Bar, Permute, Recolour, Translation"),
    ("""            else:
                new = f.read(after)
                op = f.op_between(old, new)
                if op is None:
                    c.active = False""",
     """            else:
                new = f.read(after)
                op = f.op_between(old, new)
                if op is None and f.kind == "permute":    # part 30: more cells took part; grow the support
                    old, new, op = self._grow(c, before, after)
                if op is None:
                    c.active = False"""),
    ("""        for k, site in enumerate(sites):                # the rest: a recolour, or one more cell of a counter
            if k in used:
                continue
            pairs = [(int(before[y, x]), int(after[y, x])) for x, y in site]
            m = {}
            if any(m.setdefault(a, b) != b for a, b in pairs) or self._extend_bar(site, pairs):
                continue
            self._recolour(site, before, after, action, report)""",
     """        for k, site in enumerate(sites):                # the rest: a recolour, a permutation, or one more cell of
            if k in used:                               # a counter
                continue
            pairs = [(int(before[y, x]), int(after[y, x])) for x, y in site]
            m = {}
            if any(m.setdefault(a, b) != b for a, b in pairs):
                self._permute(site, before, after, action, report)   # part 30: not one colour map
                continue
            if self._extend_bar(site, pairs):
                continue
            self._recolour(site, before, after, action, report)

    def _permute(self, cells, before, after, action, report):
        \"\"\"Part 30: a Permute column over these cells, if their colours are the same multiset before and after.\"\"\"
        cells = [tuple(map(int, p)) for p in cells]
        b = tuple(int(before[y, x]) for x, y in cells)
        a = tuple(int(after[y, x]) for x, y in cells)
        if sorted(b) != sorted(a):
            return None
        frame = Permute(cells, b)
        col = self._new(frame, b)
        if col is None:
            return None
        op = frame.op_between(b, a)
        others = self.others(col)
        col.learn(col.akey(action, b), self.context(col, b, before, others, self.v1.under_at), op, self.t)
        col.state = a
        report["recruited"].append(col.cid)
        return col

    def _grow(self, c, before, after):
        \"\"\"Part 30: a Permute column whose cells' change is not a permutation of them grows by the changed cells
        connected to it, if the union's change is one. Returns (old state, new state, operator), the operator None
        if no growth explains it.\"\"\"
        own = set(map(tuple, c.frame.cells0.tolist()))
        changed = before != after
        seen, q = set(), [p for p in own if changed[p[1], p[0]]]
        while q:
            x, y = q.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) not in own and (nx, ny) not in seen and 0 <= nx < W and 0 <= ny < H and changed[ny, nx]:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        new = sorted(seen)
        old = c.state + tuple(int(before[y, x]) for x, y in new)
        if not new or sorted(old) != sorted(c.frame.read(after) + tuple(int(after[y, x]) for x, y in new)):
            return c.state, c.frame.read(after), None
        c.grow(c.frame.extend(new, [int(before[y, x]) for x, y in new]))
        nxt = c.frame.read(after)
        return old, nxt, c.frame.op_between(old, nxt)"""),
])

patch("arc_agent/worlds.py", [
    ("""W16 fuse            colour-8 doors are open for the 4 actions after each ACTION5 (part 28: a timer)
""", """W16 fuse            colour-8 doors are open for the 4 actions after each ACTION5 (part 28: a timer)
W17 ring            8 tiles on a ring (two colours repeat); ACTION5 shifts their contents one step clockwise (part 30)
"""),
    ("""          "W8 rotating", "W9 ladders", "W10 sliding")""",
     """          "W8 rotating", "W9 ladders", "W10 sliding", "W17 ring")"""),
    ("""HIDDEN_WORLDS = ("W11 hidden toggle", "W12 lock", "W13 mode", "W14 replay", "W15 clock door", "W16 fuse")""",
     """HIDDEN_WORLDS = ("W11 hidden toggle", "W12 lock", "W13 mode", "W14 replay", "W15 clock door", "W16 fuse")
RING = [(14, 3), (15, 3), (16, 3), (16, 4), (16, 5), (15, 5), (14, 5), (14, 4)]   # W17: the ring's cells, clockwise
RING_COLOURS = [1, 1, 2, 3, 4, 4, 6, 7]"""),
    ("""        self.steps, self.fuse = 0, 0                      # W15: the steps taken; W16: actions left on the fuse""",
     """        self.steps, self.fuse = 0, 0                      # W15: the steps taken; W16: actions left on the fuse
        self.ring = list(RING_COLOURS)                    # W17: the colours on the ring's cells"""),
    ("""        return [1, 2, 3, 4, 5] if (self.switched or self.tag in ("W8", "W11", "W16")) else [1, 2, 3, 4]""",
     """        return [1, 2, 3, 4, 5] if (self.switched or self.tag in ("W8", "W11", "W16", "W17")) else [1, 2, 3, 4]"""),
    ("""        if self.tag == "W14":
            fx, fy = self.follower""",
     """        if self.tag == "W17":
            for (x, y), c in zip(RING, self.ring):
                g[y * C:y * C + C, x * C:x * C + C] = c
        if self.tag == "W14":
            fx, fy = self.follower"""),
    ("""        if self.tag == "W14" and cell == self.follower:
            return False""",
     """        if self.tag == "W14" and cell == self.follower:
            return False
        if self.tag == "W17" and cell in RING:
            return False"""),
    ("""        if a in (5, 7) and self.tag in ("W11", "W12", "W13"):""",
     """        if a == 5 and self.tag == "W17":                # the ring turns one step clockwise
            self.ring = self.ring[-1:] + self.ring[:-1]
            return self.frame()
        if a in (5, 7) and self.tag in ("W11", "W12", "W13"):"""),
])
print("part 30 spliced")

patch("arc_agent/test_conditions.py", [
    ("""from arc_agent.worlds import HIDDEN_WORLDS, WORLDS, World""",
     """from arc_agent.worlds import HIDDEN_WORLDS, RING, WORLDS, World"""),
    ("""        preds = []
        for c in targets(brain, name):""",
     """        preds = []
        if name.startswith("W17"):                      # part 30: the ring, scored on the whole predicted scene
            if a == 5 and t >= start:
                g2 = w.step(a)
                total += 1
                hits += ring_ok(brain, action, g2)
                brain.observe(g, action, g2, ("key", a))
                g = g2
                continue
            g2 = w.step(a)
            brain.observe(g, action, g2, ("key", a))
            g = g2
            continue
        for c in targets(brain, name):"""),
    ("""def run(name, conditional=True, seed=0, hidden=True):""",
     """def ring_ok(brain, action, after):
    \"\"\"W17: does the brain's predicted next scene (every planned column at its predicted state, drawn over what
    lies beneath them) show the ring's cells as they turned out? Any frame type can predict it.\"\"\"
    brain.cache.clear()                                 # the contexts cached per joint state belong to the last scene
    static = brain.static_scene()
    cols = {c.cid: c for c in brain.planning()}
    new, _ = brain.model_step(dict(brain.joint()), action, static, argmax=True)
    scene = static.copy()
    for cid, s in new.items():
        if cid in cols:
            cols[cid].frame.draw(scene, s)
    return all(np.array_equal(scene[y * 3:y * 3 + 3, x * 3:x * 3 + 3], after[y * 3:y * 3 + 3, x * 3:x * 3 + 3])
               for x, y in RING)


def run(name, conditional=True, seed=0, hidden=True):"""),
])
print("test_conditions patched")
