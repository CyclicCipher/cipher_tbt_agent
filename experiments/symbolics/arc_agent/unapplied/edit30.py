import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
# 1. no Column.grow (tokens need no remap)
a = s.index('patch("arc_agent/column.py", [')
b = s.index('patch("arc_agent/brain.py", [')
s = s[:a] + s[b:]
# 2. _grow: extend only
old = """        c.grow(c.frame.extend(new, [int(before[y, x]) for x, y in new]))"""
assert s.count(old) == 1
s = s.replace(old, """        c.frame.extend(new, [int(before[y, x]) for x, y in new])""")
# 3. more brain splices, inserted before the worlds patch
extra = r'''
patch("arc_agent/brain.py", [
    ("""MAX_SITES = 24                    # more change sites than this in one step is a scene change, not motion""",
     """MAX_SITES = 24                    # more change sites than this in one step is a scene change, not motion
PERMUTE_MAX = 600                 # part 30: the most cells a Permute rival is offered for"""),
    ("""            if c.cid in self.pfc.member:
                painted[c.cid], claimed[c.cid] = self._paint(c, action, ctxs[c.cid])""",
     """            if c.cid in self.pfc.member or c.cid in self.pfc.side:
                painted[c.cid], claimed[c.cid] = self._paint(c, action, ctxs[c.cid])"""),
    ("""        elif f.kind == "recolour":
            for (x, y), v in zip(f.place(ns).tolist(), ns):""",
     """        elif f.kind in ("recolour", "permute"):
            for (x, y), v in zip(f.place(ns).tolist(), ns):"""),
    ("""                self.last_cells[c] = len(cells)
        self.last_evidence = lp
        return lp""",
     """                self.last_cells[c] = len(cells)
        for r in self.pfc.rivals:                       # part 30: sides that are sets of columns, each scored on its
            sides = [[c for c in s if c in painted and self.pfc.responsible(c, self.last_act, sides=False)]
                     for s in r["sides"]]               # joint prediction of the same cells
            inside = {c for s in r["sides"] for c in s}
            cells = set().union(*[claimed[c] for s in sides for c in s]) if any(sides) else set()
            ex = set().union(*[v for cid, v in (occupied or {}).items() if cid not in inside]) if occupied else set()
            cells = [(x, y) for x, y in cells if 0 <= x < W and 0 <= y < H and (x, y) not in ex]
            if not cells:
                continue
            for k, s in enumerate(sides):
                joint = {}
                for c in s:
                    joint.update(painted[c])
                n = sum(1 for x, y in cells if joint.get((x, y), int(before[y, x])) == int(after[y, x]))
                r["ev"][k] += hit if n == len(cells) else miss
        self.last_evidence = lp
        return lp"""),
    ("""        original = residual.copy()
        rigid = []                                      # (column, the cells it explains)""",
     """        original = residual.copy()
        n0 = len(report["recruited"])
        rigid = []                                      # (column, the cells it explains)"""),
    ("""            if self._extend_bar(site, pairs):
                continue
            self._recolour(site, before, after, action, report)
""",
     """            if self._extend_bar(site, pairs):
                continue
            self._recolour(site, before, after, action, report)
        made = report["recruited"][n0:]                 # part 30: a change explained in pieces (several rigid ones and
        cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(original))]   # recolours) has one more rival: contents
        if len(rigid) >= 2 and len(cells) <= PERMUTE_MAX:                  # permuted among the cells, as one column
            col = self._permute(cells, before, after, action, report)
            if col is not None:
                self.pfc.rivalry([made, [col.cid]])
"""),
])
'''
pfc = r'''
patch("arc_agent/pfc.py", [
    ("""KINDS = ("translation", "recolour", "bar")""",
     """KINDS = ("translation", "recolour", "bar", "permute")"""),
    ("""        self.next, self.rng = 0, random.Random(seed)""",
     """        self.next, self.rng = 0, random.Random(seed)
        self.rivals, self.side = [], {}                 # part 30: rivalries between SETS of columns; cid -> (r, side)"""),
    ("""    def ctx(self, g, act):""",
     """    def rivalry(self, sides):
        \"\"\"Part 30: explanations that are sets of columns compete: a Permute column against the pieces the same change
        was also explained by (several rigid columns and recolours). Each side is scored on its joint prediction of
        the same cells (brain._group_evidence); only the side with the best evidence is planned, the first while
        none is ahead. Within a side, each column's own group still decides between its rivals.\"\"\"
        r = len(self.rivals)
        self.rivals.append(dict(sides=[list(s) for s in sides], ev=[0.0] * len(sides)))
        for k, s in enumerate(sides):
            for c in s:
                self.side[c] = (r, k)
        return r

    def ctx(self, g, act):"""),
    ("""    def responsible(self, cid, act=None):
        \"\"\"True if this column may be planned with: it belongs to no group, or it is its group's responsible model.\"\"\"
        gid = self.member.get(cid)
        if gid is None:
            return True""",
     """    def responsible(self, cid, act=None, sides=True):
        \"\"\"True if this column may be planned with: it belongs to no group, or it is its group's responsible model;
        and (part 30) it is on no side of a rivalry, or on the side ahead.\"\"\"
        if sides and cid in self.side:
            r, k = self.side[cid]
            ev = self.rivals[r]["ev"]
            if ev.index(max(ev)) != k:
                return False
        gid = self.member.get(cid)
        if gid is None:
            return True"""),
])
'''
k = s.index('patch("arc_agent/worlds.py", [')
s = s[:k] + extra.lstrip("\n") + "\n" + pfc.lstrip("\n") + "\n" + s[k:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
