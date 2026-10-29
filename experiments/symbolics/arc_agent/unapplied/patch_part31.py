"""Part 31's splice: displayed copies as "match" goal hypotheses (the transform from a column to its copy is the
difference), and goal world G3."""
D = r"C:/Users/julia/AppData/Local/Temp/claude/C--Users-julia-GitHub-predictive-coding-agent/f1514020-295b-4f78-8b2d-297f4e9e4687/scratchpad/displayed.py"


def patch(p, reps):
    s = open(p, encoding="utf-8").read()
    for a, b in reps:
        assert s.count(a) == 1, (p, a[:80])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)


patch("arc_agent/brain.py", [
    ("""    # ---- the model, for planning ------------------------------------------------------------------------------
    def static_scene(self):""",
     open(D, encoding="utf-8").read() + """    # ---- the model, for planning ------------------------------------------------------------------------------
    def static_scene(self):"""),
    ("""MAX_SITES""", """MATCH_MAX = 6                                           # part 31: more displayed copies than this say nothing
MAX_SITES"""),
])

patch("arc_agent/executive.py", [
    ("""TYPES, RARE, EFFORT = ("on-rare", "on-common", "at", "joint"), 4, 50""",
     """TYPES, RARE, EFFORT = ("on-rare", "on-common", "at", "joint", "match"), 4, 50   # part 31: + "match\""""),
    ("""        self.joint, self.joint_rest = {}, 0             # joint state -> errors; errors of every joint state never named
        self.wins = 0""",
     """        self.joint, self.joint_rest = {}, 0             # joint state -> errors; errors of every joint state never named
        self.match = {}                                 # part 31: (kind, state) -> errors; states a displayed copy names
        self.wins = 0"""),
    ("""    def type_prior(self, even=False):""",
     """    def register_match(self, kind, state):
        \"\"\"Part 31: a displayed copy of this kind's shape names the state that would cover it.\"\"\"
        self.match.setdefault((kind, state), self.wins)

    def type_prior(self, even=False):"""),
    ("""        self._bj = math.log(P["joint"] / JOINT_STATES)
        return w_on, base""",
     """        self._bj = math.log(P["joint"] / JOINT_STATES)
        n_m = max(1, len(self.match))
        self._wm = {key: math.log(P["match"] / n_m) + LOG_EPS * e for key, e in self.match.items()}
        return w_on, base"""),
    ("""        m = max(list(w_on.values()) + [b + LOG_EPS * self.at_rest[k] for k, b in base.items()] +
                [bj + LOG_EPS * self.joint_rest])
        z, num = 0.0, 0.0""",
     """        m = max(list(w_on.values()) + [b + LOG_EPS * self.at_rest[k] for k, b in base.items()] +
                [bj + LOG_EPS * self.joint_rest] + list(self._wm.values()))
        z, num = 0.0, 0.0
        for key, w in self._wm.items():                 # part 31: the displayed copies
            v = math.exp(w - m)
            z += v
            if ("match",) + key in held:
                num += v"""),
    ("""        cand = [(w, ("on",) + key) for key, w in w_on.items()]
        for kk, b in base.items():
            cand += [(b + LOG_EPS * e, ("at", kk, s)) for s, e in self.at[kk].items()]""",
     """        cand = [(w, ("on",) + key) for key, w in w_on.items()] + [(w, ("match",) + key) for key, w in self._wm.items()]
        for kk, b in base.items():
            cand += [(b + LOG_EPS * e, ("at", kk, s)) for s, e in self.at[kk].items()]"""),
    ("""        m = max([w for w, _ in cand] + [b + LOG_EPS * self.at_rest[kk] for kk, b in base.items()]
                + [bj + LOG_EPS * self.joint_rest])
        z = (JOINT_STATES - len(self.joint)) * math.exp(bj + LOG_EPS * self.joint_rest - m)""",
     """        m = max([w for w, _ in cand] + [b + LOG_EPS * self.at_rest[kk] for kk, b in base.items()]
                + [bj + LOG_EPS * self.joint_rest])
        z = (JOINT_STATES - len(self.joint)) * math.exp(bj + LOG_EPS * self.joint_rest - m)
        z += sum(math.exp(w - m) for w in self._wm.values())"""),
    ("""            elif h[0] == "joint":
                self.joint[h[1]] = self.joint.get(h[1], self.joint_rest) + 1""",
     """            elif h[0] == "joint":
                self.joint[h[1]] = self.joint.get(h[1], self.joint_rest) + 1
            elif h[0] == "match" and (h[1], h[2]) in self.match:
                self.match[(h[1], h[2])] += 1"""),
    ("""            elif h[0] == "joint":
                w, t = self._bj + LOG_EPS * self.joint.get(h[1], self.joint_rest), "joint\"""",
     """            elif h[0] == "joint":
                w, t = self._bj + LOG_EPS * self.joint.get(h[1], self.joint_rest), "joint"
            elif h[0] == "match" and (h[1], h[2]) in self._wm:
                w, t = self._wm[(h[1], h[2])], "match\""""),
    ("""        for key in self.on:
            if ("on",) + key not in held:
                self.on[key] += 1""",
     """        for key in self.on:
            if ("on",) + key not in held:
                self.on[key] += 1
        for key in self.match:
            if ("match",) + key not in held:
                self.match[key] += 1"""),
    ("""            k = c.kind
            out.add(("at", k, s))""",
     """            k = c.kind
            out.add(("at", k, s))
            if (k, s) in self.goals.match:              # part 31
                out.add(("match", k, s))"""),
    ("""    def register(self, static):
        kinds = [c.kind for c in self.brain.planning() if c.frame.kind == "translation"]""",
     """    def register(self, static):
        for c in self.brain.planning():                 # part 31: displayed copies, once per column and level
            key = (self.brain.level, c.cid)
            if c.frame.kind == "translation" and key not in self.matched:
                self.matched.add(key)
                for s in self.brain.displayed(c, static):
                    self.goals.register_match(c.kind, s)
        kinds = [c.kind for c in self.brain.planning() if c.frame.kind == "translation"]"""),
    ("""        self.inverse = Inverse(brain, self.goals)       # part 29: plans read off the inverted model; viability""",
     """        self.inverse = Inverse(brain, self.goals)       # part 29: plans read off the inverted model; viability
        self.matched = set()                            # part 31: (level, column) already searched for copies"""),
])

patch("arc_agent/inverse.py", [
    ("""    def holds(self, h, c, s):
        if h[0] == "at":
            return s == h[2]""",
     """    def holds(self, h, c, s):
        if h[0] in ("at", "match"):
            return s == h[2]"""),
])
print("part 31 spliced (brain, executive, inverse)")

C = r"C:/Users/julia/AppData/Local/Temp/claude/C--Users-julia-GitHub-predictive-coding-agent/f1514020-295b-4f78-8b2d-297f4e9e4687/scratchpad/conjugate.py"
s = open("arc_agent/column.py", encoding="utf-8").read()
assert s.count("class Numeric:") == 1
s = s.replace("class Numeric:", open(C, encoding="utf-8").read() + "\n\nclass Numeric:")
open("arc_agent/column.py", "w", encoding="utf-8").write(s)

patch("arc_agent/column.py", [
    ("""NEAR = 10.0                       # a head's theta learns while its log weight is within this of the best (part 28)""",
     """NEAR = 10.0                       # a head's theta learns while its log weight is within this of the best (part 28)
CONJ, CONJ_NEAR = True, 5.0       # part 31b: actions' models lent by the frame's turns; experts kept within this"""),
    ("""        self.reach = 4                                  # the context window's margin: grows with the largest move seen""",
     """        self.reach = 4                                  # the context window's margin: grows with the largest move seen
        self.relate = None                              # part 31b: akey -> [(another key, quarter turns)] (Brain.relate)
        self.conj = {}                                  # part 31b: akey -> {expert (None: its own model): log evidence}"""),
    ("""    def dist(self, akey, ctx):
        b = self.clock if self.driver() == "clock" else self.belief(akey)
        return b.dist({} if b is self.clock else ctx)""",
     """    def dist(self, akey, ctx):
        b = self.clock if self.driver() == "clock" else self.belief(akey)
        if b is self.clock:
            return b.dist({})
        ex = self.experts(akey, ctx)
        if len(ex) == 1:
            return ex[0][2]
        m = max(w for w, _, _ in ex)
        ws = [math.exp(w - m) for w, _, _ in ex]
        z = sum(ws)
        out, rest = Counter(), 0.0
        for wi, (_, _, (p, r)) in zip(ws, ex):
            for op, x in p.items():
                out[op] += wi / z * x
            rest += wi / z * r
        return dict(out), rest

    def experts(self, akey, ctx):
        \"\"\"Part 31b: [(log evidence, expert, (prediction, rest))]: this action's own model, and each other key's model
        turned by the frame's symmetry onto this key's direction (the user's square and parallelogram: the turn between
        two actions says exactly how they differ, so one action's lesson is lent to the other). Experts more than
        CONJ_NEAR nats behind the best are not asked.\"\"\"
        ev = self.conj.get(akey, {})
        out = [(ev.get(None, 0.0), None, self.belief(akey).dist(ctx))]
        if not (CONJ and self.relate is not None and akey[0] == "key" and self.frame.kind == "translation"):
            return out
        s = ctx.get(("self",))
        if s is None:
            return out
        cells = self.frame.place(s)
        centre = cells.mean(0) - cells[0]
        best = max([ev.get(None, 0.0)] + [ev.get(src, 0.0) for src, _ in self.relate(akey)])
        for src, k in self.relate(akey):
            w = ev.get(src, 0.0)
            sb = self.keys.get(src)
            if sb is None or not sb.n or w < best - CONJ_NEAR:
                continue
            p, rest = sb.dist(Turned(ctx, k, centre))
            out.append((w, src, ({turn_op(op, k): x for op, x in p.items()}, rest)))
        return out"""),
    ("""        b = self.belief(akey)
        lp_a = math.log(max(b.predictive(ctx, op), 1e-12))""",
     """        b = self.belief(akey)
        pa, ra = self.dist(akey, ctx)                   # the prediction made before seeing op (prequential)
        lp_a = math.log(max(pa.get(op, ra), 1e-12))
        if CONJ and self.relate is not None and akey[0] == "key":   # part 31b: then each expert's evidence
            ev = self.conj.setdefault(akey, {})
            for w, src, (p, rest) in self.experts(akey, ctx):
                ev[src] = ev.get(src, 0.0) + math.log(max(p.get(op, rest), 1e-12))"""),
])

patch("arc_agent/brain.py", [
    ("""        c = Column(self.next_id, frame, state, self.t, self.conditional, self.kind_prior, self.library)
        self.next_id += 1""",
     """        c = Column(self.next_id, frame, state, self.t, self.conditional, self.kind_prior, self.library)
        c.relate = self.relate                          # part 31b
        self.next_id += 1"""),
    ("""    def col_features(self, cid, state):""",
     """    def directions(self):
        \"\"\"Part 31b: each key's direction, the move it gives the most self-propelled column (its efference copy).
        Cached per step.\"\"\"
        if getattr(self, "_dirs", (None,))[0] != self.t:
            out = {}
            if self.solo:
                cid = max(self.solo, key=self.solo.get)
                c = next((c for c in self.columns if c.cid == cid), None)
                for akey, b in (c.keys.items() if c is not None else []):
                    if akey[0] == "key" and b.n:
                        op = b.n0.most_common(1)[0][0]
                        if isinstance(op, tuple) and len(op) == 3 and (op[0] or op[1]):
                            out[akey] = (op[0], op[1])
            self._dirs = (self.t, out)
        return self._dirs[1]

    def relate(self, akey):
        \"\"\"Part 31b: [(another key, k)]: k quarter turns take that key's direction onto this key's.\"\"\"
        dirs = self.directions()
        d = dirs.get(akey)
        if d is None:
            return []
        return [(src, k) for src, e in dirs.items() if src != akey for k in (1, 2, 3) if turn(e[0], e[1], k) == d]

    def col_features(self, cid, state):"""),
    ("""from arc_agent.column import PMIN, Column""", """from arc_agent.column import PMIN, Column, turn"""),
])
print("part 31b spliced (column, brain)")

patch("arc_agent/goal_worlds.py", [
    ("""G2 lock and exit  a wall of doors (8) splits the room; ACTION5 arms a hidden lock and ACTION7 opens it if armed
                  (W12's lock: 5 then 7); a level is won when the avatar stands on the exit cell (3), behind the doors.""",
     """G2 lock and exit  a wall of doors (8) splits the room; ACTION5 arms a hidden lock and ACTION7 opens it if armed
                  (W12's lock: 5 then 7); a level is won when the avatar stands on the exit cell (3), behind the doors.
G3 copy the pose  an L-shaped piece (9) moves with keys 1-4 and turns a quarter with ACTION5 (about its own pivot); a
                  copy of its shape (14) is drawn at another pose; a level is won when the piece covers the copy
                  exactly (part 31: the transform from the piece to the copy is the goal)."""),
    ("""GOAL_WORLDS = ("G1 push", "G2 lock and exit")""",
     """GOAL_WORLDS = ("G1 push", "G2 lock and exit", "G3 copy the pose")
PIECE = [(0, -1), (0, 0), (0, 1), (1, 1)]               # G3's L around its pivot (0, 0)


def piece(at, turn):
    out = []
    for x, y in PIECE:
        for _ in range(turn % 4):
            x, y = -y, x
        out.append((at[0] + x, at[1] + y))
    return out"""),
    ("""            else:
                mid = N // 2
                doors = {(mid, y) for y in range(1, N - 1)}""",
     """            elif self.tag == "G3":
                inside = lambda cells: all(0 < x < N - 1 and 0 < y < N - 1 for x, y in cells)
                poses = [((x, y), t) for x in range(1, N - 1) for y in range(1, N - 1) for t in range(4)
                         if inside(piece((x, y), t))]
                i, j = self.rng.choice(len(poses), 2, replace=False)
                (a0, t0), (g0, tg) = poses[i], poses[j]
                if set(piece(a0, t0)) & set(piece(g0, tg)) or tg == t0:
                    continue                            # the goal must differ in place and in turn
                return dict(walls=walls, avatar=a0, turn=t0, box=None, target=(g0, tg), doors=set())
            else:
                mid = N // 2
                doors = {(mid, y) for y in range(1, N - 1)}"""),
    ("""        self.avatar, self.box, self.q = self.lay["avatar"], self.lay["box"], 0""",
     """        self.avatar, self.box, self.q = self.lay["avatar"], self.lay["box"], 0
        self.turn = self.lay.get("turn", 0)"""),
    ("""        for p in self.lay["doors"]:
            cell(p, 8)
        cell(self.lay["target"], 3)""",
     """        for p in self.lay["doors"]:
            cell(p, 8)
        if self.tag == "G3":
            for p in piece(*self.lay["target"]):
                cell(p, 14)
            for p in piece(self.avatar, self.turn):
                cell(p, 9)
            return g
        cell(self.lay["target"], 3)"""),
    ("""        return [1, 2, 3, 4, 5, 7] if self.tag == "G2" else [1, 2, 3, 4]""",
     """        return [1, 2, 3, 4, 5, 7] if self.tag == "G2" else [1, 2, 3, 4, 5] if self.tag == "G3" else [1, 2, 3, 4]"""),
    ("""        elif self.tag == "G2":
            if a in MOVES:""",
     """        elif self.tag == "G3":
            at, turn = self.avatar, self.turn
            if a == 5:
                turn = (turn + 1) % 4
            elif a in MOVES:
                at = (at[0] + MOVES[a][0], at[1] + MOVES[a][1])
            if not set(piece(at, turn)) & self.lay["walls"]:
                self.avatar, self.turn = at, turn
            done = set(piece(self.avatar, self.turn)) == set(piece(*self.lay["target"]))
        elif self.tag == "G2":
            if a in MOVES:"""),
])
print("G3 added")
