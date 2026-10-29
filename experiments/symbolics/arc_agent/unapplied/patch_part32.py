"""Part 32's splice: a goal column, the reward model across games."""


def patch(p, reps):
    s = open(p, encoding="utf-8").read()
    for a, b in reps:
        assert s.count(a) == 1, (p, a[:80])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)


patch("arc_agent/executive.py", [
    ("""from arc_agent.frames import H, W
from arc_agent.inverse import Inverse""",
     """from arc_agent.column import Column
from arc_agent.frames import H, W, Choice
from arc_agent.inverse import Inverse"""),
    ("""class Goals:
    def __init__(self, type_draws=()):
        self.over = Overhypothesis()                    # the prior over goal types (part 22)
        for t in type_draws:                            # warm-up: WIN types from other games
            self.over.add("goal", t)""",
     """class Goals:
    def __init__(self, type_draws=(), examples=()):
        self.over = Overhypothesis()                    # the prior over goal types (part 22)
        for t in type_draws:                            # warm-up: WIN types from other games
            self.over.add("goal", t)
        self.col = Column(-50, Choice(TYPES), None, 0)  # part 32: the reward model, a column: which type of situation
        for ctx, t in examples:                         # wins, given the game's situation; learned across games
            self.col.learn(("goal",), {tuple(k): v for k, v in ctx}, ("to", t))
        self.ctx = {}"""),
    ("""    def type_prior(self, even=False):
        return {t: 1 / len(TYPES) for t in TYPES} if even else self.over.prior("goal", TYPES)""",
     """    def type_prior(self, even=False):
        \"\"\"P(the WIN's cause is of each type): even; or the goal column's prediction in this game's situation (part 32),
        which backs off to the overhypothesis while the column has seen nothing.\"\"\"
        if even:
            return {t: 1 / len(TYPES) for t in TYPES}
        over = self.over.prior("goal", TYPES)
        b = self.col.keys.get(("goal",))
        if not GOAL_COLUMN or b is None or not b.n:
            return over
        p, rest = self.col.dist(("goal",), self.ctx)
        n = b.n
        out = {t: (n * p.get(("to", t), rest) + over[t]) / (n + 1) for t in TYPES}
        z = sum(out.values())
        return {t: v / z for t, v in out.items()}"""),
    ("""    def win(self, held):
        t = self.win_type(held)
        if t is not None:                               # one draw for the overhypothesis over goal types
            self.over.add("goal", t)
            self.win_types.append(t)""",
     """    def win(self, held):
        t = self.win_type(held)
        if t is not None:                               # one draw for the overhypothesis over goal types
            self.over.add("goal", t)
            self.win_types.append(t)
            self.examples.append((sorted(self.ctx.items()), t))
            self.col.learn(("goal",), dict(self.ctx), ("to", t))   # part 32: and one lesson for the goal column"""),
    ("""        self.win_types = []                             # this game's draws""",
     """        self.win_types = []                             # this game's draws
        self.examples = []                              # part 32: (situation, type) of this game's WINs"""),
    ("""INVERSE, VIABLE = True, True                            # part 29: the inverse model's plans; viability's charge""",
     """INVERSE, VIABLE = True, True                            # part 29: the inverse model's plans; viability's charge
GOAL_COLUMN = True                                      # part 32: the goal-type prior is the goal column's prediction"""),
    ("""    def __init__(self, brain, type_draws=()):
        self.brain = brain
        self.goals = Goals(type_draws)""",
     """    def __init__(self, brain, type_draws=(), examples=()):
        self.brain = brain
        self.goals = Goals(type_draws, examples)"""),
    ("""    def register(self, static):""",
     """    def situation(self, static, available=None):
        \"\"\"Part 32: the game's situation as the goal column reads it (game-agnostic features).\"\"\"
        cols = self.brain.planning()
        n_col = len(np.unique(static))
        return {("propelled",): bool(self.brain.solo), ("copies",): bool(self.goals.match),
                ("columns",): min(len(cols), 4), ("colours",): 0 if n_col <= 3 else 1 if n_col <= 6 else 2,
                ("click",): self.click, ("keys",): self.keys}

    def register(self, static):
        self.goals.ctx = self.situation(static)"""),
    ("""        self.matched = set()                            # part 31: (level, column) already searched for copies""",
     """        self.matched = set()                            # part 31: (level, column) already searched for copies
        self.click, self.keys = False, 0                # part 32: the controls (Executive.actions sees them)"""),
    ("""    def actions(self, available):
        keys = [(a, None, None) for a in available if a not in (0, 6)]""",
     """    def actions(self, available):
        self.click, self.keys = 6 in available, len([a for a in available if a not in (0, 6)])
        keys = [(a, None, None) for a in available if a not in (0, 6)]"""),
])

patch("arc_agent/column.py", [
    ("""              "pfc": 0.5}                         # the PFC column: staying with the same model, before any evidence""",
     """              "pfc": 0.5,                         # the PFC column: staying with the same model, before any evidence
              "goal": 0.5}                        # part 32: the goal column"""),
])

patch("arc_agent/agent.py", [
    ("""        self.executive = Executive(self.brain, pri.get("draws", []))""",
     """        self.executive = Executive(self.brain, pri.get("draws", []), pri.get("examples", []))"""),
])

patch("arc_agent/kaggle/harness.py", [
    ("""                win_types=list(agent.executive.goals.win_types), recall=[agent.brain.hc_hits, agent.brain.hc_calls],""",
     """                win_types=list(agent.executive.goals.win_types), recall=[agent.brain.hc_hits, agent.brain.hc_calls],
                examples=list(agent.executive.goals.examples),"""),
    ("""        out.append(dict(draws=[t for h in others for t in warm[h].get("win_types", [])],""",
     """        out.append(dict(draws=[t for h in others for t in warm[h].get("win_types", [])],
                        examples=[e for h in others for e in warm[h].get("examples", [])],"""),
])

patch("arc_agent/warmup.py", [
    ("""    games = {r["game"]: dict(win_types=r["win_types"], kinds=r.get("kinds", []), corpus=r.get("corpus", []))
             for r in res}""",
     """    games = {r["game"]: dict(win_types=r["win_types"], kinds=r.get("kinds", []), corpus=r.get("corpus", []),
                             examples=r.get("examples", [])) for r in res}"""),
    ("""    json.dump(dict(draws=draws, kinds=kinds, library=library, source="warm-up on the 25 public games"),""",
     """    examples = [e for g in sorted(games) for e in games[g]["examples"]]
    json.dump(dict(draws=draws, kinds=kinds, library=library, examples=examples, source="warm-up on the 25 public games"),"""),
])
print("part 32 spliced")
