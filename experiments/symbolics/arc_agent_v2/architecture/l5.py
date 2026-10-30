"""L5: the operation (ARCHITECTURE.md section 1): what each moving input does to the place.

**Two ways to know a move:**
- **As a table:** an association from the conjunction of (place, input) (HDClibrary.Conjunction, a code that depends
  on both jointly) to the place the move leads to.
- **As one fixed operation on the code (part 4):** where the frame has coordinates, a move is a shift δ of the
  place's code (binding in its Modules), the same at every place of the orbit, including places it was never seen at.

The moves come from the frame (frameLearning.py), seen and inferred alike. L5 doesn't know what any input means: only
where an input has led, or must lead, from each place.

**The route (L5's inverse):** the moves that lead from here to a goal.
- **With coordinates:** the displacement unbind(goal, here) is looked up in a table of shortest words, built once by
  a breadth-first walk over the codes.
- **Otherwise:** a breadth-first search over the known moves.
"""
from collections import deque

from arc_agent_v2.architecture.HDClibrary import Associations, Conjunction

ROUTE_STATES = 20000


class L5:
    def __init__(self, d, w, seed=0):
        self.key = Conjunction(d, w, seed)
        self.clear()

    def clear(self):
        self.moves = Associations(self.key.d, theta=self.key.w // 2)
        self.shifts = {}                                # input -> [(modules, delta)], one per orbit it moves
        self.words = []                                 # per orbit: (modules, {displacement code: [inputs]})
        self.inputs = []                                # every input with a known move, in the order learned
        self._known = set()                             # the same, for O(1) membership
        self._moves = {}                                # move's results, memoized until the next learn

    def learn(self, place, x, nxt):
        self._moves = {}
        self.moves.store(self.key(place, x), nxt)
        if x not in self._known:
            self._known.add(x)
            self.inputs.append(x)

    def learn_many(self, moves):
        """Many (place, input, next place) at once: the same as `learn` one by one."""
        self._moves = {}
        self.moves.store_many([(self.key(place, x), nxt, 1) for place, x, nxt in moves])
        for _, x, _ in moves:
            if x not in self._known:
                self._known.add(x)
                self.inputs.append(x)

    def learn_shift(self, x, modules, delta):
        self._moves = {}
        self.shifts.setdefault(x, []).append((modules, delta))
        if x not in self._known:
            self._known.add(x)
            self.inputs.append(x)

    def move(self, place, x):
        """Where input x leads from place, or None if the frame doesn't know (memoized: a pure function of what L5
        has learned)."""
        key = (place, x)
        if key in self._moves:
            return self._moves[key]
        out = None
        for modules, delta in self.shifts.get(x, ()):
            if modules.is_code(place):
                out = modules.bind(place, delta)
                break
        else:
            out = self.moves.best(self.key(place, x))
        self._moves[key] = out
        return out

    def build_words(self, modules):
        """The shortest words over this orbit's shifts, for every displacement (a breadth-first walk from 0)."""
        gens = [(x, d) for x, sh in self.shifts.items() for m, d in sh if m is modules]
        start = modules.identity()
        word, q = {start: []}, deque([start])
        while q:
            c = q.popleft()
            for x, d in gens:
                n = modules.bind(c, d)
                if n not in word:
                    word[n] = word[c] + [x]
                    q.append(n)
        self.words.append((modules, word))

    def route(self, here, goal):
        """The moves from here to goal, or None if none is known."""
        if here is None or goal is None:
            return None
        if here == goal:
            return []
        for modules, word in self.words:
            if modules.is_code(here) and modules.is_code(goal):
                return word.get(modules.unbind(goal, here))
        prev, q = {here: None}, deque([here])
        while q and len(prev) < ROUTE_STATES:
            p = q.popleft()
            for x in self.inputs:
                n = self.move(p, x)
                if n is not None and n not in prev:
                    prev[n] = (p, x)
                    if n == goal:
                        path = []
                        while prev[n] is not None:
                            n, x = prev[n]
                            path.append(x)
                        return path[::-1]
                    q.append(n)
        return None
