"""The thalamus: working memory and its gate (ARCHITECTURE.md §4; docs/CHAINS.md §3-4 and §12; docs/LAB.md part 10).

The bus (bus.py) is the transport: every input delivered once, stamped, by receptive field. The thalamus is the active
gate on what a column holds and is fed. Following the literature (CHAINS.md §12), it learns no content. It holds,
restarts and re-feeds, and the choice of when is learned from prediction error, as the basal ganglia's gating policy
would be.

**Working memory, per hypothesis** (a "thought"):
- the column's state (its place, and its union of (phase, coordinate));
- the last input read, the state before it, and the input before that;
- a stack of held states, each with the input that led into it.

**Three operations, at each input read** (content that moves the place; amendment 3):
- **continue:** the column takes the input.
- **hold:** push the state before the last input; restart the column at its frame's start, re-feed the last input,
  then take the current one. ("a +" waits while "b × c" is computed.)
- **release** (if the stack is not empty): pop a held state and feed it the name of the current coordinate, the
  symbol whose own position it is (part 5's naming). The gate then decides again: release again, or take the input.

None of these knows what an input means.

**The gate, as hypotheses** (Bayes).
- **The branching:** at each input read, every thought branches by the operations, weighted by the gate's prior for
  the context.
- **The contexts:**
  - **for hold:** the input before the last, and the current one;
  - **for release:** the held state's input, and the current one.
- **The weighting:** each thought is weighted by how well its column state predicted the input (the posterior
  predictive, information.py), before the gate responds to the input (amendment 6), and its branches then move by
  it.
- **Pruning:** below 10⁻⁶ of the best, and at most 64 thoughts.

**What teaches the gate** (prediction error, no reward):
- **The prior:** Dirichlet counts per context, (1, 0.1) for (no, yes).
- **The update:** at an episode's end, each context's counts gain the posterior weight of the thoughts that made each
  choice there (an EM step).
- **When it branches** (amendment 2):
  - while learning: in every context;
  - when predicting without learning: only where learning raised the operation's posterior above the prior's mean;
  - so before any learning, the thalamus does nothing.
"""
import math

from arc_agent_v2.architecture.information import NEG_INF, logsumexp, mixture_bits, mixture_logp, mixture_scores

PRUNE = math.log(1e6)
MAX_THOUGHTS = 64
PRIOR = (1.0, 0.1)                                      # Dirichlet counts: (no, yes)
USE = PRIOR[1] / sum(PRIOR)                             # amendment 2: a context branches when predicting only above this


class Thought:
    __slots__ = ("place", "hyps", "last", "before", "last2", "stack", "lw", "path")

    def __init__(self, place, hyps, last=None, before=None, last2=None, stack=(), lw=0.0, path=()):
        self.place, self.hyps, self.last, self.before, self.last2 = place, hyps, last, before, last2
        self.stack, self.lw, self.path = stack, lw, path

    def copy(self, **kw):
        t = Thought(self.place, self.hyps, self.last, self.before, self.last2, self.stack, self.lw, self.path)
        for k, v in kw.items():
            setattr(t, k, v)
        return t


class Thalamus:
    def __init__(self, column, read, enabled=True):
        """column: the column whose working memory this is; read: the sensors whose inputs are read (content that moves
        the place)."""
        self.column, self.read, self.enabled = column, frozenset(read), enabled
        self.counts = {}                                # (kind, context) -> [no, yes]
        self.thoughts = None

    @property
    def trained(self):
        return bool(self.counts)

    def active(self):
        return self.thoughts is not None

    def _p(self, kind, ctx, learn):
        """P(yes) for this decision: the posterior while learning (the prior where unknown); when predicting, only a
        learned context with P(yes) >= USE branches."""
        c = self.counts.get((kind, ctx))
        if learn:
            no, yes = c if c is not None else PRIOR
            return yes / (no + yes)
        if c is None:
            return 0.0
        p = c[1] / (c[0] + c[1])
        return p if p > USE else 0.0

    # ---- an episode -----------------------------------------------------------------------------------------------
    def begin(self):
        self.thoughts = None

    def _start(self, learn):
        col = self.column
        if not self.enabled or col.fac is None or not self.read or not (learn or self.trained):
            return None                                 # nothing to hold, or nothing learned to hold with
        place, hyps = col.start_state()
        return [Thought(place, hyps)]

    def _decide(self, t, x, sensor, learn):
        """The thoughts t becomes before taking x: releases (repeatedly), then a hold or not."""
        col, out = self.column, []

        def releases(t):
            if not t.stack:
                return [t]
            held, held_last = t.stack[-1]
            ctx = ("release", (held_last, x))
            p = self._p(*ctx, learn)
            res = []
            name = col.name(t.place, t.hyps) if p > 0 else None
            if p > 0 and name is not None:
                place, hyps, _ = col.advance(held[0], held[1], name, sensor, True, True)
                res += releases(t.copy(place=place, hyps=hyps, last=name, before=held, last2=held_last,
                                       stack=t.stack[:-1], lw=t.lw + math.log(p), path=t.path + ((ctx, 1),)))
            if p < 1:
                res.append(t.copy(lw=t.lw + (math.log(1 - p) if p > 0 else 0.0),
                                  path=t.path + ((ctx, 0),) if learn else t.path))
            return res

        for t1 in releases(t):
            if t1.before is None or t1.last is None:
                out.append(t1)
                continue
            ctx = ("hold", (t1.last2, x))
            p = self._p(*ctx, learn)
            if p > 0:
                start = col.start_state()
                place, hyps, _ = col.advance(start[0], start[1], t1.last, sensor, True, True)
                out.append(t1.copy(place=place, hyps=hyps, before=start, last2=None,
                                   stack=t1.stack + ((t1.before, t1.last2),), lw=t1.lw + math.log(p),
                                   path=t1.path + ((ctx, 1),)))
            if p < 1:
                out.append(t1.copy(lw=t1.lw + (math.log(1 - p) if p > 0 else 0.0),
                                   path=t1.path + ((ctx, 0),) if learn else t1.path))
        return out

    def observe(self, x, sensor, moves, content, learn):
        """Branch by the gate and weigh by x (before moving by it). Returns the most probable place: the brain's best
        estimate of where x came (amendment 5), or None if the thalamus is inactive."""
        if self.thoughts is None:
            self.thoughts = self._start(learn)
        if self.thoughts is None:
            return None
        col, K = self.column, self.column.fac.vocab
        read = moves and content and sensor in self.read
        if content:                                     # amendment 6: x is weighed before the gate responds to it
            for t in self.thoughts:
                t.lw += mixture_logp(col.components(t.place, t.hyps, sensor), x, K)
        thoughts = [b for t in self.thoughts for b in (self._decide(t, x, sensor, learn) if read else [t])]
        self.thoughts = self._prune(thoughts)
        return self.best_place()

    def best_place(self):
        """The place with the most posterior weight, summed over the thoughts at it (amendment 5), or None. The thoughts
        that have lost their place count together as one candidate, "no known place"."""
        mass = {}
        for t in self.thoughts or ():
            mass[t.place] = mass.get(t.place, 0.0) + math.exp(t.lw)
        return max(mass, key=mass.get) if mass else None

    def advance(self, x, sensor, moves, content):
        """Every thought moves by x."""
        if not self.thoughts:
            return
        col = self.column
        read = moves and content and sensor in self.read
        out = []
        for t in self.thoughts:
            place, hyps, _ = col.advance(t.place, t.hyps, x, sensor, moves, content)
            if read:
                out.append(t.copy(place=place, hyps=hyps, last=x, before=(t.place, t.hyps), last2=t.last))
            else:
                out.append(t.copy(place=place, hyps=hyps))
        self.thoughts = out

    def _prune(self, thoughts):
        if not thoughts:
            return thoughts
        best = max(t.lw for t in thoughts)
        kept = sorted((t for t in thoughts if t.lw >= best - PRUNE), key=lambda t: -t.lw)[:MAX_THOUGHTS]
        z = logsumexp(t.lw for t in kept)
        for t in kept:
            t.lw -= z
        return kept

    def finish(self, learn):
        """The episode ended: while learning, each context's counts gain the posterior weight of each choice (EM). The
        thoughts stay until the next episode begins (a prediction may still be read from them)."""
        if learn and self.thoughts:
            for t in self.thoughts:
                w = math.exp(t.lw)
                for ctx, yes in t.path:
                    c = self.counts.setdefault(ctx, list(PRIOR))
                    c[yes] += w
                t.path = ()

    # ---- the prediction ---------------------------------------------------------------------------------------------
    def _components(self, sensor):
        col, out = self.column, []
        for t in self.thoughts or ():
            comps = col.components(t.place, t.hyps, sensor)
            if comps:
                z = logsumexp(lw for lw, _, _ in comps)
                out += [(t.lw + lw - z, c, N) for lw, c, N in comps]
        return out

    def predict(self, sensor):
        comps = self._components(sensor)
        if not comps:
            return None
        scores = mixture_scores(comps, self.column.fac.vocab)
        return max(scores, key=scores.get) if scores else None

    def bits(self, target, sensor, K):
        comps = self._components(sensor)
        return mixture_bits(comps, target, K) if comps else None

    def p_yes(self, kind, ctx):
        """The gate's learned P(yes) in a context (for the harness), or None if it never met the context."""
        c = self.counts.get((kind, ctx))
        return None if c is None else c[1] / (c[0] + c[1])
