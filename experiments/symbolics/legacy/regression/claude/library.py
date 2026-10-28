"""Library learning (docs/CURRICULUM.md): templates from the formulas of solved tasks, kept greedily while they
shorten the corpus's description (Stitch-style: one template at a time, rewrite, repeat). Polynomial throughout.

A term is a block: a product of factors with parameters. Candidate templates come from two operators:
    anti-unification   two terms with the same factor structure generalize to a template: inputs become slots; a
                       parameter whose two values agree within TOL becomes a fixed constant (their mean), otherwise a
                       free parameter. At most MAX_FACTORS factors.
    group folding      in one formula, two or more single-input terms with the same structure and parameters within TOL
                       fold into one group template, sum over a group of inputs of c_v T(v), parameters shared and fixed.
Description length (nats) of the corpus given the library (amended 2026-10-26, see docs/CURRICULUM.md; first
specified with 1/2 log n per parameter, under which no template ever paid for itself):
    sum over learned templates of factors * log(number of primitives) + PARAM_COST * fixed constants
  + sum over term instances of -log P(template) + PARAM_COST * (free parameters of the instance),
PARAM_COST = log(120): a constant coded to precision 0.1 over a range of 12, as a compression objective counts it
(Stitch, DreamCoder count each program's whole description, so a template saves its body at every use);
with each term (or set of terms, for a group) assigned the cheapest template it instantiates, and
P(template) = (uses + 1) / total over primitives and templates.
"""
import math

import numpy as np

from regression.claude.symreg2 import N_PARAMS, PRIMITIVES

TOL, MAX_FACTORS = 0.1, 5
PARAM_COST = __import__("math").log(120.0)


def structure(block):
    """Canonical structure of a single-binding term: the factor kinds with the slot pattern, factors sorted."""
    order = sorted(range(len(block.factors)), key=lambda k: (block.factors[k][0], block.bindings[0][block.factors[k][1]]))
    first = {}
    slots = []
    for k in order:
        v = block.bindings[0][block.factors[k][1]]
        first.setdefault(v, len(first))
        slots.append(first[v])
    kinds = tuple(block.factors[k][0] for k in order)
    starts = np.cumsum([0] + [N_PARAMS[block.factors[k][0]] for k in range(len(block.factors))])
    theta = np.concatenate([block.theta[starts[k]:starts[k + 1]] for k in order]) if len(order) else np.zeros(0)
    return (kinds, tuple(slots)), theta


class Template:
    def __init__(self, name, struct, theta, fixed, group=False):
        self.name, self.struct, self.group = name, struct, group
        self.theta, self.fixed = np.asarray(theta, float), np.asarray(fixed, bool)

    def size(self):
        return len(self.struct[0]) + int(self.fixed.sum())

    def matches(self, struct, theta):
        return struct == self.struct and np.all(np.abs(theta[self.fixed] - self.theta[self.fixed]) <= TOL)

    def n_free(self):
        return int((~self.fixed).sum())

    def spec(self):
        """The form symreg2.Finder takes."""
        kinds, slots = self.struct
        return dict(name=self.name, factors=list(zip(kinds, slots)), slots=len(set(slots)), theta=self.theta,
                    free=~self.fixed, group=self.group)

    def describe(self):
        kinds, slots = self.struct
        out, at = [], 0
        for kind, s in zip(kinds, slots):
            k = N_PARAMS[kind]
            ps = ["%.3g" % self.theta[at + j] if self.fixed[at + j] else "?" for j in range(k)]
            out.append(f"{kind}(v{s}{', ' if ps else ''}{', '.join(ps)})")
            at += k
        return ("sum over a group: " if self.group else "") + " * ".join(out)


class Corpus:
    """Per solved task: its terms (structure, theta, primitive name) and sample size."""
    def __init__(self):
        self.tasks = []

    def add(self, finder_record, n):
        terms = []
        for b in finder_record["state"].blocks:
            if getattr(b, "kind", None) in ("affine", "alog"):        # combined-input atoms: task-specific directions
                terms.append(dict(struct=((b.kind,), (0,)), theta=np.asarray(b.theta, float), prim=b.kind))
                continue
            for bind in b.bindings:                                   # a group block contributes one term per input
                single = type(b)(b.factors, [bind], b.theta, b.free, b.template)
                st, th = structure(single)
                prim = "pow*pow" if st[0] == ("pow", "pow") else st[0][0] if len(st[0]) == 1 else "product"
                terms.append(dict(struct=st, theta=th, prim=prim))
        self.tasks.append(dict(terms=terms, n=n))


def primitive_cost_free(term):
    return len(term["theta"])


def assign(corpus, library):
    """Each term's cheapest template; groups first. Returns (uses per name, list of (name, free params, log n))."""
    uses = {k: 0 for k in list(PRIMITIVES) + ["product"] + [t.name for t in library]}
    items = []
    for task in corpus.tasks:
        covered = set()
        for t in library:
            if not t.group:
                continue
            idx = [k for k, term in enumerate(task["terms"]) if k not in covered and t.matches(term["struct"], term["theta"])]
            if len(idx) >= 2:
                covered.update(idx)
                uses[t.name] += 1
                items.append((t.name, t.n_free(), math.log(task["n"])))
        for k, term in enumerate(task["terms"]):
            if k in covered:
                continue
            best = (term["prim"], primitive_cost_free(term))
            for t in library:
                if not t.group and t.matches(term["struct"], term["theta"]) and t.n_free() < best[1]:
                    best = (t.name, t.n_free())
            uses[best[0]] = uses.get(best[0], 0) + 1
            items.append((best[0], best[1], math.log(task["n"])))
    return uses, items


def description_length(corpus, library):
    uses, items = assign(corpus, library)
    for _ in range(2):                                               # P from uses, uses from cheapest assignment
        total = sum(u + 1 for u in uses.values())
        lp = {k: math.log((u + 1) / total) for k, u in uses.items()}
        dl = sum(len(t.struct[0]) * math.log(len(PRIMITIVES)) + PARAM_COST * int(t.fixed.sum()) for t in library)
        dl += sum(-lp[name] + PARAM_COST * free for name, free, logn in items)
    return dl, uses


def candidates(corpus, existing):
    """Anti-unifications of term pairs, and group foldings within formulas, deduplicated."""
    terms = [term for task in corpus.tasks for term in task["terms"]]
    seen, out = {(t.struct, t.group, tuple(np.round(np.where(t.fixed, t.theta, np.nan), 2))) for t in existing}, []

    def offer(struct, theta, fixed, group):
        # margins index a task's own constraints, not inputs: they stay primitives, never part of a template
        if len(struct[0]) > MAX_FACTORS or (not fixed.any() and not group) or \
                {"margin", "affine", "alog"} & set(struct[0]):
            return
        key = (struct, group, tuple(np.round(np.where(fixed, theta, np.nan), 2)))
        if key not in seen:
            seen.add(key)
            out.append(Template(f"T{len(existing) + len(out)}", struct, theta, fixed, group))

    by_struct = {}
    for term in terms:
        if not {"affine", "alog"} & set(term["struct"][0]):
            by_struct.setdefault(term["struct"], []).append(term)
    for struct, group in by_struct.items():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                ta, tb = group[a]["theta"], group[b]["theta"]
                fixed = np.abs(ta - tb) <= TOL
                offer(struct, np.where(fixed, (ta + tb) / 2, ta), fixed, False)
    for task in corpus.tasks:                                          # relational folding: two-input terms
        twos = [t for t in task["terms"] if t["struct"][1] == (0, 1)]
        for a in range(len(twos)):
            same = [twos[a]] + [s for s in twos[a + 1:] if s["struct"] == twos[a]["struct"]
                                and np.all(np.abs(s["theta"] - twos[a]["theta"]) <= TOL)]
            if len(same) >= 2:
                theta = np.mean([s["theta"] for s in same], 0)
                offer(twos[a]["struct"], theta, np.ones(len(theta), bool), True)
    for task in corpus.tasks:
        singles = [t for t in task["terms"] if len(set(t["struct"][1])) == 1 and
                   not {"affine", "alog"} & set(t["struct"][0])]
        for a in range(len(singles)):
            same = [singles[a]] + [s for s in singles[a + 1:] if s["struct"] == singles[a]["struct"]
                                   and np.all(np.abs(s["theta"] - singles[a]["theta"]) <= TOL)]
            if len(same) >= 2:
                theta = np.mean([s["theta"] for s in same], 0)
                offer(singles[a]["struct"], theta, np.ones(len(theta), bool), True)
    return out


def learn(corpus, library=None, max_new=20):
    """Greedily add the template that most shortens the description, while any does. Returns (library, uses, log)."""
    library = list(library or [])
    dl, uses = description_length(corpus, library)
    log = [dict(added=None, dl=dl)]
    for _ in range(max_new):
        best = None
        for c in candidates(corpus, library):
            dl_c, _ = description_length(corpus, library + [c])
            if dl_c < dl - 1e-9 and (best is None or dl_c < best[0]):
                best = (dl_c, c)
        if best is None:
            break
        dl, tpl = best
        tpl.name = f"T{len(library)}"
        library.append(tpl)
        log.append(dict(added=tpl.describe(), dl=dl))
    _, uses = description_length(corpus, library)
    return library, uses, log
