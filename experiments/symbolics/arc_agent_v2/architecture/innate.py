"""Innate frames (ARCHITECTURE.md §0, rule 5; docs/LAB.md part 12): where biology gives a region its frame, as V1's
retinotopy is given.

**What is given:** the region's places and moves. A retinotopic region's places are the positions of a sheet
(HDClibrary.Retinotopic), and a gaze shift's efference copy is the shift's own code, so it moves every position by it.

**What is learned:** only the content at the places, and the objects (objects.py). Nothing here is frame learning:
the replay's places are mapped to the sheet by path integration from the start, the start being the sheet's origin.

**The frame** has frameLearning's form, so the column installs it as any other:
- one orbit, coordinated by the sheet's code;
- one phase;
- every shift known at every position.
"""
from collections import Counter, defaultdict

from arc_agent_v2.architecture.frameLearning import Factored, Frame, Orbit

KINDS = ("innate:retinotopic-2d",)


def frame(kind, transitions, contents, start, sheet):
    """The innate frame of a region, from the replay of its own inputs: transitions [(place, input, next place, n)]
    and contents [(place, sensor, input, n)], episodes starting at start; sheet: its HDClibrary.Retinotopic."""
    if kind not in KINDS:
        raise ValueError("no innate frame %r" % kind)
    m = sheet.modules
    origin = sheet.code(0, 0)
    where, children = {start: origin}, defaultdict(list)
    for h, x, nxt, _ in transitions:
        children[h].append((x, nxt))
    stack = [start]
    while stack:                                        # path integration: each shift moves the position by itself
        h = stack.pop()
        for x, nxt in children.get(h, ()):
            if nxt not in where and m.is_code(x):
                where[nxt] = m.bind(where[h], x)
                stack.append(nxt)
    totals = Counter()
    for h, s, x, n in contents:
        if h in where:
            totals[(where[h], s, x)] += n
    content = [(p, s, x, n) for (p, s, x), n in totals.items()]
    grid = [sheet.code(r, c) for r, c in sheet.positions()]
    shifts = sheet.shifts()
    phase = origin                                      # the one phase, named by its first place
    counts, cells = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    for p, s, x, n in content:
        counts[(phase, s)][x] += n
        cells[(phase, p)][s][x] += n
    fac = Factored(m, {c: phase for c in grid}, (phase, origin), {(phase, g): phase for g in shifts},
                   {(0, g): g for g in shifts}, {c: c for c in grid}, {}, {}, [], [], 1, cls={phase: 0},
                   counts={k: dict(c) for k, c in counts.items()}, origins={g: {phase: 1} for g in shifts},
                   vocab=len({x for _, _, x, _ in content}),
                   cells={k: {s: dict(c) for s, c in v.items()} for k, v in cells.items()})
    orbit = Orbit(m, {c: c for c in grid}, {g: g for g in shifts}, m.periods, origin, len(grid))
    return Frame(origin, [], content, where, coords=[orbit], factored=fac)
