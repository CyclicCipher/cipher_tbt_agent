"""An inverse model for planning toward a goal (docs/AGENT.md, part 17): read a plan off the learned model instead of
searching the joint space of every object and hypothesis.

- nav: get one body to a goal by breadth-first search over its own translations (at most the board's 81 cells), using
  the actions' intended displacements (the efference copy), around cells it must not enter.
- push_plan: move an object the body pushes. The contact relation inverted: to move the object by d, the body must
  move into it by d, so it must first stand on the cell at minus d. Search over the object's translations; each push
  costs the navigation to the pushing cell plus the push itself.

Both handle a single-cell body and object (all of part 16-17's worlds); the caller falls back to the joint search
otherwise.
"""
import heapq
from collections import deque


def nav(start, cells, table, free, goal_ok, limit=400):
    """Actions taking a body from translation start to one satisfying goal_ok, or None. free(x, y): the body may stand
    on that cell."""
    if goal_ok(start):
        return []
    prev, q = {start: None}, deque([start])
    while q:
        t = q.popleft()
        for a, d in sorted(table.items()):
            if not d or d == (0, 0):
                continue
            nt = (t[0] + d[0], t[1] + d[1])
            if nt in prev or not all(free(x + nt[0], y + nt[1]) for x, y in cells):
                continue
            prev[nt] = (t, a)
            if goal_ok(nt):
                path, cur = [], nt
                while prev[cur] is not None:
                    cur, act = prev[cur]
                    path.append(act)
                return path[::-1]
            if len(prev) > limit:
                return None
            q.append(nt)
    return None


def push_plan(bcells, b0, xcells, x0, table, free_body, free_x, goal_ok, limit=3000):
    """Actions that bring object X to a translation satisfying goal_ok by pushing it with the body, or None."""
    if goal_ok(x0):
        return []
    (bcx, bcy), (xcx, xcy) = bcells[0], xcells[0]
    heap, best, n, tie = [(0, 0, x0, b0, [])], {(x0, b0): 0}, 0, 0
    while heap:
        cost, _, xt, bt, path = heapq.heappop(heap)
        if path and goal_ok(xt):
            return path
        n += 1
        if n > limit:
            return None
        X = (xcx + xt[0], xcy + xt[1])
        for a, d in sorted(table.items()):
            if not d or d == (0, 0):
                continue
            P = (X[0] - d[0], X[1] - d[1])                  # where the body must stand to push X by d
            Xn = (X[0] + d[0], X[1] + d[1])
            if not free_body(*P) or not free_x(*Xn):
                continue
            p = (P[0] - bcx, P[1] - bcy)
            route = nav(bt, bcells, table, lambda x, y: free_body(x, y) and (x, y) != X, lambda t: t == p)
            if route is None:
                continue
            nxt = ((xt[0] + d[0], xt[1] + d[1]), (p[0] + d[0], p[1] + d[1]))
            c2 = cost + len(route) + 1
            if c2 < best.get(nxt, float("inf")):
                best[nxt] = c2
                tie += 1
                heapq.heappush(heap, (c2, tie, nxt[0], nxt[1], path + route + [a]))
    return None
