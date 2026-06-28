"""Object tracking with permanence -- follow each moving object across frames so its dynamics live over a stable POSE.

Cell-level analysis cannot find a moving self: a moving object spreads its changes thinly, so no cell carries a
stable signature (measured: 0 action-selective cells on ls20). The self is an OBJECT, identified by tracking it
across frames and represented by its pose -- exactly why TBT/Monty is object-and-pose-based, not pixel-based. The
key separation falls out of pose: an autonomous animation has a FIXED pose (its cells churn but it does not move)
-> no pose-operator; the controllable self has a MOVING, action-dependent pose -> a learnable operator.

This tracker reads the moving objects from the residual (the salient cells = `tbt.retina.salient_cells`), segments
them into connected components, and links each to the nearest compatible object next frame (object permanence).
Event boundaries (`tbt.events`) RESET the linkage -- a scene-cut is not object motion, so the brain re-localises
rather than path-integrating across it. Pure stdlib.
"""

from __future__ import annotations

from collections import deque

from .retina import salient_cells


def components(cells):
    """All 4-connected components of `cells`, each as (cellset, centroid)."""
    cells = set(cells)
    seen, out = set(), []
    for s in cells:
        if s in seen:
            continue
        comp, q = set(), deque([s])
        seen.add(s)
        while q:
            x, y = q.popleft()
            comp.add((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (x + dx, y + dy)
                if p in cells and p not in seen:
                    seen.add(p)
                    q.append(p)
        cx = sum(x for x, _ in comp) / len(comp)
        cy = sum(y for _, y in comp) / len(comp)
        out.append((comp, (cx, cy)))
    return out


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ObjectTracker:
    """Track moving objects across frames by pose, with object permanence. `max_jump` = the largest centroid
    displacement still counted as the same object; `min_size` = the smallest motion blob to track; `stale` = drop a
    track from the active set after this many steps unseen (so a vanished object is not matched to a new one). An
    event boundary resets the active set (re-localisation)."""

    def __init__(self, max_jump: float = 8.0, min_size: int = 2, stale: int = 3):
        self.max_jump = max_jump
        self.min_size = min_size
        self.stale = stale
        self.tracks: dict = {}                               # id -> list of (step, pose, action)
        self._last: dict = {}                                # id -> (last_step, last_pose) for active tracks
        self._next = 0
        self._step = 0

    def observe(self, prev_frame, action, frame, boundary: bool = False) -> None:
        """One transition. On a boundary, reset linkage (do not path-integrate across it). Otherwise read the
        moving objects from the residual and greedily link each to the nearest active track (object permanence),
        starting a new track when none is within `max_jump`."""
        self._step += 1
        if boundary:
            self._last = {}
            return
        self._last = {tid: lp for tid, lp in self._last.items() if self._step - lp[0] <= self.stale}
        objs = [(c, p) for c, p in components(salient_cells(prev_frame, frame)) if len(c) >= self.min_size]
        used = set()
        for _comp, pose in objs:
            best, bd = None, self.max_jump
            for tid, (_s, lp) in self._last.items():
                if tid in used:
                    continue
                d = _dist(pose, lp)
                if d <= bd:
                    best, bd = tid, d
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = []
            used.add(best)
            self.tracks[best].append((self._step, pose, action))
            self._last[best] = (self._step, pose)

    def moving_tracks(self, min_steps: int = 3):
        """Tracks seen over >= `min_steps` -- the persistent objects (the controllable self plus autonomous movers)."""
        return {tid: t for tid, t in self.tracks.items() if len(t) >= min_steps}

    @staticmethod
    def pose_spread(track) -> float:
        """How far a track's pose roams (max centroid extent) -- high for a mover (a candidate self), ~0 for an
        autonomous animation that churns in a FIXED place."""
        xs = [p[0] for _s, p, _a in track]
        ys = [p[1] for _s, p, _a in track]
        return max(max(xs) - min(xs), max(ys) - min(ys)) if track else 0.0
