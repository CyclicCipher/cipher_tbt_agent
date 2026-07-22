"""behavior.py -- the object BEHAVIOR model: "changes@locations", the touch-conditioned forward model over objects.

TBP (https://docs.thousandbrains.org/docs/object-behaviors): object dynamics is a behavior model storing CHANGES@locations (vs
the structural model's features@locations), in an object-INDEPENDENT reference frame, state-conditioned. TBP leaves the
INTERACTION open (contact, efference, action-from-behavior are "unresolved"); we fill it with the active-touch forward model
(notes/touch_and_body_design.md §7): the CONDITION is felt contact, the CHANGE is efference-parameterised, and YIELD / RESIST /
PASS is discriminated by the PREDICTION ERROR between the operator's predicted body motion (the efference) and the actual
outcome. Learned per felt object; the change is direction-GENERAL because it lives in the invariant frame (the LID win).
"""

from __future__ import annotations

from .operator import eye, norm, rotate, sub   # rotate(M, v) IS the matrix-vector product; reuse the algebra

YIELD, RESIST, PASS, UNKNOWN = "yield", "resist", "pass", "unknown"


def _outer(u, v):
    return tuple(tuple(a * b for b in v) for a in u)


def _madd(A, B, k):
    return tuple(tuple(A[i][j] + k * B[i][j] for j in range(len(A[0]))) for i in range(len(A)))


def _zero(v):
    return tuple(0.0 for _ in v)


class ObjectBehavior:
    """The learned contact behavior of ONE felt object. Pressing into it does one of three things, DISCOVERED from the
    prediction error, never assumed: **YIELD** (it moves by `T·efference`, and the body advances), **RESIST** (it does not move
    and the body is blocked -- solidity, learned), or **PASS** (the body moves through and it does not move -- non-solid). `T`
    (the change transform, identity = co-motion by default) is fit by the Widrow-Hoff rule, so one clean YIELD gives the change
    one-shot and generalises across push directions (the change is stored in the object-independent frame)."""

    def __init__(self, dims: int = 2) -> None:
        self.kind = UNKNOWN
        self.T = eye(dims)          # object displacement = T · efference; identity prior = "moves with the pusher", revisable
        self.n = 0                  # yields observed (for reporting)

    def observe(self, efference, body_disp, obj_disp, tol: float = 1e-6) -> None:
        """One felt interaction: the body pressed with `efference` (its forward-model-PREDICTED displacement) and ACTUALLY moved
        `body_disp` while the felt object moved `obj_disp`. Discriminate by the prediction error and update. YIELD is STICKY --
        once an object has been seen to move, it IS movable, and a later non-move is OBSTRUCTION (something behind it), not
        inherent resistance; so a proven-yielding object is never downgraded (only its change `T` keeps refining)."""
        if norm(obj_disp) > tol:                 # the object gave -> YIELD; fit the change from obj_disp = T·efference
            self.kind = YIELD
            self._fit(efference, obj_disp)
        elif self.kind == YIELD:                 # already proven movable -> this non-move is obstruction, not resistance
            return
        elif norm(body_disp) <= tol:             # pressed, neither moved, never yielded -> RESIST (the body could not advance)
            self.kind = RESIST
        elif norm(sub(body_disp, efference)) <= tol:   # body advanced through, object unmoved -> PASS (non-solid)
            self.kind = PASS

    def predict(self, efference):
        """The predicted outcome of pressing into this object with `efference`: `(object_displacement, body_blocked)`. UNKNOWN is
        the honest prior -- assume it neither moves nor blocks, revised the first time it is actually felt."""
        if self.kind == YIELD:
            return rotate(self.T, efference), False
        if self.kind == RESIST:
            return _zero(efference), True
        return _zero(efference), False           # PASS or UNKNOWN

    def _fit(self, e, d, lr: float = 1.0) -> None:
        """Widrow-Hoff: fit only the column of `T` that `e` constrains, leaving the identity prior on unobserved directions
        (bold generalisation). `lr=1` -> one clean observation sets the observed direction EXACTLY (no snap, no per-direction)."""
        ee = sum(x * x for x in e)
        if ee == 0:
            return
        self.T = _madd(self.T, _outer(sub(d, rotate(self.T, e)), e), lr / ee)
        self.n += 1


class ContactDynamics:
    """The object behavior model over a scene: one `ObjectBehavior` per felt object (keyed on its identity -- a colour for a
    single-cell object, a recognised id later). `observe` learns from a felt interaction; `predict` is the forward model the
    rollout queries. Object-independent behaviour (the SAME kind+T transfers to any object that exhibits it) with per-object
    dispatch on which kind each object has -- exactly TBP's object-independent, per-object-instantiated behavior model."""

    def __init__(self, dims: int = 2) -> None:
        self.dims = int(dims)
        self._by_object: dict = {}

    def _behavior(self, object_id) -> ObjectBehavior:
        b = self._by_object.get(object_id)
        if b is None:
            b = self._by_object[object_id] = ObjectBehavior(self.dims)
        return b

    def observe(self, object_id, efference, body_disp, obj_disp) -> None:
        self._behavior(object_id).observe(efference, body_disp, obj_disp)

    def predict(self, object_id, efference):
        """`(object_displacement, body_blocked)` for pressing the felt `object_id` with `efference`. An unfelt object is UNKNOWN
        (no move, no block) -- the honest prior until contact teaches otherwise."""
        return self._behavior(object_id).predict(efference)

    def kind_of(self, object_id) -> str:
        b = self._by_object.get(object_id)
        return b.kind if b is not None else UNKNOWN
