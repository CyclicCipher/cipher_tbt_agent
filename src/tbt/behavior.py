"""behavior.py -- the object BEHAVIOR model: "changes@locations", the touch-conditioned forward model over objects.

TBP (https://docs.thousandbrains.org/docs/object-behaviors): object dynamics is a behavior model storing CHANGES@locations (vs
the structural model's features@locations), in an object-INDEPENDENT reference frame, state-conditioned. TBP leaves the
INTERACTION open (contact, efference, action-from-behavior are "unresolved"); we fill it with the active-touch forward model
(notes/touch_and_body_design.md §7): the CONDITION is felt contact and the CHANGE is efference-parameterised.

The change is learned by the L5 `Transform` below -- ONE cortical layer plus its population read-out, no kinds and no branches.
"""

from __future__ import annotations

from .htm import HTMLayer, PopulationReadout
from .operator import add, eye, norm, rotate, sub   # rotate(M, v) IS the matrix-vector product; reuse the algebra

YIELD, RESIST, PASS, UNKNOWN = "yield", "resist", "pass", "unknown"


class Transform:
    """The L5 TRANSFORM -- an ordinary cortical layer read out as a metric quantity. NO bespoke machinery: it is exactly

        cells  =  HTMLayer(proximal = the CUES present, basal = the interaction PARAMETER)
        delta  =  PopulationReadout(cells)

    i.e. the canonical pipeline (`reference_htm_canonical_pipeline`) run to its VECTOR read-out. The two dendrite zones carry
    the two halves of the old hand-written form `delta = SUM_cue W[cue].[param;1]`, and carry them for the RIGHT reason:

    * PROXIMAL = the cues. Each cue drives its own minicolumns, so a set of cues drives the UNION of their assemblies and the
      population vector SUMS their contributions. Additivity over cues is a property of the code; there is no summation to
      write. (Putting cues in the basal zone instead does not work, and the reason is instructive: HTM only mints a new
      representation on a BURST, and a superset context always matches the subset's segment -- so {support, neighbour} could
      never acquire an assembly distinct from {neighbour}. Cues must DRIVE, not contextualise.)
    * BASAL = the parameter. The context selects WHICH cell fires inside each driven column, so the same cue under a different
      parameter is a different assembly and decodes to a different delta. That is the interaction term, formed for free
      (`htm.py`: "what drives the basal context is the ONLY thing that distinguishes the layers").
    * CUE COMPETITION -- the read-out's delta rule shares the error over the active cells, so a cue that merely co-occurs is
      blocked once the cells that really predict the delta explain it (`reference_cue_competition_key_discovery`).

    THE ASSEMBLY WE READ is the DEPOLARISED one -- the cells this parameter predicts inside the cue-driven columns -- never a
    burst. A burst means "no prediction", and every cell in a column firing is the layer's ambiguity signal, not a delta:
    decoding it would make an unlearned situation read out the average of everything ever seen. Reading the depolarised set
    makes "no evidence, no effect" automatic and leaves `layer.bursting()` free to be what it is everywhere else -- novelty.

    ONE-SHOT: `init_perm > connected`, so a segment grown in a single burst is connected immediately and one clean observation
    is exact. That is a per-layer permanence setting, the same knob that separates fast episodic binding from slow cortical
    statistics -- not a mechanism change.

    PRIORS: none. Weights start at zero and an unlearned cell contributes nothing, so an unseen cue predicts no change and an
    unseen parameter is not extrapolated. HONEST LIMIT: generalisation across parameters is what the parameter's ENCODING
    grants -- a bump-coded parameter shares bits with its neighbours, so a small change decodes the same delta and a large one
    decodes nothing. Plateau then cliff, not a smooth slope; anything smoother has to come from the code, never from here.

    RESOLUTION: `activation_threshold` and the caller's parameter ENCODER are ONE design decision, as they are for every HTM
    layer -- the threshold has to sit above the overlap between parameter values that must stay DISTINCT and below the overlap
    between values that should share a delta. The defaults here are set against a 24-bit grid-coded 2-D parameter, where unit
    directions overlap 16/24 and adjacent magnitudes 20/24. `min_threshold` matters just as much and for a subtler reason: it
    is the bar for REUSING an existing segment on a burst, so leaving it low lets a novel parameter hijack a neighbouring
    parameter's segment and drag it across instead of recruiting its own cell -- distinct parameters then silently share one
    delta. Holding it at the activation threshold means a burst always recruits, which is what a conjunction wants."""

    def __init__(self, dims: int = 2, cells_per_column: int = 16, lr: float = 1.0, activation_threshold: int = 18,
                 min_threshold: int = 18, init_perm: float = 0.60) -> None:
        self.dims = int(dims)
        self.layer = HTMLayer(cells_per_column=cells_per_column, activation_threshold=activation_threshold,
                              min_threshold=min_threshold, init_perm=init_perm)
        self.readout = PopulationReadout(dims, lr)

    def _assembly(self, cues, param):
        """The cells this basal `param` DEPOLARISES inside the proximally cue-driven columns -- the sensorimotor conjunction.
        Empty for a cue or a parameter the layer has not learned."""
        cols = set(cues)
        self.layer.depolarize(param)
        return frozenset(c for c in self.layer._predictive if c[0] in cols)

    def predict(self, cues, param):
        """The predicted delta: the population vector of the assembly these cues-under-this-parameter drive."""
        return self.readout.decode(self._assembly(cues, param))

    def learn(self, cues, param, observed) -> None:
        """One observation: run the layer (a novel conjunction bursts and grows its segment), then fold the read-out's error
        into the assembly that conjunction has settled on -- the same one `predict` decodes."""
        self.layer.depolarize(param)
        self.layer.observe(cues, context=param, learn=True)
        self.readout.learn(self._assembly(cues, param), observed)


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
