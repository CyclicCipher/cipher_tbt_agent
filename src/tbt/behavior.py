"""behavior.py -- the object BEHAVIOR model: "changes@locations", the touch-conditioned forward model over objects.

TBP (https://docs.thousandbrains.org/docs/object-behaviors): object dynamics is a behavior model storing CHANGES@locations (vs
the structural model's features@locations), in an object-INDEPENDENT reference frame, state-conditioned. TBP leaves the
INTERACTION open (contact, efference, action-from-behavior are "unresolved"); we fill it with the active-touch forward model
(notes/touch_and_body_design.md §7): the CONDITION is felt contact and the CHANGE is efference-parameterised.

The whole of it is the L5 `Transform` below -- ONE cortical layer plus its population read-out. There is no taxonomy of
behaviours here and no branch on one: what used to be YIELD / RESIST / PASS, each with its own stored state and its own rule,
are just three learned deltas, and the enum, the per-object dispatch and the "sticky yield" repair that propped it up are gone.
"""

from __future__ import annotations

from .htm import HTMLayer, PopulationReadout
from .operator import invert, rotate   # rotate(M, v) IS the matrix-vector product; reuse the algebra


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
    directions overlap 16/24 and adjacent magnitudes 20/24. `max_syn` has to clear the whole context width as well, or a
    segment cannot physically hold enough synapses to reach its own threshold and NOTHING is ever predicted. `min_threshold`
    matters just as much and for a subtler reason: it
    is the bar for REUSING an existing segment on a burst, so leaving it low lets a novel parameter hijack a neighbouring
    parameter's segment and drag it across instead of recruiting its own cell -- distinct parameters then silently share one
    delta. Holding it at the activation threshold means a burst always recruits, which is what a conjunction wants.

    NO PREDICTED-COLUMN PUNISHMENT (`predicted_dec=0`). In sequence memory a column that was predicted and did not activate is
    a genuine error and its segment should be weakened. Here the proximal drive is a QUERY -- the cues that happen to be
    present in this one interaction -- so the other columns being silent says nothing at all. With punishment on, learning
    about one thing silently erases what was learned about another whenever they share a situation: teaching the agent what a
    wall does wiped out what it knew about a block, because both were queried under the same press and each query punished the
    other's cell. The layer is an association store read by subset, not a stream in which absence is evidence."""

    def __init__(self, dims: int = 2, cells_per_column: int = 16, lr: float = 1.0, activation_threshold: int = 18,
                 min_threshold: int = 18, init_perm: float = 0.60, max_syn: int = 64) -> None:
        self.dims = int(dims)
        self.layer = HTMLayer(cells_per_column=cells_per_column, activation_threshold=activation_threshold,
                              min_threshold=min_threshold, init_perm=init_perm, max_syn=max_syn, predicted_dec=0.0)
        self.readout = PopulationReadout(dims, lr)

    def _assembly(self, cues, param):
        """The cells this basal `param` DEPOLARISES inside the proximally cue-driven columns -- the sensorimotor conjunction.
        Empty for a cue or a parameter the layer has not learned."""
        cols = set(cues)
        self.layer.depolarize(param)
        return frozenset(c for c in self.layer._predictive if c[0] in cols)

    def predict(self, cues, param, frame=None):
        """The predicted delta: the population vector of the assembly these cues-under-this-parameter drive, expressed in the
        caller's frame. `frame` is a rotation this population is tuned in -- the decode happens in that frame and is rotated
        back out. A cell population tuned in some frame other than the world's is ordinary cortex (egocentric, head-centred,
        object-centred populations all exist); this is a coordinate statement about the read-out, not a second mechanism."""
        d = self.readout.decode(self._assembly(cues, param))
        return d if frame is None else rotate(frame, d)

    def learn(self, cues, param, observed, frame=None) -> None:
        """One observation: run the layer (a novel conjunction bursts and grows its segment), then fold the read-out's error
        into the assembly that conjunction has settled on -- the same one `predict` decodes. `observed` arrives in the caller's
        frame and is carried into this population's own frame first, so what is stored is frame-invariant."""
        if frame is not None:
            observed = rotate(invert(frame), observed)
        self.layer.depolarize(param)
        self.layer.observe(cues, context=param, learn=True)
        self.readout.learn(self._assembly(cues, param), observed)
