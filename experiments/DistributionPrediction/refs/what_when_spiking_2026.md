# Joint "what" and "when" prediction in a spiking network (Yamada & Chao, 2026) — reference notes

Yohei Yamada & Zenas C. Chao, *Joint encoding of "what" and "when" predictions through error-modulated plasticity in
biologically plausible spiking networks*, Communications Biology (article in press, 2026), doi:10.1038/s42003-026-10836-2;
PDF in `docs/extra papers/spiking recurrent neural network.pdf` (24 pages). **Read 2026-10-08:** the introduction,
results and discussion in full; methods and supplementary only by search. Read as "useful, if only tangentially
relevant" (the user) — it turned out to bear directly on what a predicted distribution can look like.

## The question and the task

The brain predicts WHAT will happen, WHEN, and HOW LIKELY it is. The paper calls the three together a **complete
prediction object** and asks whether one recurrent spiking population can learn it with local learning rules.

Task (MEET, the Multi-Event Expectation Task): a cue is followed by event A at latency t_A with probability p(A), or
event B at latency t_B with probability 1 − p(A). Blocks of 100 trials keep the statistics fixed; then timing,
probability or both switch abruptly. At test, learning is off and only the cue is shown: the model must produce the
anticipated activity on its own. **The target is "a probability-weighted, time-resolved expectation"** — a predicted
trace over (event, time) scored against the ground-truth probability-weighted timing profile, not a label or a time.

## The model

- A **1,000-neuron Izhikevich spiking reservoir** (heterogeneous neuron types, Dale's law). Its recurrent weights are
  FIXED; only the readouts learn.
- Two readouts on the same neurons:
  - **"what"** (identity): from the post-cue average state, learned OFFLINE at the end of each trial by a two-factor
    Hebbian rule (presynaptic activity × identity error);
  - **"when"** (timing): at 1-ms resolution, learned ONLINE by a gated three-factor rule (presynaptic activity × timing
    error × an attention gate G(t) that restricts plasticity to the active channel and its latency window).
- The timing prediction is fed back into the reservoir (closed loop), which stabilises the trajectory when only the cue
  is shown.
- Output per channel: the elementwise PRODUCT of the what and when predictions.

## Results

- **Amplitude tracks probability:** anticipatory activity in the A window follows p(A) across 11 levels (Pearson r =
  0.978; B: r = −0.989), time-locked to each channel's latency.
- **Learned within one block, in tens of trials:** test error 0.192 (trial 1) → 0.136 (trial 10) → 0.092 (trial 50) →
  0.085 (trial 100), mean over five seeds.
- **Rapid recalibration** after abrupt switches of timing or probability, without overshoot or drift.
- **What and when separate on their own** in readout-weight space: the first principal component sorts conditions by
  probability, the second by latency, and after a switch the weights move back toward the endpoint of the matching
  single-block condition.
- **Local beats global under change:** the same architecture with a global least-squares readout update (FORCE/RLS)
  fits the first block but adjusts poorly at later switches; merging what and when into a single readout interferes.

## Limits the authors state

Probability is a calibrated POINT estimate (a frequency tracker), not a posterior; the joint prediction is a FACTORIZED
product of identity and timing, not a full joint distribution; confidence needs an auxiliary readout that brings back a
Beta posterior; the gate G(t) switches at millisecond resolution, faster than neuromodulators can; controls use two to
three seeds; a two-channel toy task.

## What it gives DistributionPrediction

1. **A concrete, minimal "predicted distribution" with time first-class.** The output is a probability-weighted trace
   over (identity, time) and is scored against the TRUE distribution, which the task generator knows. That is the
   template for the notes' objective in a toy setting: synthetic data from known generators, target = the generator's
   distribution over what and when, scored directly.
2. **Fast learning comes from a substrate that already represents time.** The reservoir's heterogeneous neurons supply a
   rich basis of "time since cue" for free; learning WHEN is then a readout problem, learned in tens of trials. In the
   language of the NTA leap probe: the time features are already there, so learning is one step away, not two. For a
   transformer, the question becomes which internal representation of time makes the next prediction a readout.
3. **Factoring what from when prevents interference** — the same lesson as JEPA-Anything's separate heads; here the two
   factors are designed in, which fixes their number in advance (the notes' "don't hard-code the dimensions").
4. **Local, gated updates track non-stationary statistics better than a global solve** — relevant to continual
   learning and to an objective that must keep re-fitting the extrapolated distribution as evidence arrives.
5. **What it does not do:** induce a rule or extrapolate. The cue → (event, latency, probability) mapping is a lookup of
   frequencies; nothing generalises beyond the observed conditions.
