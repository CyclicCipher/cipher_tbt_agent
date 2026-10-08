# Distribution Prediction — the idea in my own words

My rewording of the handwritten notes (verbatim in `TRANSCRIPT.md`), written 2026-10-08 to get a feel for the idea
before anything is designed (pages 1–4). Where I add something of my own or connect to other work, I say so. Figures are in
`figures/` (made by `make_figures.py`).

## Decisions so far (the user's)

- **2026-10-08 — the experiment starts from a TRANSFORMER and modifies it** (rather than from a CTM, a spiking network
  or a new substrate). The papers read so far (`refs/`) become candidate MODIFICATIONS: an internal time axis with
  statistics-of-trajectory read-outs (CTM), per-step denoising targets instead of backpropagation through time
  (DiffusionBlocks), separate predictors per factor (JEPA-Anything; the what/when readouts of the spiking-network paper).

## Where it comes from

Our experiments keep stalling at the same place: a model sits at chance for thousands of steps, then suddenly gets the
rule — the lookup in P0 sat at chance for 12,000 steps (NTA planning §20.1). That wait, "grokking", makes every
experiment slow and fragile. The user wanted a learner that goes to the rule directly instead of waiting for it to
appear, and looked for what that would take by doing it by hand: making pencil marks without looking, then trying to
work out the rule behind them. What failed in that attempt is what the notes are about.

## 1. A cloud of marks is not a rule; the order of the marks is the missing evidence

A finished picture tells you WHERE the marks are: the distribution. It does not tell you HOW they got there: the rule.
Many different rules leave the same picture. The rule lives in the relation between one mark and the next (from here,
turn a little, step, avoid what is already filled), and that relation is visible only if you know which mark came
after which.

`figures/fig1_marks.png` runs the rule the notes describe ("one mark after another, moving in a spiral, towards
whatever section of the frame was nearest and not already filled to the appropriate density"). Statically it is an
irregular patch; coloured by drawing order, the path is obvious.

In compression terms (my framing): given the order, each mark costs a few bits — a small turn, a step, an occasional
jump to the nearest empty section. Without the order, each mark costs as much as naming a point somewhere in the
patch. The order is what makes the data compressible, and compressing it is finding the rule.

What I think is NEW here, beyond "use sequences": a transformer reading the marks in order already has their positions
in the list. The notes ask for more — time as a coordinate OF the generating process, so the model can represent "the
pen's state now, and how it moves to its next state". That is a dynamical law, not a lookup over list positions. The
notes point to the user's 2D+1T RoPE work and to Sakana AI's Continuous Thought Machine (a network with its own
internal time axis) as supporting this, and to the cost of training through time (BPTT, stiff neural ODEs) as the
obstacle — with DiffusionBlocks (Sakana AI) as one way around it (see question 3 at the end).

## 2. "Noise" can be a rule that has spread itself out

If a rule spreads its marks as evenly as possible, the finished picture looks like random noise — and the picture
alone cannot tell it apart. `figures/fig2_dispersed.png`: 720 marks from the same spiral-filling rule next to 720
uniformly random marks, both static, then the rule's marks in order. The static pictures look alike; the ordered one
shows loops spiralling out and jumps to unfilled sections.

The notes hope this could help crack pseudo-random noise. I would draw the boundary here (my addition, from the
earlier cryptography discussion): a cryptographic generator is built so that even its ORDERED output tells an
efficient observer nothing — a keystream already arrives in order. Seeing time helps against rules that were never
designed to hide: a pencil spiral, physics, most of the world. The "perfect cryptography or perfect learner" question
the notes open with stays open either way. But the observation still matters: much of what looks like noise to a model
that only sees snapshots is a simple rule seen without its time axis.

## 3. Don't hard-code the coordinates

The rule in Fig 1 is simple when written in the pen's heading, its step, and how full each section is — not in x and
y. A positional scheme like RoPE or PoPE decides in advance which axes exist (position in the sequence; a 2-D grid;
time). A rule whose natural variables are something else — angle around a centre, distance travelled, local density —
is then hard or impossible to express. So a model must FIND the coordinates in which the rule is simple, rather than be
given a fixed set. (This is the same lesson as the JEPA-Anything discussion: understanding a planet means finding the
coordinates in which its motion is trivial.)

## 4. The shape of where the marks are is already evidence about how they were made

The notes' insight into Fig 1: the marks cannot have been random, because they do not cover the frame — they form an
irregular, path-like shape. A random scatter fills its region; a moving pen leaves a trail. So even without the order,
the shape of the distribution narrows down the kinds of process that could have made it. The notes say such
geometric intuitions "may be way less trivial than they look"; I agree — the static shape is weak evidence, but it is
free, and it is where a human starts.

## 5. Going beyond the data is a choice among rules, weighed by evidence

`figures/fig3_horseshoe.png`: 40 noisy points that look like a horseshoe. `figures/fig4_extrapolations.png`: two rules
that both fit them — a logarithmic spiral (which keeps opening out into something very unlike a horseshoe, as in the
notes' Fig 4) and a circle (the notes' "donut"). On these 40 points the spiral fits slightly better (root-mean-square
distance 0.0143 against 0.0159; a BIC-style score of −325.2 against −320.4, which makes the spiral about ten times
more probable under that approximation). So the evidence leans one way without settling it, while the two predictions
disagree completely outside the seen region.

The notes draw the right conclusion: we cannot list every possible rule, but we can propose candidates and weigh each
by how probable it is given the evidence, then extrapolate with those weights. (My gloss: that is Bayesian inference
with a prior favouring simple rules — Solomonoff induction is its ideal, uncomputable form.)

## 6. The research idea: predict the distribution of continuations, then the next token

Today's objective: given the context, predict the next token. The proposed objective: given the context, output the
best extrapolated distribution — what the whole continuation of this pattern could look like, and how likely each
version is — and only then predict the next token FROM that distribution.

Why this might skip grokking (my reading, to be tested): if what is scored is a distribution that has to be right far
beyond the seen data, memorising the seen data earns nothing; only something rule-like can extrapolate. The objective
rewards the rule directly instead of waiting for it to emerge from next-token fitting.

The notes name the hard part: the output has to be a "domain and dimensionality-agnostic thing". Three candidates I
can see:
- **a weighted set of explicit rules** — e.g. "spiral, b = 0.14: weight 0.9; circle: weight 0.1". The most
  interpretable, and it fits the project's founding purpose, but it needs a space of rules to choose from;
- **a weighted set of sampled continuations** — future points in the SAME space as the data, so domain- and
  dimension-agnostic by construction (a particle representation of the extrapolated distribution);
- **a density you can query at any coordinate** — a learned function from coordinates to likelihood.

Closest existing work, to place the idea (from memory, unchecked): neural processes (a network that outputs a
distribution over functions from a handful of context points, queryable anywhere); prior-fitted networks such as
TabPFN (trained on data drawn from a prior over generating functions, so they output the Bayesian posterior
prediction); Bayesian program learning and DreamCoder (explicit rules weighed by evidence). What the notes add to these
is the emphasis on time and discovered coordinates as first-class, and on the extrapolated distribution as an explicit
intermediate the model must produce.

## 7. What the notes predict if it works (page 4)

The notes stake out four predictions for a model trained this way: far fewer parameters for the same knowledge; circuits
that are easier to decompile; much higher data efficiency; and perhaps the same on language — the user offers their own
ability to improvise speeches they never wrote or practised as a hint that a human does this. My reading of why all
four would follow from one cause: a model that stores RULES instead of their many outputs needs fewer parameters (a
rule is shorter than its table), its circuits correspond to those rules (decompilable), it needs only enough data to
pick the rule out (data efficiency), and improvising is extrapolating a learned rule to a situation never seen.
These are claims about what success would look like, so each is something an experiment can measure — parameters
needed for a fixed accuracy, data needed before the rule appears, and how far extrapolation holds.

**A further possibility the notes raise:** if the model predicts the SPACE of a problem and its extrapolated
distribution, that prediction is a map — a navigable cognitive map the model can plan in, with the benefits of GCML
(planning by "goal minus where I am" in a space where that difference means something) and of how animals navigate and
solve problems. (My connection: the NTA lab book found that GCML-style planning works exactly where a map with real
geometry exists, §13–§15; here the map would come out of the training objective rather than be bolted on.)

**Two notes on what the output must be able to express:**
- *It need not be a function.* Restricting the prediction to curves y = f(x) — one value per input, drawable as a line —
  would rule out shapes (a circle is not a function of x; the donut in Fig 4 is not one) and anything not linearly
  separable, like XOR. So the output has to be able to represent relations and regions, not just graphs of functions.
  (That favours the second and third candidates in §6 — sampled continuations, or a density over coordinates — or rules
  whose output is a set.)
- *NAND and EML might be useful primitives for extrapolations.* NAND alone can build every Boolean function. EML, as I
  understand it, is the "exp-minus-log" operator eml(x, y) = exp(x) − ln(y), reported to generate all elementary
  functions from that one operator and a constant — the continuous counterpart of NAND. I have not verified that
  source. If extrapolations are written as rules, a single universal primitive gives the rule space a uniform grammar,
  and a natural simplicity measure: the number of primitive applications.

## 8. A first experiment I could imagine (nothing started)

- Data: mark sequences from several known rules — the spiral-filling pen, uniform random, a low-discrepancy sequence
  (deterministic but evenly spread), a walk with momentum, a horseshoe that is really a spiral.
- Measures: bits per next mark with the marks given (a) as an unordered set and (b) in order; how far beyond the seen
  region the predicted continuation stays right; and how much data each learner needs before it switches from fitting
  to the rule — the grokking question itself.
- Contrast: a next-token objective against a "predict the distribution of continuations" objective, same model size.

## Questions I would want answered before designing it

1. Which distribution is the output: over the next mark, over the whole future of the pattern, or over the rules
   themselves? The three lead to different architectures.
2. What counts as success: generalising sooner (fewer steps before the rule appears), or with less data, or
   extrapolating further?
3. The notes name BPTT and stiff neural ODEs as the obstacle for models with a real time axis, and point to HRM-Text's
   training method and DiffusionBlocks as ways around it. DiffusionBlocks (Shing, Koyama & Akiba, Sakana AI, ICLR
   2026; `docs/extra papers/DiffusionBlocks.pdf`) reads each residual update as one step of a denoising process, so
   every block — or every pass of a looped model — has its own target and trains independently, with no gradient
   through the others. On Geiping et al.'s recurrent-depth model (Huginn) it replaces 32 unrolled training iterations
   with single-pass training and does better (MAUVE 0.70 vs 0.49; perplexity under Llama-2 16.08 vs 17.04). So the
   obstacle may be removable: a CTM's internal ticks could be trained the same way, one denoising step per tick
   (my extrapolation, not something the paper tests). The CTM paper is now read (`refs/ctm_2505.05522.md`): it
   supports three of the notes' premises — time is a useful computational resource; a model can build its own
   coordinates by moving (its maze solver uses NO positional encoding); statistics of a trajectory beat snapshots as
   a representation — but its time is internal, not the order of observations, and it is not data-efficient
   (~10^6 iterations for mazes). Is the CTM line meant to be the substrate, or one of several?
