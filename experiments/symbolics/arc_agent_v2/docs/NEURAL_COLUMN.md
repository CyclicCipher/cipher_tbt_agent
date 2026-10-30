# What a real column does that v2's can't (2026-10-29)

**The user (2026-10-29), on part 7's open question:** "the question exposes where your architecture is too rules-based
and not neural enough. We shouldn't have to ask the question at all if the cortical column were designed correctly.
Which leaves the question of what operation a real cortical column can do that ours can't."

**The question that exposed it** (LAB part 7, results, point 2):
- **The case:** in a world with only addition, a digit is always followed by "+" and a result always by "=".
- **What v2 did:** P3 lifted to phases saw two places that disagree and kept them apart, so the brain could not
  answer "3 + 4 + 2 =".
- **What I asked:** should the learner merge them anyway? That was a rule deciding whether two places are one.

This note is for the user to decide on. Nothing here is designed yet.

## 1. What v2's column is

- **The column's state:** one atomic hypothesis, its place (and its (phase, coordinate)).
- **How the frame is learned:** at sleep, by exact rules.
  - **P3:** same future, same place.
  - **P4:** a move is determined by where it takes one place.
  - Coordinates, shift classes, phase merges.
- **Its decisions are binary:**
  - a content is determined or not;
  - a pair is refuted or accepted;
  - two places are one or two.
- **Where it fails:** where the rules don't decide, the column is lost (its place becomes None), and the brain falls
  back on the hippocampus.

## 2. What a real column does (HTM's theory of the column)

1. **One input, many contexts, one population** (Hawkins & Ahmad 2016).
   - **The structure:** a minicolumn stands for the input. The cells inside it stand for the input in a context.
   - **Two encounters of one input:** "7 after a digit" and "7 after 3 + 4" share the same minicolumns, but use
     different cells.
   - **Two consequences:**
     - what they share generalizes (the minicolumns: the value);
     - what differs (the next symbol is "+" in one context, "=" in the other) is a different prediction from
       different cells. It is not a contradiction to be resolved.
   - **So:** nothing ever decides whether two places are one place.
2. **Surprise falls back to the general case** (bursting).
   - **The mechanism:** an input that no cell predicted activates every cell of its minicolumns. The input then stands
     for itself in all its contexts at once, and whatever was learned about it in any context applies.
   - **The context is then relearned** from what follows.
   - **For A1:** "+" after "3 + 4" would burst. "+" leaves the location where it is in every context it was seen in,
     the next digit shifts the location as digits always do, and "=" reads off the value. A chain is answered with no
     merge rule.
3. **The location moves whatever the context** (Hawkins, Ahmad & Cui 2017; Lewis et al. 2019).
   - **The mechanism:** a movement signal updates L6's location through the grid code, whether or not that movement
     was ever made from this location.
   - **v2 already does half of this:** its coordinate shifts globally. But its phase automaton can veto a move, and
     its place table can lose the column.
4. **Predictions are unions; learning is graded** (Cui, Ahmad & Hawkins 2016).
   - **The prediction:** a column predicts several next inputs at once: the union of its predictive cells.
   - **The learning:** synapses strengthen and weaken a little with each experience (permanences), online.
   - **Ambiguity:** it is held as a union, and narrowed by what comes next. v2 instead makes yes/no calls once, at
     sleep.

## 3. The gap, concretely

**The gap:** v2's state is one hypothesis, and its learning is a set of rules. A real column's state is a population,
a union of context-specific cells over shared minicolumns, and its learning is graded.

**What would change** (a proposal, not designed):
1. **The column's state** becomes a union of (phase, coordinate) hypotheses, not one.
2. **On an input it has no transition for,** the column bursts. It continues from every phase where that input is
   known, with the input's shift there, and does not get lost. Later inputs narrow the union, as predicted cells
   narrow.
3. **A prediction** is the union of what the hypotheses predict, weighted by how often each was seen.
4. **Part 7's phase merge** would become unnecessary: the union does what the merge did, and where contents
   disagree, the union keeps both. The decisive-cutover rule says to delete the merge in the same move.
5. **Further** (the larger step): replace the sleep-time rules (P3, P4) by graded, online learning in a minicolumn and
   cell substrate (HTM's temporal memory), with the coordinate as the L6 input. This would be its own part, or parts.

**What to keep:**
- the SDRs;
- the bus;
- the hippocampus;
- the coordinates. They are the "location moves whatever the context" part, and they are what made the reversal and
  the chains work.

## 4. Bayes (the user, 2026-10-29: "any use for Bayes' theorem and related here?")

**The user's order:** the union-of-hypotheses column is the next part after part 8, before precedence (B).

Bayes fits in three places. Each replaces a yes/no rule with graded evidence.

**1. The union's weights: Bayesian filtering.**
- **The state:** a union with weights is a posterior over (phase, coordinate) hypotheses.
- **On each input:**
  - every hypothesis moves by the input (its transition and shift);
  - its weight is multiplied by how well it predicted that input (the likelihood);
  - the weights are renormalized.
- **The prediction:** the posterior predictive, each hypothesis's prediction weighted by its posterior. This is the
  forward algorithm of a hidden Markov model (a particle filter when the set is sampled).
- **Bursting** is what happens when every hypothesis gave the input almost no probability: the posterior falls back
  to a broad prior, every phase where that input is known.
- **The neural version:** "probabilistic population codes" (Ma, Beck, Latham & Pouget 2006). A population's activity
  encodes a distribution, and adding activity multiplies likelihoods.
- **HTM's union** is the same thing without weights (only the support).

**2. Structure learning: model comparison instead of merge rules.**
- **The question becomes a posterior:** "are these two places one place?" is a comparison of two models by their
  marginal likelihood. Bayes' Occam factor penalizes the extra place unless the data need it.
- **The nonparametric form:** latent-cause inference with a Chinese-restaurant-process prior (Gershman, Blei & Niv
  2010). Each new situation is assigned either to an existing cause (place, context) or to a new one, in proportion to
  evidence × prior. Hippocampal remapping has been modelled exactly this way (Sanders, Wilson & Gershman 2020).
- **A link to what we have:** v2's MDL search (`_choose`) is already Bayesian in spirit (code length = −log prior).
- **For part 7's A1 question:** "merged" and "separate" would each get a posterior, not a verdict. With the union
  (§3), no single verdict is needed: both are kept, weighted.

**3. Arbitration between memory systems: part 8's recency rule, made graded.**
- **The principle:** trust whichever system is less uncertain (Daw, Niv & Dayan 2005: prefrontal against striatal
  control; Lengyel & Dayan 2007: episodic control wins when experience is scarce).
- **For part 8:** "today's memory at a location always wins" would become "the hippocampus's day memory wins where the
  cortex's predictions have been unreliable today". For example, after the relabelling, the cortex's day-1 labels keep
  failing, so its reliability falls.

**Related, already in the repo's knowledge base:**
- confidence is a population, not a scalar ([[reference_population_code_belief]] in memory);
- cue competition is the delta rule, a Kalman filter in its Bayesian form (Dayan & Kakade 2001);
- the goal cortex's active inference (ARCHITECTURE §7) is variational Bayes. Expected free energy is a Bayesian
  quantity, so "be correct" fits the same mathematics.

**For part 9 (the union column):**
- **Built:** filtering (1). The union carries weights from the start: counts as likelihoods, the posterior predictive
  as the prediction, bursting to the prior on surprise.
- **Recorded, not built:** model comparison (2) and graded arbitration (3) come later, where they replace a rule.

## Sources

- Hawkins & Ahmad 2016, "Why neurons have thousands of synapses, a theory of sequence memory in neocortex", *Frontiers
  in Neural Circuits* 10.
- Cui, Ahmad & Hawkins 2016, "Continuous online sequence learning with an unsupervised neural network model", *Neural
  Computation* 28.
- Hawkins, Ahmad & Cui 2017, "A theory of how columns in the neocortex enable learning the structure of the world",
  *Frontiers in Neural Circuits* 11.
- Lewis, Purdy, Ahmad & Hawkins 2019, "Locations in the neocortex: a theory of sensorimotor object recognition using
  cortical grid cells", *Frontiers in Neural Circuits* 13.
- Daw, Niv & Dayan 2005, "Uncertainty-based competition between prefrontal and dorsolateral striatal systems for
  behavioral control", *Nature Neuroscience* 8.
- Dayan & Kakade 2001, "Explaining away in weight space", NIPS 13.
- Gershman, Blei & Niv 2010, "Context, learning, and extinction", *Psychological Review* 117.
- Lengyel & Dayan 2007, "Hippocampal contributions to control: the third way", NIPS 20.
- Ma, Beck, Latham & Pouget 2006, "Bayesian inference with probabilistic population codes", *Nature Neuroscience* 9.
- Sanders, Wilson & Gershman 2020, "Hippocampal remapping as hidden state inference", *eLife* 9.
