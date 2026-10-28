# Attention: how transformers learn it, why it works, and what the arc_agent's version should be

Written 2026-10-28 for item 3 of the open list (attention that depends on the situation). The user: making an actual
attention mechanism, not a cargo-culting of it, would solve the problem, but it forces the question of how transformers
learn their attention weights, and why.

## 1. What the mechanism computes

A query q (the current position's state) scores every key k_j (each context item), turns the scores into weights,
and returns the weighted values:

    s_j = q^T W_Q^T W_K k_j / sqrt(d),     a = softmax(s),     output = sum_j a_j v_j,   v_j = W_V x_j

Three readings of the same formula, each well established:
- **Kernel regression** (Nadaraya–Watson; Tsai et al. 2019): the output is a kernel-weighted average of the values,
  with a learned kernel exp(q^T W k).
- **Associative memory** (modern Hopfield networks; Ramsauer et al. 2021): one attention step is one retrieval
  step of a continuous Hopfield network whose stored patterns are the keys.
- **Posterior over sources:** if each key is the centre of an equal-variance Gaussian, the softmax of the scores is
  the posterior probability that the query came from each key. Attention weights are a posterior over "which context
  item is the relevant one".

## 2. How the weights are learned: the gradient is a responsibility error

When the output is a mixture of the values' predictions, p(y) = sum_j a_j p_j(y), the log-likelihood's gradient with
respect to key j's score is

    d log p(y) / d s_j = r_j - a_j,       r_j = a_j p_j(y) / p(y)

where r_j is key j's **posterior responsibility** for the outcome. Gradient descent moves attention toward the keys that
actually explained what happened: the E-step of a mixture of experts, taken one small step at a time (Jacobs, Jordan,
Nowlan & Hinton 1991; Jordan & Jacobs 1994).

Because the score is bilinear, s_j = q^T W k_j, the update to W is local:

    delta W = eta * sum_j (r_j - a_j) q k_j^T

That is query activity × key activity × a responsibility error: a three-factor Hebbian rule (Frémaux & Gerstner 2016).
No backpropagation is needed for this layer.

The values learn too. In a mixture of experts each expert is trained on the data weighted by its responsibility, so
experts **specialise** in the situations they are attended in. Several heads over the same keys let the same item be
read differently in different situations. Together these are how a transformer expresses conjunctions: which item to
read depends on the query, and what it means depends on which head read it.

## 3. Why it works

1. **The credit signal is right:** every update moves weight toward whatever predicted the outcome, in proportion to
   how much better it predicted than the current mixture.
2. **The parametrisation shares that credit:** the score is a function of the query's and key's features, with one W
   for every position and every context. What is learned about one key transfers to every key with similar features.
   W is low-rank, so it compresses "which item is relevant to which query" into a metric: relevance becomes closeness
   in a learned space.
3. **Positions enter as features of the keys** (positional encodings), so attention learns a metric over a geometry.
   This is the thesis's point in another language: the reference frame supplies the coordinates; attention learns
   which coordinates matter for which query.
4. **Composition:** stacked layers let one head's output become the next head's query (induction heads, Olsson et al.
   2022). In-context learning in linear attention is a gradient step on an implicit model (von Oswald et al. 2023);
   linear attention with the delta rule is a fast-weight memory (Schlag, Irie & Schmidhuber 2021), the same update as
   our Spotlight cells.

## 4. What the arc_agent does now, and why it isn't attention

Each expert's weight is exp(its accumulated log predictive probability): one global track record per (column,
action). Mathematically it is Bayesian model averaging over experts **assuming one expert is right in every
situation**. The weights do not depend on the query (the situation), and nothing is shared between keys: each feature
is its own table. That is why:
- conjunctions needed separate machinery (part 21);
- "own state" (an exact-position lookup) wins when no single feature is right everywhere (part 23's warm-up).

## 5. The principled version for a column (a proposal, to test first)

For each (column, action):
- **Keys:** the column's context features, each described by key features: its kind (content at a relative position,
  another column, a quotient, numeric), and for positions, (dx, dy) in the column's own frame, relative to the move's
  direction.
- **Query:** the situation, described by query features: the action, what lies beneath the column, the states of the
  other columns.
- **Values:** each key's table, from its value to the operator. Each head's tables are learned from the data weighted
  by that head's responsibility, so heads specialise.
- **Scores:** s_e = q^T W k_e, with H heads (H small), weights a = softmax over (heads × keys).
- **Learning:**
  - W by the local rule, delta W = eta (r - a) q k^T;
  - the tables by responsibility-weighted counts (EM).
- **What it would replace:** the track-record weights, and the conjunction cases (part 21), since the situation now
  selects the reading. The description-length priors (part 23) become the prior on W (small initial weights).

**What it should fix:** a door that blocks only when the switch is off. When the switch is on, the query moves
attention away from the door cell's "blocked" reading, to a head that learned "door → passes" in exactly those
situations. A single global weight per expert can't do that.

## 6. The feel-experiment before any change to the agent

A synthetic version of W7 (part 20): the outcome of a move depends on the cell ahead (floor, wall, door) and on a
switch (doors block only while it is off). The same single-feature experts and value tables as the agent; three
weightings compared:
1. the track record (the agent's now);
2. query-dependent attention by the local rule, 1 head;
3. the same with 2 heads and responsibility-weighted tables.

Accuracy over 2,000 steps. Results below.

### Results (2026-10-28; `structure/claude/attention_feel.py`; 5 seeds × 3,000 steps; a few seconds)

| weighting | accuracy, last 1,000 steps | log loss |
|---|---|---|
| A: track record (the arc_agent now) | 0.898 | 0.143 |
| B: attention by the local rule, 1 head, tables from every step | 0.898 | 0.146 |
| **C: attention by the local rule, 2 heads, tables weighted by responsibility** | **0.966** | **0.098** |
| C with 4 heads | 0.919 | 0.144 |

**Reading:**
- **Query-dependent weights alone (B) change nothing.** The door cell's table mixes its two meanings ("blocks" while
  the switch is off, "passes" while it is on), so whichever expert attention selects, the value it reads is wrong
  half the time.
- **The conjunction is solved when the values specialise too (C, 2 heads).** Each head's tables are learned from the
  steps it is responsible for. One head comes to read the door cell while the switch is off and learns "door →
  blocked"; the other covers the switch-on steps. The query routes between them. That is how a transformer expresses
  a conjunction: which item to read depends on the query, and what it means depends on the head that read it.
- **More heads were worse here (4: 0.919).** With symmetric starts the heads split the data and specialise more
  slowly in 3,000 steps, a known mixture-of-experts problem.
- **For comparison, part 21's conjunction cases** on the real W7 reach 0.942 at 300 steps and 0.996 at 1,500.

**What this establishes:** the principled attention is three things together, none of them bespoke:
1. scores that are a bilinear function of query and key features;
2. trained by the log-likelihood's own gradient, which is the local rule (responsibility − attention) × query × key;
3. values specialised by responsibility (EM).

Each is the transformer's mechanism with its learning written as a local rule.

## 7. What the agent showed (part 25), and the in-context version (built in part 26)

Part 25 put sections 5–6 into the arc_agent's columns (`docs/AGENT.md` part 25). It was worse than the track record
within a game's few hundred steps, for two reasons:
1. **The step:** the rule moves a score by at most η(r − a) per observation, while a position key starts about 5 nats
   behind (one of about 120). The track record moves about 4 nats per informative event.
2. **Starvation:** the tables learn with weight r, so a key that is barely attended never learns what its values mean.

Transformers have the same slowness in their weights and pay for it with data. Within a task they learn in context: the
query attends over earlier tokens, and the answer is read off what followed similar tokens. The weights only set the
similarity.

**The in-context version for a column** (for one (column, action)):
- **Memory:** each observation is stored as an episode: the situation's features, every key's value, the operator seen.
- **Query:** the current situation. **Keys:** the stored situations. **Similarity:** κ(q, q_t) = exp(Σ_f θ_f [q_f =
  q_t,f]), a weight θ_f per situation feature (another column's state, what lies beneath, a quotient).
- **Values:** what a key's value meant in similar situations, P_j(op | v, q) ∝ Σ_t κ(q, q_t) [v_t,j = v] [op_t = op]
  plus the backoff (kernel regression over episodes: Nadaraya–Watson).
- **Which key to read:** each key's evidence in similar situations, E_j(q) = Σ_t κ(q, q_t) log P_j(op_t), the
  prequential log probability it gave each past outcome; attention a_j ∝ prior_j exp(E_j(q)).
- **Learning:** only θ, by the log-likelihood's gradient (again a responsibility error, now times the feature matches
  of the episodes). The fast learning is the memory itself.

**The two older mechanisms are its endpoints:**
- θ = 0 (every situation is similar): E_j is the track record of part 20.
- θ_f large for chosen features (only exactly matching situations count): the conjunction cases of part 21.

The learned θ, in between, is the part neither older mechanism had: which features of the situation matter for which
action. As in a transformer, the weights set what counts as similar, and the context does the learning.

**Built in part 26 (`docs/AGENT.md` part 26).** On the synthetic switch-door task:

| learner | accuracy at 300 steps | at 3,000 steps |
|---|---|---|
| track record (θ = 0) | 0.896 | 0.898 |
| part 25's weight learning | 0.845 | 0.966 |
| **in-context, θ learned** | **0.977** | **1.000** |

The switch's θ rose to 1.4–2.0 by 300 steps and 18.5 by 3,000; the distractor's stayed near 0. In the agent, the ten
worlds match or beat part 24 (W7 0.950, the best so far), and the PFC learns to learn again.
