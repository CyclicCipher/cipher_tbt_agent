# The preference graph: its architecture (draft, 2026-10-21)

**The question (the user):** "How exactly are we going to make the preference/goal graph? What is the architecture of
the graph itself, rather than its content?" This document answers with a concrete design. The content (which nodes
exist) comes from `BEAUTY_MATH.md` and `PREFERENCE.md`; the evidence that shaped the design comes from
`BEAUTY_RESULTS.md`. Status: a design; only its first piece is built (`prefgraph.py`, tested in E10a).

## What the architecture has to do (requirements, each with its source)

1. **No fixed argmax.** E9: a fixed score, optimized, collapses onto one motif and gets hacked. A fixed weighted sum
   of nodes is one fixed score, so the combination of nodes must depend on the agent's changing state.
2. **Several time scales.** Fluency acts at once; boredom builds over exposures; usefulness shows up per episode.
   One-shot judgments see only the fast nodes (the RLHF failure).
3. **Some preferences are constraints, not trade-offs.** Harm is not exchanged for beauty at any rate.
4. **Preferences can acquire content from experience.** Kant §41–42: interest attaches *afterwards* to what was
   appreciated. E9c showed why it is needed: an appreciator without it consumes the beauty it learned from.
5. **Values live in the network's weights** (the user's standing requirement), and each node should be a locatable
   circuit, so the mechint question (G6) can be asked.
6. **Innate structure, learned content.** Evolution supplies the node types and the wiring. Experience supplies what
   each node is about.

## The three layers (from PREFERENCE.md)

| layer | what it is | computed by |
|---|---|---|
| **preference nodes** | terminal: why anything matters | innate signal functions over the agent's own quantities |
| **values** | instrumental: how much a state or action leads to what the nodes want | learned, through the world model: one value head per node |
| **goals** | the outcome selected now as the target of action | selection among the nodes' proposals, with commitment |

"Useful" is a value, not a node. A goal is the output of a decision, not a kind of preference.

## A node: its type signature

Every node has the same five parts, whatever its content:

| part | what it is | example (appreciation) | example (preservation) |
|---|---|---|---|
| **signal** σᵢ | an innate function computing this step's amount of the thing the node cares about, from the agent's own quantities (perception, world model, body) | the compression progress of the current view | minus the loss of appreciated structure in the world |
| **internal state** hᵢ | a level with a set point, moved by the signal and by time | (not built yet) unsatisfied curiosity, growing while nothing new is learned | (none needed) |
| **urgency** uᵢ = fᵢ(hᵢ) | how much the node matters *now*: the satiation mechanism | high when bored, low just after learning | constant |
| **content** Mᵢ | what this node has learned to care about | (the perceptual model itself) | the value of each learned structure, written by appreciation |
| **value head** Qᵢ(s, a) | the expected future signal, with the node's own discount γᵢ (its time scale) | learned by TD from σᵢ | learned by TD from σᵢ |

In reinforcement-learning terms, a node with a value head is a *general value function*: a cumulant σᵢ plus a
discount γᵢ, as in the Horde architecture (Sutton et al. 2011). Separate value heads per reward component are the
hybrid reward architecture (van Seijen et al. 2017). The internal state and urgency are homeostatic RL (Keramati &
Gutkin 2014): the reward is the reduction of the distance to a set point, so overshooting does not pay.

## Edges: the graph's architecture proper

Five edge types, each with a different job. These are what make the structure a graph and not a sum.

| edge | from → to | what it does | example |
|---|---|---|---|
| **gain** | a node's state → another node's urgency | multiplies: uⱼ ← uⱼ · gᵢⱼ(hᵢ) | hunger raises food's urgency (alliesthesia); fear lowers curiosity |
| **veto** | a constraint node → the action set | an action whose predicted violation exceeds a tolerance is removed before any trade-off; or a Lagrange multiplier that grows while the constraint is violated (a constrained MDP) | harm; preservation |
| **content** | a node's experience → another node's content | one node's signal history writes what another node cares about | appreciation → preservation: what was appreciated becomes what is protected (Kant §42) |
| **competition** | every node → the selector | nodes compete for control; the winner's proposal becomes the goal | the basal-ganglia selection |
| **derivation** | terminal nodes → values | the world model turns a node's signal into values of states and actions (no new node) | "useful" |

Two more structural facts:
- **The self-loop of satiation:** a node's signal feeds back into its own internal state.
- **Which edges exist, and of which type, is innate:** the wiring.

## The decision cycle (per step)

1. **Signals:** every node computes σᵢ from the new observation and the agent's own models.
2. **States and urgencies:** hᵢ moves (satiation, deficit), uᵢ = fᵢ(hᵢ), then gain edges apply.
3. **Content edges:** nodes write into other nodes' content (appreciation's felt progress into preservation's values).
4. **Proposals:** every drive node proposes its best action or goal from its value head (its own priority map).
5. **Vetoes:** actions that a constraint node predicts would violate it beyond tolerance are removed.
6. **Selection with commitment:** among the remaining proposals, the node with the highest uᵢ × advantage wins and
   keeps control until its goal is reached or another node exceeds it by a margin (hysteresis, against dithering).
7. **Act and learn:** the planner pursues the selected goal; every value head learns by TD from its own signal.

**Why this is not a weighted sum.** It reduces to one fixed objective, the maximizer, only if all three of these
are absent:
- urgencies depend on internal state, so the weights move;
- vetoes are lexicographic, so there is no exchange rate;
- content edges make what a node cares about non-stationary.

With any of them present, there is no fixed argmax for an optimizer to collapse onto.

## Where it lives in the network (values in the weights)

- **One network, as OneNet (E6):** a shared trunk (the world model's features) plus, per node, a signal-prediction
  head and a value head Qᵢ.
- **The internal states hᵢ are inputs** (the global token), so values and policy can depend on needs, and a probe can
  ask where hunger or boredom is represented.
- **The selection is computed from the heads** (urgency × advantage); a learned gating head is a later option.

What is fixed and what is learned:

| fixed (innate) | learned (experience) | meta-learned (the evolutionary outer loop, across worlds) |
|---|---|---|
| the signal functions σᵢ, the edge wiring and types, the satiation dynamics' form | the world model, the value heads Qᵢ, each node's content Mᵢ | set points, gains, discounts γᵢ, veto tolerances: as E8 learned its prior by empirical Bayes across worlds |

**The mechint question this makes concrete (G6):** is each node's value head a separable circuit? Does appreciation
read the world model's own prediction circuits, as the unification in `BEAUTY_MATH.md` predicts?

## Three things that are not the same node

The user's caution: preserving old beauty should not be conflated with doing no harm.

| node | its content | where the content comes from | built? |
|---|---|---|---|
| **preservation** | structure the agent has appreciated | its own appreciation history (a content edge) | E10a |
| **harm avoidance** | other agents' welfare: their health, their own preferences | a model of other agents (theory of mind); needs a world with others (Village) | no |
| **low impact** | any change at all, relative to doing nothing | none: content-free (side-effect penalties: relative reachability, Krakovna et al. 2019; attainable utility, Turner et al. 2020) | no |

- They can conflict: preservation would protect a dangerous structure that harm avoidance must remove.
- Low impact would block creation that preservation allows.
- Harm avoidance and preservation may both be vetoes, but with different content and different justification.

## Build order

1. **E10a (now):** two nodes, appreciation and preservation, joined by a content edge. Two combinations are compared:
   a fixed exchange rate (sum) and a veto. Built in `prefgraph.py` as a preference for the existing planner.
2. **Satiation:** internal states and urgencies (appreciation's boredom); selection with commitment.
3. **Value heads in one network** (OneNet-style), replacing the perfect-model planner's direct reward.
4. **Harm avoidance in Village**, with the dilemmas; low impact as a contrast.
5. **Meta-learning** the parameters across a world family (E8's machinery).
