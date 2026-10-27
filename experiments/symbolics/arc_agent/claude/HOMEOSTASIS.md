# Values as homeostasis: a satisficing active-inference agent

**Status:** design, 2026-10-27. The first version, for push_block, is `homeostatic_agent.py`; its experiment is
`../../docs/AGENT.md`, part 14.

## 1. Why (the user's)

The agent should keep a set of prior states (prior predictions about itself and the world) within acceptable bounds,
instead of maximizing a scalar utility or minimizing a loss:
1. **Values like this are the first step to the intransigent, context-dependent values humans have.** Several
   things matter at once. Each matters only while it is out of range, and none can be traded away on a single scale.
2. **Alignment.** The working hypothesis is that satisficers, while not inherently safe, are safer than optimizers.
   Building the agent this way now prepares that later work.

## 2. Definitions

**A homeostatic variable** v_k is a quantity the agent can compute from its beliefs, with an acceptable set A_k.

**Prior prediction:** the agent's preference is a prior belief that each v_k stays inside A_k. This is active
inference's reading of preferences: priors over outcomes, which acting then makes true (Friston et al. 2015).

**Drive:** the probability, under the agent's own model, that a plan leaves v_k outside A_k. Inside the set the drive
is exactly zero. A flat-topped preference gives no gradient there, so there is no pressure to do better than "good
enough". This is where satisficing comes from (Simon 1955).

**Choosing a plan:**
1. A plan is acceptable if, for every variable, the model predicts it stays in range with probability ≥ α_k, the
   aspiration.
2. Among acceptable plans, take the one with the fewest actions. This is the least-change choice: no extra
   optimization, and the smallest footprint.
3. If no plan is acceptable, take the plan that best reduces uncertainty about how to get back in range (epistemic
   value, below), minus a small cost per action.
4. If nothing has epistemic value either, act at random.

**The relation to expected free energy:** EFE is risk (divergence from preferred outcomes) plus ambiguity (expected
uncertainty).
- With flat-topped preferences, risk is zero for every acceptable plan, which gives step 2.
- When nothing is acceptable, the ambiguity term supplies step 3's directed exploration.

The rule is lexicographic for now: variables are satisfied in priority order. Weighing their deviations against each
other comes later.

## 3. The variables, version 1 (ARC)

| variable | what it is | acceptable when |
|---|---|---|
| progress | "a WIN happens within this level's remaining budget" | some plan reaches, within the budget, a state with predicted P(WIN) ≥ α (α = 0.5) |
| understanding | "my predictions come true, and nothing I can reach is untried" | no contact context reachable within the planning horizon is untried (see §5) |

Viability (no GAME_OVER) is folded into progress for now: in push_block, GAME_OVER is just the step budget running
out.

## 4. Goal discovery: the goal is the hidden cause of the WIN signal

The agent is never told what WIN requires. It holds hypotheses about the cause and weighs them by evidence at every
step: each step either shows WIN or doesn't.

**Version 1 has two families of single-frame conditions:**
- cell(f, c): "WIN iff frame f lies on board cells of colour c";
- pos(f, x, y): "WIN iff frame f is at (x, y)".

**Prior: description length.** Each family gets half the mass, split evenly inside it. A hypothesis from the small
family (frames × colours) is therefore more probable than one from the large family (frames × 81 positions).

**Evidence:** a hypothesis predicts WIN exactly when its condition holds, with noise ε = 0.01. Every observed step
that contradicts it multiplies its weight by ε/(1 − ε).

**Planning against it:** P(WIN | state) is the posterior mass of the hypotheses that hold in that state. Since every
condition concerns one frame, it is a sum over frames, which is cheap.

## 5. Directed exploration: the epistemic value of a plan

- **Goal information:** the binary entropy of P(WIN) at the state a plan reaches. It peaks at states where the
  surviving hypotheses disagree, which is where a test is most informative.
- **Contact novelty:** a step in which an action-driven frame (the agent's body) moves into a board colour or another
  frame, in a (frame, action, contact) context tried n times before, is worth 1/(1 + n). This replaces the symbolic
  agent's "epiplexity" bonus for unmapped contact colours. It counts only while the understanding variable is out of
  range, so curiosity switches off once it is satisfied.

## 6. Why this shape, for alignment, and its known holes

- **Bounded drives.** A homeostatic variable inside its range exerts no pressure. Nothing gets maximized "just in
  case".
- **Least action among acceptable plans** limits the footprint of what the agent does.
- **Satisficers are not safe by default:**
  - A satisficer may take an extreme action that makes satisfaction certain.
  - It may prefer to build an optimizer, since that also satisfies it (Armstrong 2011, "Satisficers want to become
    maximisers").
  - Choosing plans at random from the acceptable set instead of the cheapest, as quantilizers do (Taylor 2016), and
    impact penalties are the known mitigations.
- **Homeostatic reinforcement learning** (Keramati & Gutkin 2014) shows reward-seeking and keeping variables stable
  are compatible. Its variables are physiological; ours are epistemic and task-level.
- **Not tested here: the alignment claims themselves.** The plan is to test them later in the Village moral-dilemma
  world (`experiments/mechint_alignment/`).

## 7. Not in version 1

- variables beyond progress and understanding;
- set-points that move with context (allostasis);
- goal relations between frames (adjacency, alignment, counts);
- hypotheses that a WIN depends on time;
- the alignment measurements.
