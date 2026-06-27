# EfficientZero-V2 — thorough notes + mapping onto our neocortex

Source: **EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data** (Wang, Liu, Ye, You,
Gao — arXiv 2403.00564). Read 2026-06-26 to inform reworking our agent/neocortex. This is the detailed record;
the headline is at the bottom (§ "What to steal").

## 0. What it IS, in one paragraph
A **model-based, search-driven, online** RL learner that is SOTA at *sample efficiency* (Atari-100k, DMControl
Proprio/Vision), beating DreamerV3 on 50/66 tasks. It is MuZero's successor: learn a latent world-model, **plan
with a tree search at every acted step**, and train all networks against **targets the search itself produces**.
V2's job was to make this work for **both discrete and continuous** actions and **few** environment steps. There
is **no separate exploration module** — exploration is a side-effect of the search sampling low-prior actions.

## 1. The four learned functions (same as MuZero)
- **Representation** `H`:  s_t = H(o_t)            (obs → latent state)
- **Dynamics** `G`:        ŝ_{t+1}, r̂_t = G(s_t, a_t)   (latent transition + reward)
- **Policy** `P`:          p_t = P(s_t)            (prior over actions; Gaussian μ,Σ for continuous)
- **Value** `V`:           v_t = V(s_t)
Acting = run a search from H(o_t) using G/P/V to pick an action; learning = regress P,V,r̂,ŝ to search/observed
targets. An **action-embedding layer** maps actions to latent vectors so *similar actions sit close* (helps large
/ continuous action spaces).

## 2. Sampling-based Gumbel search (their headline planning change — replaces MCTS)
The expensive part of MuZero is MCTS with ~50 simulations. V2 uses **Gumbel search** (from Gumbel MuZero,
Danihelka 2022) made *sampling-based* for continuous control. Per acted step:

**Root:** sample K actions (default K=16) **without replacement** via the Gumbel-Top-k trick:
  `A_i = argmax_{a∉{A_1..A_{i-1}}} ( g(a) + logits(a) )`,  g ~ Gumbel(0).
  - For **continuous**: K actions split into `A_S1` from the current policy p_t **and `A_S2` from a *flattened*
    (high-entropy) prior p'_t**. The flattened half is the *only* source of exploration — "A_S2 can introduce
    actions that have a low prior under p_t."
  - For **discrete** (Atari): sample from the policy logits.
**Allocate the simulation budget** (default **32 sims**, works at **8–16**) across the K actions with
  **Sequential Halving** (Karnin 2013): run rounds; each round splits the budget evenly over the *surviving*
  actions, evaluate their Q via 1-step model lookahead + V bootstrap, keep the top half; after log2(K) rounds one
  action survives = `a*_S`. This concentrates compute on promising actions (a pure-exploration bandit).
**Non-root nodes:** sample *fewer* actions, only from p_t → deeper search for the same budget.
**Policy-improvement guarantee:** `q(s, a*_S) ≥ E_{a~p_t}[q(s,a)]` (Eq. 7) — the search action is provably no
  worse than the current policy in expectation.

## 3. Policy target from the search (completed-Q)
The improved policy the network regresses toward is built from **completed Q-values** (Gumbel MuZero):
  - visited action: `q(a) = r(s,a) + γ·v(s')`;  unvisited: `completedQ = Σ_a p(a) q(a)` (fill with the prior-
    weighted mean so unsearched actions still get a value).
  - target:  `π' = softmax( σ(completedQ) )`,  with monotonic transform
    `σ(q) = (c_visit + max_b N(b))·c_scale·q`,  `c_visit=50, c_scale=0.1`.
  - Two loss forms: **(a) distribution** `L_P = E_{a~π'}[-log p_t(a)]`; **(b) direct/"recommended action"**
    `L_P = -log p_t(a*_S)` — used for **large action dims** to get *early exploitation* (faster convergence).

## 4. Search-Based Value Estimation (SVE) — their value contribution
Instead of (or alongside) bootstrapped TD, compute the value target by **averaging the bootstrapped returns of
the search's own imagined rollouts**:
  `V̂_S(s0) = (1/N) Σ_{n=0}^{N} V̂_n(s0)`,   `V̂_n(s0) = Σ_{t=0}^{H(n)} γ^t r̂_t + γ^{H(n)} V̂(ŝ_{H(n)})`
  (N = #simulations, H(n) = depth of the n-th). It's "free" — computed inside policy reanalysis.
  **Error bound (Cor. 4.3):** MSE ≤ (4/N²)Σ_n[ Σ_t γ^{2t}(L_r²ε_s²+ε_r²) + γ^{2H}(L_V²ε_s²+ε_v²) ] → 0 as model
  error → 0; the `γ^{2t}` weighting **damps long-horizon model error** (key: deep imagined rollouts don't blow up).

## 5. Mixed value target — robustness to a wrong/early model
Don't trust the model's rollouts when the model is young or the data is stale-but-fresh:
  `V_mix = { multi-step TD  if (i_t < T1)  or  (i_s > |D|−T2);   V̂_S  otherwise }`
  - multi-step TD: `Σ_{i=0}^{l-1} γ^i u_{t+i} + γ^l v_{t+l}`, horizon **l=5**.
  - `T1≈100k` warm-up steps (model too young → trust real returns); `T2≈50k` (very recent transitions → trust
    real returns more); the **stale middle** uses SVE. Ablation: mixed > pure-TD and > pure-SVE everywhere.

## 6. Self-supervised temporal consistency (kept from EZ-V1)
`L_G(s_{t+1}, ŝ_{t+1}) = L_cos( sg(P1(s_{t+1})), P2(P1(ŝ_{t+1})) )` — **SimSiam** (Chen&He 2021): asymmetric
projector/predictor heads `P1,P2`, negative-cosine, stop-gradient. Makes the *predicted* next latent match the
*encoded* real next latent → a far richer signal than reward+value alone (this is what made EZ-V1 sample-efficient).

## 7. Priority precalculation (a small, cheap win)
New trajectories are normally inserted at max priority. V2 instead **warms up their priority with the actual
Bellman error under the current model**, so genuinely-surprising new data is replayed more (and boring new data
less). Cheap, improves early sample efficiency.

## 8. The training objective + loop
`L_t = λ1 L_R(reward) + λ2 L_P(policy) + λ3 L_V(value) + λ4 L_G(consistency)`, unrolled `l_unroll=5` steps and
averaged. Default λ ≈ reward 1.0, policy 1.0, **value 0.03**, consistency 0.1–0.2. Online: act-by-search →
store → sample by priority → reanalyze targets with the *current* net (this is where SVE rides along) → SGD.

## 9. Results worth remembering
Atari-100k normalized mean **2.428** (>BBF 2.247, >EZ-V1 1.945, human=1.0). Proprio mean **723** matching TD-MPC2
with **100× fewer imagined states**. Vision **726 vs DreamerV3 498** (+45%). Search at **8 sims still beats Sample
MCTS at 50**. So: *small search + good targets + consistency > big search.*

---

## What to steal — mapping onto our column system (the rework)
Our pieces already line up with MuZero's: **column = representation+dynamics** (SR-frame L6 + L5 operators + the
recurrence is exactly H/G), **reward.py = value+critic**, **basal ganglia = the policy/gate**, **thalamus =
cross-column routing**. So EZ-V2 is a near-direct template. Ranked by how hard it hits a gap we actually have:

1. **Sampled search + Sequential Halving instead of the flat subgoal enumeration. [biggest]**  reward.py today
   does prioritized sweeping over *every* enumerated subgoal — the 2^K wall (`scaling_probe`). EZ-V2 says: **don't
   enumerate, SAMPLE K candidate subgoals and let a pure-exploration bandit (Sequential Halving) spend a tiny
   budget finding the best**, evaluating each by rolling the column's recurrence forward 1–step + value bootstrap.
   Breaks the wall and is how a *discovered/affordance* subgoal set (B3) gets searched without enumerating.

2. **Exploration = a flattened prior in the search, NOT random collect, NOT a separate module. [our live gap]**
   Our LockPath-L2 failure is depth-of-exploration. EZ-V2's entire exploration is: at the root, sample some
   candidates from a *high-entropy* prior so low-prior options get tried, and let value pull the agent toward
   whatever paid off. Mapped to us: the agent should **plan-and-act online with an exploratory share of subgoals
   drawn from a flattened prior** (∪ our reward.py novelty bonus), which is how it would reach + learn deep
   mechanics — replacing the oracle teacher we just deleted. This argues for **merging collect into the agent
   (online learning)**: act by search (explores), learn the model/F/value from what the search reaches.

3. **Search-Based Value Estimation = vicarious evaluation by rolling the recurrence. [already half-built]**
   The EMERGENT_PLAN already says "rolling h forward values a subgoal without acting." EZ-V2 makes it concrete:
   value target = **average of N imagined-rollout returns**, with `γ^{2t}` damping deep-rollout model error. Drop
   this into reward.py as the value target instead of (or mixed with) the abstract-MDP backup.

4. **Mixed value target.**  Blend real observed returns (TD) with the column-rollout estimate (SVE), trusting
   real returns while the column's dynamics is still being learned (young model / very fresh data) and the rollout
   in the stale middle. A 3-line guard in reward.py; directly relevant since our model is *learned online*.

5. **Temporal-consistency (SimSiam) objective for the column.**  Our recurrence already predict-corrects
   (loc_move/loc_sense), but only against a sensed node. A SimSiam-style "predicted next latent ≈ encoded real
   next latent" objective is a far denser self-supervision for the SR-frame / L5 operators — the thing that made
   EZ sample-efficient. (We don't gradient-train today; this is the strongest argument for where a learned
   component would help.)

6. **Priority precalculation.**  In our prioritized sweeping, seed a new transition's priority from its actual
   surprise (prediction error), not a constant — replay surprising experience first.

7. **Action embedding** — for the **real ARC click(x,y)** action (a huge action space): embed click targets so
   nearby clicks generalize. Parks until we wire the click; flagged so we remember it exists.

### The one-line thesis
EZ-V2 = **(small sampled search) + (model-rollout value targets) + (self-supervised consistency) + (exploration
folded into the search via a flattened prior)**, run **online**. Our column already is the model; the rework is to
drive it with this loop — which simultaneously answers our planning wall (sampled search), our exploration gap
(flattened-prior search, online), and our value/credit question (SVE + mixed target). Open question to decide
next: do we keep the offline-learn / online-solve split, or merge to one online search loop like EZ-V2 (its
exploration story only works online).

## Re-read 2026-06-26 — what they SEARCH OVER, and the full-observability assumption (the key clarification)
`H` takes a **single observation** → latent `s₀`; it is **purely feed-forward + stateless** (no RNN, no belief, no
memory). The tree search runs **entirely in latent space** (`s₀ = H(o)`, then roll `G` forward; real observations
are never re-encoded mid-search). `G` is a **deterministic** point-estimate latent transition. The latent is shaped
**end-to-end by gradient** — the consistency + value + policy + reward losses jointly push `H` to encode whatever
predicts well. **There is ZERO discussion of partial observability, aliasing, memory, belief, or POMDPs — EZ-V2
assumes FULL observability** (frame-stacking, if any, is preprocessing *before* `H`).

**What this resolves for us (the `cell × open-doors` question).** EZ-V2 never builds NOR hand-codes an augmented
state — it doesn't need to, because the relevant state is **observable**: a closed door is *in the frame*. So the
latent just encodes the current frame, and the **learned value `V`** carries the multi-step logic — `V(at the key,
door-closed)` is high *because firing the key reaches the goal*. The "fire the key first" plan lives in the
**learned V over the latent** — a function that GENERALISES (no 2^K enumeration) — not in any hand-built state. So
a hand-coded `cell × open-doors` is wrong twice: hand-coded AND unnecessary. For full-obs games the clean design is
exactly theirs: latent = the column's encoding of the *visible* frame (the SR-frame map already IS this), `G` = the
learned effects, a **TD-learned `V` over that latent**, shallow search.

**The twist that is OURS, beyond EZ-V2.** Full observability means EZ-V2 has **no answer for the toggle** — a door
*invisible when open* is a hidden, aliased state that a stateless feed-forward latent cannot represent. That is
exactly where the column's **recurrence (a belief: "I pressed the switch") + CSCG cloning (split the aliased door
cell)** go beyond EZ-V2. We are *ahead* of them on that axis — and it is the genuinely novel, harder research.

**The line, now clean:** full-obs games → copy EZ-V2 (TD-learned V over the latent + search; NO cloning, NO
hand-coded state); hidden-state / partial-obs games (the toggle) → recurrence + cloning, which EZ-V2 structurally
cannot do.
