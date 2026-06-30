# ANN / binding fork — research notes

*Started 2026-06-29, ~24h before the ARC-AGI-3 milestone-1 deadline. We will likely miss M1; the real targets are
M2 and the bonus prize. This line runs ALONGSIDE the TBT agent in `src/tbt/` (not a replacement yet) — we do science,
try alternatives, and ground every claim with a baseline before building.*

## Premise of the fork

TBT (rebuilding intelligence from scratch in `src/tbt/`) works but is painfully slow to build and brittle. The doubt:
maybe a trainable ANN (gradient + scale) with the RIGHT inductive biases is the better substrate. **TBT was not
wasted** — it isolated the actual crux (**binding**) and produced concrete priors that become the ANN's inductive
biases: object-property binding (one bump → all walls; `src/tbt/behavior.py`), predict-then-compare, path-integration
(content↔location), learning-progress curiosity, covert rollouts.

**The Voynich framing (the user's).** The Voynich manuscript has predictable statistical structure but is undecoded
because nobody grounded its symbols — humans made *zero* progress despite being data-efficient. Internet-text → a
transformer is the same: distributional structure without a key. So the gap is **not (only) data efficiency** — it is
**grounding / binding**. Two senses, both relevant:
- **Symbol grounding** (Harnad / Voynich): connect symbols to referents. LLMs partly solve via scale + multimodality.
- **Feature binding** (Treisman): bind distributed features (colour, shape, location, value) into unified, reusable
  objects/roles WITHOUT crosstalk. This is what TBT reference frames solved.
- The operational test the user cares about (ARC): **generalize from ONE interaction** — bump a wall once → *all*
  walls non-traversable. That is binding a PROPERTY to a recognized TYPE + a strong type-level prior. RL becomes
  reliable when one mistake (run through a wall, lose action-efficiency score) is never repeated.

## The unifying insight (threads most of the questions into one)

**Binding = role ⊗ filler, and a Mamba state is already a binding memory.**
- The compositional binder is the **outer product**: TPR (Smolensky), compressed as **VSA / HRR** (circular
  convolution, Plate). Bind `value⊗place`, `effect⊗object`, `time⊗number`; superpose; unbind by the role. (TBT's
  thalamus already used VSA binding.)
- **The PoPE time-series result is a special case.** A positional embedding is a *fixed role vector*; "time as a
  dimension" = `number ⊗ time-role` with the role hardcoded. That is *why* it hit 100% with **no grokking**: the
  binding prior made the systematic solution the EASY one, so SGD didn't memorize-then-grok. It is *also* why it
  doesn't scale — the role is hardcoded.
- **Scaling fix:** LEARN the role vectors (don't add an axis per factor). Same outer-product binder, inferred roles.
  Honest caveat: VSA superposition crosstalks — a single dense vector has a hard capacity ceiling; this is the real
  research question, and it is exactly why TBT used MANY sparse reference frames instead of one dense vector.
- **A linear-attention / SSM state is a sum of outer products = a fast-weight memory** (Schmidhuber fast weights =
  Hebbian outer products = TPR). So **Mamba's recurrent state is already a role-filler store**; the Mamba3
  modification is not "add binding" but "structure the state update to bind the RIGHT roles to fillers and support
  clean unbinding." Then arithmetic (`digit⊗place`, carry = role-propagation), one-shot rules (`effect⊗object`), and
  in-context learning are the SAME operation in the same substrate.

## Thread verdicts — KEEP vs PASS

**KEEP / worthwhile:**
- **Online weight updates during a private game = ALLOWED.** Kaggle sandbox is local compute (no internet, 1 GPU,
  ~8h) → gradient steps on the agent's own experience (**test-time training**) are legal. Clean design = **two
  timescales**: slow pretrained weights (priors) + a FAST binding state updated online (the wall-rule in one shot).
  Fast-weight binding IS in-context learning without touching slow weights. Meta-learn the slow weights so few online
  steps suffice (sample efficiency).
- **The elementary binding test (the first experiment).** Each sequence first *defines* random bindings (`◆→+3`,
  `★→×2`, shown once) then *queries* them (`◆ 5 → ?`); **re-randomize the binding every sequence** so memorizing
  fails — the model must read the demo, bind it into state, apply it. Train on some symbols/ops, **test on held-out
  symbol↔op assignments**. Vanilla Mamba/transformer memorizes → fails the swap; a clean role-filler binder solves it
  from one demo. Same machinery as place-value → precondition for succession/carry/arithmetic. Mirrors the wall.
- **Anticipation / active inference = a GENERATIVE objective.** Mamba is already a generative latent-dynamics SSM
  (belief + transition + emission) — the right shape. Train it to PREDICT the next observation; prediction error is
  the surprise signal (the predictive-state requirement).
- **Exploration without a harness = Plan2Explore.** Port the TBT learning-progress + covert-rollout idea: intrinsic
  reward = expected information gain (model disagreement / prediction-error REDUCTION), estimated by IMAGINED rollouts
  of the generative model (cheap compute, expensive actions). The noisy-TV gate ports directly (disagreement that
  won't drop = noise). (Sekar et al. 2020, Plan2Explore.)
- **Event separation = a predictive model's SURPRISE spike (≈ a Wasserstein/OT jump) IS the boundary.** On the
  boundary, reset the FAST episodic state, keep the SLOW weights. Verify the predictive Mamba does this naturally
  before building a separate `events.py`. This same fast/slow split fights catastrophic forgetting — event separation
  and forgetting are two views of ONE mechanism + ONE control signal.
- **Optimal transport — useful, with Sinkhorn (the naive form is infeasible; entropic-regularized OT is O(n²) and
  differentiable).** Three uses: (1) DETECT shift — a Wasserstein spike = an event boundary (ties OT to event
  separation); (2) **Gromov-Wasserstein** compares distributions across DIFFERENT spaces (when the representation
  drifts); (3) OT-regularize continual learning — penalize the Wasserstein distance between old/new feature
  distributions to keep old representations stable (the EMD-against-forgetting intuition — a real family, not a
  guaranteed fix). Honest: research, not a silver bullet.

**PASS / deprioritize (with reasons):**
- **HNN (Hamiltonian NN) — PASS.** Hamiltonian = energy CONSERVATION, the right bias for *physical* systems. ARC
  rules are abstract, not physics. The right bias is generative-SSM + a FREE-ENERGY (predictive-coding) objective +
  the binding structure — Mamba already gives the first; "energy" in the free-energy sense is right, "Hamiltonian"
  (conservation) is a red herring. Revisit only if a specific game is conservation-governed.
- **STAR (github.com/Konstantin-Sur/STAR) — mostly PASS.** It is a lightweight O(n) gated recurrence:
  `scene ← LN(scene + tanh(tok_proj(t) ⊙ (1+scene_proj(scene))))`. Its "binding" is IMPLICIT multiplicative gating of
  a token against ONE dense scene vector — not slot/role-filler. Two problems for us: (a) largely REDUNDANT with Mamba
  (both O(n) state accumulators; Mamba's selective gating already does the multiplicative interaction, more
  principled); (b) a single dense scene vector is exactly the low-capacity superposition that crosstalks. The one
  transferable intuition — explicit multiplicative token×state interaction as a binding op — is better captured by
  VSA/TPR outer products. Early single-epoch WikiText result; no online/meta/generalization claims; not slot-based.

## The three problems, and how they connect

Not fully separate — **binding is the spine**:
- **Binding** (role⊗filler, learned roles, two-timescale Mamba state) → one-shot rules + arithmetic + in-context.
- **Catastrophic forgetting** → mostly a consequence of the fast/slow split binding implies (bind the new rule into
  fast state; protect slow weights), optionally OT-regularized.
- **Distribution shift / event separation** → the CONTROL SIGNAL (surprise / OT jump) that triggers "reset fast,
  keep slow." If binding is done right with two timescales, forgetting + event-separation are mostly consequences.

## Plan (grounded, incremental)

0. **Baseline (NOW): official Mamba on the public ARC-AGI-3 games, online learning, 4GB.** Expected to FAIL — the
   point is to SEE the failure mode and why (binding / sample-efficiency / grounding), to ground the claims.
1. **The meta-binding test** (above): vanilla Mamba vs Mamba + a VSA/outer-product binding state, on held-out
   bindings. Validate the core hypothesis cheaply before the full agent.
2. From there: the generative/predictive objective, Plan2Explore exploration, the two-timescale fast/slow split,
   event-separation-by-surprise. Each grounded by a test.

## Baseline result — step 0 DONE (2026-06-29): vanilla Mamba-3, online, on real ARC frames

Setup: official **Mamba-3** (Triton, runs on the 4GB 3050 Ti — peak **344 MB**, 0.54M params, d_model=96 ×2),
streaming the captured real frames; at each step predict the NEXT frame from (current frame + action) and take ONE
online gradient step (`experiments/mamba_arc_diag.py`). Decisive metric = accuracy on the cells that actually CHANGE.

| game | changed cells/step | copy-baseline acc | overall acc (early→late) | CHANGED-cell acc (early→late) |
|------|--------------------|-------------------|--------------------------|-------------------------------|
| cn04 (movement)      | 161 / 4096 | 0.961 | 0.917 → **0.962** | 0.236 → **0.230** (flat) |
| ls20 (animation+HUD) lr2e-3 | 39 / 4096 | 0.990 | 0.100 → **0.001** (diverged) | 0.017 → 0.000 |
| ls20 lr1e-4 (stabilised) | 39 / 4096 | 0.990 | 0.642 → 0.915 | 0.122 → **0.266** |

**Two failure modes, both grounding the thesis:**
1. **cn04 → the model learns the TRIVIAL COPY, not the dynamics.** Overall acc converges to the copy baseline (0.96)
   while CHANGED-cell acc is **flat at ~0.23** (no online improvement). With ~75 examples and no priors, SGD finds the
   easy solution (next ≈ current) and cannot BIND action→effect. (0.23 > copy's 0 on changed cells = a static "what
   changes become" prior, not action-conditioned dynamics.) This is the **binding/grounding gap**.
2. **ls20 → online SGD DIVERGES on the non-stationary noisy stream** (the HUD clock + colour animation = the noisy TV).
   At normal lr (2e-3/5e-4) it mode-collapses (overall acc → ~0); only a tiny lr (1e-4) keeps it stable, and even then
   it never reaches copy (0.92<0.99) and learns no dynamics (changed acc ~0.27). This grounds the **distribution-shift /
   catastrophic-instability** problem — and the learning-progress point (vanilla SGD chases irreducible noise → blows
   up; a learning-progress gate would ignore it).

Conclusion: the capacity is fine (it runs); the FAILURE is exactly the predicted one — no inductive bias for
action→effect binding (can't learn dynamics from little data), and no mechanism to ignore irreducible noise (instability
on non-stationary streams). NEXT = the meta-binding test: does a role⊗filler binding mechanism fix the first failure?

## Binding experiment step 1 (2026-06-29): MQAR associative-recall binding probe

`experiments/binding_mqar.py` — each sequence presents m random (key,value) pairs then queries them; binding
re-randomised every sequence (memorisation impossible → must bind in-context). Vanilla Mamba-3 (d=128 ×2, d_state=64):

| binding load m | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| recall acc | 0.994 | 0.992 | 0.998 | **0.176** | 0.113 |

A CLIFF at ~16→32. Characterised m=32: **6000 steps @ d_state=64 → 0.236** (NOT undertraining); **2500 steps @
d_state=256 → 1.000** (capacity fixed it). So the binding limit is the **dense-state CAPACITY CEILING** — exactly the
VSA superposition-crosstalk ceiling, and exactly why TBT used MANY SPARSE reference frames instead of one dense vector.

**The surprise / honest nuance:** *within* its capacity, vanilla Mamba does ONE-SHOT in-context binding PERFECTLY
(m≤16 → 99.8%, and the binding is novel every sequence). So "ANNs can't bind" is too strong — they CAN bind, one-shot,
up to a capacity. Therefore the cn04 baseline failure ("learns copy, not dynamics") is NOT explained by recall-binding
capacity; it is a DEEPER problem — learning the action-conditioned RULE/dynamics from few examples (rule-binding +
structured prediction), not associative recall. Two distinct things were being conflated under "binding."

**Takeaways:** (1) binding capacity is real and is set by state size → the scalable fix is SPARSE/structured binding
(TBT's many-frames, or VSA with sparse codes), not a bigger dense state (brute force, scales poorly). (2) The
ARC-relevant gap is RULE binding (bind a transformation from one demo, apply compositionally) — the next probe.

## Binding experiment step 2 + CONCLUSION (2026-06-29): RULE binding, and the fork is SHELVED

`experiments/binding_rule.py` (vectorised, single shape): bind a cyclic-shift rule from ONE demo, apply to a NEW
input (query input forced != demo input, so copy-the-demo = 0, chance = 1/V = 0.10). Load S=4 (within recall
capacity). **Vanilla Mamba-3 application acc = 0.309** — above chance/copy (it learned *something*) but FAR from
solving one-shot bind-and-apply. So the deeper, ARC-relevant binding (infer a rule from one interaction, apply
compositionally — the wall) is genuinely where vanilla breaks, even though recall-binding worked (step 1).

**DECISION (Cipher, 2026-06-29): shelve the ANN line, return to TBT.** Reason = ITERATION LATENCY. Vectorising the
batch fixed the per-step cost (11 ms/step; 2000 steps = 22 s), but the Triton kernel COMPILE (~84 s+) recurs every
fresh process and the persistent-cache fix did NOT hold on this Windows/Triton fork — runs took even longer. ~1.5–3
min per tiny test kills iterative science. The Mamba-3 architecture itself is fine (runs on 4 GB, 11 ms/step); the
tooling tax on this box is the blocker. TBT is already a live-running agent, much closer to a real submission.

**Findings to CARRY BACK to TBT (the fork paid for itself):**
1. **Binding capacity is a DENSE-STATE ceiling** (more state fixed MQAR m=32 instantly; more training didn't). This is
   the VSA crosstalk ceiling — it EMPIRICALLY VALIDATES TBT's choice of MANY SPARSE reference frames over one dense
   vector. Sparse high-capacity codes are the scalable binder.
2. **One-shot RECALL binding works; one-shot RULE bind-and-apply does NOT** (0.31). The ARC-relevant target is RULE
   binding — and TBT's recognition-keyed, revisable object-behaviour (bind a property/effect to a TYPE, apply to new
   instances; `src/tbt/behavior.py`) is exactly a hand-built solution to this. The fork confirms it's the right target.
3. **Vanilla online SGD DIVERGES on the noisy-TV stream (ls20)** — confirming that chasing irreducible prediction
   error is catastrophic, i.e. TBT's LEARNING-PROGRESS gate (`reward.py` epistemic="progress") was the right call.
4. The Mamba/SSM substrate stays a future option IF run on a Linux/CUDA box where the kernel-compile tax disappears.

## Honest risks
- The binding problem is genuinely UNSOLVED in ANNs; 24h won't crack it. This is a research bet.
- VSA capacity/crosstalk is the make-or-break of the single-vector binder; TBT's many-sparse-frames answer may be
  needed (→ a multi-slot / object-centric Mamba, not one dense state).
- Mamba's official CUDA kernels (`mamba-ssm`, `causal-conv1d`) are hard to build on Windows; a pure-PyTorch Mamba may
  be the only runnable path on this setup (slower, fine for a diagnostic).
