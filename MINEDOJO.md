# Minecraft Agent Design

> **⚠ MineDojo is abandoned.** MineDojo 1.11.2 + Forge has unresolvable
> dependency failures on any modern system (JCenter shut down 2021,
> MixinGradle build permanently broken, `raise ValueError("TODO")` stubs in
> instance.py, daemoniker incompatible with Windows). It cannot be fixed.
>
> **Replacement (Phase K):** A custom interface using `dxcam` (screen capture),
> `pynput` (keyboard/mouse), and `mcrcon` (RCON) provides the same observation
> and action API without touching any game files. It works with **any Minecraft
> version and any modded installation**, supports **player co-op on a local
> server**, and delivers **homeostatic data** (health, food, XP, time, position)
> via the server's built-in RCON console.
>
> Everything below — biological basis, CTKG edge types, agent architecture,
> Phases A–J — remains valid. Only the implementation of `MinecraftModality`
> changes in Phase K. See MEMORY.md "Custom Minecraft Interface — Phase K Design".

---

## Vision

Build an agent that learns an unlabeled knowledge graph through purely
unsupervised play in Minecraft, driven by intrinsic motivation to complete
the achievement tree. The agent is given:

- Basic motor priors (action primitives: move, look, attack, use, craft)
- Basic perceptual priors (visual primitives: DoG, Gabor, img_diff, color)
- The achievement DAG as its initial CTKG skeleton
- No biome tags, no entity API, no labeled images, no human reward

Everything else — object concepts, causal relationships, biome identity,
entity detection, crafting recipes, navigation — is discovered by the agent
from raw experience.

**The test:** Can a purely symbolic, zero-gradient system bootstrap from
nothing to completing the Minecraft achievement tree?

---

## Current State (updated 2026-03-06 — Phases A–J complete; Phase K custom interface next)

| Component | Status |
|---|---|
| CTKG + process language execution | ✓ implemented |
| Template synthesis (exact + approximate) | ✓ implemented |
| Visual primitives: DoG, Gabor, img_crop_rel | ✓ implemented |
| face_schematic as CTKG process expression | ✓ implemented |
| Arithmetic, calculus, fluid dynamics | ✓ implemented (run_experiment.py phases 1–17 all pass) |
| Phase A — MinecraftModality + minecraft.ctkg | ✓ 25 primitives, frame buffer, 3 drives; 50 concepts, 7 types |
| Phase B — New CTKG edge types | ✓ CausalEdge, CompositionEdge, InstanceEdge, TemporalEdge; 30/30 tests |
| Phase C — Dynamic CTKG extension | ✓ engine.add_concept(), add_{prerequisite,causal,instance,composition,temporal}_edge() |
| Phase D — Effectful interpreter | ✓ DryRunError, ProcessInterpreter.dry_run=True; synthesis always safe |
| Phase E — Online synthesis | ✓ engine.observe(), engine.curiosity(), engine.offline_consolidation() |
| Phase F — Causal synthesis templates | ✓ MC-1…MC-5 templates; gated by 'minecraft' in available_ops |
| Phase G — Exploration policy | ✓ next_frontier_concept(), highest_kl_rate_concept(), priority() |
| Phase H — Checkpoint system | ✓ save/load JSON v1.0; all concepts + stores + templates serialized |
| Phase I — agent_loop.py (full loop) | ✓ Written; smoke test passes |
| Phase J — VisualCortex + visual CTKG hierarchy | ✓ Log-polar + Gabor bank; texture_patch→block_face→block_type |
| **Phase K — Custom Minecraft interface** | **NEXT — dxcam + pynput + mcrcon (any version/mods, co-op)** |

---

## Biological Basis: What We Are Actually Building

The goal is a "digital monkey." This section grounds the architecture in
known biology and computational learning theory.

### 1. Dopamine and Reward Prediction Error

Schultz (1997): dopamine neurons encode the **reward prediction error**:

    δ_t = r_t + γ · V(s_{t+1}) − V(s_t)

where V(s) is the agent's current estimate of value from state s. Key
properties directly relevant to us:

- When reward is entirely unexpected: δ ≫ 0 (maximum learning signal)
- When reward matches prediction exactly: δ = 0 (no update needed)
- When expected reward fails to arrive: δ ≪ 0 (depression — avoid this path)
- Once conditioned, dopamine fires for the **cue**, not the reward itself

**Mapping to our system:** The KL divergence of a concept is our δ.
High KL = unexpected outcomes = maximum consolidation pressure. The agent
is drawn to explore high-KL regions because they offer the largest model
improvement. Once a concept is synthesized (KL ≈ 0), it stops driving
exploration for that concept — exactly like dopamine shifting to the cue.

### 2. Curiosity as Compression Progress (Schmidhuber 2010)

The agent is intrinsically rewarded for the **rate of improvement** of its
world model — specifically for the rate of decrease in description length:

    curiosity(t) = L(M_{t-1}, data) − L(M_t, data)

where L is the compressed description length (Kolmogorov complexity
approximation). Three regions:

| Region | dL/dt | Agent behavior |
|---|---|---|
| Already understood | ≈ 0 (already compressed) | Boring — leave |
| Learning actively | < 0 (compression improving) | Interesting — stay |
| Pure noise / chaos | ≈ 0 (incompressible) | Boring — leave |

**Mapping to our system:** The KL *decrease rate* between consolidation
steps is the curiosity signal. Concepts where KL is dropping fast (synthesis
is succeeding) are being actively learned — the agent should allocate
attention there. Concepts where KL is stuck high (incomprehensible) or
already low (learned) should be deprioritized.

The agent should track `ΔKL / Δt` per concept, and bias exploration toward
concepts with the steepest negative slope.

### 3. Free Energy Principle (Friston)

The agent minimizes **expected free energy** over future action sequences:

    G(π) = epistemic_value(π) + pragmatic_value(π)

where:
- **Epistemic value** = expected information gain = E[D_KL(P(s|o,π) || P(s|π))]
  → drives exploration: actions that reduce uncertainty about world model
- **Pragmatic value** = expected proximity to preferred outcomes
  → drives exploitation: actions that lead to achievement completions

These two terms are naturally balanced: the agent explores until it can
predict outcomes reliably (epistemic becomes small), then shifts to pursuing
preferred outcomes (pragmatic dominates).

**The "not-yet-discovered reward" problem:** The agent doesn't need to know
what a reward *is* to be motivated toward it. The epistemic term drives it to
explore high-uncertainty regions, which is exactly where undiscovered rewards
live. Reward discovery is a side effect of epistemic exploration.

**Mapping to our system:**
- Epistemic: `concept_entropy()` per unexplored concept in the CTKG
  frontier. High entropy → high epistemic value of exploring this concept.
- Pragmatic: proximity to the next frontier achievement in the achievement
  DAG. `graph.frontier()` returns the achievable-but-not-mastered concepts.
- Combined policy: explore toward `argmax(concept_entropy × frontier_proximity)`

### 4. Credit Assignment: How the Agent Learns What Caused What

**Biological mechanism — eligibility traces:**
When a synapse is active, it enters a temporarily elevated state
("eligible" for potentiation). When reward arrives later, synapses that
were recently eligible get strengthened. The trace decays exponentially:

    e(s_t) ← γλ · e(s_{t−1}) + 1[visited(s_t)]

**Computational equivalent:**
The agent maintains a sliding window of recent (observation, action) pairs.
When a reward event fires (achievement completion or inventory change), all
pairs in the window are tagged as potentially causal. The CTKG adds concept
nodes for these tagged events, and synthesis attempts to find a process
consistent with the recent sequence.

Window size = credit assignment horizon. Too short: misses long-range
dependencies (can't learn "collect wood → eventually craft pickaxe").
Too long: noise overwhelms the signal.

**Practical choice for Minecraft:** Window of ~50–200 steps (≈ 5–20 seconds
at 10 fps). Most immediate causal chains in early Minecraft are short (break
block → collect drop: ≈ 3–5 steps). Longer plans (craft table → craft
pickaxe → mine stone) are handled by the CTKG's sequential composition.

### 5. Metabolic Drives and Sleep-Coupled Consolidation

Three intrinsic metabolic drives form the survival substrate beneath all higher
cognition. They are the **minimum viable biological layer** — everything else
(curiosity, achievement) runs only when these are satisfied.

**Drive 1 — Pain (nociception, seconds timescale):**
```python
def U_pain(obs):
    health = obs['life_stats']['life']  # 0–20 HP
    urgency = 1.0 - (health / 20.0)
    # Panic threshold: nonlinear acceleration below 25% health
    if urgency > 0.75:
        urgency = 0.75 + ((urgency - 0.75) ** 0.5) * 0.5
    return urgency  # range [0, 1]; > 0.50 preempts all other drives
```
Models the fight-or-flight response. Above the preemption threshold, deliberate
cognition is suspended: goal stack is cleared, compiled reflexes take over
(flee, block, eat healing food). The fastest and most preemptive drive.
Negative δ events (damage taken) produce the strongest episodic encoding —
"bad days" create the most durable memories.

**Drive 2 — Hunger (metabolic, minutes timescale):**
```python
def U_hunger(obs):
    food = obs['life_stats']['food']  # 0–20
    return 1.0 - (food / 20.0)       # linear: 0 (full) to 1 (starving)
    # > 0.70 preempts achievement/curiosity; > 0.90 causes starvation damage
```
Shapes the agent's knowledge asymmetrically: hunger directs attention to food
sources → more food examples encoded → faster consolidation of food concepts →
richer semantic world model of food chains. A hungry agent becomes an expert
forager not by design, but because hunger makes food stimuli salient.

**Drive 3 — Sleepiness (circadian, hours timescale):**
```python
def U_sleep(obs, days_since_sleep):
    time_of_day = obs['location_stats']['time']    # 0–24000 ticks
    is_night = 13000 <= time_of_day <= 23000        # ~10pm–6am Minecraft
    phantom_risk = min(1.0, days_since_sleep / 3.0) # phantoms after 3 days
    return phantom_risk * (0.5 + 0.5 * float(is_night))
    # Day + safe:     ≈0.0  (no urgency)
    # Day + risk=1:    0.5  (plan ahead — find bed before night)
    # Night + safe:    0.5  (moderate — sleep soon)
    # Night + risk=1:  1.0  (maximum — sleep NOW)
    # > 0.60 preempts achievement/curiosity; triggers shelter-seeking
```
The only **predictable** drive — the agent can observe `time_of_day` advancing
and anticipate the need before it peaks. This uniquely forces **temporal planning**:
`find_bed precedes night`. Must complete multi-step preparation before the drive
peaks — the first drive that demands forward reasoning, not just reactive behavior.

**The phantom mechanic as a Challenge edge:**
Before the first phantom encounter, the agent holds a `heuristic` concept:
"outdoor_at_night → safe (default)". The first phantom attack is evidence that
challenges this: `Challenge(source="phantom_attack", target="outdoor_night_safety",
strength=1.0)`. After sleep + consolidation: `Override` on the default —
"outdoor at night, days_since_sleep > 2 → dangerous." The phantom mechanic
enforces the sleep drive via empirical learning, not pre-programmed rules.

**The complete drive priority cascade:**
```python
def current_priority(obs, memory, engine):
    pain   = U_pain(obs)
    hunger = U_hunger(obs)
    sleep  = U_sleep(obs, memory.days_since_sleep)

    if pain   > 0.50: return ("SURVIVE",  pain,   "flee_or_heal")
    if hunger > 0.70: return ("EAT",      hunger, "find_food")
    if sleep  > 0.60: return ("SLEEP",    sleep,  "find_bed_or_shelter")
    achievement = engine.next_frontier_concept()
    kl_concept  = engine.highest_kl_rate_concept()
    if achievement:   return ("ACHIEVE",  0.40,   achievement)
    if kl_concept:    return ("EXPLORE",  0.30,   kl_concept)
    return                   ("WANDER",   0.10,   "random_direction")
```

**Sleep as primary consolidation trigger:**
Consolidation happens when the agent successfully sleeps in-game (right-clicks
bed during night or thunderstorm). This is the primary consolidation event —
not checkpoint-based scheduling. The in-game sleep skip-to-dawn creates a literal
discontinuity in the experience stream during which offline processing occurs,
exactly mirroring biological sleep. The day/night cycle (~20 real minutes per
Minecraft day) imposes a natural consolidation rhythm.

On `sleep` event, the agent runs `engine.offline_consolidation()`:
1. `consolidate()` on all concepts with KL > threshold (episodic → semantic)
2. `sheaf_merge_trigger()` on concept pairs (concept compression)
3. Prune example stores for KL ≈ 0 concepts (rule absorbed into semantic layer)
4. Update causal graph with confirmed eligibility-trace hypotheses
5. Reset `days_since_sleep`, clear phantom risk

The amount learned during the day determines what consolidates at night: a high-curiosity
day produces rich consolidation; a survival-dominated day produces sparse consolidation
but strong Challenge/Override updates from the aversive events.

**The causal subgraph the agent must learn from experience:**
```
time_approaching_night  --precedes-->   sleepiness_rises
days_since_sleep > 2    --causes-->     phantom_spawn_risk  [guard="is_night AND is_outside"]
phantom_spawn_risk      --causes-->     phantom_attack
phantom_attack          --causes-->     player_damage
use_bed                 --causes-->     sleep  [guard="is_night OR is_thunderstorm"]
sleep                   --causes-->     days_since_sleep_reset
sleep                   --causes-->     dawn_arrives
sleep                   --causes-->     CONSOLIDATION_BATCH  ← game event triggers computation
```
None of this is pre-declared — the agent discovers these causal relationships through
experience. The phantom danger is discovered when the agent fails to sleep; the
bed mechanic is discovered when the agent experimentally uses one.

### 6. Memory Architecture: Short-Term, Long-Term, and Personal Narrative

The agent requires three dissociable memory systems (Squire 1987, CLS theory:
McClelland, McNaughton, O'Reilly 1995), each with different time constants and
storage formats.

**The three systems — already partially implemented:**

| System | Biological substrate | Time constant | Agent equivalent |
|---|---|---|---|
| Working memory | Prefrontal cortex | Seconds | Frame buffer + eligibility traces + goal stack |
| Episodic memory | Hippocampus | Minutes–years | `ExampleStore` — raw (input, output) pairs |
| Semantic memory | Neocortex | Permanent | CTKG processes — synthesized rules |
| Procedural memory | Basal ganglia | Permanent, compiled | Dynamic template library + macro-actions |

**Working memory (short-term, volatile):**
- `frame_buffer` — last K=8 frames (for img_diff, img_temporal_std; entity detection)
- `eligibility_traces` — last M=50–200 (obs, action, delta_inventory) tuples
- `goal_stack` — depth ≤ 5, drive-priority-ordered (survival preempts everything)
- `active_candidates` — concepts currently accumulating examples (≤ 5, highest KL)

**Episodic → Semantic consolidation (`consolidate()` IS the HPC transfer):**
The `ExampleStore` is the hippocampus — fast, specific, context-rich.
The synthesized `concept.process` is the neocortex — slow, general, context-free.
`consolidate()` is the hippocampal replay: many specific episodes → one general rule.
The episodes are *retained* after consolidation (catastrophic forgetting avoided —
same reason the hippocampus doesn't erase memories when neocortex learns the rule).

**Semantic → Procedural (habit formation):**
When a process is invoked successfully > N times, it is promoted to a *compiled
macro-action* — a cached lookup that bypasses deliberate step-by-step execution.
Biological analog: PFC-dependent deliberate action → basal ganglia habit (no longer
requires attention). The eligibility trace then assigns credit at the macro level,
making multi-step plans treatable as atomic for higher-level planning.

**Consolidation schedule (the "sleep" analog):**
Biological consolidation happens during sleep (hippocampal SWS replay). Agent
equivalent — offline consolidation at natural rest points:
1. Achievement unlock → consolidation batch ("reflect on what just changed")
2. Death/respawn → consolidation batch ("wake up fresh")
3. Checkpoint save → consolidation batch (scheduled)
During offline phase: run `consolidate()` on high-KL concepts, run
`sheaf_merge_trigger()` on concept pairs, prune example stores for KL ≈ 0 concepts
(the rule exists; the raw episodes are no longer needed for re-derivation).

**Personal narrative — the self as causal CTKG path:**
The "narrative" is the causally-structured event log. Not a flat chronological
sequence but a causal story:
```
episode = {timestamp: 12450, event: "consolidated", concept: "log_detection",
           triggered_by: ["repeated_failures"], enabled: ["wood_collection"],
           emotional_valence: delta_kl = -3.2}  ← satisfaction of learning
```
The "chapters" are high-δ events: achievement unlocks (+), deaths (−), first
consolidations (compression progress spike). Between chapters: routine (compiled
habits). This IS the Schmidhuber curve — narrative chapters = non-flat KL regions.

The self = the causal path through the CTKG from past to present:
"Curiosity drove me to explore → found tree → broke it (learned) → got wood →
achievement unlocked → new frontier opened → now crafting." Two agents with
identical CTKG processes but different episode logs have different selves —
different histories, different learned narratives.

**The motivation-memory loop (the closed cycle):**
```
Drive state → attention direction → episodic encoding → consolidation →
semantic world model → better predictions → lower KL on known concepts →
curiosity shifts to next high-KL concept → drive state update
```
The monkey's semantic knowledge IS shaped by what it was motivated to attend to.
Hungry monkey → rich food-chain knowledge. Curious monkey → broad shallow world
model. Drive hierarchy determines the agent's "interests" which determine its
knowledge which shapes its future drives. This is not incidental — it is the
mechanism by which biological motivations shape biological intelligence.

**Curiosity gradient (the attention signal at any moment):**
```python
curiosity[concept] = kl_history[concept][-1] - kl_history[concept][-K]
# Negative slope = actively learning = stay, encode more examples
# Near-zero slope + high KL = incomprehensible noise = move on
# Near-zero slope + low KL = already mastered = move on
# Most negative slope = peak Schmidhuber "interesting zone" = maximum reward
```

### 7. The Anticipation Problem

*"How does the agent anticipate which actions lead to rewards it's still
discovering how to acquire?"*

Three mechanisms at different timescales:

**Short-term (steps):** Planning with known processes.
`lookup(action_consequence, state)` chains known causal edges: "I know
break(log) → wood_drop, and wood_drop → inventory_increase, therefore
break(log) will eventually increase inventory."

**Medium-term (minutes):** Analogy via functors.
If the agent knows `break(grass) → seed_drop`, and knows a functor mapping
grass-actions to log-actions (same verb, different object), it anticipates
`break(log) → something_drop` without knowing what the something is.
The structural relationship (causal shape) transfers even without content.

**Long-term (hours):** Epistemic drive fills the gap.
For unknown territory, the agent doesn't need to anticipate *what* reward
it will find — it just needs to be motivated to explore high-entropy regions.
The FEP epistemic term provides this without requiring any specific goal model.
The achievement DAG tells it *which direction* to bias exploration.

---

## Granularity, MDL, and Sheaf-Driven Concept Induction

The right granularity for a concept is determined by **minimum description
length (MDL)**: the concept boundary that minimizes the total description
length of (world model) + (agent's experience given the model).

**The oak/birch case — why merging is always more efficient:**

The agent learns `oak_log_presence` and `birch_log_presence` separately (different
contexts triggered different concept creation events). The synthesizer finds the
*same* process template P₁ covers both with identical accuracy. At that point:

```
Before merge:  oak_log_presence: P₁ + N causal edges
               birch_log_presence: P₁ + N causal edges     ← P₁ duplicated
               description length: 2 × (P₁ + edges)

After merge:   log_presence: P₁ + N causal edges
               oak_log instance_of log
               birch_log instance_of log
               description length: (P₁ + edges) + 2 short instance_of edges
```

The merge is always shorter *and* generalizes immediately to spruce_log, jungle_log,
etc. — the agent adds `spruce_log instance_of log` and gets the correct process for
free, without new examples.

**Merge trigger (three conditions must all hold):**
1. Same process template (synthesizer's dynamic library finds a common template)
2. Same causal signature (same set of `causes` edges: break → wood_drop, etc.)
3. Compatible interface types (A's example outputs ≈ B's example inputs)

If (1) and (2) hold but (3) fails: visual similarity is accidental — do NOT merge
(e.g., `oak_log` and `mushroom_block` may share visual texture but have different
causal effects; keeping them separate is correct).

**KL as the MDL proxy:**
High KL on a concept = the model isn't compressing this experience = the concept
boundary is wrong. The agent should split or merge when this improves total KL.
KL drop after merge = compression gain = MDL improvement confirmation.

---

## Sheaves in Unsupervised Learning: Recovering the Typed Graph

The typed CTKG relies on pre-declared types for composition compatibility and
transfer. The unsupervised MineDojo agent cannot use pre-declared types — it must
discover them. Sheaf theory is the formal framework for this.

### The Presheaf of Experience

Define a presheaf F over the poset of experience contexts (ordered by generality):

```
F(oak_log, forest, day) = {process P₁, causes: break→wood_drop, ...}
F(birch_log, forest, day) = {process P₁, causes: break→wood_drop, ...}
F(log, forest) = limit of the above — what both sections agree on
```

Restriction maps forget specificity: `F(oak_log context) → F(log context)` strips
color details while preserving structural regularity. The **sheaf condition**: if two
local sections agree on their overlap, they can be glued into a global section.

Oak_log and birch_log agree on "cubic block → wood_drop", so they glue into `log`.
Mushroom_block shares the visual process P₁ (large brownish cubic thing) but
*disagrees* on its causal edges → sheaf violation → no merge, concepts stay separate.

### Sheaf Cohomology as Error Signal

**H¹(CTKG, F) = 0**: all local observations glue cleanly → concept boundaries correct.
**H¹(CTKG, F) ≠ 0**: irreducible local conflict → one of three resolutions:

1. **Merge**: the conflict is resolved by creating a supertype (the global section exists
   at a higher level of abstraction)
2. **Split**: the concept was too coarse — it conflates two distinct things that don't
   glue; split into two disjoint concepts
3. **Override**: the conflict is a genuine exception, not a boundary error →
   add an `Override` edge (already implemented): "this instance violates the default"

The KL divergence already tracked per concept is a practical approximation of the H¹
contribution. High KL = local inconsistency = H¹ contribution from this concept.

### Recovering Types Without Pre-Declaration

Types in the typed CTKG are objects in the category. Objects are discovered bottom-up
as **interfaces between morphisms**: the implicit type at the interface of concepts A
and B is the equivalence class of all concepts that can be composed at that point.

**Interface compatibility check** (replaces declared type annotations):
```python
def compatible_interface(engine, concept_A, concept_B):
    """A's output type ≅ B's input type? Check via distribution overlap."""
    A_outputs = [ex.outputs for ex in engine.stores[concept_A].examples]
    B_inputs  = [ex.inputs  for ex in engine.stores[concept_B].examples]
    return distribution_overlap(A_outputs, B_inputs) > THRESHOLD
```

**What this recovers from the typed graph:**

| Typed graph advantage | Unsupervised recovery |
|---|---|
| Type-safe composition | Interface compatibility check on example distributions |
| Transfer via supertypes | `instance_of` edges from MDL merge trigger |
| Sheaf consistency | Same `sheaf_check()` — on *discovered* types, not declared ones |
| Type mismatch detection | H¹ ≠ 0: process agrees, interface disagrees → no merge |
| Curriculum ordering | Unchanged — `requires` edges are epistemic, orthogonal to types |

**The key result:** the unlabeled knowledge graph discovered by the agent will
develop a type structure correct for the actual Minecraft world — rather than
whatever types a human designer anticipated. If spruce logs have identical behavior
to oak logs, they merge. If a future Minecraft update gives oak logs a unique recipe,
the sheaf condition prevents the incorrect merge automatically.

---

## Category Theory Relationships (Beyond Prerequisites)

The CTKG already has a rich relationship type system. This section first
documents what **already exists**, then identifies what must be **added for
MineDojo**.

### Already Implemented in graph.py

```
requires       -- EPISTEMIC PREREQUISITE (Prerequisite class)
                  "A must be learned before B" — the core curriculum edge
                  Carries: transfer_probability (Markov kernel), invertible flag,
                  assuming context (conditional prerequisites), assumption_status
                  e.g. wooden_pickaxe_use requires wooden_pickaxe_craft [1.0]
                  e.g. swimming requires buoyancy via "physical basis" [0.75]

inverse_of     -- ADJUNCTION (Adjunction class) — ALREADY IMPLEMENTED
                  Forward/inverse pairs with unit/counit expressions
                  e.g. add adjunction sub (arithmetic.ctkg: adjunction add_sub)
                  e.g. pickup adjunction drop (when added to minecraft.ctkg)

maps_to        -- FUNCTOR (Functor class) — ALREADY IMPLEMENTED
                  Structure-preserving maps between domains
                  e.g. universal_syntax → english_syntax (english_syntax.ctkg)
                  Preserves: composition, concept structure, prerequisite ordering
                  concept_map: {source_concept: target_concept}

challenges     -- CHALLENGE EDGE (Challenge class) — ALREADY IMPLEMENTED
                  "Evidence E weakens claim C" — epistemic counter-evidence
                  strength: 0.0 = weak hint, 1.0 = full refutation
                  e.g. non_euclidean_geometry challenges parallel_postulate
                  validate() flags ChallengedConjecture on conjecture-tier concepts
                  what_if_not() answers "what's unlocked if we drop this assumption?"

overrides      -- INSTANCE EXCEPTION (Override class) — ALREADY IMPLEMENTED
                  "Instance I deviates from heuristic default D on property P"
                  The Fido problem: "dogs have 4 legs" default overridden by "Fido has 3"
                  resolve_default(concept, property, instance) applies override before default
```

Already implemented **graph operations** (also part of the answer):
- `d_separated(x, y, given)` — Bayes-ball conditional independence (Fritz & Klingler 2023)
- `concept_entropy()`, `conditional_entropy()`, `mutual_information()` — Shannon entropy
- `intervene(do_concepts)` — Pearl's do-operator / string diagram surgery (Jacobs et al. 2019)
- `sheaf_check()`, `sheaf_merge()` — cross-domain consistency
- `MasteryState` — Bayes filter for per-concept mastery (Fritz et al. 2024)
- `what_if_not()` — epistemic counterfactual ("what if we dropped this assumption?")
- `challenged_concepts()` — query all active challenge edges
- `assumption_dependents()` — what depends on a specific assumption
- `resolve_default()` — Fido problem resolution

### Edge Types to Add to graph.py (MineDojo requires)

Four new edge types. `analogous_to` was considered but is **not needed** — analogy
is derivable from `instance_of` spans (A ← log → B implies oak_log ≅ birch_log)
plus the dynamic template library already reusing proven patterns across compatible
concepts.

```
causes         -- CAUSAL EDGE (new: CausalEdge class)
                  "Action/event A physically produces state/item B"
                  Categorically: Kleisli(State × FinStoch) morphism.
                  Distinct from 'requires' (epistemic): requires is about what you
                  must *know*; causes is about what the world *does*.

                  Fields on CausalEdge:
                    source: str           # triggering concept (action or event)
                    target: str           # produced concept (state or item)
                    role: str             # human-readable description
                    guard: str = ''       # process-language boolean condition;
                                          # edge fires only when guard is truthy.
                                          # Makes the kernel state-dependent:
                                          # P(target|source,state) = probability if guard(state) else 0
                    delay_steps: int = 0  # game-tick delay (0 = same tick)
                    probability: float = 1.0  # base rate when guard satisfied

                  Simple case (unconditional):
                    break(log) causes wood_drop
                    night_start causes hostile_mob_spawn

                  Conditional case (guarded):
                    use_bed causes sleep  guard="equal(time_of_day,NIGHT)"
                    use_bed causes sleep  guard="equal(weather,STORM)"
                    (two edges, same source/target, different guards — coproduct condition)

                  Delayed case:
                    break(log) causes wood_drop  delay_steps=1

composes_into  -- COMPOSITION EDGE (new: CompositionEdge class)
                  "A₁ ⊗ A₂ ⊗ ... → B" — product/limit morphism.
                  Categorically: pullback — arrow goes FROM multiple inputs TO one output.
                  e.g. {4 × wood_plank} composes_into crafting_table [via crafting_ui]
                  e.g. {stick, 3 × wood_plank} composes_into wooden_pickaxe
                  Adjoint (decomposes_to) optional — not all composition is reversible.
                  Analogy to composes_with vs. requires: composition is physical structure;
                  requires is epistemic ordering. Both are needed.

instance_of    -- INSTANCE EDGE (new: InstanceEdge class)
                  "A is a specific case of type B" — subtyping morphism.
                  Categorically: forgetful functor from subtype to supertype category.
                  e.g. oak_log instance_of log
                  e.g. log instance_of wooden_item
                  Induces spans (A ← supertype → B) that represent analogy without
                  needing a separate analogous_to edge:
                    oak_log ← log → birch_log  (both instance_of log → they are analogous)
                  MDL signal: if one process covers all instances of a supertype → merge
                  via Functor (already implemented).

precedes       -- TEMPORAL EDGE (new: TemporalEdge class)
                  "A must occur before B" — monoidal sequential composition in time.
                  Categorically: composition in a monoidal category with time object.
                  e.g. look_at_log precedes break_log
                  e.g. craft_table precedes shaped_crafting
                  Chain of precedes edges = an action plan (sequential monad).
                  concurrent variant: A ⊗ B (independent steps, no ordering required).
```

### Why the Distinction Matters for Minecraft

With the existing edges (`requires`, `adjunction`, `functor`, `challenge`, `override`),
the CTKG is a sophisticated **learning dependency and epistemic graph** with full
probabilistic structure (Markov kernels, d-separation, do-calculus, Bayes filter).

Adding causal + compositional edges transforms it into an **executable world model**:
- "What do I need to reach X?" → traverse `requires` + `composes_into` backwards
- "What will happen if I do A?" → traverse `causes` forwards (checking guards)
- "Can I do this now?" → evaluate all `guard` conditions against current world state
- "What's the sequence to get from S to S'?" → path through `precedes` + `causes`
- "Is oak_log analogous to birch_log?" → both `instance_of` log → yes, via span

The challenge/override system already handles the epistemic side (what the agent
*knows* and how confident it is). The causal/compositional system handles the
physical side (what the world *does*). The guard field on `CausalEdge` is the
bridge: it's a physical condition expressed in the process language, evaluated
against world state at action time.

### The Monad Structure for Actions

Actions are **effectful morphisms**: `Action : State → State` (with side effects
on the Minecraft world). Sequential composition is monadic:

    do(look_at_tree) >>= do(attack) >>= do(collect_drop)

The process language is already structurally monadic (each line transforms the
environment sequentially). The interpreter needs to support **effectful
primitives** that reach outside the pure computation to mutate game state:

```python
# Pure primitive (current): input → output, no side effects
'img_to_gray': self._img_to_gray,

# Effectful primitive (new): executes action in game, returns new observation
'mc_attack':   self._mc_attack,    # calls MineDojo API, returns (success,)
'mc_forward':  self._mc_forward,   # moves agent, returns (new_position,)
```

The distinction is recorded in the primitive registry:
`_PURE_PRIMITIVES` vs `_EFFECTFUL_PRIMITIVES`. This allows the synthesizer
to flag templates containing effectful primitives as "plans" (to be executed
in the game) vs "computations" (to be evaluated locally).


---

## Observation Constraints

**Not using:** MineDojo's `biome` field, `nearby_entities` field.

**Reasoning:**
- Biome tags would short-circuit the agent learning what a biome *is* from
  visual patterns. Biomes should be learned as visual clusters.
- `nearby_entities` gives only the nearest entity — poor API for a world with
  many entities, and wrong substrate for learning (entities should be detected
  visually, not queried).

**What we use from MineDojo's structured observations:**
```python
obs['rgb']          # H×W×3 camera frame — primary perception
obs['inventory']    # {item: count} — event signal for learning
obs['location_stats']['pos']     # (x, y, z) — position for navigation
obs['location_stats']['pitch']   # head pitch — action state
obs['location_stats']['yaw']     # head yaw — action state
obs['life_stats']['life']        # health — survival drive
obs['life_stats']['food']        # hunger — survival drive
obs['achievement']  # {achievement_name: bool} — intrinsic reward signal
```

**What the agent must learn from pixels:**
- Block types (from color signature in `obs['rgb']`)
- Entity detection (from temporal difference between frames)
- Biome classification (from color distribution + block texture patterns)
- Spatial layout (from multiple frames during movement)

### Temporal Primitives (Needed)

Entity detection requires comparing frames over time. New primitives for
`modalities/vision.py`:

```python
'img_diff':         self._img_diff,      # |frame_t - frame_t-1| → motion map
'img_temporal_std': self._img_temporal_std, # std over frame window → persistence map
```

These require the modality to hold a **short frame buffer** (configurable,
default 8 frames). Moving objects appear in `img_diff`; persistent moving
objects appear in `img_temporal_std`. Minecraft entities are the primary source
of motion.

The frame buffer is stored in the `MinecraftModality` instance and updated each
time a new frame arrives via the game loop.

---

## Roadmap

### Phase A — MineDojo API Connection (unblocking) ✓ DONE

- Install MineDojo, verify Python API works end-to-end
- Write `modalities/minecraft.py` (`MinecraftModality`):
  - Pure primitives: `mc_rgb`, `mc_inventory`, `mc_health`, `mc_food`,
    `mc_position`, `mc_yaw`, `mc_pitch`, `mc_achievements`
  - Effectful primitives: `mc_forward(n)`, `mc_back(n)`, `mc_left(n)`,
    `mc_right(n)`, `mc_turn(yaw_delta, pitch_delta)`, `mc_jump()`,
    `mc_attack()`, `mc_use()`, `mc_sprint(on)`, `mc_sneak(on)`,
    `mc_select_slot(n)`, `mc_drop()`
  - Frame buffer: `mc_frame_diff()`, `mc_frame_std()` (motion detection)
- These are the `succ`/`pred` of the Minecraft domain — atomic, not learned
- Create `domains/minecraft.ctkg`:
  - Types: `mc_frame`, `item_count`, `block_type`, `position`, `angle`
  - Atomic concepts (built-in processes): `look`, `move`, `attack`, `use`
  - Achievement skeleton (DAG without processes — to be synthesized)

### Phase B — New CTKG Edge Types (Adding Only What's Missing) ✓ DONE

Already implemented — no changes needed: `Prerequisite` (requires + transfer_probability
+ Markov kernels), `Adjunction` (inverse_of, used by add/sub), `Functor` (domain maps,
used by universal→english syntax), `Challenge` (counter-evidence weakening), `Override`
(Fido problem exceptions), `Interface` (sheaf sections), and all graph operations:
`d_separated`, `intervene` (do-operator), `concept_entropy/mutual_information`,
`sheaf_check/merge`, `MasteryState` (Bayes filter), `what_if_not`, `challenged_concepts`.

Genuinely new additions needed for MineDojo (4 edge types, not 5):
- Add to `graph.py`: `CausalEdge` (with `guard`, `delay_steps`, `probability` fields),
  `CompositionEdge`, `InstanceEdge`, `TemporalEdge`
- `AnalogousEdge` NOT added — analogy is derivable from `instance_of` spans + dynamic
  template library; no explicit edge needed
- Update `parser.py` to parse new keywords: `causes`, `composes_into`, `instance_of`,
  `precedes`. Guards parsed as: `causes A → B  guard="expr"`
- Update `validate()` to type-check each new edge type; check guard expressions parse
- Add `graph.causal_descendants(name)` — traverse only `CausalEdge` (forward planning)
- Add `graph.analogous_concepts(name)` → Set[str] — concepts sharing a `instance_of`
  supertype (span-based analogy, no explicit edge needed)
- `generate_curriculum()` already uses only `requires` (Prerequisite) edges — unchanged

### Phase C — Dynamic CTKG Extension ✓ DONE

- `engine.add_concept(name, domain, description, input_type, output_type, process, tier)`
  adds a concept to the running graph; syncs `interpreter.concept_names` immediately
- `engine.add_prerequisite(source, target, role, transfer_probability)`
- `engine.add_causal_edge(source, target, role, guard, delay_steps, probability)`
- `engine.add_instance_edge(source, target, role)`
- `engine.add_composition_edge(source, target, role, probability)`
- `engine.add_temporal_edge(source, target, role)`
- Concepts added at runtime start with no process and empty example store
- `_deserialize_graph()` calls `add_concept()` (not `graph.add_concept()`) so
  interpreter stays synchronized after checkpoint load

### Phase D — Effectful Interpreter ✓ DONE

- `DryRunError(primitive_name)` — new exception class in interpreter.py
- `ProcessInterpreter._effectful_fns: Set[str]` — populated from `modality.EFFECTFUL` on register
- `ProcessInterpreter.run(..., dry_run=True)` — blocks all effectful primitives
- `_Parser` receives `dry_run` + `effectful_fns`; captured in `fn()` closures so
  nested calls are also protected
- `MinecraftModality.EFFECTFUL` = frozenset of 12 motor primitive names
- Synthesis always calls `interpreter.run(..., dry_run=True)` → DryRunError → template rejected
- Templates that require live game environment are correctly excluded at synthesis time

### Phase E — Online Synthesis ✓ DONE

- `engine.observe(concept_name, inputs, outputs, kl_threshold=0.1)`:
  calls `teach()`, appends KL to `_kl_history`, calls `consolidate()` when threshold met
- `engine.curiosity(concept_name, window=10)` → float:
  KL decrease rate (bits/step) over last `window` observations; 0 if no history
- `engine.offline_consolidation()` → Dict[str, process]:
  sleep-coupled batch; consolidates all concepts with `should_consolidate()==True`
- `engine._kl_history: Dict[str, List[float]]` — tracks KL sequence per concept

### Phase F — Causal Synthesis Templates ✓ DONE

- `_generate_minecraft_templates(input_type, available_ops)` in synthesis.py
  (gated by `'minecraft' in available_ops`)
- MC-1: Visual log detection via `log_detection(frame_var)` threshold
- MC-2: Inventory threshold (compare item_count vs 1,2,4,8,16)
- MC-3: Frame-diff magnitude (motion detector via `img_mean(img_dog(...))`)
- MC-4: Causal delta between two count variables (requires `compare`)
- MC-5: Day/night predictor from `time_ticks` (requires `compare`)
- All templates are **pure** (observation-pattern only) — synthesis stays dry
- Motor templates with effectful calls are deliberately excluded:
  knowledge (what predicts what) lives in observation templates;
  behavior (what to do) is the exploration policy's job

### Phase G — Exploration Policy ✓ DONE

- `engine.next_frontier_concept()` → Optional[str]:
  topological scan for unlearned concept with all prerequisites learned; returns first found
- `engine.highest_kl_rate_concept()` → Optional[str]:
  concept with maximum curiosity score (KL decrease rate)
- `engine.priority(modality=None)` → Tuple[str, float, str]:
  delegates to `modality.current_priority(engine=self)` if survival urgent;
  else falls back to ACHIEVE → EXPLORE → WANDER cascade
- Priority modes: SURVIVE (pain>0.5) > EAT (hunger>0.7) > SLEEP (sleep>0.6)
  > ACHIEVE (frontier concept) > EXPLORE (high curiosity) > WANDER (default)

### Phase H — Checkpoint System ✓ DONE

See **Checkpoint System** section below for full format spec.
Implementation: `engine.save_checkpoint(path)` / `engine.load_checkpoint(path)`
Format: JSON v1.0; concepts (with processes), all 5 edge types, ExampleStore, learned templates.
Migration: `_deserialize_graph()` calls `add_concept()` for new concepts (keeps interpreter in sync).
Schema version: string `"1.0"` — raises `ValueError` on mismatch.

### Phase I — Full Autonomous Run ✓ DONE (agent_loop.py written; MineDojo install pending)

`experiments/symbolic_ai/agent_loop.py` — full observe → priority → act → teach → consolidate loop.

Key components:
- `run_smoke_test()` — all checks pass without MineDojo (graph, engine, priority, delta, trace, KL, checkpoint round-trip)
- `run_agent()` — full loop: 6 priority modes (SURVIVE/EAT/SLEEP/ACHIEVE/EXPLORE/WANDER)
- `EligibilityTrace` — 100-step credit-assignment window (obs, action, inventory_snapshot)
- `CheckpointManager` — rolling (last 5) + permanent event saves (achievement, consolidation)
- `ProgressLogger` — per-step diagnostics at configurable interval
- Inventory change → `engine.observe()` per relevant concept
- Achievement completion → `engine.offline_consolidation()` + permanent checkpoint
- Sleep detection (time_of_day wraps backward) → `mc_modality.on_sleep(engine)`
- `_to_json_safe()` added to engine.py: numpy arrays → nested lists for JSON checkpointing

Bug fixed in engine.py: Phase C `add_*` methods used `from experiments.ctkg.graph import ...`
(wrong); fixed to `from ctkg.graph import ...` (correct package import).

**To run (requires MineDojo + Java 8+):**
```bash
pip install minedojo
python experiments/symbolic_ai/agent_loop.py

# Resume from checkpoint:
python experiments/symbolic_ai/agent_loop.py --checkpoint checkpoints/ckpt_latest.json

# Smoke test (no MineDojo needed):
python experiments/symbolic_ai/agent_loop.py --smoke-test
```

---

## Checkpoint System

The "model" for the symbolic system is: CTKG state + example stores +
learned template library. No weights.

### Serialization Format

```json
{
  "version": "1.0",
  "schema_hash": "<hash of interpreter primitive registry>",
  "ctkg": {
    "concepts": [{"name": "...", "input_type": [...], "process": [...]}],
    "edges": [{"source": "...", "target": "...", "type": "requires", ...}],
    "functors": [...],
    "adjunctions": [...]
  },
  "stores": {
    "concept_name": [{"inputs": [...], "outputs": [...]}]
  },
  "learned_templates": [
    {"process_lines": [...], "n_inputs": 2, "required_ops": [...], "count": 5}
  ],
  "frame_buffer": null,
  "timestamp": "2026-03-05T12:00:00",
  "achievement_state": {"obtain_wood": true, ...}
}
```

### Schema Hash and Invalidation

The `schema_hash` is computed from:
```python
hash(sorted(interpreter._pure_primitives.keys())
   + sorted(interpreter._effectful_primitives.keys())
   + process_language_version)
```

On checkpoint load:
1. Recompute current schema hash
2. If hashes match: load normally
3. If hashes differ: run `CheckpointMigrator`

### CheckpointMigrator

```python
class CheckpointMigrator:
    """Salvages what it can from a schema-incompatible checkpoint."""

    def migrate(self, checkpoint: dict, current_schema: str) -> dict:
        # 1. Remove any process that references deleted primitives
        deleted_prims = self._find_deleted_prims(checkpoint, current_schema)
        checkpoint = self._clear_processes_using(checkpoint, deleted_prims)

        # 2. Rename primitives that changed names (migration table)
        checkpoint = self._apply_renames(checkpoint, MIGRATION_TABLE)

        # 3. Keep example stores (raw (input, output) pairs are schema-independent)
        # 4. Keep CTKG structure (concept nodes and edges survive primitive changes)
        # 5. Discard learned_templates (likely incompatible)
        checkpoint['learned_templates'] = []
        return checkpoint
```

`MIGRATION_TABLE` is a dict of `{old_primitive_name: new_primitive_name}`
maintained in a `migrations.py` file, updated whenever primitives are renamed.

### When to Save

- After every successful `consolidate()` (a concept was learned)
- After every achievement completion
- After every N=100 steps as a safety net
- On graceful shutdown

Named checkpoints: `checkpoint_achieve_{achievement_name}.json` for
achievement-gated saves; `checkpoint_step_{N}.json` for interval saves.
Interval saves rotate (keep last 5). Achievement saves are permanent.

### When to Discard

Manual: user calls `engine.clear_checkpoint()` when a breaking code change
is made. This also clears the example stores (not the CTKG structure, which
is re-initialized from the `.ctkg` files).

Auto-discard: if schema_hash mismatch AND no migration path exists,
checkpoint is moved to `checkpoints/incompatible/` with a warning.
The CTKG skeleton is reloaded from `.ctkg` files; the agent restarts learning.

---

## Key Open Questions

1. **Plan length for synthesis**: How long a causal sequence can the
   synthesizer discover from examples? The current `fold_until` enables
   bounded loops. Multi-step plans with branches need explicit OR composition.

2. **Temporal concept formation**: When does a visual pattern become a
   "concept"? The current system waits for an explicit `teach()` call.
   In MineDojo, the agent must decide when to crystallize a visual observation
   into a named concept. Trigger: when an inventory event labels an observation.

3. **Entity tracking without API**: `img_diff` gives motion maps. But
   tracking a *specific* entity over multiple frames requires re-identification
   (same moving blob across frames = same entity). This needs a short-term
   visual memory that's not in the current design.

4. **Crafting recipe discovery**: Recipes are complex product morphisms (2D
   patterns of items). The agent can learn these from examples (place items in
   crafting grid, observe output) but the search space is large.
   Possible mitigation: the achievement DAG implies which recipes matter in
   which order — constrained search.

5. **Language grounding**: MineDojo has a text interface for achievements
   ("You obtained wood!"). The achievement names are human-readable. Should
   the agent use these strings as concept names? Or treat them as opaque tokens?
   Design choice with large downstream effects.

---

## References

- Schultz (1997) — "A Neural Substrate of Prediction and Reward"
- Schmidhuber (2010) — "Formal Theory of Creativity, Fun, and Intrinsic Motivation"
- Friston (2010) — "The Free Energy Principle: A Unified Brain Theory?"
- Friston et al. (2017) — "Active Inference, Curiosity and Insight"
- Fritz (2020) — Markov categories (implemented: d_separated, concept_entropy)
- Fan et al. (2022) — MineDojo paper
- MISTAKES.md — especially #42 (memorization), #44 (missing prerequisites)
