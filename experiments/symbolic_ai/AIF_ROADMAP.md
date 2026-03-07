# Phase R: Active Inference Engine — Complete Roadmap

**Status: IN PROGRESS — started 2026-03-07 — R0–R7 complete**
**Track progress at the bottom of this file.**

---

## Motivation

The existing `DecisionEngine` (planning.py) implements *reactive condition-checking*:
goals are checked against the current state; the highest-priority satisfied goal acts.
This is FEP-*inspired* (drives modulate priority) but not FEP-*faithful*.

A faithful Active Inference agent does not check goals. It:
1. Maintains beliefs `q(s)` — a probability distribution over hidden world states.
2. Evaluates candidate policies by their **Expected Free Energy** G(π).
3. Selects the policy that *minimises* G(π), balancing:
   - **Pragmatic value** — how well predicted outcomes match prior preferences.
   - **Epistemic value** — how much uncertainty the policy would reduce.

Goals are not checked; preferences are *prior expectations* P(o).
Exploration is not a special goal; it emerges from the epistemic value term.
Sub-goal sequencing is not hand-coded; it emerges from multi-step policy rollouts.

---

## Design Principle (inviolable)

> **The model must never be designed around a specific task. The model must be general.**

Every component in Phase R (GenerativeModel, VariationalBelief, AIFEngine) must
operate purely from the state dict and generative model. No game-specific logic
belongs in any of these files.

---

## Architecture Overview

```
                         ┌─────────────────────────────────────┐
                         │         Active Inference Loop        │
                         │                                      │
  observation ──────────►│  VariationalBelief.update(obs)       │
                         │    q(s) ← variational update         │
                         │                                      │
                         │  GenerativeModel.expected_free_energy │
                         │    G(π) = pragmatic + epistemic       │
                         │                                      │
                         │  AIFEngine.decide()                  │
                         │    action = argmin_π G(π)            │
                         │                                      │
  action ───────────────►│  env.step(action)                    │
                         │                                      │
                         │  GenerativeModel.transition.update() │
                         │    P(s'|s,a) ← observed transition   │
                         └─────────────────────────────────────┘

Files:
  generative_model.py   PreferenceFactor, TransitionModel, GenerativeModel
  perception.py         VariationalBelief (factored Bayesian belief state)
  planning.py           AIFEngine (added alongside DecisionEngine)
  agent.py              run_episode() — generic game-agnostic agent loop
```

---

## Phase R0: Structural Cleanup

**Goal:** Extract the generic agent loop from `textworld_test.py` into `agent.py`.

**Status:** COMPLETE (done in this session)

### Files:
- `agent.py` NEW — `run_episode()`, `EpisodeResult`
- `textworld_test.py` — `run_phase_q4()` updated to delegate to `agent.py`

### What `agent.py` contains:
```python
@dataclass
class EpisodeResult:
    score:      float
    steps:      int
    done:       bool
    trace:      List[dict]   # per-step records for analysis

def run_episode(env, build_state_fn, engine, world_model, rng,
                max_steps=200, verbose=False) -> EpisodeResult:
    """
    Generic agent-environment loop. Game-agnostic.
    Caller provides:
      - env:            any object with .step(action) -> (obs, score, done, info)
      - build_state_fn: (obs, world_model) -> state_dict
      - engine:         DecisionEngine or AIFEngine
      - world_model:    game-specific tracker (WorldModel from adapter)
    Engine knows nothing about the game. Only build_state_fn is game-specific.
    """
```

### Adapter contract:
New game = new adapter file containing:
1. `GameModality` — wraps game API, provides `.step()`, `.get_events()`
2. `GameWorldModel` — tracks game-specific derived state
3. `build_state()` — converts (obs, world_model) → state dict
4. `make_goals()` — defines FEPGoal objects (or provides GenerativeModel)
5. Calls `agent.run_episode()` with these — no changes to agent.py

---

## Phase R1: GenerativeModel

**Goal:** Implement `P(o)`, `P(s'|s,a)`, and EFE computation.

**Status:** COMPLETE (done in this session)

### File: `generative_model.py`

```
PreferenceFactor
  name:         str
  urgency:      float              # weight in total log preference
  log_pref_fn:  (state) -> float  # log P(o) for this factor; high = preferred
  log_preference(state) -> float  # urgency × log_pref_fn(state)
  prediction_error(state) -> float # -log_preference (free energy contribution)

TransitionModel
  update(prev_state, action, next_state, reward) -> None
  predict(state, action) -> (predicted_state, confidence)
  # Learned from observed transitions; confidence grows with observations

GenerativeModel
  preferences:  List[PreferenceFactor]  # replaces Goal/Drive list
  transition:   TransitionModel
  from_drives(drives) -> GenerativeModel  # convert Drive list to preferences

  log_preference(state) -> float   # sum of factor log prefs
  pragmatic_value(state) -> float  # -log_preference(state); lower = preferred
  epistemic_value(state, action, n_obs) -> float  # info gain proxy
  expected_free_energy(policy, state, horizon) -> float  # G(π) = prag + epist
```

### Drive → PreferenceFactor mapping:
```
Drive(name, measure, setpoint=1.0, urgency=0.8)
→ PreferenceFactor(name, urgency=0.8,
      log_pref_fn = lambda s: -max(0, setpoint - measure(s)) × scale)
# log P(o) is maximised (= 0) when measure(s) >= setpoint
# log P(o) is penalised (< 0) proportional to deficit
```

The conversion is exact: Drive.deficit(s) = PreferenceFactor.prediction_error(s) / urgency.
The information content is identical; the embedding is more principled.

---

## Phase R2: VariationalBelief

**Goal:** Replace deterministic state dict with probabilistic factored belief state.

**Status:** COMPLETE (done in this session)

### File: `perception.py`

```
VariationalBelief
  # Factored belief: q(s) = ∏_i q_i(s_i)
  # Each factor q_i is a categorical distribution over variable s_i's domain.
  _beliefs: Dict[str, Dict[Any, float]]  # variable -> {value: probability}

  observe(variable, value, certainty=0.97) -> None
    # Sharp update: concentrate mass on observed value.
    # certainty=0.97 leaves 3% residual uncertainty.

  update_from_obs(obs_dict) -> None
    # Bulk update from observation dict: treat each key-value as an observation.

  predict_error(generative_model) -> float
    # Perceptual free energy = KL[q(s) || P(s|o)]
    # Measures how surprising current observations are under the generative model.

  decay(rate=0.95) -> None
    # Temporal decay: beliefs drift toward uniform prior each step.

  sample() -> dict
    # Sample one world state from the factored belief distribution.

  entropy() -> float   # total uncertainty in bits
  summary() -> str     # human-readable belief state
```

### Relationship to existing BeliefState:
`BeliefState` tracks only P(item | location). `VariationalBelief` is more general:
it tracks P(any_variable | observations) for any state variable. `BeliefState`
is a special case of `VariationalBelief` where variables are `item_loc:<item>`.
Both coexist; AIFEngine uses `VariationalBelief`.

---

## Phase R3: AIFEngine

**Goal:** Replace condition-checking with EFE minimisation. Same external interface as DecisionEngine.

**Status:** COMPLETE (done in this session)

### Added to: `planning.py`

```
AIFEngine
  # Interface: identical to DecisionEngine for drop-in replacement.
  # Internally: evaluates candidate policies by G(π) instead of checking conditions.

  __init__(
    generative_model: GenerativeModel,
    affordances:      AffordanceModel,
    goal_stack:       GoalStack = None,       # kept for backward compat
    episodic:         EpisodicBuffer = None,
    belief:           VariationalBelief = None,
    goals:            List = None,            # optional: falls back to condition-check
    policy_horizon:   int = 1,               # R3: 1-step; R8: multi-step
    n_random_explore: int = 3,               # random actions sampled for epistemic
    is_nav_fn:        Callable = None,
    state_loc_key:    str = 'location',
    state_inv_key:    str = 'inventory',
    state_score_key:  str = 'score',
  )

  decide(state, rng) -> (action, reason)
    1. Enumerate candidate actions from state['admissible']
    2. For each action: predict next state via transition model
    3. Compute G(action) = pragmatic_value(predicted) + epistemic_value(state, action)
    4. Return argmin G(action), reason='AIF:G=<value>'
    # Fallback: if no admissible actions, return ('look', 'AIF:empty')
    # Backward compat: if goals provided AND model has no preferences, use condition-check

  feedback(prev_state, action, new_state, events=[]) -> None
    # Same as DecisionEngine.feedback() PLUS:
    # Updates generative_model.transition with observed transition
    # Updates variational_belief with new observations
```

### EFE computation (Phase R3 — single step):
```
G(action) = pragmatic_value(predicted_next_state)
           + epistemic_value(state, action)

pragmatic_value(s) = GenerativeModel.pragmatic_value(s)
                   = -sum_factors urgency_i × log_pref_i(s)
                   ≈ sum_factors urgency_i × drive_deficit_i(s)

epistemic_value(state, action) = confidence_gain × novelty
  confidence_gain = transition.predict(...)[1]     # 0 for unknown, → 1 with experience
  novelty         = 1 if (location, action) not in episodic history else 0
```

Phase R3 is single-step look-ahead. Multi-step rollouts come in Phase R8.

---

## Phase R4: Hierarchical Timescales

**Goal:** Implement 3-level hierarchy:
- Level 1 (fast, ~100ms): FovealAttention — minimise visual prediction error
- Level 2 (medium, ~1s): AIFEngine — minimise action-level expected free energy
- Level 3 (slow, ~10s): Deliberate planner — multi-step EFE over goals

**Status:** COMPLETE (2026-03-07)

### File: `vision_cortex.py` (extend existing)

```
FovealAttention
  fixation_point: (row, col)   # current gaze position in image coordinates
  foveal_size:    int          # pixels of high-res foveal patch (e.g. 64×64)
  peripheral_res: int          # downsampled resolution of peripheral (e.g. 16×16)

  foveal_patch(image) -> np.array    # high-res crop at fixation_point
  peripheral_patch(image) -> np.array # downsampled peripheral view

  visual_prediction_error(foveal_patch, expected_patch) -> float
    # Mean squared error or KL between observed and expected foveal content

  update_fixation(image, generative_model) -> (row, col)
    # Move fixation to maximise visual information gain
    # = argmax_candidate visual_prediction_error(foveal_at_candidate, expected)
    # Biologically: saccade to high-surprise regions

  step(image) -> dict
    # Returns: {'foveal': patch, 'peripheral': patch, 'fixation': (r,c),
    #           'prediction_error': float, 'text_region': bool}
```

### Level 3 (deliberate planning):
- Uses same AIFEngine with policy_horizon > 1
- Policy search: beam search or MCTS over action sequences
- Activated when Level 2 G(action) is uniformly high (all options look bad)
- Deactivates when a promising policy is found

---

## Phase R5: Goal Learning

**Goal:** Discover preferences from experience; eliminate manual goal specification.

**Status:** COMPLETE (2026-03-07)

### File: `goal_learning.py` (new dedicated module)

```python
# goal_learning.py — DiscoveredGoal + discover_goals() + update_generative_model()

def discover_goals(
    causal_history:      List[Tuple[dict, str, dict, float]],   # (s, a, s', reward)
    drives:              List[Drive],
    min_support:         int   = 3,
    min_confidence:      float = 0.5,
    min_lift:            float = 1.2,
    max_goals_per_drive: int   = 5,
    skip_keys:           Optional[List[str]] = None,
) -> List[DiscoveredGoal]:
    """
    Discover goal-relevant PreferenceFactor objects from causal history.

    Algorithm (per Drive D):
    1. Partition transitions: drive_reduced = deficit_before - deficit_after > 0.01
    2. Feature statistics: conditional P(feature=val | drive_reduced) over prev_state
    3. Filter by support ≥ min_support, confidence ≥ min_confidence, lift ≥ min_lift
    4. Sort by information_gain (KL divergence bits), return top-max_goals_per_drive

    Returns List[DiscoveredGoal] sorted by information_gain descending.
    Each DiscoveredGoal has: drive_name, feature_key, feature_value,
      confidence, support, base_rate, drive_urgency, evidence (≤10 transitions).
    Properties: lift (= confidence / base_rate), information_gain (KL bits).
    Method: to_preference_factor(scale=5.0) → PreferenceFactor for GenerativeModel.
    """

def update_generative_model(
    generative_model, causal_history, drives, scale=5.0, verbose=False, **kwargs
) -> int:
    """Convenience: discover_goals() → to_preference_factor() → model.preferences.append()"""
```

**Design note:** Placed in dedicated `goal_learning.py` (not `synthesis.py`) to
keep template-based program synthesis separate from statistical causal inference.

This replaces manual `FEPGoal(...)` specifications. The agent observes that
"kitchen+take_apple reduces hunger" and automatically generates a preference
for states where `location='kitchen'` when hunger drive has deficit.

---

## Phase R6: Visual Symbol Learning (Phase O applied to vision)

**Goal:** Learn letter/word recognition from pixel statistics — no OCR labels.

**Status:** COMPLETE (2026-03-07)

### File: `modalities/visual_symbol.py` (new)

```
VisualGlyphStream          — rolling patch-token stream + bigram context counts
                              (analogue of _stream_to_dists() for text)

VisualSymbolLearner        — main class: wraps stream, drives two-level discovery
  .observe(foveal)           — feed one text-region foveal patch (Level 1)
  .discover_glyphs()         — cluster patch hashes → letter-like categories
                               uses discover_categories_from_dists() UNCHANGED
  .decode(foveal)            — decode → glyph cluster ids (left-to-right)
  .observe_word(foveal)      — record glyph sequence for word-level clustering
  .discover_words()          — cluster glyph sequences → word-like categories
  .word_cluster(foveal)      — identify word cluster for one foveal patch
  .causal_crossref(history)  — map word clusters → probable command labels

discover_visual_symbols()  — module-level convenience function
```

**Key design decision:** `discover_categories_from_dists()` from `synthesis.py`
is reused **verbatim** — the only change is the tokenization (patch hashes
instead of word strings).  This is the strongest possible demonstration of
algorithmic generality: one clustering function, two domains.

**Causal cross-reference ("distributional hypothesis applied causally"):**
"You shall know a visual word by the action it accompanies."
If word cluster W precedes rewarded action A, W ← label A.

**Smoke test results:**
- 200 synthetic foveal frames (5 glyph templates): 1084 tokens, 20 unique hashes
- discover_glyphs(): 18 hashes assigned to 8 clusters
- decode([g0, g1, g2]): 3/3 patches classified correctly
- causal_crossref(): labels high-confidence (word_cluster → action) pairs

---

## Phase R7: New Game Adapter

**Goal:** Port to new text+image game using the adapter contract.

**Status:** COMPLETE (2026-03-07) — generic Action types + ScreenModality base class

### Files: `modalities/action.py` + `modalities/screen_modality.py` (new)

```
action.py:
  KeyPress(key, duration_s=0.0)     keyboard tap or hold
  KeyHold(key, steps=1)             multi-step hold (for movement)
  MouseMove(x, y, relative, dur)    cursor movement in [0,1] coords
  MouseClick(x, y, button, double)  click at normalised screen position
  MouseScroll(x, y, dx, dy, clicks) scroll wheel
  Action = Union[str, KeyPress, KeyHold, MouseMove, MouseClick, MouseScroll, List[Action]]
  flatten(action) -> List[Action]   recursive macro expansion
  Factories: tap(), hold(), click(), move(), scroll_down(), scroll_up(), macro()

screen_modality.py:
  ScreenModality(window_title, region, capture_w, capture_h,
                 frame_delay_s=0.1, dry_run=False, verbose=False)
    — Abstract base class.  Three backends tried in order:
        Screen: dxcam → mss → PIL.ImageGrab → zeros stub
        Input:  pynput → logging stub
    — Interface:
        connect() → dict          first obs; resets state
        step(action) → (obs, reward, done, info)
        dispatch(action) → None   translate Action → OS events
        disconnect() → None

  Game adapter pattern (override these methods):
    build_obs(frame, info={}) → dict  ← ONLY game-specific method
    get_reward() → float
    get_done() → bool
    get_events() → List[dict]
    get_admissible() → List[Action]
    send_text(text: str) → None
```

**Smoke test result:** mss + pynput available; live screen capture working (1920×1080).
All 8 action types dispatched cleanly in dry_run=True mode.

**Game adapter requires 3 files only:**
  1. `<game>_modality.py` — subclass ScreenModality, override build_obs() + get_reward() + get_done()
  2. `<game>_adapter.py` — WorldModel, build_state(), GenerativeModel factory
  3. `<game>_test.py`    — validate agent can complete basic objectives

---

## Phase R8: Multi-Step Policy Rollouts

**Goal:** Extend AIFEngine to evaluate policies of length > 1.

**Status:** PENDING (depends on R3, R7)

### Design:
```
policy_horizon = k  # evaluate k-step action sequences

candidate_policies = beam_search(
    root_state = current_state,
    transition = generative_model.transition,
    horizon    = k,
    beam_width = 10,
)
# Returns top-10 k-step policies by cumulative G(π)

best_policy = argmin G(π) over candidate_policies
action = best_policy[0]  # execute first step
```

For the new game: k=3 is likely sufficient for near-term planning.
For complex multi-minute decisions: Level 3 planner uses k=10+.

---

## Progress Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| R0 — agent.py extraction | COMPLETE | 2026-03-07 | `agent.py` written; `textworld_test.py` updated |
| R1 — GenerativeModel | COMPLETE | 2026-03-07 | `generative_model.py` written |
| R2 — VariationalBelief | COMPLETE | 2026-03-07 | `perception.py` written |
| R3 — AIFEngine | COMPLETE | 2026-03-07 | Added to `planning.py` |
| R4 — FovealAttention | COMPLETE | 2026-03-07 | `FovealAttention` added to `vision_cortex.py`; smoke test PASS |
| R5 — Goal Learning | COMPLETE | 2026-03-07 | `goal_learning.py` written (`discover_goals`, `DiscoveredGoal`, `update_generative_model`); smoke test PASS |
| R6 — Visual Symbol Learning | COMPLETE | 2026-03-07 | `visual_symbol.py`; VisualSymbolLearner + discover_visual_symbols(); smoke test PASS |
| R7 — New Game Adapter | COMPLETE | 2026-03-07 | `action.py` (Action type hierarchy) + `screen_modality.py` (ScreenModality base); smoke test PASS |
| R8 — Multi-step Rollouts | PENDING | — | After R3 + R7 |

---

## Validation Checklist

After each phase, verify:
- [ ] `python run_experiment.py` — all 17 phases PASS (symbolic AI unchanged)
- [ ] `python textworld_test.py` — NanoTextEnv 3/3 PASS, MicroTextWorld 5×3/3 PASS
- [ ] `python textworld_test.py --plan` — Phase Q4 quest completion 3/3 PASS
- [ ] (After R3) `python textworld_test.py --aif` — same success with AIFEngine
- [ ] (After R5) `python textworld_test.py --discover-goals` — goals discovered, not manual
