# Mamba3 + OaK: ARC-AGI Training System

## Goal

Train a non-language, non-fact-memorizing Mamba3 model to achieve 90%+ on procedurally
generated analogs of ARC-AGI-1, 2, and 3. The model is a **rule-discovering agent**,
not a retrieval system. It should learn to form hypotheses, test them, and compose
primitives it has never seen combined before.

---

## Part 1: Architecture Changes

### 1.1 Input Encoding — Grid Encoder

Mamba3's token embedding is replaced with a **CNN patch encoder** that processes
grid cells rather than text tokens. ARC grids are at most 30×30 with 10 colors (0–9).

```python
class GridEncoder(nn.Module):
    """
    Encodes a (H, W) integer grid into a sequence of patch embeddings.
    Each cell becomes one token. Color is embedded; 2D spatial position
    is handled by 2D-PoPE (see §1.2). A segment embedding distinguishes
    input grids from output grids and from the query.
    """
    def __init__(self, num_colors=10, d_model=512, n_segments=3):
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, d_model)
        # Local context: 3x3 neighborhood conv before flattening
        self.local_ctx = nn.Conv2d(d_model, d_model, kernel_size=3,
                                   padding=1, groups=d_model)
        # Segment embeddings: 0=input grid, 1=output grid, 2=query
        self.segment_embed = nn.Embedding(n_segments, d_model)

    def forward(self, grid, segment_id):
        # grid: (B, H, W) int, segment_id: int scalar
        x = self.color_embed(grid)           # (B, H, W, d_model)
        x = x.permute(0, 3, 1, 2)           # (B, d_model, H, W)
        x = self.local_ctx(x)               # local spatial context
        x = x.permute(0, 2, 3, 1)           # (B, H, W, d_model)
        x = x + self.segment_embed(torch.tensor(segment_id))
        return x.flatten(1, 2), x.shape[1], x.shape[2]  # (B, H*W, d_model), H, W
```

The sequence fed to Mamba3 for a K-shot task is:

```
[SEP] [input_grid_1_tokens] [SEP] [output_grid_1_tokens]
[SEP] [input_grid_2_tokens] [SEP] [output_grid_2_tokens]
...
[SEP] [test_input_tokens] [QUERY]
```

Special tokens SEP and QUERY are learned embeddings. 2D-PoPE handles position within
each grid (§1.2); segment embeddings distinguish roles across grids.

---

### 1.2 2D-PoPE — Factored Spatial Position Encoding

PoPE (Gopalakrishnan et al. 2024) encodes position purely in the phase of B and C
projections, leaving magnitude for content. The standard Mamba3 implementation
computes a cumulative sum over the *sequence* dimension — assigning positions
0, 1, 2, ... to flattened tokens, which is 1D-only.

For grids, this loses 2D structure: tokens at (row=0, col=5) and (row=1, col=0)
both have flattened index 5 or something else but cannot be distinguished as
being in the same column vs same row. 2D-PoPE fixes this.

**Derivation from 2D-RoPE:** Multi-dimensional RoPE is established in video
transformers (VideoRoPE, arXiv:2502.05173; CogVideoX). The key identity is that
composing two independent rotations is additive in the angle:

    e^{i(θ_row + θ_col)} = e^{iθ_row} · e^{iθ_col}

So encoding row and column as separate phase contributions that are *summed* is
mathematically equivalent to independent rotation composition. For 3D (video),
this extends to θ_t + θ_h + θ_w. VideoRoPE and CogVideoX allocate dimensions
among the axes, with temporal dimensions receiving lower frequencies to prevent
periodic aliasing over long sequences.

**2D-PoPE design:** Split the theta projection into two halves — one for row
position, one for column position. Compute cumulative sums along the respective
spatial axes, then add. The combined phase encodes both spatial coordinates
simultaneously with no information loss.

```python
def apply_pope_2d(x, theta_row, theta_col, delta, H, W):
    """
    2D Polar Positional Embedding.

    x:         (batch, H*W, d_bc) — content features
    theta_row: (batch, H*W, d_bc//2) — row-position angles (from in_proj)
    theta_col: (batch, H*W, d_bc//2) — col-position angles (from in_proj)
    delta:     (d_bc,) — learnable magnitude bias
    H, W:      spatial dimensions of the current grid

    Returns:   (batch, H*W, d_bc*2)
    """
    batch = x.shape[0]
    d_half = theta_row.shape[-1]

    # Reshape to 2D grid for axis-aligned cumulative sums
    theta_row_2d = theta_row.reshape(batch, H, W, d_half)
    theta_col_2d = theta_col.reshape(batch, H, W, d_half)

    # Cumsum along row axis (dim=1) and column axis (dim=2)
    theta_row_cs = torch.cumsum(theta_row_2d, dim=1)   # (B, H, W, d_half)
    theta_col_cs = torch.cumsum(theta_col_2d, dim=2)   # (B, H, W, d_half)

    # Additive composition — corresponds to multiplicative rotation
    theta_2d = (theta_row_cs + theta_col_cs).reshape(batch, H * W, d_half)

    # Concatenate to match d_bc for apply_pope
    theta_combined = torch.cat([theta_2d, theta_2d], dim=-1)  # (B, H*W, d_bc)

    return apply_pope(x, theta_combined, delta)
```

**Dimension allocation:** The theta projection from `in_proj` now produces
`d_state // 2` angles split equally: the first half drives row cumsum, the
second drives column cumsum. No change to `in_proj` output size — the split
happens after projection.

**For non-grid sequences** (SEP/QUERY tokens, or future language extensions):
fall back to standard 1D PoPE with the full theta cumsum over the sequence
dimension, as currently implemented.

---

### 1.3 GVF Heads — Multi-Timescale Prediction

K parallel output projections from h_t, each with its own discount γ_i and
cumulant target:

```python
class GVFHead(nn.Module):
    def __init__(self, d_model, gamma):
        super().__init__()
        self.gamma = gamma
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h):
        return self.proj(h).squeeze(-1)  # (B, T) scalar prediction
```

| GVF | Cumulant | γ | Purpose |
|-----|----------|---|---------|
| 0 | Prediction error on next grid cell | 0.5 | Short-horizon: local pattern |
| 1 | Rule-consistency signal (see §2.1) | 0.9 | Mid-horizon: rule stability |
| 2 | Episode completion signal | 0.99 | Long-horizon: task coherence |
| 3 | Δ variance (state update aggressiveness) | 0.7 | Internal dynamics monitoring |
| 4 | Option value of current option | 0.95 | Option-level credit |

**GVF training regime:** GVFs 0 and 1 are multi-scale predictive losses in all
environments. In Env 1 and 2 (passive inference — no actions), GVF targets are
computed over the example sequence: GVF-0 targets the cross-entropy on the next
grid cell; GVF-1 targets the consistency of rule predictions across example pairs.
These are auxiliary regression losses, not RL-style TD.

Full TD training (SMDP Bellman targets, option-boundary credit assignment) is only
active in Env 3 (interactive). GVFs 2, 3, 4 ramp up in weight at the start of
Phase 3 when Env 3 is introduced. Training code branches on `env.is_interactive`.

```python
gvf_loss = sum(F.mse_loss(gvf_head(h), td_target_i)
               for gvf_head, td_target_i in zip(gvf_heads, td_targets))
```

---

### 1.4 Fast Weight Hypernetwork

This is the core addition for in-episode hypothesis revision. A secondary network
maps h_t → additive offsets for B and C at each timestep.

**Mamba3-specific note:** In Mamba3, B and C are per-token projected vectors
(`shape: (batch, seqlen, d_state)`), computed by `in_proj` from the input — they
are NOT persistent weight matrices. The fast weight head adds a correction to the
already-projected B and C values at each step, conditioned on h_t. Concretely:
B_eff_t = B_t + f(h_{t-1}), where h_{t-1} is the hidden state from the previous
step (or layer output in chunked mode).

A is left fixed because StableSSM A governs memory timescales, which should be
stable. B and C govern what gets written into and read from the state, which
should adapt to the current episode's rule.

```python
class FastWeightHead(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        # Direct projection: d_model → d_state (no factorization needed;
        # d_state << d_model already makes this low-rank relative to d_model)
        self.to_dB = nn.Linear(d_model, d_state)
        self.to_dC = nn.Linear(d_model, d_state)
        self.scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, h):
        # h: (batch, seqlen, d_model)
        dB = self.to_dB(h)  # (batch, seqlen, d_state)
        dC = self.to_dC(h)  # (batch, seqlen, d_state)
        return self.scale * dB, self.scale * dC
```

Applied in the SSM forward pass as an additive correction before the SSD core:

```python
dB, dC = fast_weight_head(h_layer_prev)   # h from previous layer or step
# B, C: (batch, seqlen, 1, d_state) — unsqueeze(2) for MIMO rank dim
B_eff = B + dB.unsqueeze(2)
C_eff = C + dC.unsqueeze(2)
# B_eff, C_eff used in ssd_trapz
```

The scale parameter is initialized small so the base SSM behavior dominates early
in training. BPTT differentiates through the fast weight computation cleanly.
Full-sequence BPTT through Mamba3's chunked SSD is already the default behavior —
chunks are a computational optimization, not a truncation of gradient flow.

---

### 1.5 Option Register

Augment each input embedding with a learned option identity before it enters
the SSM layers:

```python
class OptionRegister(nn.Module):
    def __init__(self, num_options, d_model, d_option=64):
        super().__init__()
        self.embed = nn.Embedding(num_options, d_option)
        self.proj = nn.Linear(d_model + d_option, d_model)

    def forward(self, x, omega):
        # x: (B, T, d_model), omega: (B, T) int
        opt_emb = self.embed(omega)          # (B, T, d_option)
        return self.proj(torch.cat([x, opt_emb], dim=-1))
```

Two heads on h_t:

```python
# Termination: probability of ending current option
termination_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

# Option value: Q(h_t, omega) for each option
option_value_head = nn.Linear(d_model, num_options)
```

SMDP Bellman target at option boundaries: when option ω terminates at t_end
having started at t_start, the target is the discounted sum of cumulants over
the option duration, plus the bootstrapped value at t_end. Active in Env 3 only.

---

### 1.6 Epistemic Options — Uncertainty-Driven Termination

An uncertainty head produces per-GVF prediction variance estimates, enabling
termination conditions based on belief state rather than state visitation alone:

```python
class UncertaintyHead(nn.Module):
    def __init__(self, d_model, num_gvfs):
        super().__init__()
        self.proj = nn.Linear(d_model, num_gvfs)

    def forward(self, h):
        return F.softplus(self.proj(h))  # (B, T, num_gvfs) positive variance
```

Termination condition becomes a conjunction:

```
β(h_t) fires if:
  Δ_mean(t) > μ_Δ + 2σ_Δ  (event boundary: something structurally changed)
  OR
  max_i uncertainty_i(h_t) < threshold  (hypothesis confirmed/refuted)
```

Option value now includes an information gain term (Env 3 only):

```python
ig_bonus = lambda_ig * uncertainty_head(h_t).max(dim=-1).values
q_values = option_value_head(h_t) + ig_bonus
```

---

### 1.7 Full Architecture Summary

```
Input grids
    → GridEncoder (color embed + local conv + segment embed)
    → OptionRegister (injects current option identity)
    → Mamba3 layers (StableSSM A, 2D-PoPE on grid tokens, fast weights
                     modulating B/C per step from previous layer's h_t)
    → h_t at each step feeds:
        ├── Task output head (grid cell predictions)
        ├── GVF heads × K (multi-scale prediction / multi-timescale cumulants)
        ├── Termination head β(h_t)
        ├── Option value head Q(h_t, ω)
        └── Uncertainty head (per-GVF variance)
```

**Config note:** Set `mimo_rank=1` for this experiment. The MIMO implementation
in the current Mamba3 codebase creates R independent states rather than the
paper's shared-state rank-R updates (Mistake #47). Disable it to avoid the
resulting R× state explosion.

---

## Part 2: Training Environments

All three environments are procedurally generated every episode. No rule or
environment configuration is ever repeated. Rules are drawn from a compositional
DSL and sampled fresh per episode during training.

---

### 2.1 Environment 1 — Rule Inference from Examples (ARC-1 Analog)

**What it tests:** Extract a transformation rule from K input-output grid pairs.
Apply it to a novel test input. Pure inference, no interaction.

**Format:**
- Grid size: 3×3 to 10×10 (sampled per episode)
- Colors: subset of 0–9 (sampled per episode)
- K: 2–5 example pairs
- Agent predicts the output grid for the test input

**Rule DSL — Primitives:**

Spatial:
- `translate(dx, dy)` — shift all non-background cells
- `rotate(k)` — rotate grid 90°×k
- `reflect(axis)` — horizontal or vertical flip
- `crop(rect)` — extract a rectangular sub-region
- `tile(nx, ny)` — tile the pattern

Color:
- `recolor(src, tgt)` — replace one color with another
- `swap(c1, c2)` — swap two colors throughout
- `fill_region(condition, color)` — fill cells matching a spatial condition

Structural:
- `sort_rows(key)` — sort rows by count of a color
- `align(axis, anchor)` — align objects to an edge
- `symmetrize(axis)` — force symmetry
- `count_to_color(color)` — output encodes count of a color as a 1D bar

Relational:
- `apply_to_largest(primitive)` — apply transformation only to largest object
- `apply_to_color(color, primitive)` — apply only to cells of given color
- `if_count(color, n, then, else)` — conditional on object count

**Rule sampling:**

```python
def sample_rule(difficulty):
    """
    difficulty 1: 1-2 primitives, no relational
    difficulty 2: 2-3 primitives, simple relational allowed
    difficulty 3: 3-5 primitives, full relational, nested conditionals
    """
    n_primitives = sample_n(difficulty)
    primitives = [sample_primitive(difficulty) for _ in range(n_primitives)]
    return compose(primitives)  # left-to-right application
```

Episode composition rules are sampled independently each episode. With a DSL of
~20 primitives and compositions up to depth 5, the combinatorial space is
effectively inexhaustible. The model cannot memorize rules — it must learn to
discover composition structure.

**Rule-consistency signal (GVF cumulant 1):**

During the example sequence, a consistency signal is computed: does the current
h_t predict the next output grid correctly given the same rule applied to the
next input? This is the cumulant for GVF-1. Tracking it at γ=0.9 forces h_t to
maintain a stable rule hypothesis across examples rather than rederiving it from
scratch per pair.

**Reward:** Exact grid match on test output. Partial credit: fraction of correct
cells (used during early training only, removed as performance improves).

---

### 2.2 Environment 2 — Abstract Object Reasoning (ARC-2 Analog)

**What it tests:** Harder compositions with abstract object-level reasoning. Rules
reference objects, their properties, and relations between them, not raw pixels.

**New abstractions on top of Env 1:**

Objects: connected components of same-colored cells. Each object has:
- Color, size, bounding box, centroid, convex hull, axis of symmetry

Object-level primitives:
- `move_obj(obj_selector, destination_rule)` — move an object to a computed location
- `copy_obj(obj_selector, n_times, direction)` — repeat an object
- `merge_objs(selector1, selector2)` — combine two objects
- `delete_obj(obj_selector)` — remove an object
- `resize_obj(obj_selector, scale_rule)` — scale an object

Selectors (abstract references):
- `largest`, `smallest`, `leftmost`, `rightmost`, `most_common_color`
- `with_color(c)`, `with_size(n)`, `touching_edge`
- `having_symmetry(axis)`, `unique_color` (color appearing exactly once)

Relational rules:
- `align_to(obj_a, obj_b, axis)` — position a relative to b
- `replicate_pattern(source_obj, target_positions)` — stamp a pattern
- `interpolate(obj_a, obj_b, n)` — fill n objects between two anchor objects

**Key difference from Env 1:** The agent must build an object-level representation
in h_t, not a pixel-level one. The fast weight head is critical here — B and C
must adapt to the current episode's object vocabulary (which objects exist, which
relations are relevant).

**Grid size:** 6×6 to 20×20. K: 3–5 pairs.

**Difficulty ramp:** Start with one object-level primitive plus one spatial
primitive. Add relational references and multi-object interactions as performance
improves on simpler cases.

---

### 2.3 Environment 3 — Interactive Rule Discovery (ARC-3 Analog)

**What it tests:** The agent enters a novel interactive environment with no
instructions, no explicit rules, and no stated objective. It must explore,
discover the mechanics, infer the win condition, and achieve it efficiently.
Scored on action efficiency relative to a human baseline.

**Episode structure:**

Each episode is a small interactive world with:
1. A **state** — a grid or structured display that changes in response to actions
2. An **action space** — buttons, object selections, or grid cells the agent can
   activate (3–8 actions available, not labeled)
3. A **hidden mechanic** — the rule governing how actions change state
4. A **hidden win condition** — a target state the agent must reach
5. A **step budget** — agent must reach win condition in ≤ B steps (human baseline)

The agent observes the current state grid and takes discrete actions. No reward
signal is given until the win condition is reached (sparse reward). The agent
must infer what the win condition even is through exploration.

**Environment generators:**

```
Generator A — Button-State Systems
  Hidden mechanic: each button toggles a subset of cells (XOR, set, clear)
  Win condition: reach a target pattern
  Mechanics are rule-governed but not labeled

Generator B — Object Manipulation
  Hidden mechanic: selecting objects applies hidden transformations to them
  Objects have hidden properties (some are moveable, some are anchors)
  Win condition: reach a target configuration of objects

Generator C — Causal Chains
  Hidden mechanic: actions have cascading effects through a hidden causal graph
  Win condition: trigger a specific terminal state
  Requires understanding the causal graph, not just the immediate effects

Generator D — Compositional Mechanics
  Two or more mechanics from A/B/C are combined
  Win condition requires exploiting both
  Available only in late training
```

**Scoring:**

```
score = max(0, 1 - (agent_steps / human_baseline_steps))
```

Episodes where the agent fails to reach the win condition within 3× the human
baseline score 0.

**Human baseline:** Precomputed by running a deliberate BFS/MCTS solver with
full knowledge of the hidden mechanics. This is a perfect-information solver
baseline — a computational lower bound on optimal action count.

**The epistemic option loop in Env 3:**

Env 3 is where the full OaK machinery activates. Uncertainty across all GVFs is
initially maximal. The epistemic option selector chooses actions that maximally
reduce GVF uncertainty (information gain). As mechanics are discovered, GVF
uncertainty drops, triggering option termination and transition to a new option
targeting the next unknown aspect. The fast weight head continuously updates B
and C as the agent's hypothesis about the mechanic evolves.

---

## Part 3: Training Pipeline

### 3.1 Option Discovery Schedule

Before training begins, a small offline option discovery run is performed on Env 1
with random policies to initialize option centers. This gives the option register
non-trivial initial options to work with.

During training, option discovery runs continuously:

```python
# Every N forward passes:
if step % rediscover_interval == 0:
    new_centers = kmeans(boundary_state_buffer, k=num_options)
    option_register.update_centers(new_centers)
    boundary_state_buffer.clear()
```

Boundary states are collected using the Δ spike criterion: any timestep where
Δ mean across SSM heads exceeds 2 standard deviations is logged.

**Option hierarchy from A-spectrum:** In Mamba3, A is per-head (shape `nheads`),
not per-state-dimension — all d_state dimensions within a head share the same
decay scalar. The option hierarchy therefore operates over *heads* rather than
state dimensions. Sort heads by their learned A value (equivalently, by
`exp(A_i * Δ̄)` at the running-mean dt):

- Heads where `exp(A_i * Δ̄) < 0.5` — fast-decaying heads → short-option boundaries
- Heads where `exp(A_i * Δ̄) > 0.9` — slow-decaying heads → long-option boundaries

Three option levels emerge naturally from the head spectrum without explicit design.
With StableSSM reparameterization, A values spread across this range during training.

---

### 3.2 Loss Function

```python
total_loss = (
    task_loss            # exact grid match cross-entropy
  + λ_gvf  * gvf_loss   # multi-scale prediction (all envs) + TD (Env 3 only)
  + λ_term * term_loss   # termination head: BCE against Δ-detected boundaries
  + λ_opt  * opt_loss    # option value: SMDP Bellman targets (Env 3 only)
  + λ_unc  * unc_loss    # uncertainty calibration: NLL of prediction errors
)
```

Suggested initial weights: λ_gvf=0.3, λ_term=0.1, λ_opt=0.2, λ_unc=0.1.
These are annealed as training progresses — task loss weight increases relative
to auxiliary losses once basic rule inference is working. λ_opt and the SMDP
Bellman terms are zero until Phase 3.

---

### 3.3 Training Phases

**Phase 1 — Rule Inference Foundation (Env 1 only)**

Goal: The model must learn to extract and apply rules from examples.
GVFs 0 and 1 are trained as predictive losses. Options 2, 3, 4 (long-horizon)
are initialized but their weights are near zero.

Curriculum within Env 1:
- Start: 1-primitive rules, 3×3 to 5×5 grids, K=3 examples
- Advance when: 85% exact match on current difficulty
- End criterion: 85% on full Env 1 difficulty range (3 primitives, 10×10, K=2)

**Phase 2 — Object Abstraction (Env 1 + Env 2)**

Goal: The fast weight head must learn to build episode-specific object bindings.
The model sees Env 1 and Env 2 in a 1:1 ratio initially, shifting to 1:2 as
Env 2 performance lags.

Key checkpoint: correctly apply rules referencing `largest`, `smallest`, and
`unique_color` selectors before advancing. All GVF heads now active as
predictive losses.

**Phase 3 — Interactive Discovery (All Three Environments)**

Env 1 : Env 2 : Env 3 ratio starts at 2:2:1, shifts to 1:1:2. The SMDP Bellman
targets, option value loss, and full GVF TD training activate. The information
gain bonus λ_ig is annealed from 0.0 to its final value over the first 10k Env 3
episodes — prevents over-exploration before GVF predictions are meaningful.

**Phase 4 — Compositional Stress Test**

Generator D (compositional mechanics) is introduced in Env 3. Env 1 and 2 are
maintained at low sample rates to prevent forgetting. No curriculum adjustment —
this is the test of whether compositional generalization emerged from the training
distribution or requires explicit scaffolding.

---

### 3.4 What Compositional Generalization Requires of This System

The claim that this system should learn to combine rules rests on three
properties that the architecture enforces:

**1. Rule primitives are represented relationally, not as lookup entries.**

The fast weight hypernetwork maps h_t → ΔB, ΔC, which modifies how the SSM
reads inputs. A rule primitive like `apply_to_largest` is not stored as a
fact — it is encoded as a pattern of B/C adaptation that, when applied to a
sequence containing a "largest object" structure, produces the right state
transitions. This representation is compositional by construction: applying
`rotate` and then `apply_to_largest` is the sequential composition of two
B/C adaptation patterns, which h_t can hold simultaneously.

**2. GVFs enforce rule consistency across the episode, not per-step.**

GVF-1 (rule consistency, γ=0.9) penalizes h_t for representing a rule that
produces inconsistent predictions across example pairs. If the model tentatively
hypothesizes `rotate(90)` but one example contradicts this, GVF-1's error
spikes, the termination head fires, and the option transitions to a revised
hypothesis. This is hypothesis testing implemented as a training pressure on
the GVF heads.

**3. The option hierarchy separates rule discovery from rule application.**

Long-horizon options (slow-A heads, GVFs 2 and 4) represent the current rule
hypothesis at the episode level. Short-horizon options (fast-A heads, GVFs 0
and 3) represent the current step's application of that hypothesis. Novel rule
combinations don't require representing a new episode-level concept — they require
the long-horizon option to hold a composition of two known short-horizon patterns.

---

## Part 4: Evaluation

### On the ARC-AGI Analogs

Metrics computed per environment on held-out procedurally generated test sets
(episodes generated with the same DSL but seeds unseen during training):

- **Exact match rate** — fraction of episodes with perfectly correct output
- **Primitive generalization** — held-out test set uses primitive combinations
  that never co-occurred during training
- **Sample efficiency** — for Env 3, agent steps / human baseline steps

Target: 90%+ exact match on all three held-out test sets, with primitive
generalization test not more than 10% below full test set performance.

### Red Flags to Watch For

- **GVF loss not decreasing by Phase 2**: fast weight head is not learning
  useful episode-specific representations. Increase d_state offset size or add
  a reconstruction auxiliary loss.
- **Option re-clustering destabilizing training**: option identity is changing
  faster than the option value head can track. Increase rediscover_interval
  or add momentum to option center updates.
- **Env 3 exploration is random (uncertainty not dropping)**: GVF heads have
  not learned to predict environment dynamics. Enforce Phase 2 completion
  criteria more strictly before introducing Env 3.
- **Compositional generalization gap > 20%**: the model is partially memorizing
  rule co-occurrences rather than representing rules as independent composable
  units. Increase DSL diversity and enforce a minimum Hamming distance between
  training and test rule compositions.
- **2D-PoPE not helping over learned embeddings**: ablate by replacing 2D-PoPE
  with a simple learned (H_max, W_max, d_model) position table. If performance
  is identical, the data-dependent angle structure is not being used.
