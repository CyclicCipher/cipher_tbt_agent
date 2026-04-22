# TBT Architecture: HDC Binding, Grid-Cell Locations, and CHL

*Reference document. Last updated 2026-04-14.*

---

## Implementation Status

| Biology | CipherNet | Status |
|---------|-----------|--------|
| CHL / predictive coding at V1 | `commit_chl()` — free phase WTA + clamped unlearn | **DONE** (`chl: true` in config) |
| Lateral voting (L2/3 horizontal connections) | `_lateral_pass()` ±1 grid neighbours + `_global_lateral_pass()` full-layer plurality | **DONE** |
| Top-down feedback (L6: upper → lower predictions) | `_feedback_pass()` + `MacroColumn.receive_feedback_by_index()` | **DONE** (Phase 2) |
| Unsupervised IT + OutputCortex readout | `commit()` at IT + `OutputCortex.learn/classify()` | **DONE** (Phase 3) |
| Temporal memory (N context cells per location) | Single `Counter` per location | **TODO (Phase 5)** |
| Grid-cell location vectors (periodic, high-D) | Integer `(int, int)` tuples with Chebyshev-decay similarity | **TODO (Phase 6)** |
| HDC feature-location binding (`f ⊗ l`) | Nested dict `_model[loc][feat]` | **TODO (deferred)** |

---

## 1. Temporal Disambiguation (Layer 4 / Temporal Memory)

In biology, L4 is a **Temporal Memory (TM)**. Each location in the object model has *N cells* that fire based on which cell was active in the previous timestep — i.e., *which sequence context led to this location*.

Without TM:
- Column at location L stores `P(feature | location)` — one distribution
- Two objects sharing a feature at the same location are indistinguishable after 1 fixation

With TM (Sequence Memory):
- Column at location L stores `P(feature | location, context)` — context = prior active cell
- Same feature at same location activates *different cells* in different object sequences
- How a "4" is distinguished from a "9" after 2–3 fixations

**Current CipherNet:** `MiniColumn._model: dict[loc → Counter[feat]]` has no temporal dimension.

**Plan (Phase 5):** `_model: dict[(loc, context_cell) → Counter[feat]]`, where `context_cell` is the previous fixation's tentative winner index mod `n_context`.

---

## 2. Grid-Cell Location Encoding

**Biology:** Grid cells (entorhinal cortex) provide high-dimensional periodic location vectors:
- Nearby locations → overlapping, similar vectors (metric structure built-in)
- Distant locations → nearly orthogonal
- Multi-scale: several spatial frequencies simultaneously (Nobel Prize 2014)

**Current CipherNet:** Integer `(gy, gx)` tuples. `(3, 4)` and `(3, 5)` share zero information. The ±1 Chebyshev-distance similarity in `_loc_sim` compensates, but it's a hardcoded hack with a fixed radius.

**Correct approach** (Phase 6): Fourier feature encoding:
```
loc_vec(x, y) = [cos(2π f₁ x/S), sin(2π f₁ x/S), ..., cos(2π fₙ y/S), sin(2π fₙ y/S)]
```
`_loc_sim` becomes cosine similarity. Nearby-location generalization is automatic and scale-parameterised.

---

## 3. HDC Feature-Location Binding

**The binding operation** (Kanerva 2009):
- `f ⊗ l` = element-wise product (bipolar) of feature vector `f` and location vector `l`
- Result is dissimilar to both `f` and `l` individually — encodes *the conjunction*

**Superposition (object model):**
```
object_vec = Σᵢ (fᵢ ⊗ lᵢ)
```

**Query:** `object_vec ⊗ l` → noisy version of `f_l`; clean up via item memory.

**Current CipherNet:** `_model[loc][feat]` is associative storage, not binding. Features and locations are retrieved independently. This works for the current scale but cannot support the high-dimensional vector operations needed for HDC-style querying.

**Plan:** Deferred. Full HDC binding requires replacing the MiniColumn API entirely. Gate on Phases 5 and 6 first.

---

## 4. Contrastive Hebbian Learning

CHL is equivalent to backprop (Xie & Seung 2003) and to predictive coding (Whittington & Bogacz 2017):

**Two-phase algorithm:**
1. **Free phase:** WTA on bottom-up input → record winner `w_free`
2. **Clamped phase:** force correct output → record winner `w_clamped`
3. **Update:** `Δ = learn(w_clamped) − unlearn(w_free)`

**Status:** Implemented at V1 via `commit_chl()`. V1 `chl: true` in `guided.yaml`. The unlearn step subtracts from `w_free` when it differs from `w_clamped`.

**Interaction with feedback (Phase 2):** The IT→V1 feedback pass (implemented) biases V1 toward IT's current hypothesis before each fixation. This is *analogous to the clamped phase propagating backward* — it doesn't do full CHL but achieves a similar effect: the column that IT expects to win gets a prior boost, and if it wins (clamped = free), no unlearn occurs.

---

## 5. Lateral Voting

**Biology:** L2/3 horizontal fiber tracts within a cortical area allow distal columns to vote.

**Current CipherNet:**
- `_lateral_pass()`: ±1 grid-adjacent neighbors boost each other's tentative winner (local)
- `_global_lateral_pass()`: full-layer plurality winner gets a small bonus in every column (global)

Both run after each fixation's observe step. The global pass implements the long-range horizontal connections; the local pass implements the denser short-range connections.

---

## 6. Top-Down Feedback (White Matter / L6)

**Biology:** Every feedforward connection has a feedback counterpart. L5/6 of a higher area sends predictions back to L2/3 of a lower area. Feedback arrives *slower* than feedforward (~beta vs gamma).

**Current CipherNet:** `_feedback_pass()` runs after each fixation's feedforward sweep. IT's tentative winner's model (`_best[loc]`) predicts which V1 minicolumn should win at the corresponding RF slot. That minicolumn gets `receive_feedback_by_index(bonus)` added to its evidence — effectively a top-down prior that persists until the next `begin_image()`.

---

## References

- O'Keefe & Moser (2014) — Nobel Prize: grid cells and place cells
- Xie & Seung (2003) — "Equivalence of Backpropagation and Contrastive Hebbian Learning in a Layered Network"
- Whittington & Bogacz (2017) — "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network"
- Shouval, Bear & Cooper (2002) — "A unified model of NMDA receptor-dependent bidirectional synaptic plasticity"
- Constantinescu et al. (2016) — "Organizing conceptual knowledge in humans with a gridlike code" (Science)
- Kanerva (2009) — "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation"
- Hawkins et al. (2019) — "A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex"
