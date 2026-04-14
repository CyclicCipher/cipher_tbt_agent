# TBT Architecture: HDC Binding, Grid-Cell Locations, and Contrastive Hebbian Learning

*Notes from architectural review, 2026-04-12.*

---

## Three Failure Sources in the Current Design

### 1. No Temporal Disambiguation (missing Layer 4 / Temporal Memory)

In biology, each cortical column has **four layers with distinct roles**:

| Layer | Function |
|-------|----------|
| L4    | Feedforward input from thalamus / lower areas |
| L2/3  | Lateral connections to neighboring columns; sends output to higher areas |
| L5    | Output to motor/sub-cortical; predictive voting |
| L6    | Feedback to lower areas |

The critical piece we are missing: **Layer 4 is a Temporal Memory (TM)**.  
Each location in the object model is not stored as a single value but as **N cells** that fire depending on *which cell was active in the previous time step* — i.e., which sequence context led to this location.

Without TM:
- A column at location L stores `P(feature | location)` — a single distribution
- Two different objects that share the feature at the same location are **ambiguous after 1 fixation**

With TM (Sequence Memory):
- A column at location L stores `P(feature | location, context)` — context = prior active cells
- Temporal disambiguation is built into the representation: the same feature at the same location activates *different cells* in different object sequences
- This is how a "4" that looks like part of both a "4" and a "9" is distinguished after 2–3 fixations

**Implication for CipherNet:** `MiniColumn._model: dict[location → Counter[feature]]` has no temporal dimension. Each minicolumn is a bag-of-(feature, location) pairs with no sequence context.

---

### 2. Missing Location Signal (no grid-cell-like encoding)

In biology, **grid cells in the entorhinal cortex** provide the location signal to cortical columns. Grid cells are:
- **Nobel Prize established** (O'Keefe, Moser & Moser 2014)
- High-dimensional vectors with **periodic, multi-scale firing fields**
- Nearby locations produce **overlapping, similar** vectors (metric structure)
- Distant locations produce **dissimilar, nearly orthogonal** vectors

The location key in a biological column is a **hyperdimensional vector** (HDV), not a scalar coordinate.

**Current implementation** uses retinal pixel coordinates quantized to 2-pixel bins:
```python
RESOLUTION: int = 2
return (round(raw_x / r), round(raw_y / r))  # integer tuple
```

This is a **scalar coordinate**, not a metric-preserving HDV. Problems:
- No overlap between nearby locations: `(3, 4)` and `(3, 5)` share zero information
- `overlap_score` checks `±1` cardinal neighbours as a hack to compensate, but this is not the same as a metric-preserving encoding
- The ±1 neighbour search is hardcoded; a grid-cell encoding would make nearby-location generalization automatic

**The correct approach:** Fourier feature encoding (or random Fourier features) at multiple spatial scales:
```
loc_vec(x, y) = [cos(2π x/λ₁), sin(2π x/λ₁), ..., cos(2π y/λₙ), sin(2π y/λₙ)]
```
This produces vectors where `||loc_vec(x₁) - loc_vec(x₂)||` varies smoothly with `||x₁ - x₂||`, which is the defining property of grid cells.

---

### 3. WTA Credit Assignment at V1

**The problem:** V1 is unsupervised. The WTA winner at V1 is determined by overlap with *whatever minicolumn happened to win before* — there is no guarantee that the minicolumn index that wins for digit "3" features is consistent across training images.

**Consequences:**
- V1 minicolumn → feature-at-location models are learned correctly per-minicolumn
- But IT receives `str(sorted(active))` as the feature token — e.g., `"[3]"` for minicolumn 3
- If digit "3" sometimes maps to V1 minicolumn 3 and sometimes to minicolumn 7, IT sees inconsistent tokens for the same class
- IT CAN compensate (it is supervised), but only if V1 representations are at least **consistent** — same structural feature → same minicolumn, across all images of that class

**Current V1 is purely competitive (WTA) with no class supervision.**  
Without supervision (teacher forcing or CHL), there is no mechanism ensuring "the stroke junction at top-right of a 3" always wins in minicolumn 3 vs. minicolumn 7.

---

## HDC Feature-Location Binding

**Hyperdimensional Computing (HDC)** is the mathematical framework for binding.

### The Binding Operation

For binary/bipolar vectors of dimension D (D = 1000–10000):
- `f ⊗ l` = **XOR** (binary) or **element-wise product** (bipolar) of feature vector `f` and location vector `l`
- The result is a **new vector** that is dissimilar to both `f` and `l` individually
- It encodes the **conjunction**: "feature f at location l"

### Superposition (Object Model)

An object model is the **sum** of all bindings seen during exploration:
```
object = (f₁ ⊗ l₁) + (f₂ ⊗ l₂) + (f₃ ⊗ l₃) + ...
```

To query: "what feature is at location L?"
```
query_result = object * l  # XOR/bind with location (self-inverse for XOR)
# → noisy version of f_L; clean up with item memory
```

### Biological Evidence

| Structure | HDC Role |
|-----------|----------|
| Grid cells (entorhinal) | Location vectors (periodic, multi-scale) |
| Place cells (hippocampus) | Superposition of bindings (object/scene model) |
| Theta sequences | Sequential reactivation = reading out bindings |
| Sharp-wave ripples | Offline consolidation = binding reinforcement |

The **Nobel Prize result** (Constantinescu et al., Science 2016): entorhinal grid cells fire for abstract conceptual spaces, not just physical space. This means grid cells are a **general-purpose location encoding**, not specific to navigation.

### Current Code vs. Correct Binding

**Current `column.py`:**
```python
def learn_one(self, feat: str, loc: tuple) -> None:
    # Stores (feat, loc) as SEPARATE pieces of information
    self._model[loc][feat] += 1
```

The feature and location are stored in a nested dict — they are retrievable independently. This is **not binding**; it is associative storage.

**Correct HDC binding:**
```python
def learn_one(self, feat_vec: np.ndarray, loc_vec: np.ndarray) -> None:
    binding = xor(feat_vec, loc_vec)   # or hadamard for bipolar
    self._object_vec += binding        # superposition
```

With HDC binding:
- `overlap_score` would be a **dot product** between query binding and stored object vector
- Nearby-location generalization is **automatic** because nearby `loc_vec` are similar → binding with the same feature produces similar result vectors
- The `±1 neighbour hack` in `overlap_score` is no longer needed

---

## Contrastive Hebbian Learning (CHL)

**Why CHL matters for V1 credit assignment:**

CHL is a biologically grounded learning rule proven equivalent to backpropagation (Xie & Seung 2003):

### Two-Phase Algorithm

**Free phase:** Let the network settle with only bottom-up input.
- Record activity: `x_free`

**Clamped phase:** Force the correct output (teacher forcing at output layer).
- Let activity propagate backward (recurrent/lateral connections)
- Record activity: `x_clamped`

**Weight update:**
```
ΔW = η * (x_clamped ⊗ x_clamped - x_free ⊗ x_free)
```
= Hebbian(clamped) − Hebbian(free)

### Biological Grounding

- The **calcium concentration signal** (Shouval et al. 2002) implements the two-phase gate biologically: low calcium → LTD (free phase subtraction), high calcium → LTP (clamped phase addition)
- CHL is **mathematically equivalent to predictive coding** (Whittington & Bogacz 2017): the free phase is prediction, the clamped phase is correction
- The error signal propagates via **lateral and feedback connections**, not a separate error pathway

### CHL in Our Context

For V1 credit assignment, CHL would work as follows:

1. **Free phase:** V1 runs WTA normally on the input patch → winner = `w_free`
2. **Clamped phase:** IT layer (which knows the label) sends feedback to V1 forcing the "correct" minicolumn to be active → winner = `w_clamped = label mod n_mini` (or some mapping)
3. **Update:** V1 minicolumn `w_clamped` strengthens its model for this (feature, location); V1 minicolumn `w_free` weakens

**Simpler approximation (Teacher Forcing):**
- V1 minicolumn index = IT label's winning minicolumn projected down (or just `label mod n_mini`)
- Skip the free phase; always write to the minicolumn that IT would assign
- Already used at IT layer (`commit_supervised`) — could be propagated down to V1

---

## Summary: What the Current Design Is Missing

| Biology | Current CipherNet | Impact |
|---------|------------------|--------|
| Grid-cell location vectors (periodic, high-D) | Integer coordinate tuple `(int, int)` | No metric structure; nearby-location generalization only via hardcoded ±1 neighbour hack |
| HDC binding: `f ⊗ l` | Separate nested dict `_model[loc][feat]` | Features and locations not truly bound; no vector-space query mechanism |
| Temporal memory (N cells per minicolumn, context-dependent) | Single Counter per location | No temporal disambiguation; two objects sharing a feature at a location are indistinguishable after 1 fixation |
| CHL / predictive coding credit assignment | Unsupervised WTA at V1 | No guarantee V1 minicolumns learn consistent class-specific representations |
| Lateral voting (L2/3 connections between columns) | Independent column decisions | No cross-column consistency enforcement |

---

## References

- O'Keefe & Moser (2014) — Nobel Prize in Physiology/Medicine: grid cells and place cells
- Xie & Seung (2003) — "Equivalence of Backpropagation and Contrastive Hebbian Learning in a Layered Network"
- Whittington & Bogacz (2017) — "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network"
- Shouval, Bear & Cooper (2002) — "A unified model of NMDA receptor-dependent bidirectional synaptic plasticity"
- Constantinescu et al. (2016) — "Organizing conceptual knowledge in humans with a gridlike code" (Science)
- Kanerva (2009) — "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation"
- Hawkins et al. (2019) — "A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex"
