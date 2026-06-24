# Grid ↔ Place: the Fourier Duality of the Reference Frame (gap C reference)

Research synthesis (2026-06-16) for the **grid→place orthogonalisation** step of gap C
(feature⊗location binding; `BLUEPRINT.md` Finding 8). The central result reframes the whole step:

> **The grid code and the place code are the same information in two bases, related by the Fourier
> transform on the group.** So "orthogonalise the location code while preserving movement" is not an open
> engineering problem — it is the **inverse group-Fourier transform**, analytically exact and
> movement-preserving by construction.

This ties directly to `PURE_MATH_FOR_ML.md §3` (the generalising circuit = the group's *irreducible*
representations) and extends it: **place = the *regular* representation.**

## 1 · The duality (the headline)

| | **Grid code** (gaps A/B) | **Place code** (gap C) |
|---|---|---|
| basis | Fourier / character = **irreducible reps** | localized / delta = **regular representation** |
| dimensionality | low, distributed | high, sparse |
| geometry | **metric** (nearby ≈ similar) | **orthogonal** (distinct = separable) |
| group action | **diagonal** — a rotation per frequency | **permutation** — cyclic shift |
| good for | path integration, generalisation | **binding, memory** (orthogonal keys) |
| in our system | the discovered rings/tori, generator `R` | the binding keys for `S = Σ fᵢ ⊗ z(locᵢ)` |

**Place = inverse-Fourier of grid.** A place field is a weighted sum of multi-scale grid modes; the two
bases are *unitarily equivalent* (the DFT for cyclic groups; Peter-Weyl in general). Because both are
representations of the **same group**, the change of basis **preserves the group action (movement)** — the
permutation action on places and the diagonal (rotation) action on grids are the same operator in two
bases. That is exactly the property our next step needed.

## 2 · Neuroscience — the mechanism

- **Grid→place transform.** Place fields form by **linear summation of 10–50 multi-scale grid cells**
  (Solstad, Moser & Einevoll 2006). That sum *is* the inverse Fourier synthesis. Network-level structure:
  Si, Treves and others.
- **Dentate-gyrus pattern separation.** DG **sparsifies, orthogonalises, and expansion-recodes** EC input
  (more granule cells than inputs), so overlapping grid inputs become non-overlapping CA3 codes for
  memory. This is the grid(overlapping)→place(orthogonal) step — and it is **nonlinear** (a sparsifying
  threshold), not a bare linear sum.
- **Grid as a Fourier basis (explicit).** Rodriguez & Caplan 2019 (hexagonal Fourier model of grid cells);
  "Does the entorhinal cortex use the Fourier transform?" (2013). Grid population ≈ an inverse Fourier
  transform of a sampled signal; place fields = linearly weighted combinations of the modes.
- **Tolman-Eichenbaum Machine** (Whittington et al. 2020) is the integrated model: structural `g` (grid)
  ⊗ sensory `x` → conjunctive `p` (place). That **is** our feature⊗location binding, and place cells are
  the bound representation — direct architectural precedent.

## 3 · Pure math — capacity, and the required nonlinearity

- **Group representation theory.** Grid = the **irreducible** representations (characters / Fourier modes;
  the generalising circuit of `PURE_MATH §3`). Place = the **regular representation** (the action of the
  group on itself; the localized/one-hot basis), which *decomposes into the direct sum of all irreps* —
  so grid (a few irreps) and place (the regular rep) are two views of one object, the **group Fourier
  transform** between them. Movement is preserved because it is the same group element acting. (Gao, Xie,
  Zhu & Wu 2020 model grid cells as exactly this — block-diagonal generators, one rotation per module,
  our multi-generator structure.)
- **VSA/HDC capacity theory** (Clarkson, Frady, Kleyko, Sommer et al., "Capacity Analysis of Vector
  Symbolic Architectures", 2023). Formal bounds on **how many bindings a superposition holds before
  crosstalk**, as a function of dimensionality. This *predicts* the Finding-8 capacity curves and gives
  the place-code dimension needed for a target object count `K`.
- **Expansion recoding / sparse coding** (Babadi & Sompolinsky 2014, *Sparseness and Expansion in Sensory
  Representations*; cerebellum-as-kernel-machine 2022). Projecting to a **high-dim sparse** code makes
  patterns orthogonal/separable — but with a **crucial caveat: linear expansion amplifies noise**; a clean
  sparse-orthogonal code needs a **nonlinearity (threshold / top-k)** that exploits signal structure.

## 4 · What it gives gap C (now a concrete, theory-backed construction)

The grid→place orthogonalisation becomes mostly analytic:

1. **Inverse group-Fourier** — sum the multi-scale grid modules with localized weights to synthesise a
   place field peaked at each location. **Movement-preserving by construction** (same group; permutation
   action on places). For our cyclic/abelian setup this is literally the inverse DFT of the module phases.
2. **Sparsify (nonlinear, DG-style)** — threshold / top-k the place activations → **sparse orthogonal
   keys**. This is the required nonlinearity (Babadi-Sompolinsky); it avoids the linear noise blow-up and
   produces near-orthonormal keys (the idealised ceiling of Finding 8).
3. **Capacity is set by the number of modules/frequencies** — more grid modules → sharper place fields →
   more orthogonal → higher binding capacity. This *is* Finding 8 (multi-scale beat single-ring), now with
   a theory: capacity ≈ a VSA bound in the effective place dimension.

**Upshot:** we do not need to *learn* a mysterious orthogonaliser. Build place as **inverse-DFT +
sparsification**; the group-rep framing guarantees it preserves movement, and VSA theory predicts the
capacity. The brain implements the same thing (grid summation in CA1/DG + sparsification); we get it in
closed form.

**Validated (BLUEPRINT Finding 9, `binding.py:place_from_grid`).** Built exactly this — place field =
similarity profile across the grid code (circulant, peaked) + top-k; grid `R` → cyclic-shift permutation in
place space; top-k commutes with the shift so movement is preserved. Result: top-k≤3 → **binding
what/where/move = 1.00 at every K≤8** (raw grid code: 0.87/0.83; orthonormal ceiling: 1.00). Closes the gap;
sparser k → higher capacity (the Babadi-Sompolinsky / DG sparse regime). The theory delivered, no learning.

**Division of labour (the design rule this settles):** keep **two coupled codes** — the **grid** basis for
*path integration / generalisation / movement* (gaps A, B) and the **place** basis for *binding / memory*
(gap C) — related by the Fourier transform, not one code forced to do both. This is the brain's
entorhinal↔hippocampal split, and it resolves the metric-vs-orthogonal tension we hit in Finding 8.

## References

Neuroscience:
- Solstad, Moser & Einevoll 2006, *From grid cells to place cells: a mathematical model* (Hippocampus 16)
  — https://pubmed.ncbi.nlm.nih.gov/17094145/
- *The structure of networks that produce the grid-to-place-cell transformation* —
  https://ncbi.nlm.nih.gov/pmc/articles/PMC3210383
- *Reassessing pattern separation in the dentate gyrus* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3726960/
- Rodriguez & Caplan 2019, *A hexagonal Fourier model of grid cells* — https://pubmed.ncbi.nlm.nih.gov/30216605/
- *Does the entorhinal cortex use the Fourier transform?* (Front. Comput. Neurosci. 2013) —
  https://www.frontiersin.org/articles/10.3389/fncom.2013.00179/full
- Whittington et al. 2020, *The Tolman-Eichenbaum Machine* (Cell) — grid ⊗ sensory → place.

Pure math / ML:
- *Regular representation* — https://en.wikipedia.org/wiki/Regular_representation (irreps ⊕ → regular rep;
  the group Fourier transform / Peter-Weyl).
- Gao, Xie, Zhu & Wu 2020, *On Path Integration of Grid Cells: Group Representation & Isotropic Scaling* —
  https://arxiv.org/pdf/2006.10259 (block-diagonal generators = our multi-module structure).
- *Capacity Analysis of Vector Symbolic Architectures* (2023) — https://arxiv.org/abs/2301.10352
- Babadi & Sompolinsky 2014, *Sparseness and Expansion in Sensory Representations* (Neuron) —
  https://www.cell.com/neuron/fulltext/S0896-6273(14)00646-1
- *Cerebellum as a kernel machine: expansion recoding in the granule cell layer* (2022) —
  https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1062392/full
- *GridPE: Unifying Positional Encoding in Transformers with a Grid-Cell framework* (2024) —
  https://arxiv.org/pdf/2406.07049 (grid-cell PE in transformers; relevant to our PoPE substrate).
- *Place Cells as Multi-Scale Position Embeddings: Random Walk Transition Kernels* (2025) —
  https://arxiv.org/pdf/2505.14806.
