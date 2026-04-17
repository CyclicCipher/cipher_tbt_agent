# GrokkingMamba3 — Research Ideas

Three ideas from a Claude analysis session (2026-04-14) on why the grokking
transition is slow and probabilistic under standard SGD, and how to make it
fast and deterministic.

---

## Problem 1: Does Fourier Rotation Generalize?

The Fourier basis works for modular addition because addition is a **group
homomorphism on Z/pZ**, and the Fourier basis is exactly the set of irreducible
representations (irreps) of that group. The rotation in representation space is
the action of the group on its own irreducibles. This is not a coincidence —
it is a theorem.

The general principle is **representation theory of symmetry groups**. Every
problem has a symmetry group G — the set of transformations that leave the
problem's structure invariant. The right weight structure for that problem is a
G-equivariant map: one that commutes with G's action. Schur's lemma then says
that equivariant maps between irreducible representations of G are either zero
or scalar multiples of the identity — so the optimal weight matrix is
block-diagonal in the irrep basis, with one scalar per irrep.

This is why the Fourier basis emerges for addition: it diagonalizes the cyclic
group action. The generalization:

| Problem structure       | Symmetry group G        | Right basis               |
|-------------------------|-------------------------|---------------------------|
| Modular addition        | Z/pZ (cyclic)           | Discrete Fourier          |
| Spatial rotation        | SO(2) or SO(3)          | Fourier / spherical harm. |
| Permutation-inv. sets   | S_n                     | Specht modules            |
| Grid transforms (ARC)   | Dihedral x translation  | Steerable CNNs            |
| General relational      | Unknown, per-episode    | Must be discovered        |

Fourier rotation does not generalize directly, but **the principle behind it
does**: find the irreps of the problem's symmetry group, and the generalized
solution will be block-diagonal in that basis.

For ARC-AGI, the symmetry group is not fixed — it is part of what the agent
must discover per episode. The fast weight head (adapting B and C) is doing
implicit group discovery when it adapts to each episode's rule.

---

## Problem 2: Forcing Layer Coordination

Under standard SGD, each layer's gradient is computed independently. There is
no signal telling layer 3 "layer 2 is implementing a Fourier basis, you
should too." Layers must stumble into mutual consistency, which is exponentially
unlikely in the number of layers.

### Equivariance Constraints (Structural Forcing)

If G is known, constrain each weight matrix to be G-equivariant by construction.
A G-equivariant linear map W between two G-representations decomposes as:

    W = sum_rho  (M_rho (x) I_{dim(rho)})

where the sum is over irreps rho and M_rho are small unconstrained matrices.
This parameterization makes it impossible to implement a non-equivariant
function. For ARC grid encoders, D_4 equivariance (rotations and reflections
of a square grid) is a natural hard constraint.

### Alignment Losses Between Layers (Shared T_R)

Add auxiliary losses that penalize inconsistency between layers' representations
of structurally related inputs. If inputs A and B are related by rule R, the
hidden representations at EVERY layer should reflect that relationship:

    L_align = sum_l  || h_l(B) - T_R @ h_l(A) ||^2

where T_R is a per-rule linear operator **shared across ALL layers**. This is
the key: T_R is shared. If layer 7's gradient is pulling T_R in one direction,
layer 3 feels the same pull. Layers are forced to agree on T_R simultaneously.

For modular arithmetic, R = "successor" (a -> a+1). For both addition and
subtraction, shifting a by +1 shifts the result by +1. So a single T_succ is
the coordinating operator across all layers.

### Probing Losses at Intermediate Layers

Add lightweight prediction heads at each intermediate layer that must all solve
the same structural task. The gradient from each probe flows back through its
own layer and those before it, collectively imposing a consistent structural
signal throughout the network.

---

## Problem 3: Starting Low-Rank and Growing

This is the most important idea. Under standard training, the model starts
full-rank (high memorization capacity) and grokking requires weight decay to
grind down singular values until only the structural solution remains. This is
slow because you are subtracting from a large random initial spectrum.

### Incremental Rank Expansion (GrowableLinear)

Parameterize each key weight matrix as an explicit low-rank factorization:

    W = U @ diag(S) @ V

where U: (out, r), S: (r,), V: (r, in). Start at r=1 or r=small.

Growth criterion: when the loss plateau persists for N steps AND gradient
magnitude on existing singular values S has dropped below threshold (the
current rank is genuinely insufficient, not just under-optimized), add one
new singular triplet initialized near zero:

    U_new = cat([U, u_new])      u_new ~ N(0, epsilon)
    S_new = cat([S, s_new])      s_new ~ epsilon
    V_new = cat([V, v_new])      v_new ~ N(0, epsilon)

The near-zero initialization ensures the new component starts as a small
perturbation of the current solution, preserving already-learned structure.

### Why This Makes Grokking Deterministic

Under standard training: start full-rank, need weight decay to decay to
structural solution. Probabilistic because starting point is random and high
dimensional.

Under rank expansion:
- Rank 1: the model finds the BEST RANK-1 APPROXIMATION to the task.
  For addition, this is the dominant Fourier frequency.
- Rank 1 plateaus -> add rank 2: finds the NEXT Fourier frequency, constrained
  by the already-found rank-1 structure.
- Each growth step asks: "given the current r-rank structural solution, what
  is the best rank-1 addition to it?" The search space is constrained by
  existing structure, so the probability of finding a structurally consistent
  extension is orders of magnitude higher than finding the full solution at once.
- Converge to the full Fourier solution in exactly as many growth steps as
  there are relevant frequencies.
- Never pass through the memorization phase because you never had the rank
  capacity for it.

### Nuclear Norm Penalty (Convex Rank Regularization)

The nuclear norm ||W||_* = sum_i sigma_i (sum of singular values) is the convex
relaxation of matrix rank. Adding it as a regularizer directly penalizes rank:

    L_total = task_loss + lambda_nuclear * sum_layers nuclear_norm(layer.weight)

For GrowableLinear, the nuclear norm approximation is S.abs().sum() (exact when
U, V have orthonormal columns). This is O(rank) — negligible cost.

Nuclear norm penalization biases toward the minimum-rank solution that fits the
data. Combined with rank expansion, it acts as a "ratchet": grow when needed,
but don't keep unnecessary rank components.

---

## How They Compose

**Problem 1 (what basis)**: The symmetry group determines the right basis.
For arithmetic, Z/pZ -> Fourier. For ARC, the group must be discovered
per-episode by the fast weight head.

**Problem 2 (layer coordination)**: Alignment losses with shared T_R force
all layers to simultaneously implement the same structural transformation.
The shared matrix is the coordinating signal.

**Problem 3 (avoiding memorization)**: Rank expansion means the model finds
structural solutions first, by construction. Combined with nuclear norm
regularization, the model is biased toward minimum-complexity solutions.

Together: the model is parameterized to find minimal-complexity solutions first
(rank expansion), trained to maintain structural consistency across layers
(alignment losses), and the structural solution is architecturally preferred
over memorization from the very first step.

---

## Experimental Setup

Task: Modular arithmetic on Z/pZ, p=97 (standard grokking benchmark).
Operations: addition (a + b mod p) and subtraction (a - b mod p), interleaved.
Train/test split: 50% of all (a, b, op) triples in train, 50% held out.
Evaluation: exact-match accuracy on held-out test triples (no data leakage).

Conditions tested (ablation):
1. Baseline: standard Mamba3 (full-rank, no alignment, no nuclear norm)
2. Nuclear norm only: standard Mamba3 + nuclear norm penalty
3. GrowableLinear only: low-rank init, rank expansion, no alignment
4. Alignment only: standard Mamba3 + alignment loss
5. All combined: GrowableLinear + nuclear norm + alignment loss

Diagnostics:
- Train accuracy, test accuracy, generalization gap over time
- Singular value spectrum of key weight matrices at init / mid / end
- Rank growth events (when, which layer, what triggered growth)
- Alignment loss per layer over time
- Nuclear norm of weight matrices over time
- Grokking step: first step where test accuracy exceeds 95%
