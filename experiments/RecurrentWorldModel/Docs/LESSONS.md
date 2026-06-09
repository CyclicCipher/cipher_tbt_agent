# Architectural Lessons — Experiments & Papers

> A running log of what we've actually learned building the Clamped Settling Core,
> from our own runs and from the literature. Each entry: the lesson, the evidence,
> and the source. Newest sections appended; keep it honest (record what failed).

## The working recipe so far (quick reference)

For the Stage-0 settling core on a small model + small data:
- **Positional encoding: PoPE** (`pos_mode="pope"`). Clean what/where split; big OOD/length-extrapolation win.
- **Convergence: `state_norm`** (RMS-normalize the state each iteration) — without it the residual block drifts and never settles.
- **Gradient: BPTT** (`grad_mode="bptt"`), not IFT, at this scale — matches "Solve the Loop" practice for tiny models.
- **Stability levers (being tried):** warm-start the settle from a proposal (not zeros); QK-norm on raw features before PoPE's softplus.
- Open: the BPTT result generalizes OOD *better than the matched fixed-depth baseline* (0.57 vs 0.47) but oscillates — stability work in progress.

---

## 1 · Convergence & settling dynamics

**A single-block DEQ is not Ouro — expect strange dynamics.** One weight-tied map iterated to a fixed point is the *pure* DEQ form (Bai 2019); Ouro loops a multi-*layer* stack. Low per-iteration expressivity. Block depth `k` is a knob, not a constraint.

**A pre-norm *residual* block iterated is an integrator, not a contraction.** `h ← h + g(h)` drifts (we measured state RMS blowing up to ~60), never reaching a fixed point — so there's nothing for the gradient to stand on. *Fix:* `state_norm` (RMS-normalize the state each iteration) → converges in ~7 iters untrained. Convergence (Risk 1) is the linchpin the whole architecture rests on.

**The settling map is contractive but *very slowly* (spectral radius ≈ 0.99), and that rate is intrinsic.** The forward residual decreases monotonically but crawls (5.8e-3 @ 40 iters → 1.6e-3 @ 200; would hit tol ≈ 1e-3 around ~260 iters). Anderson acceleration *diverges* on this map. **Empirically falsified fixes:** QK-norm-on-raw and warm-start did **not** change the asymptotic convergence rate (all still ~1.6e-3 @ 200) — the slow rate lives in the Jacobian near the fixed point, not in the magnitude scale or the starting point. Warm-start changes *where you start*, not *how fast it contracts*. So the genuine convergence lever (needed for IFT) is a *structural* one — spectral/weight norm or a gated/damped update that reduces the spectral radius — not normalization of the features. For BPTT this doesn't matter (it differentiates a fixed unroll, convergence-agnostic).

**The BPTT "oscillation" was a train/eval mismatch, not a dynamics problem.** BPTT *trains* on a fixed N-step unroll, but `evaluate()` ran under `no_grad`, which skipped the BPTT path and fell through to the Picard solver — reading the state at a different, non-converged point than training optimized. *Fix:* run the same fixed N-step unroll in eval. Consistency, not convergence. (Echoes Solve-the-Loop's point that train/test settling must match.)

## 2 · Gradient choice depends on scale

**`one_step` gradient is too weak.** It backprops through a single block step at the equilibrium and can't shape the deep recurrent computation → stuck at chance loss (ln 7).

**IFT (implicit) is correct *only at a converged fixed point*.** Verified exact vs a full-BPTT reference (cosine 1.00000) *when the forward converges*. But in the full PoPE model the forward doesn't converge within budget, so the `(I−J_f)⁻¹` adjoint is ill-conditioned and training collapses to a trivial attractor at chance. IFT needs: a contractive map + tight convergence tolerance (DEQ trains at ε≈√T·10⁻⁵, not 10⁻³).

**BPTT/phantom is the right tool at small scale — and this is SOTA practice, not a compromise.** "Solve the Loop" (Attractor Models, 2605.12466) uses one-step IFT for large LMs but **switches to the phantom/unrolled gradient for tiny models + ~1k examples** because "the one-step surrogate can provide too crude a training signal." Our IFT-fails / BPTT-works outcome *is* their conclusion for our regime. IFT is a large-model memory tool; revisit it there.

## 3 · Positional encoding (RoPE → PoPE)

**PoPE cleanly decouples what (content) from where (position) — RESOLVED, see architecture.md §2.2.** Magnitude=`softplus(features)`=content (position-independent); phase=position; score factorizes *exactly* into (what-match)×(where-match). RoPE couldn't — its rotary inner product carries a content-dependent phase cross-term.

**PoPE was night-and-day for OOD.** OOD accuracy went from flat-at-chance (~0.14) to 0.47–0.60 and climbing, for everything that learns. Its proven strengths (10× zero-shot length extrapolation; indirect-indexing 11%→95%) map exactly onto our failure mode.

**Three distinct "what/where" splits — don't conflate them.** (a) content vs *sequence position* = PoPE; (b) essence vs accident *within* content (the "cat independent of texture") = JEPA's job, no positional revision needed; (c) the *coordinate system* of "where" — PoPE's is the 1D token axis; generalizing it to a learned reference frame (grid-cell / Yoneda) is agenda item 29, a known future evolution.

**PoPE + QK-norm:** the PoPE paper uses *no* QK-norm (it's feed-forward; global pre-norm RMSNorm suffices). For our recurrent setting, apply QK-norm to the **raw features before softplus** — bounds magnitudes and preserves the decoupling (magnitude stays position-independent). Not in the paper; the natural extrapolation, and what the repo's Mamba3 `apply_pope` already does. *Caveat (measured):* it is principled but did **not** measurably improve convergence — keep it for logit stability, but don't expect it to fix the slow contraction (§1).

## 4 · Stable training of equilibrium / attractor models (DEQ + Solve the Loop)

**Warm-start from a coherent proposal, not zeros.** "Solve the Loop"'s central stability trick: a backbone proposes `ỹ₀`, the attractor refines *from it* with `ỹ₀` persistently injected. Initializing from a coherent proposal "makes training stable compared to DEQ," which "blows up in the number of iterations" from zeros.

**Operator-norm control.** DEQ: "the weights' operator norm is directly related to the stability of root-finding; weight normalization typically finds more stable parameters" — plus small init and LayerNorm/gating to constrain output ranges. Our levers: QK-norm-on-raw (§3) and weight/spectral norm.

**Robust root-finder + tight training tolerance.** DEQ uses Broyden (quasi-Newton); "deeply stacked self-attention tends to oscillate around the fixed point" with naive iteration (our exact symptom). Training needs tight tol; inference can be loose.

**The single-operator purity has a cost.** The SOTA-stable design (Solve the Loop) uses *two* modules — backbone proposes, attractor refines — a deliberate relaxation of pure weight-tying. Our "one operator" thesis is harder to train; warm-start (via one pass of the *same* block) and the `k`-layer knob bridge most of the gap without abandoning the thesis. Worth tracking as a known tension.

## 5 · Process / methodology

**If the baseline fails identically, the architecture isn't the variable — the test is.** Both our settling core *and* the fixed-depth baseline failed OOD at chance until PoPE. That's the signature of a confounded test (length/position extrapolation), not an architecture verdict. (Echoes repo Mistake #42: memorization vs generalization on small algorithmic tasks.) Always train+eval the matched baseline.

**Read the paper, not just the code.** The repo's Mamba3 `apply_pope` puts the learnable `δ` in the softplus (magnitude); the *paper* puts `δ_c` in the **phase**. We follow the paper. Reading caught it.

**How to read papers here:** no local PDF tooling (`pdftoppm`/`pip` unavailable offline; zlib-from-PDF-bytes choked on a 3.9 MB file). `WebFetch` on the arXiv **HTML** version (`arxiv.org/html/<id>` or `ar5iv.labs.arxiv.org/html/<id>`) is the reliable path.

---

## Papers referenced

- **Bai, Kolter, Koltun 2019 — Deep Equilibrium Models** (arXiv:1909.01377). Implicit differentiation through the equilibrium; constant memory; Broyden solver; weight-norm for stability.
- **Solve the Loop: Attractor Models for Language and Reasoning** (arXiv:2605.12466) — backbone proposal + attractor refinement; warm-start stability; phantom gradient for small models. The closest paper to our design.
- **Gopalakrishnan et al. 2025 — PoPE** (arXiv:2509.10534). Polar coordinate positional embeddings; exact what/where decoupling; length extrapolation.
