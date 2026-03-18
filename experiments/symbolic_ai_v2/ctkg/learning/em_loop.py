"""
Phase 2+4 EM: iterate fca_discover <-> graph_grammar to convergence.

The EM loop coordinates Phase 2 (FCA on H — concept discovery) and Phase 4
(SEQUITUR — compression tree recovery).

E-step (t=0): assign types using flat HankelCount → ConceptLattice.
M-step:       run SEQUITUR on typed corpus → grammar.
              discover morphisms using current type assignment.
E-step (t≥2): use compression-tree-position contexts (architecture §Phase 2
              iteration n) — FCA on H_tree instead of flat H.

Convergence: BOTH of the following must hold (architecture §Phase 2):
    (a) Relative DL change < tol: |DL(t) - DL(t-1)| / DL(t-1) < tol
    (b) Productivity threshold: top-100 concept productivity scores all ≥
        productivity_threshold (composition-based, not entropy-based).

If (a) holds but (b) fails, λ is doubled and EM resumes (architecture:
"multiply λ by 2 and resume from current model — not from scratch").
At most one λ-doubling is attempted before proceeding with a warning.

Description length (MDL proxy):
    DL = grammar_encode_length + n_concepts * log2(n_concepts)

See CTKG_ARCHITECTURE.md §Phase 2 for the full specification.
"""

from __future__ import annotations

import math

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import (
    discover_concepts,
    discover_concepts_from_tree,
    composition_productivity,
    all_concepts_productive,
)
from experiments.symbolic_ai_v2.ctkg.learning.graph_grammar import (
    corpus_grammar,
    grammar_description_length,
    typed_corpus_from_lattice,
)
from experiments.symbolic_ai_v2.ctkg.learning.morphism_discover import (
    discover_morphisms,
)
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import ConceptLattice
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph


def em_loop(
    corpus: list[list[str]],
    r_levels: list[int] | None = None,
    n_em_max: int = 50,
    lambda_productivity: float = 0.1,
    merge_threshold: float = 0.15,
    min_support: float = 3.0,
    max_contexts: int = 80,
    tol: float = 0.001,
    productivity_threshold: float = 0.1,
    max_grammar_seqs: int = 200,
    verbose: bool = False,
) -> tuple[list[ConceptLattice], MorphismGraph, list[float]]:
    """Run the Phase 2↔4 EM loop to convergence.

    Parameters
    ----------
    corpus:
        Raw string sequences (training data).
    r_levels:
        Radius levels for FCA.  Defaults to [1, 2, 3].
    n_em_max:
        Hard cap on EM iterations.
    lambda_productivity:
        MDL regularisation strength for FCA and morphism pruning.
    merge_threshold:
        JSD threshold for concept FCA.
    min_support:
        Minimum raw context count for FCA (filters noise).
    max_contexts:
        Hard cap on contexts per FCA level (keeps complexity O(k² log k)).
    tol:
        Relative convergence threshold for DL: converged when
        |DL(t) - DL(t-1)| / DL(t-1) < tol.
    productivity_threshold:
        Composition-based productivity threshold for convergence criterion (b).
        All top-100 concepts must score >= this before EM halts.
        Architecture: productivity = novel_compositions / all_compositions.
    max_grammar_seqs:
        Maximum number of sequences to pass to SEQUITUR for grammar induction.
        SEQUITUR is O(N²) in corpus length; capping at 200 sequences gives a
        ~100x speedup with negligible loss of grammar quality (the grammar
        discovers the same repeated digrams from a representative subsample).
        Set to None or 0 to disable subsampling.
    verbose:
        If True, print DL value at each iteration.

    Returns
    -------
    lattices:
        Final ConceptLattices, one per r_level.
    morphism_graph:
        Final MorphismGraph after convergence.
    dl_history:
        List of description lengths at each EM iteration.
    """
    if r_levels is None:
        r_levels = [1, 2, 3]

    # Grammar subsample helper — SEQUITUR is O(N²); cap to keep it fast.
    def _grammar_corpus(tc: list[list[int]]) -> list[list[int]]:
        """Return a representative subsample of typed_corpus for SEQUITUR."""
        if not max_grammar_seqs or len(tc) <= max_grammar_seqs:
            return tc
        # Evenly-spaced indices (deterministic, covers full range)
        step = len(tc) / max_grammar_seqs
        return [tc[int(i * step)] for i in range(max_grammar_seqs)]

    # Phase 1: build HankelCount (fixed across EM iterations)
    hc = HankelCount(r_max=max(r_levels))
    hc.update_batch(corpus)

    # t=0 E-step: initial FCA from flat co-occurrence (bootstrap)
    lattices = discover_concepts(
        hankel=hc,
        r_levels=r_levels,
        lambda_productivity=lambda_productivity,
        merge_threshold=merge_threshold,
        min_support=min_support,
        max_contexts=max_contexts,
    )

    primary_r_idx = 0
    primary_r = r_levels[primary_r_idx]

    dl_history: list[float] = []
    morphism_graph = MorphismGraph()

    # Compute initial DL
    typed_corpus = typed_corpus_from_lattice(corpus, hc, lattices[primary_r_idx], r=primary_r)
    grammar = corpus_grammar(_grammar_corpus(typed_corpus))
    n_concepts = len(lattices[primary_r_idx].concepts)
    dl = (grammar_description_length(grammar, typed_corpus)
          + n_concepts * math.log2(max(n_concepts, 2)))
    dl_history.append(dl)

    if verbose:
        print(f"  EM t=0: DL={dl:.1f}, n_concepts={n_concepts}, "
              f"n_grammar_rules={grammar.n_rules()}")

    lambda_current = lambda_productivity
    lambda_doubled = False  # track whether we've already doubled λ

    for t in range(1, n_em_max + 1):
        # M-step: grammar on typed corpus
        typed_corpus = typed_corpus_from_lattice(
            corpus, hc, lattices[primary_r_idx], r=primary_r
        )
        grammar = corpus_grammar(_grammar_corpus(typed_corpus))

        # Phase 3: morphism discovery on current type assignment
        morphism_graph = discover_morphisms(
            corpus=corpus,
            hankel=hc,
            lattice=lattices[primary_r_idx],
            r=primary_r,
            merge_threshold=merge_threshold,
            lambda_productivity=lambda_current,
        )

        # E-step: FCA
        if t == 1:
            # First iteration: still use flat co-occurrence (grammar not yet stable)
            lattices = discover_concepts(
                hankel=hc,
                r_levels=r_levels,
                lambda_productivity=lambda_current,
                merge_threshold=merge_threshold,
                min_support=min_support,
                max_contexts=max_contexts,
            )
        else:
            # t >= 2: use compression-tree-position contexts (architecture §Phase 2)
            # The primary lattice (r_levels[0]) is rebuilt from tree contexts;
            # other r_levels continue to use flat FCA.
            tree_lattice = discover_concepts_from_tree(
                corpus_grammar=grammar,
                typed_corpus=typed_corpus,
                lattice=lattices[primary_r_idx],
                lambda_productivity=lambda_current,
                merge_threshold=merge_threshold,
                max_contexts=max_contexts,
                min_support=min_support,
            )
            # For multi-radius: rebuild other levels with flat FCA.
            # Single-radius case (len==1): skip the dead flat_lattices call.
            if len(r_levels) > 1:
                flat_lattices = discover_concepts(
                    hankel=hc,
                    r_levels=r_levels[1:],
                    lambda_productivity=lambda_current,
                    merge_threshold=merge_threshold,
                    min_support=min_support,
                    max_contexts=max_contexts,
                )
                lattices = [tree_lattice] + flat_lattices
            else:
                lattices = [tree_lattice]

        # Compute DL with updated lattice
        typed_corpus = typed_corpus_from_lattice(
            corpus, hc, lattices[primary_r_idx], r=primary_r
        )
        grammar = corpus_grammar(_grammar_corpus(typed_corpus))
        n_concepts = len(lattices[primary_r_idx].concepts)
        new_dl = (grammar_description_length(grammar, typed_corpus)
                  + n_concepts * math.log2(max(n_concepts, 2)))
        dl_history.append(new_dl)

        if verbose:
            print(f"  EM t={t}: DL={new_dl:.1f}, n_concepts={n_concepts}, "
                  f"n_morphisms={len(list(morphism_graph.morphisms()))}, "
                  f"n_grammar_rules={grammar.n_rules()}, lambda={lambda_current:.3f}")

        # --- Convergence check (architecture: BOTH criteria must hold) ---
        prev_dl = dl_history[-2]
        dl_converged = (prev_dl > 0
                        and abs(new_dl - prev_dl) / prev_dl < tol)

        if dl_converged:
            # Check criterion (b): composition-based productivity
            prod_scores = composition_productivity(
                grammar, lattices[primary_r_idx], typed_corpus
            )
            prod_ok = all_concepts_productive(
                prod_scores,
                threshold=productivity_threshold,
                top_k=100,
            )
            if prod_ok:
                if verbose:
                    print(f"  EM converged at t={t} "
                          f"(DL OK, productivity OK)")
                break
            else:
                # Criterion (a) met but (b) fails → double λ and resume
                if not lambda_doubled:
                    lambda_current *= 2.0
                    lambda_doubled = True
                    if verbose:
                        min_prod = min(prod_scores.values()) if prod_scores else 0.0
                        print(f"  EM: DL converged but productivity insufficient "
                              f"(min={min_prod:.3f} < {productivity_threshold}). "
                              f"Doubling lambda to {lambda_current:.3f}.")
                else:
                    # Already doubled once — proceed with warning
                    if verbose:
                        print(f"  EM: productivity still insufficient after "
                              f"lambda doubling. Proceeding with current model.")
                    break

    return lattices, morphism_graph, dl_history
