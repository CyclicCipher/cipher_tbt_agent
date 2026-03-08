"""e6_meta_synthesis.py — Phase E6: Meta-Synthesis (Composition Search over CTKG)

Given a few examples of a NEW concept, search for a factorisation through
existing CTKG concepts. This is the symbolic analogue of what backprop does
implicitly: finding the composition of primitive operations that explains
observed input-output pairs.

Architecture:

    CTKG DAG search:
        The existing AI stores form a DAG of concepts:
            next_word_hier  ←  next_cat  ←  word_pos (assignment)
            word_given_cat  ←  next_cat  ←  word_pos
            slot_occupants  ←  word_pos (from E4)
            ...
        A new concept C: X → Y can sometimes be expressed as:
            C(x) = f_k(f_{k-1}(...f_1(x)...))
        where each f_i is an existing CTKG concept.

    Search algorithm (beam search over concept chains):
        1. Start from input type X.
        2. At each step, try all stored concepts whose input type matches
           the current intermediate type.
        3. Apply the concept to the intermediate output of each example.
        4. Score the chain by: coverage × accuracy on the given examples.
        5. Beam search (width B) explores the K best partial chains.
        6. Return the highest-scoring chain of length 1..max_depth.

    Example:
        New concept: "word_after_verb" (words following verbs in Latin)
        Examples: [('dicere', 'quod'), ('esse', 'iam'), ('fuit', 'ergo')]
        Search finds: word_pos ∘ next_cat ∘ word_given_cat  (E1 chain)
        OR discovers that the examples come from a specific cluster pattern.

    Key insight:
        Transformers learn composition implicitly via gradient descent.
        We search composition EXPLICITLY using the existing CTKG concepts.
        If the examples are explained by an existing chain → no new parameters.
        If not → add a minimal new concept to the DAG.

Usage:
    python e6_meta_synthesis.py --corpus EarlyModernLatin --n_train 5000
    python e6_meta_synthesis.py --corpus EarlyModernLatin --demo next_word
    python e6_meta_synthesis.py --corpus EarlyModernLatin --demo slot_fill
"""
from __future__ import annotations

import argparse
import collections
import itertools
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

import io
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', 'utf-8').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from discover_structure import run_pipeline, _stream_texts, _banner, _DATA_DIR
from language_pipeline import build_assignment, train_chain
from e4_paradigmatic import train_slot_occupants

try:
    from ocr_test import find_pairs
except ImportError:
    import glob as _glob, random as _random
    def find_pairs(d, max_n=None, shuffle=False, seed=0):
        pairs = []
        for gt in _glob.glob(os.path.join(d, '**', '*.gt.txt'), recursive=True):
            pairs.append((None, gt))
        if shuffle:
            rng = _random.Random(seed)
            rng.shuffle(pairs)
        return pairs[:max_n] if max_n else pairs


# ---------------------------------------------------------------------------
# Concept graph traversal
# ---------------------------------------------------------------------------

class ConceptNode:
    """A single step in a composition chain."""

    def __init__(self, name: str, ai):
        self.name = name
        self._ai  = ai

    def apply(self, inputs: tuple) -> tuple | None:
        """Apply this concept to an input tuple. Returns output tuple or None."""
        result = self._ai.ask(self.name, inputs)
        if result is None:
            return None
        return result if isinstance(result, tuple) else (result,)

    def apply_dist(self, inputs: tuple) -> dict | None:
        """Return output distribution for this input."""
        return self._ai.ask_dist(self.name, inputs)

    def __repr__(self):
        return f'ConceptNode({self.name!r})'


class CompositionChain:
    """An ordered sequence of concept nodes forming a composition."""

    def __init__(self, nodes: list[ConceptNode]):
        self.nodes = nodes

    @property
    def name(self) -> str:
        return ' ∘ '.join(n.name for n in self.nodes)

    def apply(self, inputs: tuple) -> tuple | None:
        """Apply each concept in sequence, threading output → input."""
        current = inputs
        for node in self.nodes:
            current = node.apply(current)
            if current is None:
                return None
        return current

    def score(self, examples: list[tuple]) -> tuple[float, float]:
        """Score this chain on (input, expected_output) examples.

        Returns (coverage, accuracy):
            coverage = fraction of examples where chain returned an answer
            accuracy = fraction of answered examples where answer is correct
        """
        n_answered = 0
        n_correct  = 0
        for inp, expected in examples:
            out = self.apply(inp)
            if out is not None:
                n_answered += 1
                if out == expected or (len(out) >= 1 and out[0] == expected[0]):
                    n_correct += 1
        n = len(examples)
        coverage = n_answered / n if n else 0.0
        accuracy = n_correct / n_answered if n_answered else 0.0
        return coverage, accuracy

    def __repr__(self):
        return f'CompositionChain([{self.name}])'


# ---------------------------------------------------------------------------
# Beam search over concept chains
# ---------------------------------------------------------------------------

def get_all_concepts(ai) -> list[str]:
    """Return all concept names currently stored in the AI."""
    return list(ai.stores.keys())


def enumerate_single_step_chains(
    ai,
    input_examples: list[tuple],
    min_coverage:   float = 0.1,
) -> list[tuple[CompositionChain, float, float]]:
    """Try all single-concept chains on the examples.

    Returns [(chain, coverage, accuracy), ...] for concepts with coverage ≥ min_coverage.
    """
    results = []
    for name in get_all_concepts(ai):
        chain = CompositionChain([ConceptNode(name, ai)])
        cov, acc = chain.score(input_examples)
        if cov >= min_coverage:
            results.append((chain, cov, acc))
    return sorted(results, key=lambda x: -(x[1] * x[2]))


def _extend_chain(
    ai,
    base_chain:     CompositionChain,
    examples:       list[tuple],
    min_coverage:   float = 0.05,
) -> list[tuple[CompositionChain, float, float]]:
    """Extend a chain by one concept and score."""
    # First, collect intermediate outputs from the base chain
    intermediates = []
    for inp, expected in examples:
        mid = base_chain.apply(inp)
        if mid is not None:
            intermediates.append((mid, expected))

    if not intermediates:
        return []

    results = []
    for name in get_all_concepts(ai):
        extended = CompositionChain(base_chain.nodes + [ConceptNode(name, ai)])
        cov, acc = extended.score(examples)
        if cov >= min_coverage:
            results.append((extended, cov, acc))
    return results


def beam_search_composition(
    ai,
    examples:     list[tuple],     # [(input_tuple, output_tuple), ...]
    max_depth:    int   = 3,
    beam_width:   int   = 5,
    min_coverage: float = 0.05,
    verbose:      bool  = True,
) -> list[tuple[CompositionChain, float, float]]:
    """Beam search for a composition chain explaining the given examples.

    At each depth, keeps the best `beam_width` chains by coverage×accuracy.
    Returns all chains found (sorted by score), not just the best.

    Args:
        examples:     list of (input_tuple, expected_output_tuple) pairs.
        max_depth:    maximum composition length (1 = single concept).
        beam_width:   number of partial chains to keep at each depth.
        min_coverage: minimum fraction of examples a chain must cover to remain.
    """
    if verbose:
        _banner('Phase E6: Beam-search composition synthesis')
        print(f'  Examples: {len(examples)}')
        print(f'  Max depth: {max_depth}   Beam width: {beam_width}')
        print(f'  Available concepts: {len(get_all_concepts(ai))}')
        print()

    all_found: list[tuple[CompositionChain, float, float]] = []

    # Depth 1: single concepts
    beam = enumerate_single_step_chains(ai, examples, min_coverage=min_coverage)
    if verbose:
        print(f'  Depth 1: {len(beam)} chains with coverage≥{min_coverage:.0%}')

    all_found.extend(beam)
    beam = beam[:beam_width]

    # Depth 2+: extend
    for depth in range(2, max_depth + 1):
        candidates = []
        for chain, cov, acc in beam:
            extensions = _extend_chain(ai, chain, examples, min_coverage=min_coverage)
            candidates.extend(extensions)

        if not candidates:
            if verbose:
                print(f'  Depth {depth}: no extensions with coverage≥{min_coverage:.0%}')
            break

        # Sort by coverage × accuracy
        candidates.sort(key=lambda x: -(x[1] * x[2]))
        all_found.extend(candidates)

        if verbose:
            print(f'  Depth {depth}: {len(candidates)} chains')
            for chain, cov, acc in candidates[:3]:
                print(f'    [{cov:.0%} cov, {acc:.0%} acc] {chain.name}')

        beam = candidates[:beam_width]

    # Deduplicate and sort final results
    seen_names: set = set()
    unique: list = []
    for chain, cov, acc in sorted(all_found, key=lambda x: -(x[1] * x[2])):
        if chain.name not in seen_names:
            seen_names.add(chain.name)
            unique.append((chain, cov, acc))

    return unique


# ---------------------------------------------------------------------------
# Demo tasks
# ---------------------------------------------------------------------------

def make_next_word_examples(
    train_texts: list[str],
    assignment:  dict,
    ai,
    n: int = 20,
) -> list[tuple]:
    """Build (input=(w1, w2), expected=(w3,)) examples for next-word prediction.

    Tests whether beam search rediscovers the E1 chain from raw examples.
    """
    examples = []
    seen: set = set()
    for text in train_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            c1 = assignment.get(w1)
            c2 = assignment.get(w2)
            if c1 is None or c2 is None:
                continue
            key = (w1, w2)
            if key in seen:
                continue
            seen.add(key)
            examples.append(((w1, w2), (w3,)))
            if len(examples) >= n:
                return examples
    return examples


def make_slot_fill_examples(
    train_texts: list[str],
    assignment:  dict,
    n: int = 20,
) -> list[tuple]:
    """Build (input=(w1, w3), expected=(w2,)) examples for slot filling.

    Tests whether beam search rediscovers slot_occupants from raw examples.
    """
    examples = []
    seen: set = set()
    for text in train_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            c1 = assignment.get(w1)
            c3 = assignment.get(w3)
            if c1 is None or c3 is None:
                continue
            key = (w1, w3)
            if key in seen:
                continue
            seen.add(key)
            examples.append(((w1, w3), (w2,)))
            if len(examples) >= n:
                return examples
    return examples


def make_custom_examples(
    description: str,
    pairs: list[tuple[str, str]],
) -> list[tuple]:
    """Convert user-provided (input_word, output_word) pairs to example format."""
    return [((inp,), (out,)) for inp, out in pairs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Phase E6: Meta-Synthesis (Composition Search over CTKG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--corpus',      default='EarlyModernLatin')
    p.add_argument('--n_train',     type=int,   default=5000)
    p.add_argument('--n_test',      type=int,   default=None)
    p.add_argument('--split',       type=float, default=0.8)
    p.add_argument('--n_clusters',  type=int,   default=12)
    p.add_argument('--demo',        choices=['next_word', 'slot_fill', 'both'],
                   default='both',
                   help='Which demo task to run (default: both)')
    p.add_argument('--n_examples',  type=int, default=20,
                   help='Number of examples to use for synthesis (default: 20)')
    p.add_argument('--max_depth',   type=int, default=3,
                   help='Max composition chain length (default: 3)')
    p.add_argument('--beam_width',  type=int, default=5,
                   help='Beam width for search (default: 5)')
    p.add_argument('--load',        metavar='PATH')
    p.add_argument('--save',        metavar='PATH')
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    _banner('Phase E6: Meta-Synthesis (Composition Search)')
    print(f'  Corpus:      {args.corpus}')
    print(f'  Train lines: {args.n_train}')
    print(f'  Demo:        {args.demo}')
    print(f'  n_examples:  {args.n_examples}')
    print(f'  max_depth:   {args.max_depth}  beam_width: {args.beam_width}')

    # ---- Corpus ----
    d = (args.corpus if os.path.isdir(args.corpus)
         else os.path.join(_DATA_DIR, args.corpus))
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {args.corpus!r}')
        sys.exit(1)

    total_n = args.n_train + (args.n_test or max(args.n_train // 4, 100))
    pairs   = find_pairs(d, max_n=total_n, shuffle=True, seed=args.seed)
    texts   = _stream_texts(pairs)

    n_train     = min(args.n_train, int(len(texts) * args.split))
    train_texts = texts[:n_train]
    print(f'  Lines: {len(train_texts)} train')

    # ---- Discovery + E1 + E4 ----
    print('\n  Running multi-scale discovery...')
    ai, _ = run_pipeline(
        texts=train_texts, n_levels=3, max_merges=500,
        save_path=None, load_path=args.load, verbose=False,
    )

    _banner('Building E1 + E4 concepts')
    assignment, clusters = build_assignment(ai, n_clusters=args.n_clusters)
    print(f'  Vocab: {len(assignment):,}   Clusters: {len(clusters)}')

    train_chain(ai, train_texts, assignment, verbose=False)
    train_slot_occupants(ai, train_texts, assignment, verbose=False)

    all_concepts = get_all_concepts(ai)
    print(f'  Concepts available for search: {all_concepts}')

    # ---- Demo: next-word synthesis ----
    if args.demo in ('next_word', 'both'):
        _banner('E6 Demo 1: Rediscover next-word prediction chain from examples')
        print('  Task: given (w1, w2) pairs → predict w3')
        print('  Goal: find chain that reproduces E1 next-word prediction')
        print()

        nw_examples = make_next_word_examples(
            train_texts, assignment, ai, n=args.n_examples)
        print(f'  Example inputs (first 5):')
        for inp, out in nw_examples[:5]:
            print(f'    {inp!r} → {out!r}')
        print()

        results = beam_search_composition(
            ai, nw_examples,
            max_depth=args.max_depth, beam_width=args.beam_width,
            min_coverage=0.05, verbose=True,
        )

        _banner('E6 Results: Next-word synthesis')
        print(f'  Top chains found:')
        for chain, cov, acc in results[:5]:
            combined = cov * acc
            print(f'    [{cov:.0%} cov × {acc:.0%} acc = {combined:.0%}]  {chain.name}')

        if results:
            best_chain, best_cov, best_acc = results[0]
            print(f'\n  Best: [{best_cov:.0%}×{best_acc:.0%}] {best_chain.name}')
            print(f'  Interpretation: beam search found {len(best_chain.nodes)}-step composition')
            if 'next_cat' in best_chain.name and 'word_given_cat' in best_chain.name:
                print(f'  ✓ Correctly rediscovered E1 category chain!')
            else:
                print(f'  ✗ Did not find E1 chain — may need more examples or deeper search')

    # ---- Demo: slot fill synthesis ----
    if args.demo in ('slot_fill', 'both'):
        _banner('E6 Demo 2: Rediscover slot filling chain from examples')
        print('  Task: given (w1, w3) pairs → predict w2 (fill-in-the-blank)')
        print('  Goal: find chain that reproduces E4 slot_occupants')
        print()

        sf_examples = make_slot_fill_examples(
            train_texts, assignment, n=args.n_examples)
        print(f'  Example inputs (first 5):')
        for inp, out in sf_examples[:5]:
            print(f'    {inp!r} → {out!r}')
        print()

        results_sf = beam_search_composition(
            ai, sf_examples,
            max_depth=args.max_depth, beam_width=args.beam_width,
            min_coverage=0.05, verbose=True,
        )

        _banner('E6 Results: Slot-fill synthesis')
        print(f'  Top chains found:')
        for chain, cov, acc in results_sf[:5]:
            combined = cov * acc
            print(f'    [{cov:.0%} cov × {acc:.0%} acc = {combined:.0%}]  {chain.name}')

        if results_sf:
            best_chain, best_cov, best_acc = results_sf[0]
            print(f'\n  Best: [{best_cov:.0%}×{best_acc:.0%}] {best_chain.name}')
            if 'slot_occupants' in best_chain.name:
                print(f'  ✓ Correctly rediscovered E4 slot chain!')
            else:
                print(f'  Closest match: {best_chain.name}')

    # ---- Summary ----
    _banner('Phase E6: Summary')
    print('  Meta-synthesis: beam search over composition chains')
    print(f'  Search space: {len(all_concepts)} concepts × {args.max_depth} depth levels')
    print('  If beam search rediscovers E1/E4 chains → existing concepts explain new examples.')
    print('  If not → new concept needed; min extension = 1 new ai.teach call.')
    print()
    print('  Implication: any new observed regularity is either:')
    print('    (a) A composition of existing CTKG concepts → zero new parameters')
    print('    (b) Genuinely novel → add one minimal concept to the DAG')
    print('  This is the symbolic analogue of neural implicit composition.')

    if args.save:
        ai.save_checkpoint(args.save)
        print(f'\n  Checkpoint saved: {args.save}')


if __name__ == '__main__':
    main()
