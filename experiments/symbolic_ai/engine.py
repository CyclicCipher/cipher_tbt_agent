"""SymbolicAI engine — the main interface for the symbolic reasoning system.

Ties together interpreter, memory, and synthesis.

The engine distinguishes two kinds of knowledge:

    Built-in knowledge: concepts with pre-defined process expressions
    (loaded from a .ctkg file).  These execute immediately via the
    interpreter.  They represent "instincts" — pre-loaded capability
    that doesn't need training.

    Learned knowledge: concepts without a process expression.  The system
    accumulates (inputs, outputs) examples via teach(), then consolidates
    them into a process rule via consolidate().  Before consolidation,
    queries are answered by exact-match example lookup.  After
    consolidation, queries use the discovered rule.

KL divergence (from memory.py) acts as the consolidation signal:
    high KL → the current rule (or absence of a rule) is surprising
              given the stored examples — consolidation is needed.
    low KL  → the rule explains the examples — consolidated.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from interpreter import ProcessInterpreter
from memory import ExampleStore
from modalities.base import Modality
from synthesis import (CONCEPT_TO_PRIM, Synthesizer,
                       discover_categories, discover_categories_from_dists,
                       chunk_sequences, apply_chunks)


def _inputs_equal(a: tuple, b: tuple) -> bool:
    """Element-wise equality that handles numpy arrays safely."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray) or isinstance(y, _np.ndarray):
                if not (isinstance(x, _np.ndarray) and isinstance(y, _np.ndarray)):
                    return False
                if x.shape != y.shape or not _np.array_equal(x, y):
                    return False
                continue
        except ImportError:
            pass
        if x != y:
            return False
    return True


class SymbolicAI:
    """Symbolic AI backed by a CTKG knowledge graph.

    Usage:
        import sys; sys.path.insert(0, '..')
        from ctkg.parser import parse_file
        graph = parse_file('ctkg/domains/arithmetic.ctkg')

        ai = SymbolicAI(graph)

        # Built-in ops work immediately:
        ai.ask('successor', (4,))          # → (5,)
        ai.ask('comparison', (7, 3))       # → ('GT',)

        # Clear the pre-loaded process to simulate "not yet learned":
        ai.clear_process('single_digit_addition')

        # Teach some examples:
        ai.teach('single_digit_addition', (3, 'ADD', 4), (0, 7))
        ai.teach('single_digit_addition', (8, 'ADD', 5), (1, 3))
        ...

        # Consolidate examples → discovers the process rule:
        rule = ai.consolidate('single_digit_addition')
        # rule = ['result = fold(a, b, succ)', ..., 'emit(c, ones)']

        # Now generalizes perfectly:
        ai.ask('single_digit_addition', (9, 'ADD', 1))  # → (1, 0)
    """

    def __init__(self, graph, modalities: Optional[List[Modality]] = None) -> None:
        self.graph = graph
        self.stores: Dict[str, ExampleStore] = {}
        self._synthesizer = Synthesizer()
        self._interp = ProcessInterpreter(
            engine_ask=self.ask,
            concept_names=frozenset(graph.concepts.keys()),
        )
        # Register any input modalities (vision, audio, etc.)
        for modality in (modalities or []):
            self._interp.register_modality(modality)
        # Phase E: KL history for curiosity() — dict of concept -> [kl_0, kl_1, ...]
        self._kl_history: Dict[str, List[float]] = {}
        # Phase L: pre-built frequency tables for distributional concepts.
        # Populated by freq_consolidate(). Used by ask() as priority-3 fallback.
        # Format: concept_name -> {inputs_tuple -> {outputs_tuple -> probability}}
        self._dist_tables: Dict[str, Dict[tuple, Dict[tuple, float]]] = {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def teach(
        self,
        concept_name: str,
        inputs: tuple,
        outputs: tuple,
    ) -> None:
        """Record one (inputs, outputs) training example for a concept."""
        if concept_name not in self.stores:
            self.stores[concept_name] = ExampleStore(concept_name)
        self.stores[concept_name].add(tuple(inputs), tuple(outputs))

    def ask(
        self,
        concept_name: str,
        inputs: tuple,
    ) -> Optional[tuple]:
        """Answer a query about a concept given inputs.

        Priority:
            1. If the concept has a process expression, execute it.
            2. If there are stored examples, do exact-match lookup.
            3. If a frequency table exists (freq_consolidate was called),
               return the mode of the empirical output distribution.
            4. If examples exist but no freq table, compute freq on the fly.
            5. Return None (cannot answer).

        For distributional concepts (language, music, action selection) the
        answer at priority 3/4 is the most likely output given experience —
        equivalent to the Markov model's argmax prediction.  Use ask_dist()
        to retrieve the full probability distribution instead of the mode.
        """
        concept = self.graph.concepts.get(concept_name)
        if concept is None:
            return None

        # 1. Execute process expression if one exists (built-in or consolidated).
        if concept.process:
            try:
                return self._interp.run(
                    concept.process, tuple(inputs), concept.input_type
                )
            except Exception:
                # Process failed (e.g. wrong input type) — fall through.
                pass

        store = self.stores.get(concept_name)
        query = tuple(inputs)

        # 2. Exact-match lookup in example store.
        if store is not None:
            for stored_inputs, stored_outputs in store.examples:
                if _inputs_equal(stored_inputs, query):
                    return stored_outputs

        # 3. Pre-built freq table (freq_consolidate was called — fast path).
        dist_table = self._dist_tables.get(concept_name)
        if dist_table is not None:
            dist = dist_table.get(query)
            if dist:
                return max(dist, key=dist.get)

        # 4. On-the-fly frequency prediction from ExampleStore (slower).
        if store is not None:
            return store.freq_predict(query)

        return None

    def ask_dist(
        self,
        concept_name: str,
        inputs: tuple,
    ) -> Optional[Dict[tuple, float]]:
        """Return the full probability distribution over outputs.

        For deterministic concepts (process or exact-match), wraps the single
        output as a Dirac delta distribution {output: 1.0}.

        For distributional concepts (freq table or ExampleStore), returns the
        empirical conditional distribution {output: probability}.

        Returns None if the concept cannot answer at all.

        Use this instead of ask() when you need:
          - Log-likelihood scoring (language model evaluation)
          - Sampling (generation / exploration)
          - Entropy measurement (curiosity about uncertain predictions)
        """
        concept = self.graph.concepts.get(concept_name)
        if concept is None:
            return None

        query = tuple(inputs)

        # Try deterministic sources first — wrap as Dirac delta.
        if concept.process:
            try:
                result = self._interp.run(
                    concept.process, query, concept.input_type
                )
                if result is not None:
                    return {result: 1.0}
            except Exception:
                pass

        store = self.stores.get(concept_name)

        # Exact-match -> Dirac delta.
        if store is not None:
            for stored_inputs, stored_outputs in store.examples:
                if _inputs_equal(stored_inputs, query):
                    return {stored_outputs: 1.0}

        # Pre-built freq table — full distribution.
        dist_table = self._dist_tables.get(concept_name)
        if dist_table is not None:
            dist = dist_table.get(query)
            if dist:
                return dict(dist)

        # On-the-fly freq distribution from ExampleStore.
        if store is not None:
            return store.freq_dist(query)

        return None

    def consolidate(self, concept_name: str) -> Optional[List[str]]:
        """Discover a process rule from stored examples (consolidation).

        Runs the synthesizer to find the shortest process consistent with
        all stored examples.  On success, the discovered process is stored
        in the concept so that future ask() calls use it.

        Returns the process lines on success, None on failure.

        Failure means: insufficient examples, missing prerequisites, or
        ambiguous examples that don't uniquely determine a rule.
        """
        store = self.stores.get(concept_name)
        if store is None or len(store) == 0:
            return None

        process = self._synthesizer.synthesize(
            concept_name=concept_name,
            store=store,
            graph=self.graph,
            interpreter=self._interp,
            engine_ask=self.ask,
        )

        if process is not None:
            concept = self.graph.concepts[concept_name]
            concept.process = process
            # Register with the dynamic template library so future synthesis
            # attempts on other concepts can re-use this proven pattern.
            available_ops = {
                prim
                for anc in self.graph.ancestors(concept_name)
                if (prim := CONCEPT_TO_PRIM.get(anc)) is not None
            }
            self._synthesizer.register_success(
                concept_name, process, concept.input_type, available_ops
            )

        return process

    def consolidate_approx(
        self,
        concept_name: str,
        accuracy_threshold: float = 0.85,
        subsample: Optional[int] = 300,
        verbose: bool = False,
    ) -> Optional[tuple]:
        """Approximate consolidation for statistical / visual concepts.

        Like consolidate(), but accepts any template achieving >= accuracy_threshold
        on stored examples.  Used when the concept's output is statistical
        (e.g. cat_detection, image_brightness).

        Returns (process_lines, accuracy) on success, None on failure.
        On success, the concept's process is updated so future ask() calls
        use the discovered rule.
        """
        store = self.stores.get(concept_name)
        if store is None or len(store) == 0:
            return None

        result = self._synthesizer.synthesize_approx(
            concept_name=concept_name,
            store=store,
            graph=self.graph,
            interpreter=self._interp,
            engine_ask=self.ask,
            accuracy_threshold=accuracy_threshold,
            subsample=subsample,
            verbose=verbose,
        )

        if result is not None:
            process, accuracy = result
            self.graph.concepts[concept_name].process = process
            # Register in learned template library
            available_ops = {
                prim
                for anc in self.graph.ancestors(concept_name)
                if (prim := CONCEPT_TO_PRIM.get(anc)) is not None
            }
            self._synthesizer.register_success(
                concept_name, process,
                self.graph.concepts[concept_name].input_type,
                available_ops,
            )

        return result

    def freq_consolidate(
        self,
        concept_name: str,
    ) -> Dict[tuple, Dict[tuple, float]]:
        """Consolidate a statistical concept into an empirical frequency table.

        Unlike consolidate() (which searches for a deterministic process),
        freq_consolidate() always succeeds.  It builds a snapshot of the
        empirical conditional distribution from the stored examples and
        caches it in self._dist_tables[concept_name].

        After freq_consolidate():
          - ask() returns the mode (argmax) prediction for any seen input
          - ask_dist() returns the full probability distribution
          - kl() reports the residual entropy (mean cross-entropy between
            the empirical distribution and itself = 0 for seen contexts)

        Returns the frequency table dict, or {} if no examples exist.

        The residual entropy reported by kl() after freq_consolidate is the
        genuine task difficulty — the irreducible stochasticity of the concept.
        It is NOT a synthesis failure; it is the information-theoretic lower
        bound on prediction error for this concept.

        Example use case:
            The 'next_word' concept cannot be synthesised (same bigram ->
            many next words).  After freq_consolidate(), ask() returns the
            most frequent next word for each seen bigram context.  For unseen
            contexts, ask() still returns None (generalisation requires the
            hierarchy — see Phase M).
        """
        store = self.stores.get(concept_name)
        if store is None or len(store) == 0:
            return {}

        table = store.build_full_freq_table()
        self._dist_tables[concept_name] = table
        return table

    def induce_hierarchy(
        self,
        flat_concept:  str,
        n_clusters:    int           = 9,
        min_examples:  int           = 1,
        vocab_size:    Optional[int] = None,
        method:        str           = 'auto',
        domain:        str           = 'discovered',
    ) -> Dict:
        """Discover latent categories from flat sequential prediction examples.

        Given a flat next-word concept whose ExampleStore contains
        (word1, word2) -> (next_word,) examples, discovers POS-like latent
        categories -- without being told what categories exist.

        Algorithm (distributional hypothesis, Firth 1957):
            1. Extract unigram forward distributions:
               From (w1, w2) -> (w3,) examples, derive (w2,) -> (w3,) pairs.
               This gives P(next | word) -- the signal for POS-like clustering.
            2. Cluster words by JS-divergence / cosine distance.
               Words followed by similar words cluster together:
                 DET cluster -> all precede NOUN/ADJ
                 NOUN cluster -> all precede VERB/PREP
            3. Add discovered category concepts to the CTKG.
            4. Return cluster membership, compression metrics, and KL.

        Cross-domain: replace 'word' with 'action', 'note', or 'state' and
        the same algorithm discovers Minecraft action types, musical harmonic
        roles, or motor movement primitives from sequential experience alone.

        Args:
            flat_concept:  Name of the flat sequential concept to analyse.
                           Its ExampleStore must have multi-word context inputs.
            n_clusters:    Target number of latent categories.
            min_examples:  Minimum word frequency to include in clustering.
            vocab_size:    Cap the output vocabulary to top-N words before
                           clustering.  Recommended for large corpora
                           (e.g. vocab_size=2000 for WikiText-2).
                           None = use all observed output words.
            method:        Clustering algorithm ('auto', 'agglomerative',
                           'kmeans').  'auto' uses agglomerative for n<=200
                           words and k-means (numpy) for larger vocabularies.
            domain:        Domain label for newly created CTKG concepts.

        Returns dict with keys:
            'clusters':      {cluster_id: [word_list]} -- discovered groupings
            'assignment':    {word: cluster_id} -- mapping for every eligible word
            'n_eligible':    Number of unique words included in clustering
            'n_clusters':    Actual clusters formed (<= n_clusters requested)
            'concept_names': List of CTKG concept names added ('cat_C0', ...)
            'kl_flat':       KL (bits/step) of flat model before hierarchy
        """
        store = self.stores.get(flat_concept)
        if store is None or len(store) == 0:
            return {'error': f'No examples for concept {flat_concept!r}'}

        # Extract unigram forward distribution:
        # From each (w1, w2) -> (w3,) example, yield (w2,) -> (w3,).
        # This captures "given word w2 appears at position t, what follows?"
        # which is exactly the distributional signal for POS-like clustering.
        unigram_store = ExampleStore(f'{flat_concept}__unigram')
        for inputs, outputs in store.examples:
            if len(inputs) >= 1 and len(outputs) >= 1:
                context_word = (inputs[-1],)   # Last input token
                unigram_store.add(context_word, outputs)

        # Cluster words by forward-distribution similarity.
        raw_assignment: Dict[tuple, int] = discover_categories(
            unigram_store,
            n_clusters   = n_clusters,
            min_examples = min_examples,
            vocab_size   = vocab_size,
            method       = method,
        )
        if not raw_assignment:
            return {'error': 'Not enough examples for clustering'}

        # Invert: cluster_id → [word_list].
        import collections as _col
        clusters_raw: Dict[int, List[str]] = {}
        for inp_tuple, cid in raw_assignment.items():
            word = inp_tuple[0]
            clusters_raw.setdefault(cid, []).append(word)

        # Renumber clusters by size (largest = C0) for readable output.
        by_size = sorted(clusters_raw.items(), key=lambda kv: -len(kv[1]))
        renumber: Dict[int, int] = {old: new for new, (old, _) in enumerate(by_size)}
        clusters: Dict[int, List[str]] = {
            renumber[old]: sorted(members)
            for old, members in clusters_raw.items()
        }
        assignment: Dict[str, int] = {
            inp_tuple[0]: renumber[raw_cid]
            for inp_tuple, raw_cid in raw_assignment.items()
        }

        # Add one CTKG concept per discovered category.
        concept_names: List[str] = []
        for cid in sorted(clusters.keys()):
            cname = f'cat_C{cid}'
            if cname not in self.graph.concepts:
                sample = clusters[cid][:4]
                desc = f'Discovered category {cid}: {", ".join(sample)}'
                if len(clusters[cid]) > 4:
                    desc += '...'
                self.add_concept(
                    name=cname,
                    domain=domain,
                    description=desc,
                    input_type=['word'],
                    output_type=['category_id'],
                    tier='theorem',
                )
            concept_names.append(cname)

        return {
            'clusters':      clusters,
            'assignment':    assignment,
            'n_eligible':    len(raw_assignment),
            'n_clusters':    len(clusters),
            'concept_names': concept_names,
            'kl_flat':       self.kl(flat_concept),
        }

    def induce_hierarchy_bidir(
        self,
        fwd_concept:  str,
        bwd_concept:  str,
        n_clusters:   int           = 9,
        min_examples: int           = 1,
        vocab_size:   Optional[int] = None,
        method:       str           = 'auto',
    ) -> Dict:
        """Discover latent categories using bidirectional context.

        Combines forward distributions (from ``fwd_concept``) and backward
        distributions (from ``bwd_concept``) before clustering.  Atoms with
        similar successors AND similar predecessors cluster together, producing
        richer categories than forward-only clustering.

        At each scale of the multi-scale discovery pipeline::

            Level 0: fwd='next_char',      bwd='prev_char'
            Level 1: fwd='next_morph',     bwd='prev_morph'
            Level 2: fwd='next_word_hier', bwd='prev_word_hier'
            Level 3: fwd='next_phrase',    bwd='prev_phrase'

        The forward distribution captures *what an atom predicts*.
        The backward distribution captures *what predicts an atom*.
        Together they identify the atom's full distributional signature.

        Args:
            fwd_concept:  Concept name whose ExampleStore holds forward bigrams
                          ``(atom,) → (next_atom,)``.
            bwd_concept:  Concept name whose ExampleStore holds backward bigrams
                          ``(atom,) → (prev_atom,)``.
            n_clusters:   Target number of latent categories.
            min_examples: Minimum total observation count to include an atom.
            vocab_size:   Cap vocabulary to top-N atoms before clustering.
            method:       Clustering algorithm ('auto', 'agglomerative', 'kmeans').

        Returns dict with keys:
            'clusters':   {cluster_id: [atom_list]} sorted by frequency.
            'assignment': {atom: cluster_id}
            'n_eligible': Number of unique atoms included.
            'n_clusters': Actual number of clusters formed.
        """
        import collections as _col

        fwd_store = self.stores.get(fwd_concept)
        bwd_store = self.stores.get(bwd_concept)
        have_fwd  = fwd_store is not None and len(fwd_store) > 0
        have_bwd  = bwd_store is not None and len(bwd_store) > 0
        if not have_fwd and not have_bwd:
            return {'error': f'No examples for {fwd_concept!r} or {bwd_concept!r}'}

        # Extract raw counts from each store.
        fwd_raw: Dict[str, Dict[str, int]] = _col.defaultdict(_col.Counter)
        if have_fwd:
            for inputs, outputs in fwd_store.examples:
                if len(inputs) >= 1 and len(outputs) >= 1:
                    fwd_raw[inputs[-1]][outputs[0]] += 1

        bwd_raw: Dict[str, Dict[str, int]] = _col.defaultdict(_col.Counter)
        if have_bwd:
            for inputs, outputs in bwd_store.examples:
                if len(inputs) >= 1 and len(outputs) >= 1:
                    bwd_raw[inputs[-1]][outputs[0]] += 1

        # Build joint distributions with namespaced context keys.
        # Forward context: ('fwd', next_atom) — what X predicts.
        # Backward context: ('bwd', prev_atom) — what predicts X.
        # Namespacing prevents forward and backward contexts from colliding.
        all_atoms = set(fwd_raw) | set(bwd_raw)
        combined_dists:  Dict[tuple, Dict[tuple, float]] = {}
        combined_counts: Dict[tuple, int] = {}

        for atom in all_atoms:
            d: Dict[tuple, float] = {}
            fwd_total = sum(fwd_raw[atom].values()) if fwd_raw[atom] else 0
            bwd_total = sum(bwd_raw[atom].values()) if bwd_raw[atom] else 0
            for nxt, cnt in fwd_raw[atom].items():
                d[('fwd', nxt)] = cnt / fwd_total
            for prv, cnt in bwd_raw[atom].items():
                d[('bwd', prv)] = cnt / bwd_total
            if not d:
                continue
            combined_dists[( atom,)] = d
            combined_counts[(atom,)] = fwd_total + bwd_total

        raw_assignment = discover_categories_from_dists(
            combined_dists, combined_counts,
            n_clusters=n_clusters, min_examples=min_examples,
            vocab_size=vocab_size, method=method,
        )
        if not raw_assignment:
            return {'error': 'Not enough examples for bidirectional clustering'}

        # Build cluster → member list, sort each cluster by total frequency.
        clusters_raw: Dict[int, List[str]] = {}
        for (atom,), cid in raw_assignment.items():
            clusters_raw.setdefault(cid, []).append(atom)

        freq = {atom: combined_counts.get((atom,), 0) for atom in all_atoms}
        for cid in clusters_raw:
            clusters_raw[cid].sort(key=lambda a: -freq.get(a, 0))

        # Renumber clusters by size (largest = C0) for consistent display.
        by_size = sorted(clusters_raw.items(), key=lambda kv: -len(kv[1]))
        renumber: Dict[int, int] = {old: new for new, (old, _) in enumerate(by_size)}
        clusters = {renumber[old]: members for old, members in clusters_raw.items()}
        assignment = {
            atom: renumber[raw_cid]
            for (atom,), raw_cid in raw_assignment.items()
        }

        return {
            'clusters':   clusters,
            'assignment': assignment,
            'n_eligible': len(raw_assignment),
            'n_clusters': len(clusters),
        }

    def chunk_store(
        self,
        concept_name: str,
        min_pmi:      float = 3.0,
        min_count:    int   = 5,
        max_merges:   int   = 500,
        separator:    str   = '',
    ) -> List[Tuple]:
        """Discover high-PMI adjacent atom pairs in a concept's ExampleStore.

        Reads the (input, output) bigram pairs from ``concept_name``'s
        ExampleStore, computes Pointwise Mutual Information for each pair,
        and returns all pairs exceeding ``min_pmi`` as merge rules for the
        next level of the scale hierarchy.

        This is the bridge between scales: after teaching ``next_char`` with
        character bigrams and calling ``induce_hierarchy``, call
        ``chunk_store('next_char')`` to discover which character pairs should
        be compressed into morpheme-level atoms.

        Args:
            concept_name:  Name of a bigram concept (next_char, next_word, …).
                           Its ExampleStore must store single-element inputs and
                           outputs: teach('next_X', (a,), (b,)).
            min_pmi:       Minimum PMI threshold in bits.  3.0 for char level,
                           2.0 for word level.
            min_count:     Minimum bigram occurrence count.  Filters noise.
            max_merges:    Maximum merge rules to return (highest PMI first).
            separator:     String between merged atoms.  '' for char→morph
                           (concatenation), ' ' for word→phrase.

        Returns:
            List of (atom_a, atom_b, compound, pmi) sorted by PMI descending.
            Build a chunk_map with::

                chunk_map = {(a, b): c for a, b, c, _ in ai.chunk_store('next_char')}
                morphemes = apply_chunks(char_sequence, chunk_map)

            Returns [] if the concept store is empty or missing.
        """
        store = self.stores.get(concept_name)
        if store is None or len(store) == 0:
            return []

        bigrams: Dict[Tuple, int] = {}
        for inputs, outputs in store.examples:
            if len(inputs) == 1 and len(outputs) == 1:
                key = (inputs[0], outputs[0])
                bigrams[key] = bigrams.get(key, 0) + 1

        return chunk_sequences(
            bigrams,
            min_pmi=min_pmi,
            min_count=min_count,
            max_merges=max_merges,
            separator=separator,
        )

    # ------------------------------------------------------------------
    # KL divergence / consolidation trigger
    # ------------------------------------------------------------------

    def kl(self, concept_name: str) -> float:
        """KL divergence of stored examples from the current model (in bits).

        For deterministic concepts:
            KL = -log2(accuracy) using exact-match / process prediction.
            KL ≈ 0   = rule perfectly explains all examples (consolidated)
            KL -> inf = no rule, or rule is wrong

        For distributional concepts (after freq_consolidate):
            KL = mean negative log-likelihood under the empirical freq table.
            This equals the empirical entropy H(output | input) — the residual
            stochasticity of the concept.  It is NOT a synthesis failure.

        The transition from deterministic to distributional KL happens
        automatically when freq_consolidate() has been called.
        """
        store = self.stores.get(concept_name)
        if store is None or len(store) == 0:
            return float('inf')

        # Distributional path: use mean negative log-likelihood.
        if concept_name in self._dist_tables:
            dist_table = self._dist_tables[concept_name]
            def dist_fn(inp):
                return dist_table.get(inp)
            ll = store.mean_log_likelihood(dist_fn)
            # ll is negative (bits); convert to positive KL-like measure.
            # For a perfect distributional model, ll = -H (empirical entropy).
            # We report H (entropy) as the KL for distributional concepts.
            return -ll   # always non-negative; equals empirical entropy after consolidation

        # Deterministic path: use accuracy-based KL.
        return store.kl_divergence(lambda inp: self.ask(concept_name, inp))

    def should_consolidate(
        self,
        concept_name: str,
        threshold: float = 0.1,
    ) -> bool:
        """True if KL exceeds threshold — consolidation is recommended."""
        return self.kl(concept_name) > threshold

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_process(self, concept_name: str) -> None:
        """Clear a concept's process expression (makes it 'not yet learned').

        Used in experiments to simulate learning from scratch even when
        the .ctkg file contains a pre-defined process.
        """
        concept = self.graph.concepts.get(concept_name)
        if concept is not None:
            concept.process = []

    def reset_concept(self, concept_name: str) -> None:
        """Clear both examples and process for a concept.

        Used in minimum-examples sweep experiments where the concept
        is repeatedly tested with different training set sizes.
        """
        self.stores.pop(concept_name, None)
        concept = self.graph.concepts.get(concept_name)
        if concept is not None:
            concept.process = []

    def example_count(self, concept_name: str) -> int:
        store = self.stores.get(concept_name)
        return len(store) if store else 0

    def has_process(self, concept_name: str) -> bool:
        concept = self.graph.concepts.get(concept_name)
        return bool(concept and concept.process)

    # ==================================================================
    # Phase C: Dynamic CTKG extension
    # ==================================================================

    def add_concept(
        self,
        name: str,
        domain: str = 'minecraft',
        description: str = '',
        input_type: Optional[List[str]] = None,
        output_type: Optional[List[str]] = None,
        process: Optional[List[str]] = None,
        tier: str = 'theorem',
    ) -> None:
        """Add a new concept to the CTKG at runtime.

        Used when the agent discovers a new concept from visual experience
        (e.g. 'oak_log' emerges from visual clustering) and needs to add
        it to the graph before accumulating examples and synthesizing a rule.

        The interpreter's concept_names set is updated so that the new name
        can be used as a concept reference in lookup() process expressions.
        """
        from ctkg.graph import Concept
        concept = Concept(
            name=name,
            domain=domain,
            description=description,
            input_type=input_type or [],
            output_type=output_type or [],
            process=process or [],
            status='planned',
            tier=tier,
        )
        self.graph.add_concept(concept)
        # Keep interpreter in sync with the graph's concept set.
        self._interp.concept_names = frozenset(self.graph.concepts.keys())

    def add_prerequisite(
        self,
        source: str,
        target: str,
        role: str = '',
        transfer_probability: float = 1.0,
    ) -> None:
        """Add a prerequisite edge (epistemic ordering) at runtime."""
        from ctkg.graph import Prerequisite
        self.graph.add_prerequisite(Prerequisite(
            source=source,
            target=target,
            role=role,
            transfer_probability=transfer_probability,
        ))

    def add_causal_edge(
        self,
        source: str,
        target: str,
        role: str = '',
        guard: str = '',
        delay_steps: int = 0,
        probability: float = 1.0,
    ) -> None:
        """Add a causal edge (physical causation) at runtime."""
        from ctkg.graph import CausalEdge
        self.graph.add_causal_edge(CausalEdge(
            source=source,
            target=target,
            role=role,
            guard=guard,
            delay_steps=delay_steps,
            probability=probability,
        ))

    def add_instance_edge(
        self,
        source: str,
        target: str,
        role: str = '',
    ) -> None:
        """Add an instance_of edge (subtype hierarchy) at runtime."""
        from ctkg.graph import InstanceEdge
        self.graph.add_instance_edge(InstanceEdge(
            source=source, target=target, role=role
        ))

    def add_composition_edge(
        self,
        source: str,
        target: str,
        role: str = '',
        probability: float = 1.0,
    ) -> None:
        """Add a composes_into edge (multi-input product morphism) at runtime."""
        from ctkg.graph import CompositionEdge
        self.graph.add_composition_edge(CompositionEdge(
            source=source, target=target, role=role, probability=probability
        ))

    def add_temporal_edge(
        self,
        source: str,
        target: str,
        role: str = '',
    ) -> None:
        """Add a precedes edge (temporal ordering) at runtime."""
        from ctkg.graph import TemporalEdge
        self.graph.add_temporal_edge(TemporalEdge(
            source=source, target=target, role=role
        ))

    # ==================================================================
    # Phase E: Online synthesis
    # ==================================================================

    def observe(
        self,
        concept_name: str,
        inputs: tuple,
        outputs: tuple,
        kl_threshold: float = 0.1,
    ) -> Optional[List[str]]:
        """Streaming teach() + auto-consolidation.

        Records the (inputs, outputs) pair, appends the current KL to the
        history buffer, then attempts consolidation if KL > kl_threshold.

        Returns the discovered process on successful consolidation, else None.

        Design: the agent calls observe() continuously as it plays.  The
        KL history drives curiosity() for the exploration policy.
        """
        self.teach(concept_name, inputs, outputs)
        kl = self.kl(concept_name)
        self._kl_history.setdefault(concept_name, []).append(kl)
        if self.should_consolidate(concept_name, kl_threshold):
            return self.consolidate(concept_name)
        return None

    def curiosity(self, concept_name: str, window: int = 10) -> float:
        """KL decrease rate (bits/step) over the last `window` observations.

        High curiosity (positive rate) = KL is dropping fast = actively learning.
        Zero curiosity  = already consolidated or examples too noisy.
        Negative rate   = KL increasing (conflicting examples — investigate).

        Used by highest_kl_rate_concept() to direct exploration.
        """
        history = self._kl_history.get(concept_name, [])
        if len(history) < 2:
            return 0.0
        recent = history[-window:]
        if len(recent) < 2:
            return 0.0
        # Positive = decreasing KL = learning; clamp to 0 if increasing
        rate = (recent[0] - recent[-1]) / len(recent)
        return max(0.0, rate)

    def offline_consolidation(self) -> Dict[str, Optional[List[str]]]:
        """Consolidate all high-KL concepts (sleep-coupled replay).

        Called by MinecraftModality.on_sleep() when the agent goes to bed.
        Iterates over all stores with KL > 0.1 and attempts consolidation.

        Returns a dict of concept_name → discovered process (or None).
        """
        results: Dict[str, Optional[List[str]]] = {}
        for name in list(self.stores.keys()):
            if self.should_consolidate(name):
                process = self.consolidate(name)
                if process is not None:
                    results[name] = process
        return results

    # ==================================================================
    # Phase G: Exploration policy
    # ==================================================================

    def next_frontier_concept(self) -> Optional[str]:
        """Next unlearned concept whose prerequisites are all learned.

        Scans the topological sort of the CTKG.  Returns the first concept
        that (a) has no process yet and (b) all its prerequisite concepts
        have processes.  This is the next achievement to attempt.

        Returns None if all concepts are learned or no concept is unblocked.
        """
        try:
            order = self.graph.topological_sort()
        except Exception:
            return None

        for name in order:
            concept = self.graph.concepts.get(name)
            if concept is None or concept.process:
                continue  # Already learned
            # Check all prerequisites have processes
            prereq_names = self.graph._parents.get(name, set())
            if all(
                self.graph.concepts[p].process
                for p in prereq_names
                if p in self.graph.concepts
            ):
                return name

        return None

    def highest_kl_rate_concept(self) -> Optional[str]:
        """Concept with the highest curiosity (fastest KL decrease rate).

        Used to direct exploration toward concepts that are actively
        being learned — maximum compression progress (Schmidhuber 2010).

        Returns None if no concept has a non-zero curiosity score.
        """
        best_name: Optional[str] = None
        best_score: float = 0.0
        for name in self.stores:
            score = self.curiosity(name)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def priority(self, modality=None) -> tuple:
        """Combined exploration priority (Mode, urgency, target).

        If a modality is supplied and has current_priority(), defers to it
        (survival drives take precedence over epistemic drives).
        Otherwise falls back to pure epistemic priority:
            ACHIEVE > EXPLORE > WANDER
        """
        if modality is not None and hasattr(modality, 'current_priority'):
            mode, urgency, target = modality.current_priority(engine=self)
            if mode in ('SURVIVE', 'EAT', 'SLEEP'):
                return (mode, urgency, target)

        frontier = self.next_frontier_concept()
        curious  = self.highest_kl_rate_concept()

        if frontier is not None:
            return ('ACHIEVE', 0.40, frontier)
        if curious is not None:
            return ('EXPLORE', 0.30, curious)
        return ('WANDER', 0.10, 'random_direction')

    # ==================================================================
    # Phase H: Checkpoint system
    # ==================================================================

    def save_checkpoint(self, path: str) -> None:
        """Save the full engine state to a JSON checkpoint.

        The "model" is: CTKG process expressions + example stores +
        learned template library.  No gradient weights — the checkpoint
        is human-readable and version-tagged.

        Format:
            {version, timestamp, ctkg: {...}, stores: {...},
             learned_templates: [...]}

        Named checkpoint conventions (caller's responsibility):
            checkpoint_achieve_{name}.json  — permanent, after achievement
            checkpoint_step_{N}.json        — rotating, every N steps
        """
        data = {
            'version': '1.0',
            'timestamp': time.time(),
            'ctkg': self._serialize_graph(),
            'stores': self._serialize_stores(),
            'learned_templates': self._serialize_templates(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_checkpoint(self, path: str) -> None:
        """Load engine state from a JSON checkpoint.

        Restores concept processes (synthesized rules), example stores
        (raw episode memory), and the learned template library.

        Migration strategy on schema mismatch: processes that reference
        still-valid primitive names are restored; invalid processes are
        dropped (the agent re-learns them from remaining examples).
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        version = data.get('version', '0')
        if version != '1.0':
            raise ValueError(
                f"Checkpoint version {version!r} not supported "
                f"(expected '1.0')"
            )

        self._deserialize_graph(data.get('ctkg', {}))
        self._deserialize_stores(data.get('stores', {}))
        self._deserialize_templates(data.get('learned_templates', []))

    # ------------------------------------------------------------------
    # Checkpoint serialization helpers
    # ------------------------------------------------------------------

    def _serialize_graph(self) -> dict:
        """Serialise concept processes and all Phase B edge types."""
        concepts = {}
        for name, c in self.graph.concepts.items():
            concepts[name] = {
                'process': c.process,
                'domain':  c.domain,
                'description': c.description,
                'input_type':  c.input_type,
                'output_type': c.output_type,
                'tier': getattr(c, 'tier', 'theorem'),
            }

        def _edge_list(edges, fields):
            return [
                {f: getattr(e, f) for f in fields}
                for e in edges
            ]

        return {
            'concepts': concepts,
            'prerequisites': _edge_list(
                self.graph.prerequisites,
                ('source', 'target', 'role', 'transfer_probability'),
            ),
            'causal_edges': _edge_list(
                self.graph.causal_edges,
                ('source', 'target', 'role', 'guard', 'delay_steps', 'probability'),
            ),
            'composition_edges': _edge_list(
                self.graph.composition_edges,
                ('source', 'target', 'role', 'probability'),
            ),
            'instance_edges': _edge_list(
                self.graph.instance_edges,
                ('source', 'target', 'role'),
            ),
            'temporal_edges': _edge_list(
                self.graph.temporal_edges,
                ('source', 'target', 'role'),
            ),
        }

    def _deserialize_graph(self, ctkg: dict) -> None:
        """Restore concept processes from checkpoint."""
        for name, info in ctkg.get('concepts', {}).items():
            if name in self.graph.concepts:
                self.graph.concepts[name].process = info.get('process', [])
            else:
                # Dynamically add concepts that weren't in the original graph.
                self.add_concept(
                    name=name,
                    domain=info.get('domain', 'unknown'),
                    description=info.get('description', ''),
                    input_type=info.get('input_type', []),
                    output_type=info.get('output_type', []),
                    process=info.get('process', []),
                    tier=info.get('tier', 'theorem'),
                )

    @staticmethod
    def _to_json_safe(obj):
        """Recursively convert numpy arrays (and other non-JSON types) to lists.

        Handles: ndarray → nested list, tuple → list, bool/int/float/str pass-through.
        Visual examples (frame arrays) are stored as nested Python lists in the
        checkpoint; they will remain as lists on deserialization (correct for
        symbolic knowledge checkpointing — pixel-level replay is not needed).
        """
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, _np.generic):      # numpy scalars
                return obj.item()
        except ImportError:
            pass
        if isinstance(obj, (list, tuple)):
            return [SymbolicAI._to_json_safe(v) for v in obj]
        return obj

    def _serialize_stores(self) -> dict:
        """Serialise all ExampleStore instances.

        Tuples become JSON arrays.  Numpy arrays (e.g. image frames) are
        recursively converted to nested Python lists so the checkpoint file
        is plain JSON.
        """
        result = {}
        for name, store in self.stores.items():
            result[name] = [
                [
                    SymbolicAI._to_json_safe(list(inp)),
                    SymbolicAI._to_json_safe(list(out)),
                ]
                for inp, out in store.examples
            ]
        return result

    def _deserialize_stores(self, stores: dict) -> None:
        """Restore ExampleStore instances from checkpoint."""
        for name, examples in stores.items():
            self.stores[name] = ExampleStore(name)
            for inp_list, out_list in examples:
                self.stores[name].add(tuple(inp_list), tuple(out_list))

    def _serialize_templates(self) -> list:
        """Serialise the learned template library."""
        result = []
        for tmpl in self._synthesizer._learned:
            result.append({
                'process_lines':  tmpl.process_lines,
                'n_digit_inputs': tmpl.n_digit_inputs,
                'required_ops':   sorted(tmpl.required_ops),
                'success_count':  tmpl.success_count,
                'source_concept': tmpl.source_concept,
            })
        return result

    def _deserialize_templates(self, templates: list) -> None:
        """Restore learned templates into the synthesizer."""
        from synthesis import _LearnedTemplate
        self._synthesizer._learned.clear()
        for t in templates:
            self._synthesizer._learned.append(_LearnedTemplate(
                process_lines  = t['process_lines'],
                n_digit_inputs = t['n_digit_inputs'],
                required_ops   = frozenset(t['required_ops']),
                success_count  = t['success_count'],
                source_concept = t['source_concept'],
            ))
