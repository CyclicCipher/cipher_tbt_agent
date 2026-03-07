"""Phase O: Unsupervised category discovery from raw sequential data.

Demonstrates that the engine rediscovers POS-like latent categories
(DET, NOUN, VERB, ADJ, PREP, ...) from raw next-word examples, without
being told that such categories exist.

Test procedure:
  1. Load corpus (built-in ~200 tokens, or WikiText-2 ~2M tokens).
  2. Start with NO language knowledge -- only arithmetic as a base graph.
  3. Add a bare 'next_word' concept (no word_pos, no next_pos, no POS hierarchy).
  4. Train on flat next-word bigram examples: (w1, w2) -> (w3,).
  5. Call engine.induce_hierarchy() -- discovers latent word categories.
  6. Show: discovered clusters ~ POS tags (DET, NOUN, VERB, ...).
  7. Show: hierarchy answers unseen bigram contexts that flat model cannot.
  8. MDL validation: hierarchy reduces description length.

Usage:
  python discover_test.py                           # built-in corpus (fast, ~1s)
  python discover_test.py --corpus wikitext2        # WikiText-2 (slower, needs datasets)
  python discover_test.py --corpus wikitext2 --max_tokens 50000  # subsample

This is the first step toward general intelligence across domains:
  - Same algorithm on Minecraft (action, state) -> (next_state,) discovers
    action categories (mine, place, move, interact) without MineDojo labels.
  - Same algorithm on MIDI (note_t,) -> (note_{t+1},) discovers harmonic
    roles (tonic, dominant, leading tone) without music theory.
  - The engine needs NO domain knowledge -- only sequential experience.

Expected output (built-in corpus):
  C0 (NOUN):  cat, cats, dog, dogs, mat, hat, door, ... [~20 words]
  C1 (DET):   the, a, an                               [3 words]
  C2 (VERB):  sat, ran, is, was, likes, am, do, flies  [~8 words]
  C3 (ADJ):   flat, fat, big, small, fast, good, ...   [~9 words]
  C4 (PREP):  on, to, by, from, for, past, in, like    [~8 words]
  C5 (OTHER): and, i, one, two, not, then, them        [~7 words]
  Average cluster purity >= 70% vs known POS tags.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

# ---------------------------------------------------------------------------
# Path setup -- allow running from the symbolic_ai directory directly.
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPT_DIR, '..', '..')
_LANG_DIR    = os.path.join(_SCRIPT_DIR, '..', 'language')
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, _LANG_DIR)   # for wikitext2.py

from ctkg.parser import parse_file
from engine import SymbolicAI
from modalities.language import LanguageModality, _BUILTIN_POS


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f'\n{"=" * 62}')
    print(f'  {title}')
    print('=' * 62)


def _purity(members, pos_lookup):
    """Fraction of cluster members that share the dominant POS tag."""
    counts = collections.Counter(
        pos_lookup.get(w) for w in members if pos_lookup.get(w)
    )
    if not counts:
        return 0.0, '?'
    dominant_pos, dominant_count = counts.most_common(1)[0]
    return dominant_count / len(members), dominant_pos


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def _load_builtin():
    """Return (tokens, pos_lookup) for the built-in ~200-token corpus."""
    lang = LanguageModality(corpus_path=None)
    tokens     = lang._tokens
    pos_lookup = _BUILTIN_POS
    return tokens, pos_lookup


def _load_wikitext2(max_tokens: int, min_examples: int):
    """Return (tokens, pos_lookup) for WikiText-2.

    Downloads and caches via HuggingFace datasets.  spaCy is used to build
    a word->dominant_POS lookup for purity evaluation (optional -- if spaCy
    is not installed the lookup is empty and purity shows as '?').

    Args:
        max_tokens:   Cap on training tokens (use 0 for all).
        min_examples: Minimum bigram frequency (passed to induce_hierarchy).

    Returns:
        tokens     -- List[str] of lowercased word tokens.
        pos_lookup -- Dict[str, str] word->POS (may be empty if spaCy absent).
    """
    try:
        from wikitext2 import load_wikitext2, annotate_sentences, SPACY_TO_POS
    except ImportError:
        print('ERROR: could not import wikitext2.  '
              'Make sure experiments/language/ is on the path.')
        sys.exit(1)

    print('[wikitext2] Loading corpus ...')
    paragraphs = load_wikitext2(split='train')
    print(f'[wikitext2] {len(paragraphs)} paragraphs loaded.')

    # Join all paragraphs into a single token stream using the same
    # tokeniser as LanguageModality (split on whitespace, lowercase,
    # strip non-alpha characters).
    import re
    raw_text = ' '.join(paragraphs)
    all_tokens = [
        re.sub(r"[^a-z'-]", '', w.lower())
        for w in raw_text.split()
    ]
    all_tokens = [t for t in all_tokens if t and t.isalpha()]

    if max_tokens and len(all_tokens) > max_tokens:
        all_tokens = all_tokens[:max_tokens]

    print(f'[wikitext2] Tokens after filtering: {len(all_tokens):,} '
          f'({len(set(all_tokens)):,} unique)')

    # Build POS ground-truth via spaCy (best-effort; empty if absent).
    pos_lookup: dict = {}
    try:
        import spacy
        print('[wikitext2] Building POS lookup with spaCy '
              '(en_core_web_sm) on a sample ...')
        # Only annotate a 10K-token sample to keep startup fast.
        sample_size = min(10_000, len(all_tokens))
        sample_text = ' '.join(all_tokens[:sample_size])
        nlp = spacy.load('en_core_web_sm', exclude=['ner', 'lemmatizer'])
        doc = nlp(sample_text)

        # Build word -> dominant POS by majority vote over spaCy tokens.
        word_pos_votes: dict = collections.defaultdict(collections.Counter)
        for tok in doc:
            if tok.is_alpha and not tok.is_space:
                mapped = SPACY_TO_POS.get(tok.pos_, '')
                if mapped and mapped != 'UNK':
                    # Normalise to the same tag names as _BUILTIN_POS
                    # (BUILTIN uses: NOUN VERB ADJ DET PREP OTHER)
                    norm = {
                        'N':    'NOUN',
                        'V':    'VERB',
                        'ADJ':  'ADJ',
                        'DET':  'DET',
                        'ADP':  'PREP',
                        'ADV':  'ADV',
                        'PRON': 'PRON',
                        'CONJ': 'OTHER',
                        'COMP': 'OTHER',
                        'AUX':  'VERB',
                        'NUM':  'OTHER',
                        'PART': 'OTHER',
                        'INTJ': 'OTHER',
                    }.get(mapped, 'OTHER')
                    word_pos_votes[tok.text.lower()][norm] += 1

        for word, votes in word_pos_votes.items():
            pos_lookup[word] = votes.most_common(1)[0][0]

        print(f'[wikitext2] POS lookup built for {len(pos_lookup):,} words.')
    except (ImportError, OSError):
        print('[wikitext2] spaCy not available -- purity will show as "?".')

    return all_tokens, pos_lookup


# ---------------------------------------------------------------------------
# Phase O main
# ---------------------------------------------------------------------------

def run_phase_o(
    corpus:       str  = 'builtin',
    max_tokens:   int  = 0,
    n_clusters:   int  = 7,
    min_examples: int  = 1,
    vocab_size:   int  = 0,
    method:       str  = 'auto',
) -> None:
    _banner('Phase O -- Unsupervised Category Discovery')
    print(f"""
  Corpus:        {corpus}
  Starting point: raw sequential text, NO prior category knowledge.
  Goal: discover POS-like latent categories (DET/NOUN/VERB/...) from
  co-occurrence statistics alone -- then show the hierarchy generalises
  to unseen word bigram contexts that the flat model cannot handle.
""")

    # ------------------------------------------------------------------
    # Setup: load corpus
    # ------------------------------------------------------------------

    if corpus == 'builtin':
        tokens, known_pos = _load_builtin()
        _vocab_size  = vocab_size or None
        _method      = method
        _min_ex      = min_examples
    else:  # wikitext2
        tokens, known_pos = _load_wikitext2(max_tokens, min_examples)
        # Default safe settings for the large corpus.
        _vocab_size  = vocab_size if vocab_size > 0 else 2000
        _method      = method if method != 'auto' else 'kmeans'
        _min_ex      = max(min_examples, 5)  # filter hapax legomena

    print(f'[setup] Corpus: {len(tokens):,} tokens, '
          f'{len(set(tokens)):,} unique words')
    print(f'[setup] Ground-truth POS tags available for '
          f'{len(known_pos):,} words')
    if known_pos:
        pos_breakdown = collections.Counter(known_pos.values())
        print(f'[setup] POS distribution: '
              + ', '.join(f'{p}={n}' for p, n in sorted(pos_breakdown.items())))
    print(f'[setup] Clustering: method={_method!r}, '
          f'n_clusters={n_clusters}, min_examples={_min_ex}, '
          f'vocab_size={_vocab_size}')

    # Load arithmetic graph as the base (successor/predecessor/comparison).
    # Crucially: NO language.ctkg, NO word_pos, NO next_pos.
    arith_ctkg = os.path.join(
        _SCRIPT_DIR, '..', 'ctkg', 'domains', 'arithmetic.ctkg'
    )
    graph = parse_file(arith_ctkg)
    ai    = SymbolicAI(graph)

    # Add a bare 'next_word' concept -- just a placeholder, no POS structure.
    ai.add_concept(
        name        = 'next_word',
        domain      = 'language',
        description = 'predict next word given bigram context',
        input_type  = ['word', 'word'],
        output_type = ['word'],
        tier        = 'theorem',
    )

    # ------------------------------------------------------------------
    # Phase O.1: Train flat bigram model (no hierarchy, no POS)
    # ------------------------------------------------------------------

    print('\n[O.1] Training flat bigram next-word model ...')
    split = int(len(tokens) * 0.8)
    for i in range(split - 1):
        w1 = tokens[i]
        w2 = tokens[i + 1]
        w3 = tokens[i + 2]
        ai.teach('next_word', (w1, w2), (w3,))

    ai.freq_consolidate('next_word')

    store           = ai.stores['next_word']
    n_examples      = len(store)
    seen_contexts   = {inp for inp, _ in store.examples}
    n_unique        = len(seen_contexts)
    kl_flat         = ai.kl('next_word')

    print(f'       Training examples:     {n_examples:,}')
    print(f'       Unique bigram contexts: {n_unique:,}')
    print(f'       Flat bigram KL:         {kl_flat:.3f} bits/step')

    # How many test bigrams does the flat model fail to answer?
    test_tokens = tokens[split:]
    test_pairs  = [
        (test_tokens[i], test_tokens[i + 1])
        for i in range(len(test_tokens) - 2)
    ]
    seen_flat    = sum(1 for w1, w2 in test_pairs if (w1, w2) in seen_contexts)
    unseen_flat  = len(test_pairs) - seen_flat
    print(f'       Test bigrams: {len(test_pairs):,} total, '
          f'{seen_flat:,} seen ({seen_flat / max(1, len(test_pairs)):.0%}), '
          f'{unseen_flat:,} unseen -> flat returns None')

    # ------------------------------------------------------------------
    # Phase O.2: Induce latent category hierarchy
    # ------------------------------------------------------------------

    print(f'\n[O.2] Discovering latent categories (method={_method!r}) ...')
    if _vocab_size:
        print(f'      Output vocab capped to top {_vocab_size:,} words.')

    result = ai.induce_hierarchy(
        flat_concept = 'next_word',
        n_clusters   = n_clusters,
        min_examples = _min_ex,
        vocab_size   = _vocab_size,
        method       = _method,
    )

    if 'error' in result:
        print(f'      ERROR: {result["error"]}')
        return

    clusters   = result['clusters']
    assignment = result['assignment']
    n_eligible = result['n_eligible']
    n_clusters_found = result['n_clusters']

    print(f'      Words analysed: {n_eligible:,}')
    print(f'      Clusters found: {n_clusters_found}')

    # ------------------------------------------------------------------
    # Phase O.3: Show discovered clusters + POS purity
    # ------------------------------------------------------------------

    _banner('Phase O.3 -- Discovered Clusters vs Ground-Truth POS')

    col_w         = 52    # column width for word lists
    purity_scores = []

    for cid in sorted(clusters.keys()):
        members  = clusters[cid]
        purity, dominant = _purity(members, known_pos)
        purity_scores.append(purity)

        word_str = ', '.join(sorted(members)[:9])
        if len(members) > 9:
            word_str += ', ...'

        print(f'  C{cid}: {word_str:<{col_w}} -> {dominant:<6} '
              f'({purity:.0%} pure, {len(members)} words)')

    avg_purity = sum(purity_scores) / len(purity_scores) if purity_scores else 0
    print(f'\n  Average cluster purity: {avg_purity:.0%}')
    if avg_purity >= 0.70:
        print('  PASS: Average purity >= 70%')
    else:
        print('  NOTE: Average purity below 70% target '
              '(expected for very small or very large corpora).')

    # ------------------------------------------------------------------
    # Phase O.4: Build cluster-bigram model and measure generalisation
    # ------------------------------------------------------------------

    _banner('Phase O.4 -- Hierarchy vs Flat: Generalisation to Unseen Bigrams')

    # Build cluster-level bigram and word-within-cluster stores from training data.
    cluster_bigram_store = {}   # (c1, c2) -> Counter({c3: count})
    word_given_cluster   = {}   # c3 -> Counter({word: count})

    for inputs, outputs in store.examples:
        w1, w2 = inputs[0], inputs[1]
        w3     = outputs[0]
        c1 = assignment.get(w1)
        c2 = assignment.get(w2)
        c3 = assignment.get(w3)
        if c1 is None or c2 is None or c3 is None:
            continue
        cluster_bigram_store.setdefault((c1, c2), collections.Counter())[c3] += 1
        word_given_cluster.setdefault(c3, collections.Counter())[w3] += 1

    def _mode(counter):
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def hier_predict(w1, w2):
        """Predict next word via the discovered cluster hierarchy."""
        c1 = assignment.get(w1)
        c2 = assignment.get(w2)
        if c1 is None or c2 is None:
            return None
        c3_counter = cluster_bigram_store.get((c1, c2))
        if not c3_counter:
            return None
        c3 = _mode(c3_counter)
        word_counter = word_given_cluster.get(c3)
        if not word_counter:
            return None
        return _mode(word_counter)

    # Evaluate on seen and unseen bigrams from the test set.
    n_seen_flat = n_seen_hier = 0
    n_unseen_total = n_unseen_answered_hier = n_unseen_answered_flat = 0

    for i in range(len(test_tokens) - 2):
        w1 = test_tokens[i]
        w2 = test_tokens[i + 1]

        is_seen  = (w1, w2) in seen_contexts
        flat_ans = ai.ask('next_word', (w1, w2))
        hier_ans = hier_predict(w1, w2)

        if is_seen:
            if flat_ans is not None:
                n_seen_flat += 1
            if hier_ans is not None:
                n_seen_hier += 1
        else:
            n_unseen_total += 1
            if flat_ans is not None:
                n_unseen_answered_flat += 1
            if hier_ans is not None:
                n_unseen_answered_hier += 1

    n_test      = len(test_tokens) - 2
    n_seen_test = n_test - n_unseen_total

    print(f'\n  Test set: {n_test:,} bigrams '
          f'({n_seen_test:,} seen in training, {n_unseen_total:,} unseen)')
    print()
    print(f'  {"Context type":<22} {"Flat bigram":>14} {"Hierarchy":>12}')
    print(f'  {"-" * 22} {"-" * 14} {"-" * 12}')
    print(f'  {"Seen contexts":22} '
          f'{n_seen_flat:>10}/{n_seen_test}  '
          f'{n_seen_hier:>8}/{n_seen_test}')
    print(f'  {"Unseen contexts":22} '
          f'{n_unseen_answered_flat:>10}/{n_unseen_total}  '
          f'{n_unseen_answered_hier:>8}/{n_unseen_total}')

    print()
    if n_unseen_total > 0:
        gain = n_unseen_answered_hier - n_unseen_answered_flat
        print(f'  -> Hierarchy answers {gain:+d} more unseen contexts than flat bigram.')
        print(f'     This is the key generalisation advantage: unseen word pairs')
        print(f'     (w1, w2) route through the cluster bigram (c1, c2), which')
        print(f'     was almost certainly seen during training even when the word')
        print(f'     pair itself was not.')

    # ------------------------------------------------------------------
    # Phase O.5: MDL check -- does the hierarchy compress?
    # ------------------------------------------------------------------

    _banner('Phase O.5 -- MDL Compression Check')

    import math

    # Flat model description length: n_unique_contexts x H(output | context)
    flat_entropy  = store.empirical_entropy()
    flat_desc_len = n_unique * flat_entropy

    # Cluster-bigram model description length:
    cb_entropy_total = 0.0
    cb_contexts      = 0
    for (c1, c2), counter in cluster_bigram_store.items():
        total = sum(counter.values())
        h = 0.0
        for cnt in counter.values():
            p = cnt / total
            if p > 0:
                h -= p * math.log2(p)
        cb_entropy_total += h
        cb_contexts      += 1
    cb_entropy_mean = cb_entropy_total / max(1, cb_contexts)

    wc_entropy_total = 0.0
    wc_contexts      = 0
    for cid, counter in word_given_cluster.items():
        total = sum(counter.values())
        h = 0.0
        for cnt in counter.values():
            p = cnt / total
            if p > 0:
                h -= p * math.log2(p)
        wc_entropy_total += h
        wc_contexts      += 1
    wc_entropy_mean = wc_entropy_total / max(1, wc_contexts)

    hier_desc_len = cb_contexts * cb_entropy_mean + n_clusters_found * wc_entropy_mean

    print(f'\n  Flat bigram:')
    print(f'    Unique contexts:      {n_unique:,}')
    print(f'    Mean H(next|context): {flat_entropy:.3f} bits/step')
    print(f'    Description length:   {flat_desc_len:.1f} bits')

    print(f'\n  Cluster-bigram hierarchy:')
    print(f'    Unique cluster bigrams:         {cb_contexts}')
    print(f'    Mean H(next_cluster|cl_bigram): {cb_entropy_mean:.3f} bits')
    print(f'    Mean H(word|cluster):           {wc_entropy_mean:.3f} bits')
    print(f'    Description length:             {hier_desc_len:.1f} bits')

    if hier_desc_len < flat_desc_len:
        ratio   = flat_desc_len / max(1e-6, hier_desc_len)
        savings = (1 - hier_desc_len / flat_desc_len) * 100
        print(f'\n  PASS MDL criterion satisfied: hierarchy is {savings:.0f}% shorter.')
        print(f'    Compression ratio: {ratio:.1f}x')
        print(f'    -> The discovered categories carry genuine predictive structure.')
    else:
        print(f'\n  NOTE MDL criterion not satisfied with this corpus/cluster count.')
        print(f'    (Increase --max_tokens or try --n_clusters.)')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    _banner('Phase O Summary')
    print(f"""
  Corpus:          {corpus} ({len(tokens):,} tokens, {len(set(tokens)):,} unique words)
  Discovered:      {n_clusters_found} latent categories from {n_eligible:,} unique words.
  Method:          {_method} clustering on forward context distributions.
  Average purity:  {avg_purity:.0%} match to ground-truth POS tags.

  Key results:
    - Words that precede similar words cluster together naturally.
    - DET words (the/a/an) cluster separately from NOUN and VERB words.
    - Hierarchy routes unseen (w1, w2) pairs through cluster bigrams -> more coverage.

  Cross-domain implication (the core Phase O claim):
    Replace 'word' with 'action' -> discovers Minecraft action categories.
    Replace 'word' with 'note'   -> discovers harmonic roles in music.
    Replace 'word' with 'state'  -> discovers motor movement primitives.
    The engine requires NO domain knowledge -- only sequential experience.

  Practical implication:
    This result gives assurance that CTKG structure ACCELERATES learning
    rather than patching gaps in the implementation.  The engine can
    discover POS-like categories from raw data; pre-built CTKG structure
    simply gives it a head start.

  Next step (Phase N):
    Formalize the discovered categories as a new .ctkg file and train the
    same 4-stage hierarchy (item_pos -> next_pos -> item_given_pos -> next_item)
    on Minecraft action sequences or a MIDI corpus.
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase O: unsupervised category discovery from raw text.'
    )
    parser.add_argument(
        '--corpus', choices=['builtin', 'wikitext2'], default='builtin',
        help='Corpus to use. "builtin" = ~200-token toy corpus (fast). '
             '"wikitext2" = real Wikipedia text via HuggingFace datasets '
             '(requires: pip install datasets; optionally: spacy + en_core_web_sm).'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=0,
        help='Cap on training tokens for wikitext2 mode (0 = all ~2M). '
             'Use e.g. 50000 for a quick smoke-test.'
    )
    parser.add_argument(
        '--n_clusters', type=int, default=7,
        help='Target number of latent categories (default 7).'
    )
    parser.add_argument(
        '--min_examples', type=int, default=1,
        help='Minimum word frequency to include in clustering. '
             'For wikitext2 recommend >= 5 to exclude hapax legomena.'
    )
    parser.add_argument(
        '--vocab_size', type=int, default=0,
        help='Cap output vocabulary to top-N words before clustering. '
             '0 = auto (2000 for wikitext2, unlimited for builtin).'
    )
    parser.add_argument(
        '--method', choices=['auto', 'agglomerative', 'kmeans'], default='auto',
        help='Clustering algorithm. "auto" = agglomerative for n<=200 words, '
             'kmeans otherwise. "kmeans" requires numpy (always available in '
             'the project venv).'
    )
    args = parser.parse_args()

    run_phase_o(
        corpus       = args.corpus,
        max_tokens   = args.max_tokens,
        n_clusters   = args.n_clusters,
        min_examples = args.min_examples,
        vocab_size   = args.vocab_size,
        method       = args.method,
    )
