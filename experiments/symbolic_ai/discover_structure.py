"""discover_structure.py -- Multi-scale hierarchical structure discovery.

Applies the symbolic AI's distributional clustering bottom-up from characters
to abstract concepts, discovering structure at each scale automatically:

    char -> morpheme -> word -> phrase -> clause -> ...

At each scale the pipeline runs three steps:

  1. Stream atoms -> teach next_X bigrams to the AI
  2. Discover distributional categories: ai.induce_hierarchy('next_X')
  3. Find high-PMI adjacent pairs: ai.chunk_store('next_X', min_pmi)
     These pairs become the atoms at the next scale.

No labels.  No grammar rules.  No external tools.
The same algorithm that finds Q-before-U at the char scale finds
"per se" and "id est" at the word scale and topic structure at the
paragraph scale.

Scale levels
------------
  0  char      -- individual Unicode code points
  1  morpheme  -- high-PMI char n-grams (syllables / bound morphemes)
  2  word      -- whitespace-delimited tokens (switches to token granularity)
  3  phrase    -- high-PMI word n-grams (fixed collocations / constituencies)
  4  clause    -- high-PMI phrase n-grams
  5+ sentence, paragraph, concept  (for very large corpora)

The transition from level 1 to level 2 is a mode switch: levels 0-1
process char-by-char (merging chars into morphemes within words); level 2+
process token-by-token (merging tokens into phrases).

Usage
-----
    # Run 3 levels (chars -> morphs -> words) on 200 lines:
    python discover_structure.py --corpus EarlyModernLatin --levels 3 --n 200

    # Full corpus, 4 levels, save checkpoint:
    python discover_structure.py --corpus EarlyModernLatin --levels 4 --save hier.json

    # Load existing checkpoint and add more levels:
    python discover_structure.py --corpus EarlyModernLatin --levels 6 --load hier.json
"""
from __future__ import annotations

import argparse
import collections
import json
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

from ctkg.parser import parse_file, merge as ctkg_merge
from engine import SymbolicAI
from synthesis import apply_chunks

try:
    from ocr_test import find_pairs, normalise_gt          # type: ignore[import]
except ImportError:
    def find_pairs(d, max_n=None, shuffle=False, seed=0):
        """Fallback: glob all .gt.txt files under d."""
        import glob, random
        pairs = []
        for gt in glob.glob(os.path.join(d, '**', '*.gt.txt'), recursive=True):
            pairs.append((None, gt))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(pairs)
        return pairs[:max_n] if max_n else pairs

    def normalise_gt(raw: str) -> str:
        return raw.strip()


_DATA_DIR = os.path.join(_HERE, 'data', 'GT4HistOCR', 'corpus')

# Cluster counts at each level (can be overridden by --n_clusters).
_DEFAULT_CLUSTERS = [12, 32, 64, 32, 24, 16, 12, 8]

# PMI thresholds at each level (higher = stricter chunking).
_DEFAULT_PMI = [3.0, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8]

# Level names for display and CTKG concept naming.
_LEVEL_NAMES = [
    'char', 'morpheme', 'word', 'phrase',
    'clause', 'sentence', 'paragraph', 'concept',
]


def _banner(title: str) -> None:
    print(f'\n{"=" * 62}')
    print(f'  {title}')
    print('=' * 62)


# ---------------------------------------------------------------------------
# Corpus streaming
# ---------------------------------------------------------------------------

def _stream_texts(pairs: list) -> list[str]:
    """Read all GT text lines; return list of normalised strings."""
    texts = []
    for _, gt_path in pairs:
        if gt_path is None:
            continue
        try:
            with open(gt_path, encoding='utf-8', errors='replace') as f:
                raw = f.read()
            text = normalise_gt(raw)
            if text:
                texts.append(text)
        except Exception:
            pass
    return texts


def _char_sequences(texts: list[str]) -> list[list[str]]:
    """Split each text into a list of individual characters."""
    return [list(t) for t in texts]


def _token_sequences(texts: list[str]) -> list[list[str]]:
    """Split each text into whitespace-delimited word tokens."""
    return [t.split() for t in texts if t.split()]


# ---------------------------------------------------------------------------
# Level runners
# ---------------------------------------------------------------------------

def run_char_level(
    ai:        SymbolicAI,
    texts:     list[str],
    n_clusters: int = 12,
    min_pmi:   float = 3.0,
    max_merges: int = 500,
    verbose:   bool = True,
) -> dict:
    """Level 0: character bigrams -> char categories + char chunk_map.

    Collects forward (next_char) AND backward (prev_char) bigrams in a single
    corpus pass, then clusters using bidirectional context for richer categories.

    Returns:
        {
          'clusters':   {cid: [char_list]}
          'chunk_rules': [(a, b, compound, pmi), ...]
          'chunk_map':   {(a, b): compound}
          'n_pairs':    int
          'n_unique':   int
        }
    """
    _banner('Level 0: Character bigrams (bidirectional)')

    fwd_concept = 'next_char'
    bwd_concept = 'prev_char'
    for cname, desc in [(fwd_concept, 'Character bigram forward distribution'),
                        (bwd_concept, 'Character bigram backward distribution')]:
        if cname not in ai.stores:
            ai.add_concept(
                name=cname, domain='scale_hierarchy',
                description=desc,
                input_type=['char'], output_type=['char'], tier='theorem',
            )

    n_pairs = 0
    for text in texts:
        for i in range(len(text) - 1):
            a, b = text[i], text[i + 1]
            ai.teach(fwd_concept, (a,), (b,))   # a → b
            ai.teach(bwd_concept, (b,), (a,))   # b ← a
            n_pairs += 1

    n_unique = len(ai.stores[fwd_concept])
    if verbose:
        print(f'  {len(texts)} lines, {n_pairs:,} char pairs, '
              f'{n_unique:,} unique forward transitions')

    result = ai.induce_hierarchy_bidir(fwd_concept, bwd_concept, n_clusters=n_clusters)
    clusters = result.get('clusters', {})

    _print_clusters(clusters, ai, fwd_concept, label='char', top_n=6)

    chunk_rules = ai.chunk_store(fwd_concept, min_pmi=min_pmi, max_merges=max_merges)
    chunk_map   = {(a, b): c for a, b, c, _ in chunk_rules}

    if verbose:
        print(f'\n  PMI chunks (min_pmi={min_pmi:.1f}): {len(chunk_rules)} merge rules')
        for a, b, compound, pmi in chunk_rules[:15]:
            print(f"    '{a}'+'{b}' -> '{compound}'  (PMI={pmi:.2f})")
        if len(chunk_rules) > 15:
            print(f'    ... and {len(chunk_rules) - 15} more')

    return {
        'clusters':    clusters,
        'chunk_rules': chunk_rules,
        'chunk_map':   chunk_map,
        'n_pairs':     n_pairs,
        'n_unique':    n_unique,
    }


def run_subword_level(
    ai:         SymbolicAI,
    texts:      list[str],
    chunk_maps: list[dict],
    level:      int,
    n_clusters: int = 32,
    min_pmi:    float = 2.5,
    max_merges: int = 500,
    verbose:    bool = True,
) -> dict:
    """Level 1 (morpheme): apply char chunk_maps, teach morph bigrams.

    Processes within-word char sequences: spaces delimit words; chunks
    are applied within each word but not across spaces.

    Returns same structure as run_char_level.
    """
    name = _LEVEL_NAMES[level]
    _banner(f'Level {level}: {name} bigrams (bidirectional)')

    fwd_concept = f'next_{name}'
    bwd_concept = f'prev_{name}'
    for cname, desc in [
        (fwd_concept, f'{name.capitalize()} bigram forward distribution'),
        (bwd_concept, f'{name.capitalize()} bigram backward distribution'),
    ]:
        if cname not in ai.stores:
            ai.add_concept(
                name=cname, domain='scale_hierarchy',
                description=desc,
                input_type=[name], output_type=[name], tier='theorem',
            )

    n_pairs = 0
    for text in texts:
        # Split into words, apply chunk_maps within each word, then sequence morphemes.
        words = text.split()
        morph_seq: list[str] = []
        for word in words:
            chars = list(word)
            compressed = chars
            for cm in chunk_maps:
                compressed = apply_chunks(compressed, cm)
            morph_seq.extend(compressed)
            morph_seq.append(' ')   # word boundary marker

        for i in range(len(morph_seq) - 1):
            a, b = morph_seq[i], morph_seq[i + 1]
            ai.teach(fwd_concept, (a,), (b,))   # a → b
            ai.teach(bwd_concept, (b,), (a,))   # b ← a
            n_pairs += 1

    n_unique = len(ai.stores[fwd_concept])
    if verbose:
        print(f'  {len(texts)} lines, {n_pairs:,} {name} pairs, '
              f'{n_unique:,} unique {name}s')

    result = ai.induce_hierarchy_bidir(fwd_concept, bwd_concept, n_clusters=n_clusters)
    clusters = result.get('clusters', {})

    _print_clusters(clusters, ai, fwd_concept, label=name, top_n=8)

    chunk_rules = ai.chunk_store(fwd_concept, min_pmi=min_pmi, max_merges=max_merges)
    chunk_map   = {(a, b): c for a, b, c, _ in chunk_rules}

    if verbose:
        print(f'\n  PMI chunks (min_pmi={min_pmi:.1f}): {len(chunk_rules)} merge rules')
        for a, b, compound, pmi in chunk_rules[:15]:
            print(f"    '{a}'+'{b}' -> '{compound}'  (PMI={pmi:.2f})")
        if len(chunk_rules) > 15:
            print(f'    ... and {len(chunk_rules) - 15} more')

    return {
        'clusters':    clusters,
        'chunk_rules': chunk_rules,
        'chunk_map':   chunk_map,
        'n_pairs':     n_pairs,
        'n_unique':    n_unique,
    }


def run_token_level(
    ai:         SymbolicAI,
    texts:      list[str],
    chunk_maps: list[dict],
    level:      int,
    n_clusters: int = 64,
    min_pmi:    float = 2.0,
    max_merges: int = 500,
    separator:  str = ' ',
    verbose:    bool = True,
) -> dict:
    """Level 2+ (word, phrase, clause…): token-granularity bigrams.

    At level 2 (word): atoms are whitespace-delimited tokens.
    At level 3+ (phrase): apply previous token-level chunk_maps to get
    phrase-level atoms.

    Args:
        chunk_maps: List of chunk_maps to apply in sequence to compress
                    word sequences into higher-level atoms.  Empty at level 2.
    """
    name = _LEVEL_NAMES[level]
    _banner(f'Level {level}: {name} bigrams (bidirectional)')

    fwd_concept = f'next_{name}'
    bwd_concept = f'prev_{name}'
    if name == 'word':
        fwd_concept = 'next_word_hier'   # avoids conflict with language.ctkg's next_word
        bwd_concept = 'prev_word_hier'
    for cname, desc in [
        (fwd_concept, f'{name.capitalize()} bigram forward distribution'),
        (bwd_concept, f'{name.capitalize()} bigram backward distribution'),
    ]:
        if cname not in ai.stores:
            ai.add_concept(
                name=cname, domain='scale_hierarchy',
                description=desc,
                input_type=[name], output_type=[name], tier='theorem',
            )

    n_pairs = 0
    for text in texts:
        tokens = text.split()
        if not tokens:
            continue
        # Apply any token-level chunk_maps in sequence.
        seq = tokens
        for cm in chunk_maps:
            seq = apply_chunks(seq, cm)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            ai.teach(fwd_concept, (a,), (b,))   # a → b
            ai.teach(bwd_concept, (b,), (a,))   # b ← a
            n_pairs += 1

    n_unique = len(ai.stores[fwd_concept])
    if verbose:
        print(f'  {len(texts)} lines, {n_pairs:,} {name} pairs, '
              f'{n_unique:,} unique {name}s')

    result = ai.induce_hierarchy_bidir(fwd_concept, bwd_concept, n_clusters=n_clusters)
    clusters = result.get('clusters', {})

    _print_clusters(clusters, ai, fwd_concept, label=name, top_n=10)

    chunk_rules = ai.chunk_store(
        fwd_concept, min_pmi=min_pmi, max_merges=max_merges, separator=separator,
    )
    chunk_map = {(a, b): c for a, b, c, _ in chunk_rules}

    if verbose:
        print(f'\n  PMI chunks (min_pmi={min_pmi:.1f}): {len(chunk_rules)} merge rules')
        for a, b, compound, pmi in chunk_rules[:15]:
            print(f"    '{a}' + '{b}' -> '{compound}'  (PMI={pmi:.2f})")
        if len(chunk_rules) > 15:
            print(f'    ... and {len(chunk_rules) - 15} more')

    return {
        'clusters':    clusters,
        'chunk_rules': chunk_rules,
        'chunk_map':   chunk_map,
        'n_pairs':     n_pairs,
        'n_unique':    n_unique,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _entropy(dist: dict) -> float:
    total = sum(dist.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in dist.values() if v > 0)


def _print_clusters(
    clusters: dict,
    ai:       SymbolicAI,
    concept:  str,
    label:    str = 'atom',
    top_n:    int = 8,
) -> None:
    """Print a compact cluster summary with entropy info."""
    store = ai.stores.get(concept)
    if not clusters or store is None:
        return

    # Build forward distributions per atom.
    fwd: dict = collections.defaultdict(collections.Counter)
    for (inp,), (out,) in store.examples:
        fwd[inp][out] += 1

    print(f'\n  Discovered {len(clusters)} {label} clusters:')
    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        entropies = []
        for m in members:
            d = fwd.get(m, {})
            if d:
                entropies.append(_entropy(d))
        h_range = (
            f'H={min(entropies):.1f}-{max(entropies):.1f}'
            if entropies else 'H=?'
        )
        sample = members[:top_n]
        rest   = f'  +{len(members) - top_n} more' if len(members) > top_n else ''
        atoms_str = ', '.join(repr(m) for m in sample)
        print(f'    C{cid:02d} ({len(members):3d} members, {h_range}): {atoms_str}{rest}')


# ---------------------------------------------------------------------------
# Checkpoint serialisation
# ---------------------------------------------------------------------------

def _save_chunk_maps(chunk_maps: list[dict], path: str) -> None:
    """Save chunk maps to a JSON file (separate from AI checkpoint)."""
    data = [
        [[list(k), v] for k, v in cm.items()]
        for cm in chunk_maps
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_chunk_maps(path: str) -> list[dict]:
    """Load chunk maps from JSON file."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [
        {tuple(k): v for k, v in level}
        for level in data
    ]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    texts:         list[str],
    n_levels:      int = 3,
    n_clusters:    list[int] | None = None,
    min_pmis:      list[float] | None = None,
    max_merges:    int = 500,
    save_path:     str | None = None,
    load_path:     str | None = None,
    chunk_map_path: str | None = None,
    verbose:       bool = True,
) -> tuple[SymbolicAI, list[dict]]:
    """Run the full multi-scale discovery pipeline.

    Args:
        texts:          Normalised text lines.
        n_levels:       Number of hierarchy levels to discover (1-8).
        n_clusters:     Clusters per level.  Defaults to _DEFAULT_CLUSTERS.
        min_pmis:       PMI thresholds per level.  Defaults to _DEFAULT_PMI.
        max_merges:     Max PMI merge rules per level.
        save_path:      Save AI checkpoint after all levels.
        load_path:      Load AI checkpoint before running.
        chunk_map_path: JSON file to save/load chunk maps.
        verbose:        Print diagnostics.

    Returns:
        (ai, chunk_maps) where chunk_maps[i] is the {(a,b): compound} map
        for level i.
    """
    n_clusters = (n_clusters or _DEFAULT_CLUSTERS)[:n_levels]
    min_pmis   = (min_pmis   or _DEFAULT_PMI)[:n_levels]

    # Build AI with scale_hierarchy CTKG.  merge() mutates in place.
    arith  = parse_file(os.path.join(_HERE, '..', 'ctkg', 'domains', 'arithmetic.ctkg'))
    hier   = parse_file(os.path.join(_HERE, '..', 'ctkg', 'domains', 'scale_hierarchy.ctkg'))
    ctkg_merge(arith, hier)   # merges hier into arith in place
    graph  = arith
    ai     = SymbolicAI(graph)

    if load_path and os.path.exists(load_path):
        ai.load_checkpoint(load_path)
        if verbose:
            print(f'  Loaded checkpoint: {load_path}')

    # Load existing chunk maps if provided.
    chunk_maps: list[dict] = []
    if chunk_map_path and os.path.exists(chunk_map_path):
        chunk_maps = _load_chunk_maps(chunk_map_path)
        if verbose:
            print(f'  Loaded {len(chunk_maps)} chunk map(s) from {chunk_map_path}')

    # Decide which levels still need to be run.
    start_level = len(chunk_maps)

    # ---- Level 0: characters ------------------------------------------------
    if start_level <= 0 and n_levels >= 1:
        r = run_char_level(
            ai, texts,
            n_clusters=n_clusters[0],
            min_pmi=min_pmis[0],
            max_merges=max_merges,
            verbose=verbose,
        )
        chunk_maps.append(r['chunk_map'])

    # ---- Level 1: morphemes (within-word char chunking) ---------------------
    if start_level <= 1 and n_levels >= 2:
        r = run_subword_level(
            ai, texts,
            chunk_maps=chunk_maps[:1],
            level=1,
            n_clusters=n_clusters[1] if len(n_clusters) > 1 else 32,
            min_pmi=min_pmis[1]      if len(min_pmis)   > 1 else 2.5,
            max_merges=max_merges,
            verbose=verbose,
        )
        chunk_maps.append(r['chunk_map'])

    # ---- Level 2+: token-level (word, phrase, clause …) --------------------
    for level in range(2, n_levels):
        if start_level > level:
            continue
        nc = n_clusters[level] if level < len(n_clusters) else 32
        mp = min_pmis[level]   if level < len(min_pmis)   else 1.5
        # chunk_maps that apply at this level: only those from level 2 onward.
        token_chunk_maps = chunk_maps[2:level] if level > 2 else []
        r = run_token_level(
            ai, texts,
            chunk_maps=token_chunk_maps,
            level=level,
            n_clusters=nc,
            min_pmi=mp,
            max_merges=max_merges,
            separator=' ',
            verbose=verbose,
        )
        chunk_maps.append(r['chunk_map'])

    # ---- Save ---------------------------------------------------------------
    if save_path:
        ai.save_checkpoint(save_path)
        if verbose:
            print(f'\n  Checkpoint saved: {save_path}')

    if chunk_map_path:
        _save_chunk_maps(chunk_maps, chunk_map_path)
        if verbose:
            print(f'  Chunk maps saved: {chunk_map_path}')

    return ai, chunk_maps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Multi-scale hierarchical structure discovery.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--corpus', default='EarlyModernLatin',
                   help='Corpus name or path (default: EarlyModernLatin)')
    p.add_argument('--n', type=int, default=None,
                   help='Max lines to use (default: all)')
    p.add_argument('--levels', type=int, default=3,
                   help='Number of hierarchy levels (default: 3)')
    p.add_argument('--save', metavar='PATH',
                   help='Save AI checkpoint after discovery')
    p.add_argument('--load', metavar='PATH',
                   help='Load existing checkpoint before discovery')
    p.add_argument('--chunks', metavar='PATH',
                   help='JSON file to save/load chunk maps')
    p.add_argument('--max_merges', type=int, default=500,
                   help='Max PMI merge rules per level (default: 500)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    _banner('Multi-Scale Hierarchical Structure Discovery')
    print(f'  Corpus:    {args.corpus}')
    print(f'  Levels:    {args.levels}  ({" -> ".join(_LEVEL_NAMES[:args.levels])})')
    print(f'  Max lines: {args.n or "all"}')

    # Locate corpus.
    d = (args.corpus if os.path.isdir(args.corpus)
         else os.path.join(_DATA_DIR, args.corpus))
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {args.corpus!r}')
        sys.exit(1)

    pairs = find_pairs(d, max_n=args.n, shuffle=bool(args.n), seed=args.seed)
    print(f'  Pairs:     {len(pairs)}')

    texts = _stream_texts(pairs)
    print(f'  Lines:     {len(texts)}')

    ai, chunk_maps = run_pipeline(
        texts=texts,
        n_levels=args.levels,
        max_merges=args.max_merges,
        save_path=args.save,
        load_path=args.load,
        chunk_map_path=args.chunks,
        verbose=True,
    )

    _banner('Summary')
    print(f'\n  Levels completed: {len(chunk_maps)}')
    for i, cm in enumerate(chunk_maps):
        name = _LEVEL_NAMES[i]
        print(f'  Level {i} ({name}): {len(cm):,} chunk rules')
    print()
    learned = [n for n, s in ai.stores.items() if len(s) > 0]
    print(f'  AI stores: {", ".join(learned[:8])}{"..." if len(learned) > 8 else ""}')
    print()
    print('  The symbolic AI now has distributional knowledge at all discovered scales.')
    print('  Call ai.ask("next_word_hier", (word,)) to predict the most likely next word.')
    print('  Call apply_chunks(sequence, chunk_maps[0]) to compress chars to morphemes.')


if __name__ == '__main__':
    main()
