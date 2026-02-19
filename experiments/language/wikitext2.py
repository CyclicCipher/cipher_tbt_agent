"""WikiText-2 data loader with spaCy annotation pipeline.

Downloads WikiText-2 (raw), segments into sentences, annotates with spaCy
for POS tags and dependency structure, then extracts phrase spans for
syntax curriculum training.

Dependencies:
    pip install spacy datasets
    python -m spacy download en_core_web_sm

Caches annotated sentences as JSON to avoid re-running spaCy.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------------------------------------------------
# Annotated sentence
# -----------------------------------------------------------------------

@dataclass
class AnnotatedSentence:
    """A sentence with linguistic annotations from spaCy."""
    text: str
    words: List[str]           # lowercased word tokens (no punct)
    pos_tags: List[str]        # universal POS: N, V, ADJ, DET, ADP, ...
    dep_labels: List[str]      # dependency labels: nsubj, dobj, ...
    dep_heads: List[int]       # head index per word (-1 = root)
    np_spans: List[List[int]]  # [[start, end], ...] NP spans (word indices)
    pp_spans: List[List[int]]  # PP spans
    vp_spans: List[List[int]]  # VP spans
    subj_span: Optional[List[int]] = None   # [start, end] or None
    pred_span: Optional[List[int]] = None   # [start, end] or None

    @property
    def n_words(self) -> int:
        return len(self.words)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnnotatedSentence':
        return cls(**d)


# -----------------------------------------------------------------------
# spaCy POS mapping
# -----------------------------------------------------------------------

SPACY_TO_POS = {
    'NOUN': 'N', 'PROPN': 'N', 'PRON': 'PRON',
    'VERB': 'V', 'AUX': 'AUX',
    'ADJ': 'ADJ', 'ADV': 'ADV',
    'DET': 'DET', 'ADP': 'ADP',
    'CCONJ': 'CONJ', 'SCONJ': 'COMP',
    'NUM': 'NUM', 'PART': 'PART',
    'INTJ': 'INTJ', 'PUNCT': 'PUNCT',
    'SYM': 'UNK', 'X': 'UNK', 'SPACE': 'UNK',
}


# -----------------------------------------------------------------------
# Dependency tree helpers
# -----------------------------------------------------------------------

def _find_children(token_idx: int, dep_heads: List[int]) -> List[int]:
    """Find immediate children of a token in the dep tree."""
    return [i for i, h in enumerate(dep_heads)
            if h == token_idx and i != token_idx]


def _find_subtree(token_idx: int, dep_heads: List[int]) -> Set[int]:
    """Find all descendants of a token (including itself)."""
    desc = set()
    stack = [token_idx]
    while stack:
        idx = stack.pop()
        if idx in desc:
            continue
        desc.add(idx)
        stack.extend(_find_children(idx, dep_heads))
    return desc


def _span_from_indices(indices: Set[int]) -> Optional[List[int]]:
    """Convert a set of word indices to a contiguous [start, end) span."""
    if not indices:
        return None
    return [min(indices), max(indices) + 1]


def _extract_pp_spans(pos_tags: List[str],
                      dep_heads: List[int]) -> List[List[int]]:
    """Extract PP spans: each ADP's subtree forms a PP."""
    spans = []
    for i, pos in enumerate(pos_tags):
        if pos == 'ADP':
            subtree = _find_subtree(i, dep_heads)
            span = _span_from_indices(subtree)
            if span and span[1] - span[0] >= 2:  # PP needs prep + complement
                spans.append(span)
    return spans


def _extract_vp_spans(pos_tags: List[str], dep_labels: List[str],
                      dep_heads: List[int]) -> List[List[int]]:
    """Extract VP spans: root verb's subtree minus subject."""
    spans = []
    for i, (pos, dep) in enumerate(zip(pos_tags, dep_labels)):
        if pos in ('V', 'AUX') and dep in ('ROOT', 'root'):
            subtree = _find_subtree(i, dep_heads)
            # Remove subject NP from VP
            for j, dl in enumerate(dep_labels):
                if dl in ('nsubj', 'nsubjpass') and dep_heads[j] == i:
                    subj_tree = _find_subtree(j, dep_heads)
                    subtree -= subj_tree
            span = _span_from_indices(subtree)
            if span:
                spans.append(span)
    return spans


def _extract_clause_spans(pos_tags: List[str], dep_labels: List[str],
                          dep_heads: List[int]
                          ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Extract subject and predicate spans for clause structure."""
    subj_span = None
    pred_span = None

    for i, dep in enumerate(dep_labels):
        if dep in ('nsubj', 'nsubjpass'):
            subtree = _find_subtree(i, dep_heads)
            subj_span = _span_from_indices(subtree)
            break  # take first subject

    for i, (pos, dep) in enumerate(zip(pos_tags, dep_labels)):
        if pos in ('V', 'AUX') and dep in ('ROOT', 'root'):
            subtree = _find_subtree(i, dep_heads)
            # Remove subject from predicate
            for j, dl in enumerate(dep_labels):
                if dl in ('nsubj', 'nsubjpass') and dep_heads[j] == i:
                    subtree -= _find_subtree(j, dep_heads)
            pred_span = _span_from_indices(subtree)
            break

    return subj_span, pred_span


# -----------------------------------------------------------------------
# WikiText-2 loading
# -----------------------------------------------------------------------

def load_wikitext2(data_dir: Optional[str] = None,
                   split: str = 'train') -> List[str]:
    """Load WikiText-2 raw text, returning list of non-empty paragraphs.

    Tries HuggingFace datasets first, falls back to local cache.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

    cache_path = os.path.join(data_dir, f'wikitext2_{split}.txt')

    # Check cache — require minimum size to guard against corrupt files
    # (e.g. from a previous run that crashed mid-write)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        with open(cache_path, 'r', encoding='utf-8') as f:
            paragraphs = [line.strip() for line in f
                          if line.strip() and not line.strip().startswith('=')]
        if paragraphs:
            return paragraphs
        # Cache exists but empty/corrupt — re-download
        os.remove(cache_path)

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        paragraphs = [
            row['text'].strip() for row in ds
            if row['text'].strip() and not row['text'].strip().startswith('=')
        ]
    except (ImportError, Exception) as e:
        raise RuntimeError(
            f"Cannot load WikiText-2. Install datasets: pip install datasets\n"
            f"Or place wikitext2_{split}.txt in {data_dir}/\n"
            f"Error: {e}"
        )

    if not paragraphs:
        raise RuntimeError("WikiText-2 loaded but produced 0 paragraphs")

    # Cache via temp file to avoid corrupt partial writes
    os.makedirs(data_dir, exist_ok=True)
    tmp_path = cache_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            f.write(p + '\n')
    os.replace(tmp_path, cache_path)
    print(f"  Cached {len(paragraphs)} paragraphs -> {cache_path}")

    return paragraphs


# -----------------------------------------------------------------------
# Annotation
# -----------------------------------------------------------------------

def annotate_sentences(paragraphs: List[str],
                       max_words: int = 12,
                       min_words: int = 3,
                       spacy_model: str = 'en_core_web_sm'
                       ) -> List[AnnotatedSentence]:
    """Segment paragraphs into sentences and annotate with spaCy.

    Filters to sentences with min_words <= n_words <= max_words.
    Strips punctuation from word lists (syntax operates on words).
    """
    try:
        import spacy
    except ImportError:
        raise RuntimeError(
            "spaCy is required for annotation.\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm"
        )

    nlp = spacy.load(spacy_model)
    sentences: List[AnnotatedSentence] = []

    for para in paragraphs:
        doc = nlp(para)
        for sent in doc.sents:
            # Filter to content tokens (no punct, no whitespace)
            content_toks = [tok for tok in sent
                            if not tok.is_punct and not tok.is_space]
            if len(content_toks) < min_words or len(content_toks) > max_words:
                continue

            # Build token-to-index map (content tokens only)
            tok_to_idx = {tok.i: i for i, tok in enumerate(content_toks)}

            word_texts = [tok.text.lower() for tok in content_toks]
            pos_tags = [SPACY_TO_POS.get(tok.pos_, 'UNK')
                        for tok in content_toks]
            dep_labels = [tok.dep_ for tok in content_toks]

            # Dep heads: map spaCy global indices to our local indices
            dep_heads = []
            for tok in content_toks:
                if tok.head == tok:  # root
                    dep_heads.append(-1)
                elif tok.head.i in tok_to_idx:
                    dep_heads.append(tok_to_idx[tok.head.i])
                else:
                    dep_heads.append(-1)  # head was punctuation (rare)

            # NP spans from spaCy noun_chunks
            np_spans = []
            for chunk in sent.noun_chunks:
                chunk_words = [tok for tok in chunk
                               if not tok.is_punct and not tok.is_space]
                if chunk_words:
                    start = tok_to_idx.get(chunk_words[0].i)
                    end = tok_to_idx.get(chunk_words[-1].i)
                    if start is not None and end is not None:
                        np_spans.append([start, end + 1])

            # PP and VP spans from dependency parse
            pp_spans = _extract_pp_spans(pos_tags, dep_heads)
            vp_spans = _extract_vp_spans(pos_tags, dep_labels, dep_heads)
            subj_span, pred_span = _extract_clause_spans(
                pos_tags, dep_labels, dep_heads)

            sentences.append(AnnotatedSentence(
                text=sent.text.strip(),
                words=word_texts,
                pos_tags=pos_tags,
                dep_labels=dep_labels,
                dep_heads=dep_heads,
                np_spans=np_spans,
                pp_spans=pp_spans,
                vp_spans=vp_spans,
                subj_span=subj_span,
                pred_span=pred_span,
            ))

    return sentences


# -----------------------------------------------------------------------
# Cache management
# -----------------------------------------------------------------------

def _cache_path(data_dir: str, split: str, max_words: int) -> str:
    return os.path.join(data_dir, f'annotated_{split}_max{max_words}.json')


def save_annotations(sentences: List[AnnotatedSentence],
                     path: str) -> None:
    """Save annotated sentences to JSON cache."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in sentences], f)


def load_annotations(path: str) -> List[AnnotatedSentence]:
    """Load annotated sentences from JSON cache."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [AnnotatedSentence.from_dict(d) for d in data]


def load_or_annotate(data_dir: Optional[str] = None,
                     split: str = 'train',
                     max_words: int = 12,
                     min_words: int = 3,
                     spacy_model: str = 'en_core_web_sm'
                     ) -> List[AnnotatedSentence]:
    """Load annotated sentences from cache, or annotate from scratch.

    First checks for a JSON cache of annotated sentences. If not found,
    loads WikiText-2, runs spaCy annotation, and caches the result.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

    cache = _cache_path(data_dir, split, max_words)
    # Guard against empty/corrupt annotation cache
    if os.path.exists(cache) and os.path.getsize(cache) > 100:
        print(f"  Loading cached annotations: {cache}")
        sents = load_annotations(cache)
        if sents:
            return sents
        print(f"  WARNING: cache was empty, re-annotating...")
        os.remove(cache)

    print(f"  Annotating WikiText-2 ({split})... this may take a few minutes.")
    paragraphs = load_wikitext2(data_dir, split)
    print(f"  Loaded {len(paragraphs)} paragraphs")
    sentences = annotate_sentences(
        paragraphs, max_words=max_words, min_words=min_words,
        spacy_model=spacy_model)

    if not sentences:
        raise RuntimeError(
            f"Annotation produced 0 sentences from {len(paragraphs)} paragraphs "
            f"(filter: {min_words}-{max_words} words). "
            f"Try increasing --max_words (default 12)."
        )

    save_annotations(sentences, cache)
    print(f"  Cached {len(sentences)} annotated sentences -> {cache}")
    return sentences


# -----------------------------------------------------------------------
# Vocabulary helpers
# -----------------------------------------------------------------------

def build_word_list(sentences: List[AnnotatedSentence],
                    max_vocab: int = 2000) -> List[str]:
    """Build a sorted list of the top N most frequent words.

    Returns words sorted alphabetically for deterministic vocab IDs.
    Words not in this list will be mapped to UNK at training time.
    """
    freq: Counter = Counter()
    for sent in sentences:
        freq.update(sent.words)

    # Take top N by frequency, then sort alphabetically
    top_words = [w for w, _ in freq.most_common(max_vocab)]
    top_words.sort()
    return top_words
