"""LanguageModality — text corpus for language-modelling experiments.

Tests whether the symbolic AI can learn hierarchical, statistical prediction
tasks via a latent POS-tag decomposition.

Design
------
The modality tokenises a text corpus and iterates through it one position
at a time, yielding (context, next_word) examples for the engine to observe.

  context   : tuple of the last N words  (N = context_size, default 2)
  next_word : the word that follows in the corpus

The engine's ExampleStore accumulates these pairs.  For flat next_word
prediction the synthesiser cannot fire (same bigram -> many next words).
The hierarchy fixes this via a 4-stage POS decomposition:

    word_pos        : word -> POS tag           (deterministic)
    next_pos        : pos, pos -> next POS       (near-deterministic)
    word_given_pos  : word, word, pos -> word    (distributional)
    next_word       : word, word -> word         (composed via CTKG chain)

See domains/language.ctkg for the CTKG definition of this hierarchy.

Alongside the engine, the modality maintains an explicit frequency table
(bigram / n-gram counts) so we can always compute a baseline Markov-model
prediction for comparison.

POS tagging
-----------
Two backends, selected automatically:
  - Built-in corpus: hardcoded lookup table (no spaCy needed, smoke-test friendly)
  - Real corpus    : spaCy en_core_web_sm if available, else lookup-based fallback

The 9-tag POS set used (simplified from Universal Dependencies):
  DET  NOUN  VERB  ADJ  ADV  PREP  CONJ  PRON  NUM

Usage
-----
  # Built-in 200-word sample corpus (for smoke tests):
  python agent_loop.py --task language

  # Real corpus -- any plain-text file:
  python agent_loop.py --task language --corpus /path/to/book.txt

  # Hierarchy mode (POS decomposition):
  python agent_loop.py --task language --hierarchy

The agent_loop controls the training / evaluation split and calls:
  mod = LanguageModality(corpus_path, context_size)
  context, next_word = mod.current_example()
  mod.advance()
  mod.reset(pos)
  mod.split_point(0.8)  # index of first test token
"""

from __future__ import annotations

import collections
import math
import re
from typing import Dict, Generator, List, Optional, Tuple

from modalities.base import Modality

# Optional spaCy for real-corpus POS tagging.
try:
    import spacy as _spacy
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False


# ---------------------------------------------------------------------------
# Built-in sample corpus
# ---------------------------------------------------------------------------
# A small structured text with clear n-gram patterns, for smoke tests.
# Intentionally repetitive so that even a tiny synthesiser can find signal.

_BUILTIN_CORPUS = """\
the cat sat on the mat the cat sat on a hat the hat is flat the mat is fat
a fat cat sat on a flat mat a flat hat sat on a fat cat the cat likes the mat
the cat likes the hat the cat on the mat sat on the flat hat
the dog ran to the door the dog sat by the door the dog ran past the cat
the cat ran from the dog the dog and the cat sat by the door
a big dog sat by the flat door a small cat sat on the big dog
the dog is big and fast the cat is small and fast the cat ran fast to the mat
the big cat sat the small dog ran the flat mat sat on the fat hat
time flies like an arrow fruit flies like a banana time flies fast
the time is now the time was then the now is the time for the cat
a rose is a rose is a rose the cat is a cat the dog is a dog
one fish two fish red fish blue fish one cat two cats one dog two dogs
the cat in the hat the cat sat in the hat the hat sat on the cat
green eggs and ham i do not like green eggs i do not like them sam i am
in the beginning was the word and the word was good the word was the cat
"""

# ---------------------------------------------------------------------------
# Built-in POS lookup table
# ---------------------------------------------------------------------------
# Covers all ~55 unique words in _BUILTIN_CORPUS.
# Used as the primary backend for built-in corpus POS tagging.
# For ambiguous words (e.g. 'flies' as VERB or NOUN), the dominant
# reading in this corpus is chosen.

_BUILTIN_POS: Dict[str, str] = {
    # Determiners
    'the': 'DET',  'a': 'DET',  'an': 'DET',
    # Nouns
    'cat': 'NOUN', 'cats': 'NOUN', 'dog': 'NOUN', 'dogs': 'NOUN',
    'mat': 'NOUN', 'hat': 'NOUN', 'door': 'NOUN', 'time': 'NOUN',
    'rose': 'NOUN', 'fish': 'NOUN', 'word': 'NOUN', 'arrow': 'NOUN',
    'banana': 'NOUN', 'ham': 'NOUN', 'eggs': 'NOUN',
    'fruit': 'NOUN',   # "fruit flies" — treated as NOUN-NOUN compound
    'flies': 'VERB',   # "time flies" — dominant reading is VERB
    'beginning': 'NOUN', 'sam': 'NOUN', 'now': 'NOUN',  # "the now"
    # Verbs
    'sat': 'VERB', 'ran': 'VERB', 'likes': 'VERB', 'is': 'VERB',
    'was': 'VERB', 'do': 'VERB', 'am': 'VERB',
    # Adjectives
    'flat': 'ADJ', 'fat': 'ADJ', 'big': 'ADJ', 'small': 'ADJ',
    'fast': 'ADJ', 'good': 'ADJ', 'red': 'ADJ', 'blue': 'ADJ',
    'green': 'ADJ',
    # Adverbs
    'not': 'ADV', 'then': 'ADV',
    # Prepositions
    'on': 'PREP', 'to': 'PREP', 'by': 'PREP', 'from': 'PREP',
    'for': 'PREP', 'past': 'PREP', 'in': 'PREP', 'like': 'PREP',
    # Conjunctions
    'and': 'CONJ',
    # Pronouns
    'i': 'PRON', 'them': 'PRON',
    # Numbers
    'one': 'NUM', 'two': 'NUM',
}

# Mapping from spaCy Universal Dependencies POS tags to our 9-tag set.
_SPACY_TO_SIMPLE: Dict[str, str] = {
    'DET':   'DET',
    'NOUN':  'NOUN',
    'PROPN': 'NOUN',   # proper nouns treated as nouns
    'VERB':  'VERB',
    'AUX':   'VERB',   # auxiliary verbs (is, was, am, do)
    'ADJ':   'ADJ',
    'ADV':   'ADV',
    'ADP':   'PREP',   # adpositions (prepositions)
    'CCONJ': 'CONJ',
    'SCONJ': 'CONJ',
    'PRON':  'PRON',
    'NUM':   'NUM',
    'INTJ':  'EXCL',
    'PART':  'PART',
    'X':     'NOUN',   # unknown — default to NOUN
}
_DEFAULT_POS = 'NOUN'   # fallback for unrecognised tokens


# ---------------------------------------------------------------------------
# LanguageModality
# ---------------------------------------------------------------------------

class LanguageModality(Modality):
    """Text corpus modality for language modelling."""

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        context_size: int = 2,
        vocab_max: int = 10_000,
    ) -> None:
        self._context_size = context_size
        self._vocab_max    = vocab_max
        self._using_builtin = (corpus_path is None)

        # Load and tokenise
        if corpus_path:
            with open(corpus_path, encoding='utf-8', errors='replace') as f:
                raw = f.read()
        else:
            raw = _BUILTIN_CORPUS

        self._tokens: List[str] = self._tokenise(raw)
        if len(self._tokens) < context_size + 1:
            raise ValueError(
                f'Corpus too short: {len(self._tokens)} tokens, '
                f'need at least {context_size + 1}.'
            )

        self._build_vocab()
        self._cursor: int = 0

        # POS tags aligned with self._tokens (built lazily).
        self._pos_tags: Optional[List[str]] = None
        self._spacy_nlp = None   # lazy-loaded spaCy model

        # Frequency table for Markov-model baseline.
        # Maps context_tuple -> Counter({next_word: count})
        self._freq: Dict[Tuple, collections.Counter] = collections.defaultdict(
            collections.Counter
        )

        # No effectful primitives — language modelling is purely cognitive.
        self._primitives: Dict = {
            'lm_context_size': lambda: self._context_size,
            'lm_vocab_size':   lambda: self.vocab_size,
            'lm_corpus_size':  lambda: self.corpus_size,
        }

    # ------------------------------------------------------------------
    # Tokenisation and vocabulary
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lowercase + strip punctuation tokeniser."""
        return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())

    def _build_vocab(self) -> None:
        counts = collections.Counter(self._tokens)
        most_common = counts.most_common(self._vocab_max)
        self._inv_vocab: List[str] = ['<UNK>'] + [w for w, _ in most_common]
        self._vocab: Dict[str, int] = {
            w: i for i, w in enumerate(self._inv_vocab)
        }

    # ------------------------------------------------------------------
    # POS tagging (Phase M)
    # ------------------------------------------------------------------

    def _build_pos_tags(self) -> None:
        """Build self._pos_tags aligned with self._tokens.

        Uses:
          1. _BUILTIN_POS lookup table when using the built-in corpus.
          2. spaCy en_core_web_sm when available and corpus_path is set.
          3. Lookup-table heuristic as final fallback.
        """
        if self._using_builtin:
            # Fast path: every token in the built-in corpus is in _BUILTIN_POS.
            self._pos_tags = [
                _BUILTIN_POS.get(tok, _DEFAULT_POS) for tok in self._tokens
            ]
            return

        # Try spaCy for real corpora.
        if _HAS_SPACY and self._spacy_nlp is None:
            try:
                self._spacy_nlp = _spacy.load('en_core_web_sm',
                                               disable=['ner', 'parser'])
            except OSError:
                pass   # model not installed — fall through to heuristic

        if self._spacy_nlp is not None:
            # Re-join tokens (spaCy prefers raw text) but we only have tokens.
            # Process in batches of 5K to avoid memory issues.
            chunk = 5_000
            pos_list: List[str] = []
            for start in range(0, len(self._tokens), chunk):
                batch_text = ' '.join(self._tokens[start:start + chunk])
                doc = self._spacy_nlp(batch_text)
                for tok in doc:
                    if not tok.is_space and not tok.is_punct:
                        tag = _SPACY_TO_SIMPLE.get(tok.pos_, _DEFAULT_POS)
                        pos_list.append(tag)
            # Align length (spaCy may differ by a token on boundaries).
            if len(pos_list) >= len(self._tokens):
                self._pos_tags = pos_list[:len(self._tokens)]
            else:
                # Pad with _DEFAULT_POS if short.
                pos_list += [_DEFAULT_POS] * (len(self._tokens) - len(pos_list))
                self._pos_tags = pos_list
            return

        # Fallback: lookup table for known words, _DEFAULT_POS for unknowns.
        self._pos_tags = [
            _BUILTIN_POS.get(tok, _DEFAULT_POS) for tok in self._tokens
        ]

    def pos_of(self, word: str) -> str:
        """Return the POS tag for a word token.

        For built-in corpus: uses _BUILTIN_POS lookup (deterministic).
        For real corpus: uses spaCy or heuristic lookup.

        Returns _DEFAULT_POS ('NOUN') for unknown words.
        """
        if self._using_builtin:
            return _BUILTIN_POS.get(word, _DEFAULT_POS)
        # For real corpora, use the pre-built per-position tags only at cursor.
        # For isolated word queries, use the lookup table as approximation.
        return _BUILTIN_POS.get(word, _DEFAULT_POS)

    @property
    def pos_tags(self) -> List[str]:
        """POS tags aligned with self._tokens (built lazily on first access)."""
        if self._pos_tags is None:
            self._build_pos_tags()
        return self._pos_tags   # type: ignore[return-value]

    # ------------------------------------------------------------------
    # POS-hierarchy example generators (Phase M)
    # ------------------------------------------------------------------

    def pos_tag_examples(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Generator[Tuple[Tuple[str, ...], Tuple[str, ...]], None, None]:
        """Yield (word,) -> (pos_tag,) examples for the 'word_pos' concept.

        Iterates over tokens[start:end], yielding one example per token.
        Deduplication is handled by the engine's ExampleStore (repeated
        examples for the same word reinforce the frequency count but don't
        change exact-match lookup).
        """
        tags = self.pos_tags
        end = end if end is not None else len(self._tokens)
        for i in range(start, min(end, len(self._tokens))):
            yield (self._tokens[i],), (tags[i],)

    def next_pos_examples(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Generator[Tuple[Tuple[str, ...], Tuple[str, ...]], None, None]:
        """Yield (pos1, pos2) -> (next_pos,) examples for 'next_pos' concept.

        Uses a bigram POS context (matching context_size=2 default).
        For context_size=3, the caller should use context_size-1 POS inputs.
        Always uses a fixed 2-POS context for the POS hierarchy regardless of
        the word context_size (POS sequences are smoother and bigrams suffice).
        """
        tags = self.pos_tags
        end = end if end is not None else len(self._tokens)
        for i in range(start, min(end, len(self._tokens)) - 2):
            pos_ctx   = (tags[i], tags[i + 1])
            next_pos  = (tags[i + 2],)
            yield pos_ctx, next_pos

    def word_given_pos_examples(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Generator[Tuple[Tuple[str, ...], Tuple[str, ...]], None, None]:
        """Yield (pos1, pos2, next_pos) -> (next_word,) examples.

        This is the interface for the 'word_given_pos' concept.  The input
        is a POS TRIGRAM (not a word context), which is the key design
        choice that enables generalisation to unseen word bigrams.

        With 9 POS tags: only 9^3 = 729 possible inputs vs 10K^2 for flat
        word bigrams.  Most POS trigrams are seen during training, so even
        when the word bigram (word1, word2) is new at test time, the POS
        trigram (pos1, pos2, next_pos) was likely seen -> prediction available.

        The distributional question answered: "Given that we just saw a DET
        followed by a NOUN and the next word should be a VERB, which verb is
        most likely?"  This is compressed but meaningful context.
        """
        tags = self.pos_tags
        end = end if end is not None else len(self._tokens)
        ctx = self._context_size
        for i in range(start, min(end, len(self._tokens)) - ctx):
            pos_ctx   = tuple(tags[i : i + ctx])   # (pos1, pos2) bigram
            next_pos  = tags[i + ctx]               # pos of next token
            next_word = self._tokens[i + ctx]
            # Input: POS bigram context + predicted next POS (all POS-level)
            inputs  = pos_ctx + (next_pos,)
            outputs = (next_word,)
            yield inputs, outputs

    # ------------------------------------------------------------------
    # Corpus iteration
    # ------------------------------------------------------------------

    def current_example(self) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
        """Return (context_tuple, next_word) at the current cursor.

        Returns (None, None) when the corpus is exhausted.
        """
        end = self._cursor + self._context_size
        if end >= len(self._tokens):
            return None, None
        context   = tuple(self._tokens[self._cursor:end])
        next_word = self._tokens[end]
        return context, next_word

    def advance(self, n: int = 1) -> None:
        """Move cursor forward by n positions."""
        self._cursor = min(self._cursor + n, len(self._tokens))

    def reset(self, pos: int = 0) -> None:
        """Reset cursor to pos (default: beginning)."""
        self._cursor = max(0, min(pos, len(self._tokens)))

    def split_point(self, train_frac: float = 0.8) -> int:
        """Cursor index that divides training / test.

        Returns the first cursor position whose context falls entirely in
        the test region.
        """
        return int(len(self._tokens) * train_frac) - self._context_size

    # ------------------------------------------------------------------
    # Frequency-table baseline (Markov model)
    # ------------------------------------------------------------------

    def record_example(self, context: Tuple[str, ...], next_word: str) -> None:
        """Update the internal bigram frequency table with one example."""
        self._freq[context][next_word] += 1

    def freq_predict(self, context: Tuple[str, ...]) -> Optional[str]:
        """Return the most frequent next word for this context.

        Returns None if the context has never been seen.
        """
        counter = self._freq.get(context)
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def freq_top_k(
        self, context: Tuple[str, ...], k: int = 5
    ) -> List[str]:
        """Return top-k predicted next words for this context."""
        counter = self._freq.get(context)
        if not counter:
            return []
        return [w for w, _ in counter.most_common(k)]

    def empirical_entropy(self, context: Tuple[str, ...]) -> float:
        """Shannon entropy H(next | context) in bits.

        Returns 0 if the context is unseen.
        """
        counter = self._freq.get(context)
        if not counter:
            return 0.0
        total = sum(counter.values())
        h = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def mean_entropy(self, contexts: Optional[List[Tuple]] = None) -> float:
        """Mean empirical entropy over all seen contexts (or a given list)."""
        targets = contexts if contexts is not None else list(self._freq.keys())
        if not targets:
            return 0.0
        return sum(self.empirical_entropy(c) for c in targets) / len(targets)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def unigram_baseline_acc(self) -> float:
        """Accuracy of always predicting the most common word (unigram LM)."""
        if not self._tokens:
            return 0.0
        most_common_word = collections.Counter(self._tokens).most_common(1)[0][0]
        hits = sum(1 for t in self._tokens if t == most_common_word)
        return hits / len(self._tokens)

    def pos_coverage(self) -> Dict[str, int]:
        """Count of each POS tag in the corpus (diagnostics)."""
        return dict(collections.Counter(self.pos_tags))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        return len(self._tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._inv_vocab)

    @property
    def context_size(self) -> int:
        return self._context_size

    @property
    def cursor(self) -> int:
        return self._cursor

    # ------------------------------------------------------------------
    # Modality interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'language'

    @property
    def primitives(self) -> Dict:
        return self._primitives

    # Language modelling has no metabolic drives.
    def U_pain(self) -> float:   return 0.0
    def U_hunger(self) -> float: return 0.0
    def U_sleep(self) -> float:  return 0.0

    def current_priority(self, engine) -> Tuple[str, float, str]:
        return 'LEARN', 1.0, 'next_word'
