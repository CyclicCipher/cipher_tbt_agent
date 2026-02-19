"""Syntax parsing generators for the scratchpad framework.

Each generator corresponds to a CTKG concept in the English syntax domain.
All stages use the same sentence pool — the output format changes between
stages, while the model's learned representations carry forward.

Stages:
  1. POS tagging       — word → POS tag per word
  2. NP chunking       — word → BIO-NP tag per word
  3. PP chunking       — word → BIO-PP tag per word
  4. VP chunking       — word → BIO-VP tag per word
  5. Clause structure   — word → SUBJ/PRED/OTHER per word

All outputs are padded to max_words with STOP tokens for fixed n_result.

Requires annotated sentences from experiments.language.wikitext2.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Set

from ..framework import Problem, ProblemGenerator, Step, Vocab


# -----------------------------------------------------------------------
# POS tag set (matches universal_syntax.ctkg)
# -----------------------------------------------------------------------

POS_TAGS = [
    'N', 'V', 'ADJ', 'ADV', 'DET', 'ADP', 'CONJ', 'COMP',
    'PRON', 'AUX', 'NUM', 'PART', 'INTJ', 'UNK',
]

# BIO chunk tags
NP_TAGS = ['B_NP', 'I_NP', 'O_NP']
PP_TAGS = ['B_PP', 'I_PP', 'O_PP']
VP_TAGS = ['B_VP', 'I_VP', 'O_VP']

# Clause structure tags
CLAUSE_TAGS = ['SUBJ', 'PRED', 'OTHER']

# All grammar tokens (added to vocab in deterministic order)
ALL_GRAMMAR_TOKENS = (
    POS_TAGS + NP_TAGS + PP_TAGS + VP_TAGS + CLAUSE_TAGS + ['STOP']
)


def setup_syntax_vocab(vocab: Vocab, word_list: List[str]) -> None:
    """Register all syntax tokens in deterministic order.

    Must be called BEFORE any generator's generate() method to ensure
    consistent token IDs across train and test splits.

    Args:
        vocab: shared vocabulary
        word_list: sorted list of in-vocabulary words
    """
    # Grammar tokens first (fixed, small set)
    for tok in ALL_GRAMMAR_TOKENS:
        vocab.add(tok)

    # UNK_WORD for out-of-vocabulary words
    vocab.add('UNK_WORD')

    # Word tokens (sorted alphabetically for determinism)
    for w in word_list:
        vocab.add(w)


def _word_to_id(word: str, vocab: Vocab, word_set: Set[str]) -> int:
    """Map a word to its vocab ID, or UNK_WORD if not in vocabulary."""
    if word in word_set:
        return vocab[word]
    return vocab['UNK_WORD']


# -----------------------------------------------------------------------
# BIO tag conversion helpers
# -----------------------------------------------------------------------

def _spans_to_bio(n_words: int, spans: List[List[int]],
                  b_tag: str, i_tag: str, o_tag: str) -> List[str]:
    """Convert a list of [start, end) spans to BIO tag sequence."""
    tags = [o_tag] * n_words
    for span in spans:
        start, end = span[0], span[1]
        if start < n_words:
            tags[start] = b_tag
            for k in range(start + 1, min(end, n_words)):
                tags[k] = i_tag
    return tags


# -----------------------------------------------------------------------
# Base class for syntax generators
# -----------------------------------------------------------------------

class _SyntaxGeneratorBase(ProblemGenerator):
    """Base class for syntax generators with shared sentence pool logic."""

    def __init__(self, sentences: List[Any], max_words: int = 12,
                 word_list: Optional[List[str]] = None):
        """
        Args:
            sentences: list of AnnotatedSentence objects
            max_words: pad/truncate output to this length
            word_list: sorted vocabulary list (for UNK mapping)
        """
        self._sentences = sentences
        self._max_words = max_words
        self._word_set = set(word_list) if word_list else set()
        self._word_list = word_list or []

    def enumerate_all(self) -> List[Any]:
        """Each sentence index is a unique problem spec."""
        return list(range(len(self._sentences)))

    def _input_tokens(self, sent: Any, vocab: Vocab) -> List[int]:
        """Convert sentence words to input token IDs."""
        words = sent.words[:self._max_words]
        return [_word_to_id(w, vocab, self._word_set) for w in words]

    def _pad_tags(self, tags: List[str], vocab: Vocab) -> List[int]:
        """Convert tag strings to IDs, pad with STOP to max_words."""
        n = len(tags)
        ids = [vocab[t] for t in tags[:self._max_words]]
        ids += [vocab['STOP']] * (self._max_words - n)
        return ids


# -----------------------------------------------------------------------
# Stage 1: POS Tagging
# -----------------------------------------------------------------------

class PosTagGenerator(_SyntaxGeneratorBase):
    """Stage 1: Assign POS tags to words.

    Input:  the big cat sat
    Output: DET ADJ N V STOP STOP ... (padded to max_words)

    Corresponds to CTKG concept: en_lexical_category
    """

    @property
    def name(self) -> str:
        return 'pos_tagging'

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        setup_syntax_vocab(vocab, self._word_list)
        problems = []
        for _ in range(n_samples):
            idx = random.choice(specs)
            sent = self._sentences[idx]
            input_toks = self._input_tokens(sent, vocab)
            tag_ids = self._pad_tags(sent.pos_tags, vocab)

            problems.append(Problem(
                question=input_toks,
                steps=[Step('pos_tags', tag_ids, weight=1.0)],
                metadata={'sentence_idx': idx},
            ))
        return problems


# -----------------------------------------------------------------------
# Stage 2: NP Chunking
# -----------------------------------------------------------------------

class NpChunkGenerator(_SyntaxGeneratorBase):
    """Stage 2: Identify NP boundaries with BIO tags.

    Input:  the big cat sat
    Output: B_NP I_NP I_NP O_NP STOP STOP ... (padded to max_words)

    Corresponds to CTKG concept: en_np_structure
    """

    @property
    def name(self) -> str:
        return 'np_chunking'

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        setup_syntax_vocab(vocab, self._word_list)
        problems = []
        for _ in range(n_samples):
            idx = random.choice(specs)
            sent = self._sentences[idx]
            input_toks = self._input_tokens(sent, vocab)

            n = min(len(sent.words), self._max_words)
            bio_tags = _spans_to_bio(n, sent.np_spans, 'B_NP', 'I_NP', 'O_NP')
            tag_ids = self._pad_tags(bio_tags, vocab)

            problems.append(Problem(
                question=input_toks,
                steps=[Step('np_chunks', tag_ids, weight=1.0)],
                metadata={'sentence_idx': idx},
            ))
        return problems


# -----------------------------------------------------------------------
# Stage 3: PP Chunking
# -----------------------------------------------------------------------

class PpChunkGenerator(_SyntaxGeneratorBase):
    """Stage 3: Identify PP boundaries with BIO tags.

    Input:  sat on the mat
    Output: O_PP B_PP I_PP I_PP STOP STOP ... (padded to max_words)

    Corresponds to CTKG concept: en_pp_structure
    """

    @property
    def name(self) -> str:
        return 'pp_chunking'

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        setup_syntax_vocab(vocab, self._word_list)
        problems = []
        for _ in range(n_samples):
            idx = random.choice(specs)
            sent = self._sentences[idx]
            input_toks = self._input_tokens(sent, vocab)

            n = min(len(sent.words), self._max_words)
            bio_tags = _spans_to_bio(n, sent.pp_spans, 'B_PP', 'I_PP', 'O_PP')
            tag_ids = self._pad_tags(bio_tags, vocab)

            problems.append(Problem(
                question=input_toks,
                steps=[Step('pp_chunks', tag_ids, weight=1.0)],
                metadata={'sentence_idx': idx},
            ))
        return problems


# -----------------------------------------------------------------------
# Stage 4: VP Chunking
# -----------------------------------------------------------------------

class VpChunkGenerator(_SyntaxGeneratorBase):
    """Stage 4: Identify VP boundaries with BIO tags.

    Input:  the cat chased the dog
    Output: O_VP O_VP B_VP I_VP I_VP STOP ... (padded to max_words)

    Corresponds to CTKG concept: en_vp_structure
    """

    @property
    def name(self) -> str:
        return 'vp_chunking'

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        setup_syntax_vocab(vocab, self._word_list)
        problems = []
        for _ in range(n_samples):
            idx = random.choice(specs)
            sent = self._sentences[idx]
            input_toks = self._input_tokens(sent, vocab)

            n = min(len(sent.words), self._max_words)
            bio_tags = _spans_to_bio(n, sent.vp_spans, 'B_VP', 'I_VP', 'O_VP')
            tag_ids = self._pad_tags(bio_tags, vocab)

            problems.append(Problem(
                question=input_toks,
                steps=[Step('vp_chunks', tag_ids, weight=1.0)],
                metadata={'sentence_idx': idx},
            ))
        return problems


# -----------------------------------------------------------------------
# Stage 5: Clause Structure
# -----------------------------------------------------------------------

class ClauseStructureGenerator(_SyntaxGeneratorBase):
    """Stage 5: Identify subject and predicate regions.

    Input:  the cat chased the dog
    Output: SUBJ SUBJ PRED PRED PRED STOP ... (padded to max_words)

    Corresponds to CTKG concept: en_simple_clause
    """

    @property
    def name(self) -> str:
        return 'clause_structure'

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        setup_syntax_vocab(vocab, self._word_list)
        problems = []
        for _ in range(n_samples):
            idx = random.choice(specs)
            sent = self._sentences[idx]
            input_toks = self._input_tokens(sent, vocab)

            n = min(len(sent.words), self._max_words)
            tags = ['OTHER'] * n

            # Mark subject span
            if sent.subj_span:
                s, e = sent.subj_span[0], sent.subj_span[1]
                for k in range(s, min(e, n)):
                    tags[k] = 'SUBJ'

            # Mark predicate span
            if sent.pred_span:
                s, e = sent.pred_span[0], sent.pred_span[1]
                for k in range(s, min(e, n)):
                    tags[k] = 'PRED'

            tag_ids = self._pad_tags(tags, vocab)

            problems.append(Problem(
                question=input_toks,
                steps=[Step('clause_struct', tag_ids, weight=1.0)],
                metadata={'sentence_idx': idx},
            ))
        return problems
