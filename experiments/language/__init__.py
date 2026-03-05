"""WikiText-2 data loading and annotation for syntax curriculum learning."""

from .wikitext2 import (
    AnnotatedSentence,
    load_wikitext2,
    annotate_sentences,
    load_or_annotate,
    build_word_list,
)

__all__ = [
    'AnnotatedSentence',
    'load_wikitext2',
    'annotate_sentences',
    'load_or_annotate',
    'build_word_list',
]
