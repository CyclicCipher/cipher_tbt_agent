"""agent_loop.py — Symbolic AI experiment runner.

Tasks
-----
  language   (default) — n-gram language modelling on a text corpus.
             Tests whether the symbolic AI can learn fuzzy, statistical tasks.
             Use --hierarchy to enable the 4-stage POS decomposition.

  minecraft  (stub)    — future dxcam / pynput / RCON agent.
             Not yet implemented; see modalities/minecraft.py for plan.

Language modelling experiment (flat bigram mode)
-------------------------------------------------
The engine observes (context, next_word) pairs from a training corpus.
Two prediction mechanisms are compared:

  Markov table   : explicit frequency-count bigram model (perfect baseline)
  Engine ask()   : exact-match lookup or synthesised process in the engine

Key diagnostic questions:
  1. Does KL decrease during training?  (engine is compressing the data)
  2. Does the synthesiser ever fire?    (engine found a reusable process)
  3. What accuracy does the Markov model achieve vs unigram baseline?
  4. For unseen test contexts, does the engine generalise?

Language modelling experiment (--hierarchy mode)
-------------------------------------------------
Uses the 4-stage POS hierarchy from domains/language.ctkg:

  word_pos        : word -> POS tag           (deterministic, synthesisable)
  next_pos        : pos, pos -> next POS       (near-deterministic)
  word_given_pos  : word, word, pos -> word   (distributional, freq_consolidate)
  next_word       : word, word -> word        (composed via CTKG lookup chain)

Key advantage: next_word via the hierarchy generalises to unseen bigram
contexts by routing through POS categories that were seen in training.
Flat bigram returns None for unseen contexts; the hierarchy predicts.

Storage comparison (V=vocab, C=POS categories):
  Flat bigram:   O(V^2) entries - 100M for V=10K
  POS hierarchy: O(V*C + C^2)  - 150K for C=15, ~665x smaller

Run
---
  # Built-in 200-word sample (smoke test, no corpus file needed):
  python agent_loop.py

  # Hierarchy mode (POS decomposition):
  python agent_loop.py --hierarchy

  # Real corpus:
  python agent_loop.py --corpus /path/to/book.txt

  # Trigram context:
  python agent_loop.py --corpus book.txt --context-size 3

  # Resume from checkpoint:
  python agent_loop.py --checkpoint checkpoints/ckpt_latest.json

  # Minecraft stub (not yet implemented):
  python agent_loop.py --task minecraft
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE        = os.path.dirname(os.path.abspath(__file__))
_WORKTREE    = os.path.abspath(os.path.join(_HERE, '..', '..'))
_EXPERIMENTS = os.path.join(_WORKTREE, 'experiments')

sys.path.insert(0, _HERE)
sys.path.insert(0, _EXPERIMENTS)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from engine import SymbolicAI
from modalities.language import LanguageModality

try:
    from ctkg.graph import KnowledgeGraph
    _HAS_CTKG = True
except ImportError:
    _HAS_CTKG = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_EVERY           = 100    # print diagnostics every N training examples
CHECKPOINT_INTERVAL = 1_000  # rolling checkpoint every N steps
MAX_ROLLING_CKPTS   = 5
KL_THRESHOLD        = 0.05   # consolidation trigger
TRAIN_FRAC          = 0.8    # fraction of corpus used for training


# ---------------------------------------------------------------------------
# Graph bootstrap (no .ctkg file needed for language task)
# ---------------------------------------------------------------------------

def _make_language_graph() -> 'KnowledgeGraph':
    """Return a minimal KnowledgeGraph with a single 'next_word' concept."""
    kg = KnowledgeGraph()
    return kg   # 'next_word' concept added dynamically via engine.add_concept()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, ckpt_dir: str) -> None:
        self._dir     = ckpt_dir
        self._rolling: List[str] = []
        os.makedirs(ckpt_dir, exist_ok=True)

    def save_event(self, engine: SymbolicAI, label: str) -> str:
        path = os.path.join(self._dir, f'ckpt_{label}.json')
        engine.save_checkpoint(path)
        print(f'  [ckpt] {path}')
        return path

    def save_rolling(self, engine: SymbolicAI, step: int) -> None:
        path = os.path.join(self._dir, f'ckpt_step_{step:07d}.json')
        engine.save_checkpoint(path)
        self._rolling.append(path)
        while len(self._rolling) > MAX_ROLLING_CKPTS:
            old = self._rolling.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
        import shutil
        try:
            shutil.copy2(path, os.path.join(self._dir, 'ckpt_latest.json'))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Language modelling loop
# ---------------------------------------------------------------------------

def run_language(
    corpus_path:     Optional[str],
    context_size:    int,
    max_steps:       int,
    checkpoint_path: Optional[str],
    checkpoint_dir:  str,
    seed:            int,
    no_checkpoint:   bool,
) -> None:
    random.seed(seed)

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    print('=' * 64)
    print('  Symbolic AI — Language Modelling Experiment')
    print(f'  context_size={context_size}  seed={seed}')
    if corpus_path:
        print(f'  corpus={corpus_path}')
    else:
        print('  corpus=<built-in sample>')
    print('=' * 64)

    if not _HAS_CTKG:
        raise RuntimeError(
            'CTKG imports failed.  Make sure experiments/ctkg/ is on the path.'
        )

    mod = LanguageModality(corpus_path, context_size=context_size)
    print(f'\n[init] Corpus: {mod.corpus_size} tokens, '
          f'{mod.vocab_size} unique words')

    split = mod.split_point(TRAIN_FRAC)
    n_train = split
    n_test  = mod.corpus_size - context_size - split
    print(f'[init] Train: {n_train} examples, Test: {n_test} examples')
    print(f'[init] Unigram baseline accuracy: '
          f'{mod.unigram_baseline_acc():.1%}')

    graph  = _make_language_graph()
    engine = SymbolicAI(graph, modalities=[mod])

    # Add the 'next_word' concept dynamically — input types are N word strings.
    engine.add_concept(
        'next_word',
        domain='language',
        description='predict next word given n-gram context',
        input_type=['word'] * context_size,
        output_type=['word'],
    )

    ckpt_mgr = CheckpointManager(checkpoint_dir) if not no_checkpoint else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'[init] Resuming from: {checkpoint_path}')
        engine.load_checkpoint(checkpoint_path)

    print()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print('[train] Starting...')
    t0              = time.time()
    n_synth         = 0      # times synthesiser fired
    steps_since_ckpt= 0

    for step in range(min(max_steps, n_train)):
        context, next_word = mod.current_example()
        if context is None:
            break

        # Teach both the engine and the Markov table.
        engine.observe(
            'next_word',
            inputs  = context,
            outputs = (next_word,),
            kl_threshold = KL_THRESHOLD,
        )
        mod.record_example(context, next_word)

        mod.advance()
        steps_since_ckpt += 1

        # --- diagnostics ---
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            kl      = engine.kl('next_word')

            # Accuracy of Markov table on the last LOG_EVERY training positions.
            window_start  = max(0, step + 1 - LOG_EVERY)
            window_tokens = mod._tokens[window_start : step + 1 + context_size]
            win_correct, win_total = 0, 0
            for i in range(len(window_tokens) - context_size):
                ctx  = tuple(window_tokens[i : i + context_size])
                true = window_tokens[i + context_size]
                pred = mod.freq_predict(ctx)
                if pred == true:
                    win_correct += 1
                win_total += 1
            freq_acc = win_correct / win_total if win_total else 0.0

            # Mean empirical entropy over all seen contexts.
            mean_h = mod.mean_entropy()

            # Has the synthesiser fired?
            concept = engine.graph.concepts.get('next_word')
            has_process = bool(concept and concept.process)

            print(
                f'  step {step+1:6d}/{n_train}'
                f'  {elapsed:5.1f}s'
                f'  kl={kl:.4f}'
                f'  freq_acc={freq_acc:.1%}'
                f'  H={mean_h:.2f}bits'
                f'  synth={"YES" if has_process else "no"}'
            )

            if has_process and n_synth == 0:
                n_synth += 1
                print(f'  *** Synthesiser fired! Process: '
                      f'{concept.process}')
                if ckpt_mgr:
                    ckpt_mgr.save_event(engine, 'first_synthesis')

        # --- rolling checkpoint ---
        if not no_checkpoint and ckpt_mgr and steps_since_ckpt >= CHECKPOINT_INTERVAL:
            ckpt_mgr.save_rolling(engine, step + 1)
            steps_since_ckpt = 0

    elapsed = time.time() - t0
    print(f'\n[train] Done in {elapsed:.1f}s')

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print('\n[eval] Evaluating on held-out test set...')
    mod.reset(split)

    markov_correct,  markov_total   = 0, 0   # Markov table prediction
    engine_correct,  engine_total   = 0, 0   # engine.ask() prediction
    seen_ctx_correct, seen_ctx_total = 0, 0   # Markov on seen contexts only
    unseen_count                     = 0      # test contexts not in train

    while True:
        context, next_word = mod.current_example()
        if context is None:
            break

        # Markov table prediction (frequency-table baseline)
        markov_pred = mod.freq_predict(context)
        if markov_pred is None:
            unseen_count += 1
        else:
            seen_ctx_total += 1
            if markov_pred == next_word:
                seen_ctx_correct += 1

        markov_total += 1
        if markov_pred == next_word:
            markov_correct += 1

        # Engine prediction (process or exact-match lookup)
        engine_result = engine.ask('next_word', context)
        if engine_result is not None:
            engine_pred = engine_result[0] if isinstance(engine_result, tuple) else engine_result
            engine_total += 1
            if engine_pred == next_word:
                engine_correct += 1

        mod.advance()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print()
    print('=' * 64)
    print('  Results')
    print('=' * 64)
    print(f'  Corpus size          : {mod.corpus_size} tokens')
    print(f'  Vocabulary           : {mod.vocab_size} types')
    print(f'  Context size         : {context_size}-gram')
    print(f'  Training examples    : {n_train}')
    print(f'  Test examples        : {markov_total}')
    print()
    print(f'  Unigram baseline     : {mod.unigram_baseline_acc():.1%}  '
          f'(always predict most common word)')
    if markov_total > 0:
        print(f'  Markov (all test)    : {markov_correct/markov_total:.1%}  '
              f'(bigram freq table, unseen->None)')
    if seen_ctx_total > 0:
        print(f'  Markov (seen ctx)    : {seen_ctx_correct/seen_ctx_total:.1%}  '
              f'({seen_ctx_total} contexts seen in training)')
    print(f'  Unseen test contexts : {unseen_count}  '
          f'({unseen_count/markov_total:.0%} of test set)')
    if engine_total > 0:
        print(f'  Engine ask()         : {engine_correct/engine_total:.1%}  '
              f'(answered {engine_total}/{markov_total} queries)')
    else:
        print(f'  Engine ask()         : no predictions (no matching examples)')
    print()
    print(f'  Synthesiser fired    : {"YES" if n_synth > 0 else "NO"}')
    kl_final = engine.kl('next_word')
    print(f'  Final KL             : {kl_final:.4f}')
    print(f'  Mean context entropy : {mod.mean_entropy():.2f} bits')
    print('=' * 64)

    if not no_checkpoint and ckpt_mgr:
        ckpt_mgr.save_event(engine, 'final')


# ---------------------------------------------------------------------------
# Hierarchy language experiment (Phase M)
# ---------------------------------------------------------------------------

def run_language_hierarchy(
    corpus_path:     Optional[str],
    context_size:    int,
    max_steps:       int,
    checkpoint_path: Optional[str],
    checkpoint_dir:  str,
    seed:            int,
    no_checkpoint:   bool,
) -> None:
    """Run the 4-stage POS hierarchy language modelling experiment.

    Trains each layer of the hierarchy in order:
        1. word_pos        (word -> POS)             deterministic
        2. next_pos        (pos, pos -> next POS)    near-deterministic
        3. word_given_pos  (word, word, pos -> word) distributional
        4. next_word       (word, word -> word)      composed (CTKG process)

    Then evaluates flat bigram vs. hierarchy on the test set.
    Key metric: hierarchy predicts on unseen contexts; flat bigram cannot.
    """
    import os as _os
    random.seed(seed)

    print('=' * 64)
    print('  Symbolic AI — Language Hierarchy Experiment (Phase M)')
    print(f'  context_size={context_size}  seed={seed}')
    if corpus_path:
        print(f'  corpus={corpus_path}')
    else:
        print('  corpus=<built-in sample>')
    print('=' * 64)

    if not _HAS_CTKG:
        raise RuntimeError(
            'CTKG imports failed.  Make sure experiments/ctkg/ is on the path.'
        )

    # Load the language CTKG hierarchy.
    import sys as _sys
    _ctkg_dir = os.path.join(_WORKTREE, 'experiments', 'ctkg')
    if _ctkg_dir not in _sys.path:
        _sys.path.insert(0, _ctkg_dir)
    try:
        from ctkg.parser import parse_file as _parse_file
        lang_ctkg_path = os.path.join(_ctkg_dir, 'domains', 'language.ctkg')
        graph = _parse_file(lang_ctkg_path)
        print(f'[init] Loaded language.ctkg: '
              f'{len(graph.concepts)} concepts, '
              f'{len(graph.prerequisites)} prerequisites')
    except Exception as exc:
        print(f'[warn] Could not load language.ctkg ({exc}). '
              f'Using empty graph.')
        from ctkg.graph import KnowledgeGraph
        graph = KnowledgeGraph()

    mod = LanguageModality(corpus_path, context_size=context_size)
    print(f'\n[init] Corpus: {mod.corpus_size} tokens, '
          f'{mod.vocab_size} unique words')

    # Trigger POS tag build and show distribution.
    pos_dist = mod.pos_coverage()
    print(f'[init] POS distribution: '
          + '  '.join(f'{tag}={cnt}' for tag, cnt in
                      sorted(pos_dist.items(), key=lambda x: -x[1])[:6]))

    split  = mod.split_point(TRAIN_FRAC)
    n_test = mod.corpus_size - context_size - split
    print(f'[init] Train tokens: {split},  Test tokens: {n_test}')
    print()

    engine = SymbolicAI(graph, modalities=[mod])
    ckpt_mgr = CheckpointManager(checkpoint_dir) if not no_checkpoint else None

    # Add hierarchy concepts if not already in graph (fallback for empty graph).
    for cname in ('word_pos', 'next_pos', 'word_given_pos', 'next_word'):
        if cname not in engine.graph.concepts:
            engine.add_concept(cname, domain='language',
                               description=cname.replace('_', ' '))

    # Ensure next_word prerequisite chain exists (for the lookup process).
    existing_prereqs = {(p.source, p.target)
                        for p in engine.graph.prerequisites}
    for src, tgt in [('word_pos', 'next_pos'),
                     ('next_pos', 'word_given_pos'),
                     ('word_given_pos', 'next_word')]:
        if (src, tgt) not in existing_prereqs:
            engine.add_prerequisite(src, tgt)

    # Set the next_word composed process if not already defined.
    nw = engine.graph.concepts.get('next_word')
    if nw and not nw.process:
        nw.process = [
            'p1 = lookup(word_pos, a)',
            'p2 = lookup(word_pos, b)',
            'p3 = lookup(next_pos, p1, p2)',
            'result = lookup(word_given_pos, a, b, p3)',
            'emit(result)',
        ]

    t0 = time.time()

    # ------------------------------------------------------------------
    # Stage 1: Learn word_pos (word -> POS)
    # ------------------------------------------------------------------
    print('[stage 1] Training word_pos (word -> POS tag)...')
    for inputs, outputs in mod.pos_tag_examples(0, split):
        engine.teach('word_pos', inputs, outputs)

    n_wp = engine.example_count('word_pos')
    print(f'  Examples: {n_wp}')

    # Try deterministic synthesis first.
    rule_wp = engine.consolidate('word_pos')
    if rule_wp:
        kl_wp = engine.kl('word_pos')
        print(f'  Synthesis: YES  KL={kl_wp:.4f}')
        print(f'  Rule: {rule_wp}')
    else:
        # Fall back to distributional consolidation.
        engine.freq_consolidate('word_pos')
        kl_wp = engine.kl('word_pos')
        store_wp = engine.stores['word_pos']
        print(f'  Synthesis: NO  -> freq_consolidate  '
              f'KL(entropy)={kl_wp:.4f} bits')

    # Quick accuracy test on training data.
    store_wp = engine.stores.get('word_pos')
    if store_wp:
        correct_wp = sum(
            1 for inp, out in store_wp.examples
            if engine.ask('word_pos', inp) == out
        )
        print(f'  Train accuracy: {correct_wp}/{len(store_wp)} '
              f'= {correct_wp/len(store_wp):.1%}')
    print()

    # ------------------------------------------------------------------
    # Stage 2: Learn next_pos (pos, pos -> next POS)
    # ------------------------------------------------------------------
    print('[stage 2] Training next_pos (POS bigram -> next POS)...')
    for inputs, outputs in mod.next_pos_examples(0, split):
        engine.teach('next_pos', inputs, outputs)

    n_np = engine.example_count('next_pos')
    print(f'  Examples: {n_np}  Unique contexts: '
          f'{len(set(inp for inp, _ in engine.stores["next_pos"].examples))}')

    rule_np = engine.consolidate('next_pos')
    if rule_np:
        kl_np = engine.kl('next_pos')
        print(f'  Synthesis: YES  KL={kl_np:.4f}')
    else:
        engine.freq_consolidate('next_pos')
        kl_np = engine.kl('next_pos')
        store_np = engine.stores['next_pos']
        ent_np = store_np.empirical_entropy()
        print(f'  Synthesis: NO  -> freq_consolidate  '
              f'KL(entropy)={kl_np:.4f}  task_entropy={ent_np:.4f} bits')
        correct_np = sum(
            1 for inp, out in store_np.examples
            if engine.ask('next_pos', inp) == out
        )
        print(f'  Train accuracy (mode): {correct_np}/{len(store_np)} '
              f'= {correct_np/len(store_np):.1%}')
    print()

    # ------------------------------------------------------------------
    # Stage 3: Learn word_given_pos (word, word, pos -> word)
    # ------------------------------------------------------------------
    print('[stage 3] Training word_given_pos (pos, pos, pos -> word)...')
    print('  (POS trigram input -- enables generalisation to unseen word bigrams)')
    for inputs, outputs in mod.word_given_pos_examples(0, split):
        engine.teach('word_given_pos', inputs, outputs)

    n_wgp = engine.example_count('word_given_pos')
    store_wgp = engine.stores['word_given_pos']
    unique_wgp = len(set(inp for inp, _ in store_wgp.examples))
    print(f'  Examples: {n_wgp}  Unique (pos, pos, pos) contexts: {unique_wgp}')

    # This concept is inherently distributional — always freq_consolidate.
    engine.freq_consolidate('word_given_pos')
    kl_wgp = engine.kl('word_given_pos')
    ent_wgp = store_wgp.empirical_entropy()
    print(f'  freq_consolidate  KL(entropy)={kl_wgp:.4f}  '
          f'task_entropy={ent_wgp:.4f} bits')

    correct_wgp = sum(
        1 for inp, out in store_wgp.examples
        if engine.ask('word_given_pos', inp) == out
    )
    print(f'  Train accuracy (mode): {correct_wgp}/{len(store_wgp)} '
          f'= {correct_wgp/len(store_wgp):.1%}')
    print()

    # Also train next_word store for KL tracking (optional flat store).
    print('[flat]  Training flat next_word store for comparison...')
    mod.reset(0)
    for step in range(min(max_steps, split)):
        ctx, nw_word = mod.current_example()
        if ctx is None:
            break
        engine.teach('next_word', ctx, (nw_word,))
        mod.record_example(ctx, nw_word)
        mod.advance()
    engine.freq_consolidate('next_word')
    print(f'  next_word store: {engine.example_count("next_word")} examples  '
          f'KL(entropy)={engine.kl("next_word"):.4f} bits')
    print()

    elapsed = time.time() - t0
    print(f'[train] All stages done in {elapsed:.1f}s')

    # ------------------------------------------------------------------
    # Hierarchy prediction helper
    # ------------------------------------------------------------------
    # The CTKG process chain for next_word uses interpreter lookup().
    # lookup() returns tuples, which don't match the flat-string inputs
    # stored in ExampleStore (interpreter 1-tuple wrapping issue).
    # Work around by routing in Python, unwrapping explicitly.
    # This is equivalent to the CTKG process:
    #   p1 = lookup(word_pos, a)
    #   p2 = lookup(word_pos, b)
    #   p3 = lookup(next_pos, p1, p2)
    #   result = lookup(word_given_pos, p1, p2, p3)
    #   emit(result)
    def _hier_predict(ctx: tuple) -> Optional[str]:
        # Step 1: word -> POS for each context word.
        pos1_t = engine.ask('word_pos', (ctx[0],))
        pos2_t = engine.ask('word_pos', (ctx[1],))
        if pos1_t is None or pos2_t is None:
            return None
        pos1 = pos1_t[0] if isinstance(pos1_t, tuple) else pos1_t
        pos2 = pos2_t[0] if isinstance(pos2_t, tuple) else pos2_t
        # Step 2: POS bigram -> next POS.
        pos3_t = engine.ask('next_pos', (pos1, pos2))
        if pos3_t is None:
            return None
        pos3 = pos3_t[0] if isinstance(pos3_t, tuple) else pos3_t
        # Step 3: POS trigram -> next word.
        word_t = engine.ask('word_given_pos', (pos1, pos2, pos3))
        if word_t is None:
            return None
        return word_t[0] if isinstance(word_t, tuple) else word_t

    # ------------------------------------------------------------------
    # Evaluation: flat bigram vs. hierarchy on test set
    # ------------------------------------------------------------------
    print('\n[eval] Evaluating on held-out test set...')
    mod.reset(split)

    flat_correct = flat_total = 0
    hier_correct = hier_total = 0
    unseen_flat  = 0   # contexts where flat bigram returned None
    hier_on_unseen_correct = hier_on_unseen_total = 0

    while True:
        ctx, true_next = mod.current_example()
        if ctx is None:
            break

        flat_pred = mod.freq_predict(ctx)
        if flat_pred is None:
            unseen_flat += 1
        else:
            flat_total += 1
            if flat_pred == true_next:
                flat_correct += 1

        # Hierarchy prediction via Python routing (bypasses tuple-wrap issue).
        hier_pred = _hier_predict(ctx)
        if hier_pred is not None:
            hier_total += 1
            if hier_pred == true_next:
                hier_correct += 1
            # Track performance specifically on unseen flat contexts.
            if flat_pred is None:
                hier_on_unseen_total += 1
                if hier_pred == true_next:
                    hier_on_unseen_correct += 1

        mod.advance()

    test_total = flat_total + unseen_flat

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print()
    print('=' * 64)
    print('  Hierarchy Experiment Results')
    print('=' * 64)
    print(f'  Corpus        : {mod.corpus_size} tokens, {mod.vocab_size} types')
    print(f'  Context size  : {context_size}-gram')
    print(f'  Train / Test  : {split} / {test_total} examples')
    print()
    print(f'  Unigram baseline  : {mod.unigram_baseline_acc():.1%}  '
          f'(always predict most common word)')
    if flat_total > 0:
        print(f'  Flat bigram       : {flat_correct/flat_total:.1%}  '
              f'(on {flat_total} seen contexts)')
    print(f'  Unseen contexts   : {unseen_flat}  '
          f'({unseen_flat/test_total:.0%} of test)  -- flat returns None')
    print()
    if hier_total > 0:
        print(f'  Hierarchy (all)   : {hier_correct/hier_total:.1%}  '
              f'(answered {hier_total}/{test_total} queries)')
    if hier_on_unseen_total > 0:
        print(f'  Hierarchy (unseen): {hier_on_unseen_correct/hier_on_unseen_total:.1%}  '
              f'({hier_on_unseen_total} contexts unseen by flat bigram)')
        print(f'  *** These are queries the flat model CANNOT answer.')
        print(f'  *** Hierarchy generalises via POS categories.')
    else:
        print(f'  Hierarchy (unseen): N/A (flat model answered all test contexts)')
    print()
    print('  KL by layer (residual entropy after consolidation):')
    print(f'    word_pos       : {engine.kl("word_pos"):.4f} bits  '
          f'(0 = deterministic, >0 = ambiguous words)')
    print(f'    next_pos       : {engine.kl("next_pos"):.4f} bits  '
          f'(irreducible POS bigram entropy)')
    print(f'    word_given_pos : {engine.kl("word_given_pos"):.4f} bits  '
          f'(irreducible lexical entropy per POS)')
    print(f'    next_word(flat): {engine.kl("next_word"):.4f} bits  '
          f'(irreducible flat bigram entropy)')
    print()
    print('  Storage (seen contexts):')
    for cname, desc in [
        ('word_pos',       'word -> POS'),
        ('next_pos',       'POS bigram -> next POS'),
        ('word_given_pos', 'word+word+POS -> word'),
        ('next_word',      'flat bigram -> word'),
    ]:
        store = engine.stores.get(cname)
        n = len(store) if store else 0
        unique = len(set(inp for inp, _ in store.examples)) if store else 0
        print(f'    {cname:20s}: {unique:5d} unique contexts  ({n} examples)')
    print('=' * 64)

    if not no_checkpoint and ckpt_mgr:
        ckpt_mgr.save_event(engine, 'hierarchy_final')


# ---------------------------------------------------------------------------
# Minecraft stub
# ---------------------------------------------------------------------------

def run_minecraft(**kwargs) -> None:
    print('=' * 64)
    print('  Minecraft task — NOT YET IMPLEMENTED')
    print('=' * 64)
    print()
    print('  The Minecraft agent will use:')
    print('    Perception : dxcam DirectX screen capture')
    print('    Drives     : mcrcon RCON queries (health / food / XP)')
    print('    Events     : server log tail (advancements, deaths)')
    print('    Actions    : pynput keyboard + mouse')
    print()
    print('  See modalities/minecraft.py for the implementation plan.')
    print('  See MINEDOJO.md for the full Phase K design document.')
    print()
    print('  To set up the server when ready:')
    print('    pip install dxcam mcrcon pynput pygetwindow')
    print('    java -Xmx2G -jar minecraft_server.1.20.1.jar nogui')
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Symbolic AI experiment runner.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--task', choices=['language', 'minecraft'], default='language',
        help='Which experiment to run.',
    )
    p.add_argument(
        '--corpus', type=str, default=None,
        help='Path to plain-text corpus file (language task only). '
             'Defaults to built-in 200-word sample.',
    )
    p.add_argument(
        '--context-size', type=int, default=2,
        help='N-gram context length (language task only).',
    )
    p.add_argument(
        '--steps', type=int, default=10_000,
        help='Maximum training steps.',
    )
    p.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint JSON to resume from.',
    )
    p.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory for checkpoint files.',
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help='Random seed.',
    )
    p.add_argument(
        '--no-checkpoint', action='store_true',
        help='Disable checkpoint saving.',
    )
    p.add_argument(
        '--hierarchy', action='store_true',
        help='(language task) Use 4-stage POS hierarchy instead of flat bigram. '
             'Trains word_pos -> next_pos -> word_given_pos -> next_word in order. '
             'Demonstrates generalisation to unseen bigram contexts via POS routing.',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = _parse_args()

    if args.task == 'language':
        if args.hierarchy:
            run_language_hierarchy(
                corpus_path     = args.corpus,
                context_size    = args.context_size,
                max_steps       = args.steps,
                checkpoint_path = args.checkpoint,
                checkpoint_dir  = args.checkpoint_dir,
                seed            = args.seed,
                no_checkpoint   = args.no_checkpoint,
            )
        else:
            run_language(
                corpus_path     = args.corpus,
                context_size    = args.context_size,
                max_steps       = args.steps,
                checkpoint_path = args.checkpoint,
                checkpoint_dir  = args.checkpoint_dir,
                seed            = args.seed,
                no_checkpoint   = args.no_checkpoint,
            )
    else:
        run_minecraft()
