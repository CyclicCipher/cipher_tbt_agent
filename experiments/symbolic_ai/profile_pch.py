#!/usr/bin/env python3
"""
profile_pch.py — CPU + memory profiler for PredictiveCodingHierarchy.

Usage:
    python profile_pch.py [--books N]   (default: 2 books)

Outputs:
  - Per-book timing and tokens/s
  - Memory snapshots (tracemalloc) at end of each book
  - Per data-structure deep-size breakdown
  - cProfile top-25 hotspots at the end
"""
import argparse
import cProfile
import gc
import glob
import io
import os
import pstats
import sys
import threading
import time
import tracemalloc

_parser = argparse.ArgumentParser()
_parser.add_argument('--books', type=int, default=2)
_args, _ = _parser.parse_known_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_T0 = time.perf_counter()

def elapsed() -> float:
    return time.perf_counter() - _T0

def stamp(label: str, extra: str = '') -> None:
    print(f'  [{elapsed():6.2f}s]  {label}  {extra}', flush=True)

def _mb(n): return f'{n/1024**2:.1f} MB'

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _deep_size(obj, seen=None):
    """Rough deep-size estimate without external libs."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        size += sum(_deep_size(i, seen) for i in obj)
    elif hasattr(obj, '__dict__'):
        size += _deep_size(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):
        size += sum(_deep_size(getattr(obj, s, None), seen)
                    for s in obj.__slots__ if hasattr(obj, s))
    return size


def memory_snapshot(pch, label=''):
    gc.collect()
    snap = tracemalloc.take_snapshot()
    top = snap.statistics('lineno')
    tm_total = sum(s.size for s in top)

    print(f'\n=== Memory snapshot: {label} — tracemalloc total {_mb(tm_total)} ===')

    # per data-structure deep sizes
    rows = []
    rows.append(('vocab._by_surface', _deep_size(pch.vocab._by_surface)))
    rows.append(('vocab._merges',     _deep_size(pch.vocab._merges)))
    rows.append(('vocab._segments',   _deep_size(pch.vocab._segments)))
    for lvl, lrn in enumerate(pch.learners):
        for attr in ('_atom_counts', '_rel_counts', '_atom_bigrams', '_atom_totals', '_rel_totals'):
            d = getattr(lrn, attr, None) or {}
            if d:
                rows.append((f'L{lvl}.{attr}', _deep_size(d)))
        h = pch._surp_hist[lvl]
        rows.append((f'L{lvl}._surp_hist', _deep_size(h)))
    # chunk sequences (needed for R0-R6 sense splitter)
    for lvl, seqs in enumerate(pch._chunk_seqs):
        if seqs:
            rows.append((f'L{lvl}._chunk_seqs', _deep_size(seqs)))
    # surface string lengths by level (to understand exponential growth)
    surf_by_level: dict = {}
    for seg in pch.vocab._segments:
        lv = getattr(seg, 'level', 1)
        surf_by_level.setdefault(lv, []).append(len(seg.surface))
    for lv, lens in sorted(surf_by_level.items()):
        avg = sum(lens) / len(lens) if lens else 0
        rows.append((f'L{lv}.segment_surfaces(n={len(lens)},avg={avg:.0f}ch)', sum(lens)))

    rows.sort(key=lambda r: -r[1])
    ds_total = sum(r[1] for r in rows)
    print(f'  Data-struct total (deep_size): {_mb(ds_total)}')
    for name, sz in rows[:15]:
        if sz > 1024:
            print(f'    {name:<40} {_mb(sz):>10}')

    print('  tracemalloc top 8 allocators:')
    for stat in top[:8]:
        print(f'    {stat}')

    # vocab / learner quick stats
    print(f'  vocab: {pch.vocab.n_merges()} merges, {pch.vocab.n_segments()} segments, '
          f'{len(pch.vocab._by_surface)} surface entries')
    for lvl, lrn in enumerate(pch.learners):
        ac = getattr(lrn, '_atom_counts', {}) or {}
        rc = getattr(lrn, '_rel_counts',  {}) or {}
        if ac or rc:
            # count total Counter entries
            ac_entries = sum(len(v) for v in ac.values())
            rc_entries = sum(len(v) for v in rc.values())
            print(f'  L{lvl}: _atom_counts {len(ac)} keys / {ac_entries} entries; '
                  f'_rel_counts {len(rc)} keys / {rc_entries} entries')


# ---------------------------------------------------------------------------
# cProfile
# ---------------------------------------------------------------------------
_pr = cProfile.Profile()

def _print_cprofile():
    print(f'\n=== cProfile top 25 (cumtime) ===')
    s = io.StringIO()
    try:
        ps = pstats.Stats(_pr, stream=s).sort_stats('cumulative')
        ps.print_stats(25)
        print(s.getvalue(), flush=True)
    except Exception as e:
        print(f'  (stats unavailable: {e})')

# ---------------------------------------------------------------------------
# Load sequences
# ---------------------------------------------------------------------------
CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'GT4HistOCR', 'corpus', 'EarlyModernLatin',
)

stamp('load_sequences — start')
from char_latin import load_sequences   # noqa: E402
sequences = load_sequences(CORPUS_DIR, n_books=_args.books)
total_chars = sum(len(s) for s in sequences)
stamp('load_sequences — done', f'{len(sequences)} books, {total_chars:,} chars')

# ---------------------------------------------------------------------------
# PCH setup + run
# ---------------------------------------------------------------------------
from relational_pipeline import PredictiveCodingHierarchy   # noqa: E402

tracemalloc.start()

pch = PredictiveCodingHierarchy(
    n_levels=5,
    max_chunk_size=7,
    adaptive_threshold=True,
    surprise_k=0.5,
    min_tokens_active=20,
)

_pr.enable()
book_speeds = []

for i, seq in enumerate(sequences):
    t_book = time.perf_counter()
    pch._reset_buffers()
    for token in seq:
        pch._process_level(0, str(token))
    for lvl in range(pch.n_levels):
        pch._emit_buffer(lvl)
    dt = time.perf_counter() - t_book
    speed = len(seq) / dt
    book_speeds.append(speed)
    stamp(f'book {i+1}/{len(sequences)}',
          f'{len(seq):,} chars  {dt:.2f}s  {speed:,.0f} char/s  '
          f'L0:{pch._seen[0]:,}  L1:{pch._seen[1]:,}  L2:{pch._seen[2]:,}')
    memory_snapshot(pch, label=f'after book {i+1}')

_pr.disable()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
avg_speed = sum(book_speeds) / len(book_speeds) if book_speeds else 0
print(f'\n  Average speed: {avg_speed:,.0f} char/s across {len(book_speeds)} books')
pch.level_summary()
_print_cprofile()
tracemalloc.stop()
print(f'\nTotal elapsed: {elapsed():.2f}s', flush=True)
