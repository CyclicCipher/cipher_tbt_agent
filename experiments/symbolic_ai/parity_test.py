"""parity_test.py — Parity test: symbolic category chain vs 2-layer transformer.

Trains a minimal causal-LM transformer (d=64, 2 heads, 2 layers) on the same
EarlyModernLatin corpus as language_pipeline.py, then compares per-token
log-likelihood on:
  (a) all test trigrams
  (b) test trigrams where (w2, w3) was NOT seen in training  ← key test

The symbolic model is built inline (re-uses language_pipeline functions) or
loaded from a saved checkpoint (--load).

Thesis: on UNSEEN word pairs the category chain should match or exceed a
2-layer transformer, because category transitions generalise via syntactic
structure rather than word co-occurrence memorisation.

Usage:
    # Full comparison (train both models from scratch)
    python parity_test.py --corpus EarlyModernLatin --n_train 5000

    # Load saved symbolic checkpoint (skips slow discovery step)
    python parity_test.py --corpus EarlyModernLatin --n_train 5000 --load chain.pkl

    # Quick test on small corpus
    python parity_test.py --corpus EarlyModernLatin --n_train 1000 --epochs 10

    # Skip transformer training (symbolic side only)
    python parity_test.py --corpus EarlyModernLatin --no_transformer
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

import io
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', 'utf-8').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from discover_structure import run_pipeline, _stream_texts, _banner, _DATA_DIR
from language_pipeline import (
    build_assignment, train_chain,
    build_trigram_assignment, train_chain_ctx,
    build_cluster_succ_dists, build_cluster_similarity_matrix,
    precompute_dist_cache, precompute_all_soft_dists,
    tune_mixture_alpha,
    evaluate_all, logprob_chain, logprob_flat,
    logprob_chain_ctx, logprob_chain_e3,
    predict_chain_e3,
)

try:
    from ocr_test import find_pairs
except ImportError:
    import glob as _glob
    def find_pairs(d, max_n=None, shuffle=False, seed=0):
        pairs = []
        for gt in _glob.glob(os.path.join(d, '**', '*.gt.txt'), recursive=True):
            pairs.append((None, gt))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(pairs)
        return pairs[:max_n] if max_n else pairs


# ---------------------------------------------------------------------------
# Transformer implementation
# ---------------------------------------------------------------------------

def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class Vocabulary:
    """Simple word↔index vocabulary with UNK."""

    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self):
        self._w2i: dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self._i2w: list[str] = [self.PAD, self.UNK]

    def add(self, word: str) -> int:
        if word not in self._w2i:
            self._w2i[word] = len(self._i2w)
            self._i2w.append(word)
        return self._w2i[word]

    def __getitem__(self, word: str) -> int:
        return self._w2i.get(word, 1)  # 1 = UNK

    def __len__(self) -> int:
        return len(self._i2w)

    def decode(self, idx: int) -> str:
        return self._i2w[idx] if idx < len(self._i2w) else self.UNK


def build_vocab(train_texts: list[str], min_count: int = 1) -> Vocabulary:
    """Build vocabulary from training texts."""
    freq: collections.Counter = collections.Counter()
    for text in train_texts:
        freq.update(text.split())
    vocab = Vocabulary()
    for word, count in freq.items():
        if count >= min_count:
            vocab.add(word)
    return vocab


def texts_to_trigrams(texts: list[str], vocab: Vocabulary):
    """Convert texts to (w1_idx, w2_idx, w3_idx) trigrams."""
    trigrams = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            trigrams.append((
                vocab[tokens[i]],
                vocab[tokens[i + 1]],
                vocab[tokens[i + 2]],
            ))
    return trigrams


def build_transformer(vocab_size: int, d_model: int = 64, n_heads: int = 2,
                      n_layers: int = 2, d_ff: int = 256,
                      context_len: int = 4, dropout: float = 0.1):
    """Build a minimal causal-LM transformer.

    Architecture:
        - Token embedding: vocab_size → d_model
        - Sinusoidal positional encoding (no learned params)
        - n_layers × TransformerEncoderLayer (causal mask, d_model, n_heads, d_ff)
        - Linear head: d_model → vocab_size

    Returns a torch.nn.Module (or raises ImportError if torch unavailable).

    Context window = context_len tokens.  For trigram comparison we only need
    3 tokens, but the LM is trained on full sequences for richer supervision.
    """
    torch = _try_import_torch()
    if torch is None:
        raise ImportError('PyTorch is required for the transformer comparison.')
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d: int, max_len: int = 512):
            super().__init__()
            pos = torch.arange(max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
            pe = torch.zeros(max_len, d)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:d // 2])
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:x.size(1)]

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed    = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_enc  = PositionalEncoding(d_model, max_len=context_len + 4)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            nn.init.normal_(self.embed.weight, std=0.02)
            nn.init.zeros_(self.head.weight)

        def forward(self, x, causal_mask=None):
            # x: (B, T)
            T = x.size(1)
            if causal_mask is None:
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    T, device=x.device
                )
            h = self.pos_enc(self.embed(x))
            h = self.transformer(h, mask=causal_mask, is_causal=True)
            return self.head(h)   # (B, T, V)

    return CausalLM()


def train_transformer_model(
    train_texts: list[str],
    vocab: Vocabulary,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    context_len: int = 4,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-3,
    verbose: bool = True,
) -> object:
    """Train the causal-LM transformer on train_texts.

    Training signal: next-token prediction (standard LM objective).
    Each token in the sequence predicts the next, giving length-1 supervision
    per token — much denser than trigram-only supervision.

    Returns the trained model (or None if torch unavailable).
    """
    torch = _try_import_torch()
    if torch is None:
        print('  WARNING: PyTorch not available — skipping transformer.')
        return None

    _banner(f'Training 2-layer transformer (d={d_model}, {n_heads} heads, '
            f'{epochs} epochs, context={context_len})')

    # Convert training texts to token-index sequences.
    sequences = []
    for text in train_texts:
        tokens = text.split()
        if len(tokens) < 2:
            continue
        idxs = [vocab[w] for w in tokens]
        # Chunk into context_len+1 windows (input + target).
        for start in range(0, len(idxs) - context_len, context_len // 2):
            chunk = idxs[start: start + context_len + 1]
            if len(chunk) == context_len + 1:
                sequences.append(chunk)

    if not sequences:
        print('  ERROR: no training sequences produced.')
        return None

    if verbose:
        print(f'  Training sequences: {len(sequences):,}')
        print(f'  Vocabulary size:    {len(vocab):,}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'  Device: {device}')

    model = build_transformer(
        vocab_size=len(vocab), d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_model * 4, context_len=context_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f'  Parameters: {n_params:,}')

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    data = torch.tensor(sequences, dtype=torch.long)  # (N, context+1)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(data))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[perm[i: i + batch_size]].to(device)
            x, y = batch[:, :-1], batch[:, 1:]  # (B, T)

            logits = model(x)                    # (B, T, V)
            loss   = criterion(
                logits.reshape(-1, len(vocab)),
                y.reshape(-1),
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs):
            avg_loss = epoch_loss / max(n_batches, 1)
            ppl      = 2 ** avg_loss
            elapsed  = time.time() - t0
            print(f'  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.3f}  '
                  f'PPL={ppl:.1f}  ({elapsed:.1f}s)')

    model.eval()
    return model


def eval_transformer_trigrams(
    model,
    test_texts: list[str],
    train_pairs: set[tuple[str, str]],
    vocab: Vocabulary,
    verbose: bool = True,
) -> dict:
    """Evaluate transformer on trigram prediction task.

    For each test trigram (w1, w2, w3):
        - Feed [w1, w2] as context to the transformer.
        - Extract P(w3 | w1, w2) from the output at position 1 (after w2).
        - Record accuracy (argmax == w3) and log-probability.

    This is an apples-to-apples comparison with the category chain, which also
    uses (w1, w2) as context.  The transformer can additionally use any learned
    position-specific patterns.
    """
    torch = _try_import_torch()
    if torch is None or model is None:
        return {}

    import torch.nn.functional as F

    _banner('Evaluating transformer on trigram task')

    device = next(model.parameters()).device
    model.eval()

    r = dict(
        total=0, tf_correct=0, tf_answered=0,
        unseen_total=0, tf_unseen_correct=0, tf_unseen_answered=0,
        tf_logloss=0.0, tf_logloss_n=0,
    )

    def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'

    with torch.no_grad():
        for text in test_texts:
            tokens = text.split()
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
                is_unseen = (w2, w3) not in train_pairs

                r['total'] += 1
                if is_unseen:
                    r['unseen_total'] += 1

                idx1 = vocab[w1]
                idx2 = vocab[w2]
                idx3 = vocab[w3]

                # Context: [w1, w2] → predict at position 1 (= after w2).
                x = torch.tensor([[idx1, idx2]], dtype=torch.long, device=device)
                logits = model(x)              # (1, 2, V)
                log_probs = F.log_softmax(logits[0, 1], dim=-1)   # (V,)

                pred_idx = log_probs.argmax().item()
                pred_word = vocab.decode(pred_idx)

                r['tf_answered'] += 1
                if pred_word == w3:
                    r['tf_correct'] += 1
                    if is_unseen:
                        r['tf_unseen_correct'] += 1
                if is_unseen:
                    r['tf_unseen_answered'] += 1

                lp = log_probs[idx3].item() / math.log(2)  # convert to log₂
                r['tf_logloss'] -= lp
                r['tf_logloss_n'] += 1

    if verbose:
        T  = r['total']
        UT = r['unseen_total']
        TA = r['tf_answered']
        TC = r['tf_correct']
        TUA = r['tf_unseen_answered']
        TUC = r['tf_unseen_correct']
        ppl = 2 ** (r['tf_logloss'] / max(r['tf_logloss_n'], 1))

        print(f'\n  Test trigrams:    {T:,}  (unseen pairs: {UT:,} = {pct(UT, T)})')
        print(f'  Coverage:         {pct(TA, T)} (transformer always answers)')
        print(f'  Top-1 accuracy:   {pct(TC, TA)} (of answered)')
        print(f'  Unseen accuracy:  {pct(TUC, TUA)} (of answered unseen)')
        print(f'  Perplexity:       {ppl:.1f}')

    return r


# ---------------------------------------------------------------------------
# Joint comparison table
# ---------------------------------------------------------------------------

def print_comparison(
    symbolic_results: dict,
    tf_results: dict,
    run_e3: bool = False,
    verbose: bool = True,
) -> None:
    """Print a unified comparison table: flat | E1 | E3 | Transformer."""
    _banner('Parity Test Results: Symbolic Chain vs Transformer')

    def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'
    def ppl_str(loss, n): return f'{2**(loss/n):.1f}' if n else 'N/A'

    sr = symbolic_results  # from evaluate_all()
    tr = tf_results        # from eval_transformer_trigrams()
    has_tf = bool(tr)
    has_e3 = run_e3 and 'e3_logloss' in sr

    T  = sr['total']
    UT = sr['unseen_total']

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    col_names = ['Flat bigram', 'E1 chain']
    if has_e3:
        col_names.append('E3 soft')
    if has_tf:
        col_names.append('Transformer')

    w = 12
    lbl_w = 36

    def row(label, *vals):
        return f'  {label:<{lbl_w}}' + ''.join(f'{v:>{w}}' for v in vals)

    header = row('Metric', *col_names)
    sep    = '  ' + '-' * (lbl_w + w * len(col_names))

    print(f'\n  Test trigrams: {T:,}   Unseen pairs: {UT:,} ({pct(UT, T)})\n')
    print(header)
    print(sep)

    # Coverage
    flat_cov  = pct(sr['flat_answered'], T)
    chain_cov = pct(sr['chain_answered'], T)
    e3_cov    = pct(sr['e3_answered'], T) if has_e3 else None
    tf_cov    = pct(tr.get('tf_answered', 0), T) if has_tf else None
    vals = [flat_cov, chain_cov]
    if has_e3: vals.append(e3_cov)
    if has_tf: vals.append(tf_cov)
    print(row('Coverage (answered/total)', *vals))

    # Overall accuracy (of answered)
    flat_acc  = pct(sr['flat_correct'],  sr['flat_answered'])
    chain_acc = pct(sr['chain_correct'], sr['chain_answered'])
    e3_acc    = pct(sr['e3_correct'],   sr['e3_answered'])   if has_e3 else None
    tf_acc    = pct(tr.get('tf_correct', 0), tr.get('tf_answered', 1)) if has_tf else None
    vals = [flat_acc, chain_acc]
    if has_e3: vals.append(e3_acc)
    if has_tf: vals.append(tf_acc)
    print(row('Top-1 accuracy (of answered)', *vals))

    # Perplexity
    flat_ppl  = ppl_str(sr['flat_logloss'],  sr['flat_logloss_n'])
    chain_ppl = ppl_str(sr['chain_logloss'], sr['chain_logloss_n'])
    e3_ppl    = ppl_str(sr['e3_logloss'],    sr['e3_logloss_n'])  if has_e3 else None
    tf_ppl    = ppl_str(tr.get('tf_logloss', 0), tr.get('tf_logloss_n', 1)) if has_tf else None
    vals = [flat_ppl, chain_ppl]
    if has_e3: vals.append(e3_ppl)
    if has_tf: vals.append(tf_ppl)
    print(row('Perplexity (lower=better)', *vals))

    print()
    print(f'  {"=== UNSEEN WORD PAIRS (key test) ===":<{lbl_w + w * len(col_names)}}')
    print(sep)

    # Unseen coverage
    fua = sr['flat_unseen_answered']; cua = sr['chain_unseen_answered']
    eua = sr.get('e3_unseen_answered', 0); tua = tr.get('tf_unseen_answered', 0)
    vals = [pct(fua, UT), pct(cua, UT)]
    if has_e3: vals.append(pct(eua, UT))
    if has_tf: vals.append(pct(tua, UT))
    print(row('Coverage (answered/unseen)', *vals))

    # Unseen accuracy (of answered)
    fuc = sr['flat_unseen_correct']; cuc = sr['chain_unseen_correct']
    euc = sr.get('e3_unseen_correct', 0); tuc = tr.get('tf_unseen_correct', 0)
    vals = [pct(fuc, fua), pct(cuc, cua)]
    if has_e3: vals.append(pct(euc, eua))
    if has_tf: vals.append(pct(tuc, tua))
    print(row('Acc. of answered (unseen)', *vals))

    # Unseen accuracy (of total)
    vals = [pct(fuc, UT), pct(cuc, UT)]
    if has_e3: vals.append(pct(euc, UT))
    if has_tf: vals.append(pct(tuc, UT))
    print(row('Acc. of total (unseen)', *vals))

    print()

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    _banner('Verdict')

    scores = {'Flat': fuc / max(fua, 1), 'E1': cuc / max(cua, 1)}
    if has_e3:
        scores['E3'] = euc / max(eua, 1)
    if has_tf:
        scores['Transformer'] = tuc / max(tua, 1)

    best = max(scores, key=scores.get)
    best_pct = 100 * scores[best]
    flat_pct = 100 * scores['Flat']

    print(f'  Unseen pair accuracy ranking:')
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        marker = ' ← BEST' if name == best else ''
        print(f'    {name:<14}: {100*s:.1f}%{marker}')

    print()
    if has_tf and scores.get('E3', scores.get('E1', 0)) >= scores['Transformer']:
        symbolic_best = 'E3' if has_e3 else 'E1'
        sym_pct = 100 * scores[symbolic_best]
        tf_pct  = 100 * scores['Transformer']
        print(f'  PARITY ACHIEVED: Symbolic ({symbolic_best}) matches/exceeds Transformer '
              f'on unseen pairs ({sym_pct:.1f}% vs {tf_pct:.1f}%).')
        print(f'  Compression: K^3 category entries vs full V-dimensional attention weights.')
    elif has_tf:
        symbolic_best = 'E3' if has_e3 else 'E1'
        sym_pct = 100 * scores[symbolic_best]
        tf_pct  = 100 * scores['Transformer']
        gap = tf_pct - sym_pct
        print(f'  GAP: Transformer leads symbolic by {gap:.1f}pp on unseen pairs '
              f'({tf_pct:.1f}% vs {sym_pct:.1f}%).')
        print(f'  Next step: Phase E4 (frame semantics) or larger K to close the gap.')
    else:
        print(f'  Symbolic best ({best}): {best_pct:.1f}%  vs  Flat: {flat_pct:.1f}%')
        print(f'  Run with PyTorch installed to add transformer comparison.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Parity test: symbolic category chain vs 2-layer transformer.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--corpus',         default='EarlyModernLatin')
    p.add_argument('--n_train',        type=int, default=5000)
    p.add_argument('--n_test',         type=int, default=None)
    p.add_argument('--split',          type=float, default=0.8)
    p.add_argument('--n_clusters',     type=int, default=12,
                   help='E1 base clusters (default: 12)')
    p.add_argument('--n_ctx_clusters', type=int, default=None)
    p.add_argument('--ctx_min_count',  type=int, default=3)
    p.add_argument('--e3_temperature', type=float, default=2.0,
                   help='E3 JSD similarity temperature (default: 2.0)')
    # Transformer args
    p.add_argument('--no_transformer', action='store_true',
                   help='Skip transformer training (symbolic only)')
    p.add_argument('--d_model',    type=int, default=64,
                   help='Transformer embedding dimension (default: 64)')
    p.add_argument('--n_heads',    type=int, default=2,
                   help='Transformer attention heads (default: 2)')
    p.add_argument('--n_layers',   type=int, default=2,
                   help='Transformer layers (default: 2)')
    p.add_argument('--epochs',     type=int, default=30,
                   help='Transformer training epochs (default: 30)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr',         type=float, default=3e-3)
    p.add_argument('--context_len', type=int, default=4,
                   help='Transformer context window (default: 4)')
    # Symbolic checkpoint
    p.add_argument('--load',  metavar='PATH',
                   help='Load symbolic AI checkpoint (skips discovery step)')
    p.add_argument('--save',  metavar='PATH',
                   help='Save symbolic AI checkpoint after training')
    p.add_argument('--seed',  type=int, default=42)
    args = p.parse_args()

    n_ctx  = args.n_ctx_clusters or args.n_clusters
    run_e3 = True   # always run E3 for the parity test

    _banner('Parity Test: Symbolic Category Chain vs 2-layer Transformer')
    print(f'  Corpus:           {args.corpus}')
    print(f'  Train lines:      {args.n_train}')
    print(f'  E1 clusters (K):  {args.n_clusters}')
    print(f'  E3 temperature:   {args.e3_temperature}')
    print(f'  Transformer:      {"disabled" if args.no_transformer else f"d={args.d_model}, {args.n_heads}h, {args.n_layers}L, {args.epochs}ep"}')

    # ---- Locate corpus ----
    d = (args.corpus if os.path.isdir(args.corpus)
         else os.path.join(_DATA_DIR, args.corpus))
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {args.corpus!r}')
        sys.exit(1)

    # ---- Load corpus ----
    total_n = args.n_train + (args.n_test or max(args.n_train // 4, 100))
    pairs   = find_pairs(d, max_n=total_n, shuffle=True, seed=args.seed)
    texts   = _stream_texts(pairs)

    n_train     = min(args.n_train, int(len(texts) * args.split))
    n_test_size = args.n_test or max(n_train // 4, 50)
    train_texts = texts[:n_train]
    test_texts  = texts[n_train: n_train + n_test_size]
    print(f'  Lines:            {len(train_texts)} train  {len(test_texts)} test')

    # ---- Training word bigrams (for seen/unseen split) ----
    train_pairs: set[tuple[str, str]] = set()
    for text in train_texts:
        toks = text.split()
        for i in range(len(toks) - 1):
            train_pairs.add((toks[i], toks[i + 1]))

    # ========================================================
    # Symbolic side
    # ========================================================

    # Step 1: Discovery
    print('\n  [Symbolic] Running multi-scale discovery...')
    ai, chunk_maps = run_pipeline(
        texts=train_texts,
        n_levels=3,
        max_merges=500,
        save_path=None,
        load_path=args.load,
        verbose=False,
    )

    # Step 2: E1 clusters
    _banner('Building E1 word cluster assignment')
    assignment, clusters = build_assignment(ai, n_clusters=args.n_clusters)
    K = len(clusters)
    print(f'  Vocab: {len(assignment):,}   Clusters: {K}')

    # Step 3: Train E1 chain
    train_chain(ai, train_texts, assignment, verbose=True)

    # Step 4: E2 context-sensitive
    context_assignment, ctx_clusters = build_trigram_assignment(
        train_texts, assignment, n_clusters=n_ctx,
        min_examples=args.ctx_min_count, verbose=False,
    )
    if context_assignment:
        train_chain_ctx(ai, train_texts, assignment, context_assignment, verbose=False)

    # Step 5: E3 soft retrieval
    _banner('Phase E3: Soft Retrieval (successor-distribution similarity)')
    succ_dists = build_cluster_succ_dists(ai, clusters, verbose=False)
    sim_matrix = build_cluster_similarity_matrix(
        succ_dists, K=K, temperature=args.e3_temperature, verbose=True)

    nc_cache  = precompute_dist_cache(ai, 'next_cat')
    wgc_cache = precompute_dist_cache(ai, 'word_given_cat')
    nc_soft   = precompute_all_soft_dists(nc_cache,  sim_matrix, K, arity=2, verbose=True)
    wgc_soft  = precompute_all_soft_dists(wgc_cache, sim_matrix, K, arity=3, verbose=True)

    # Step 6: Dev split for α tuning (10% of test)
    n_dev = max(len(test_texts) // 10, 20)
    dev_texts  = test_texts[:n_dev]
    eval_texts = test_texts[n_dev:]
    alpha = tune_mixture_alpha(
        ai, dev_texts, assignment, nc_soft, wgc_soft, verbose=False)
    print(f'  Mixture α = {alpha:.2f}')

    # Step 7: Evaluate symbolic model
    print()
    symbolic_results = evaluate_all(
        ai, eval_texts, assignment, context_assignment, train_pairs,
        nc_soft=nc_soft, wgc_soft=wgc_soft, verbose=True,
    )

    if args.save:
        ai.save_checkpoint(args.save)
        print(f'  Symbolic checkpoint saved: {args.save}')

    # ========================================================
    # Transformer side
    # ========================================================
    tf_results: dict = {}

    if not args.no_transformer:
        torch = _try_import_torch()
        if torch is None:
            print('\n  [Transformer] PyTorch not installed — skipping.')
            print('  Install with: pip install torch')
        else:
            # Build vocab from training texts
            vocab = build_vocab(train_texts)
            print(f'\n  [Transformer] Vocabulary: {len(vocab):,} words')

            tf_model = train_transformer_model(
                train_texts, vocab,
                d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                context_len=args.context_len,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                verbose=True,
            )

            if tf_model is not None:
                tf_results = eval_transformer_trigrams(
                    tf_model, eval_texts, train_pairs, vocab, verbose=True)

    # ========================================================
    # Comparison table
    # ========================================================
    print_comparison(symbolic_results, tf_results, run_e3=run_e3, verbose=True)


if __name__ == '__main__':
    main()
