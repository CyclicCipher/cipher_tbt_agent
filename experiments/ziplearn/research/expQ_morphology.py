"""expQ (S4, DESIGN §19.1) -- LEVEL-3 NAMING: can a pattern with HOLES pay for itself by description length?

§19.1 says the library problem and the missing word inventory sit at different heights of one ladder:
  level 1  values in a fixed address space            -- every count table we have
  level 2  a name for an EXACT recurrence             -- §7 step 3 on matrices (E3, E8); BPE on text
  level 3  a name for a PATTERN WITH HOLES            -- nowhere in the codebase
and that a Goldsmith signature (a set of stems crossed with a set of suffixes) IS an anti-unification with two
argument slots, so morphology is level 3 on the cheapest available domain.

Three encodings of the same object -- the word-TYPE inventory of `corpora/latin_classical` -- all two-part codes,
all gradient-free, no network anywhere:
  A  literal     every type spelled out                                    (no naming at all: level 1)
  B  signatures  stems x suffixes, the model GENERATES a set of words and
                 pays a correction bitmap for what it gets wrong           (level 3)
  C  bpe         a subword vocabulary at matched size, each type spelled
                 as a sequence of units                                    (level 2: exact recurrences only)

Pass (pre-registered §19.1): B beats C on total description length AND raises tokens per unit by >= 3x over the
surface type. Refute: B no better than C -- the holes buy nothing and level 2 is the whole story.

Usage: `python experiments/ziplearn/research/expQ_morphology.py [--types N] [--quick]`
Writes runs/research/expQ.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent.parent.parent / "corpora" / "latin_classical"
MIN_STEM = 3                       # a stem shorter than this is not a stem, it is a fragment
MAX_SUF = 7                        # Latin inflectional endings are short
MIN_STEMS_PER_SIG = 2              # Goldsmith's rule: a signature needs at least 2 stems ...
MIN_SUFS_PER_SIG = 2               # ... and at least 2 suffixes, or it is just a word


def load_types(cap=None):
    """Word types and their token counts over the classical corpus, lowercased, letters only."""
    counts = Counter()
    files = sorted(CORPUS.glob("*.txt"))
    for f in files:
        txt = unicodedata.normalize("NFC", f.read_text(encoding="utf-8", errors="replace")).lower()
        cur = []
        for ch in txt:
            if ch.isalpha():
                cur.append(ch)
            elif cur:
                counts[("".join(cur))] += 1
                cur = []
        if cur:
            counts["".join(cur)] += 1
    alphabet = sorted({c for w in counts for c in w})
    if cap:
        counts = Counter(dict(counts.most_common(cap)))
    return counts, alphabet, len(files)


def literal_bits(word, A):
    """Spell a word out: one symbol per character plus a terminator, at log2(A + 1) bits each."""
    return (len(word) + 1) * math.log2(A + 1)


# ----------------------------------------------------------------------------------------- A: no naming
def cost_literal(types, A):
    return sum(literal_bits(w, A) for w in types)


# ----------------------------------------------------------------------------------- B: stems x signatures
def build_signatures(types, A, top_suffixes):
    """Goldsmith-style: candidate suffixes by how many distinct stems take them, then stems grouped by the SET of
    suffixes they take (that set is the signature -- the pattern, with the stem as its argument)."""
    W = set(types)
    # 1. candidate suffixes, scored by the number of distinct stems they attach to
    suf_stems = defaultdict(set)
    for w in W:
        for k in range(1, min(MAX_SUF, len(w) - MIN_STEM) + 1):
            suf_stems[w[-k:]].add(w[:-k])
    ranked = sorted(suf_stems, key=lambda f: -len(suf_stems[f]))[:top_suffixes]
    F = set(ranked) | {""}
    # 2. for every candidate stem, which of those suffixes complete it to a real word
    stem_sufs = defaultdict(set)
    for w in W:
        for k in range(0, min(MAX_SUF, len(w) - MIN_STEM) + 1):
            stem, suf = (w, "") if k == 0 else (w[:-k], w[-k:])
            if suf in F:
                stem_sufs[stem].add(suf)
    # 3. keep stems with enough suffixes; group into signatures; drop signatures with too few stems
    sigs = defaultdict(list)
    for s, fs in stem_sufs.items():
        if len(fs) >= MIN_SUFS_PER_SIG:
            sigs[frozenset(fs)].append(s)
    sigs = {sig: ss for sig, ss in sigs.items() if len(ss) >= MIN_STEMS_PER_SIG}
    # 4. a word is covered by the LONGEST stem that has a signature (greedy, one parse per word)
    stem_of_sig = {}
    for sig, ss in sigs.items():
        for s in ss:
            stem_of_sig[s] = sig
    used_stems, covered = {}, set()
    for w in sorted(W, key=len, reverse=True):
        for k in range(0, min(MAX_SUF, len(w) - MIN_STEM) + 1):
            stem, suf = (w, "") if k == 0 else (w[:-k], w[-k:])
            sig = stem_of_sig.get(stem)
            if sig is not None and suf in sig:
                used_stems[stem] = sig
                covered.add(w)
                break
    return used_stems, covered, F


def cost_signatures(types, A, used_stems, covered, F):
    """model = stems + suffixes + signature table + stem->signature; data = a keep/drop bitmap over what the model
    generates, plus every observed type it fails to generate, spelled out."""
    W = set(types)
    S = list(used_stems)
    sig_set = {}
    for s, sig in used_stems.items():
        sig_set.setdefault(sig, len(sig_set))
    used_sufs = sorted({f for sig in sig_set for f in sig})
    nF = max(1, len(used_sufs))
    bits_stems = sum(literal_bits(s, A) for s in S)
    bits_sufs = sum(literal_bits(f, A) for f in used_sufs)
    bits_sigtab = sum(math.log2(nF + 1) + len(sig) * math.log2(nF) for sig in sig_set)
    bits_assign = len(S) * math.log2(max(1, len(sig_set)))
    generated = set()
    for s, sig in used_stems.items():
        for f in sig:
            generated.add(s + f)
    spurious = generated - W
    missing = W - covered
    bits_bitmap = float(len(generated))                       # 1 bit per generated word: keep or drop
    bits_missing = sum(literal_bits(w, A) for w in missing)
    total = bits_stems + bits_sufs + bits_sigtab + bits_assign + bits_bitmap + bits_missing
    return total, dict(stems=len(S), suffixes=len(used_sufs), signatures=len(sig_set),
                       generated=len(generated), spurious=len(spurious), covered=len(covered),
                       missing=len(missing), bits_stems=bits_stems, bits_suffixes=bits_sufs,
                       bits_sigtable=bits_sigtab, bits_assign=bits_assign, bits_bitmap=bits_bitmap,
                       bits_missing=bits_missing)


# ------------------------------------------------------------------------------------------- C: BPE (level 2)
def bpe_vocab(types, target, A):
    """Byte-pair encoding over word types (type-level, to match B, which is also a type-level model).

    Incremental: a pair -> {word indices} index, so a merge touches only the words that contain that pair. The
    naive rescan is O(merges * |types|) and cannot reach a vocabulary matched to B's stem count."""
    seqs = [list(w) + ["</w>"] for w in types]
    pair_counts = Counter()
    pair_words = defaultdict(set)
    for i, s in enumerate(seqs):
        for j in range(len(s) - 1):
            p = (s[j], s[j + 1])
            pair_counts[p] += 1
            pair_words[p].add(i)
    vocab = set(A)
    merges = []
    import heapq
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    while len(vocab) < target and heap:
        negc, p = heapq.heappop(heap)
        if pair_counts.get(p, 0) != -negc:                    # stale entry
            if pair_counts.get(p, 0) > 0:
                heapq.heappush(heap, (-pair_counts[p], p))
            continue
        if -negc < 2:
            break
        a, b = p
        ab = a + b
        merges.append(p)
        vocab.add(ab)
        touched = set()
        for i in list(pair_words[p]):
            s = seqs[i]
            for j in range(len(s) - 1):                        # remove this word's old pairs
                q = (s[j], s[j + 1])
                pair_counts[q] -= 1
                pair_words[q].discard(i)
            out, j = [], 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                    out.append(ab); j += 2
                else:
                    out.append(s[j]); j += 1
            seqs[i] = out
            for j in range(len(out) - 1):                      # add its new ones
                q = (out[j], out[j + 1])
                pair_counts[q] += 1
                pair_words[q].add(i)
                touched.add(q)
        pair_counts.pop(p, None)
        pair_words.pop(p, None)
        for q in touched:
            if pair_counts.get(q, 0) > 1:
                heapq.heappush(heap, (-pair_counts[q], q))
    return merges, [tuple(s) for s in seqs], vocab


def cost_bpe(types, A, merges, words, vocab):
    units = sorted(u for u in vocab if u != "</w>")
    nV = max(2, len(vocab))
    bits_vocab = sum(literal_bits(u.replace("</w>", ""), A) for u in units)
    bits_words = sum((len(w) + 1) * math.log2(nV) for w in words)
    return bits_vocab + bits_words, dict(vocab=len(vocab), merges=len(merges),
                                         bits_vocab=bits_vocab, bits_words=bits_words,
                                         mean_units_per_type=sum(len(w) for w in words) / max(1, len(words)))


def main():
    t0 = time.time()
    cap = None
    if "--types" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--types") + 1])
    quick = "--quick" in sys.argv
    counts, alphabet, n_files = load_types(cap)
    A = len(alphabet)
    types = list(counts)
    n_tokens = sum(counts.values())
    out = {"corpus": str(CORPUS.name), "files": n_files, "alphabet": A, "types": len(types),
           "tokens": n_tokens, "tokens_per_surface_type": n_tokens / len(types),
           "min_stem": MIN_STEM, "max_suffix": MAX_SUF}
    print(json.dumps({k: out[k] for k in ("files", "alphabet", "types", "tokens", "tokens_per_surface_type")},
                     indent=1), flush=True)

    base = cost_literal(types, A)
    out["A_literal_bits"] = base
    print(f"A literal: {base/8/1e6:.2f} MB ({base:.0f} bits)", flush=True)

    grid = [200, 400] if quick else [100, 200, 400, 800, 1600, 3200]
    out["B_grid"] = []
    best = None
    for K in grid:
        used, covered, F = build_signatures(types, A, K)
        tot, det = cost_signatures(types, A, used, covered, F)
        rec = {"top_suffixes": K, "bits": tot, "ratio_vs_literal": tot / base, **det}
        out["B_grid"].append(rec)
        print(f"  B K={K:<5} {tot/8/1e6:6.2f} MB  ratio {tot/base:.3f}  stems {det['stems']:>7} "
              f"sufs {det['suffixes']:>4} sigs {det['signatures']:>5} missing {det['missing']:>7} "
              f"spurious {det['spurious']:>7}", flush=True)
        if best is None or tot < best["bits"]:
            best = rec
    out["B_best"] = best
    out["B_signatures_bits"] = best["bits"]

    v_target = best["stems"] + best["suffixes"]        # matched to B, or the control proves nothing
    merges, words, vocab = bpe_vocab(types, v_target, alphabet)
    totc, detc = cost_bpe(types, A, merges, words, vocab)
    out["C_bpe"] = {"bits": totc, "ratio_vs_literal": totc / base, "target_vocab": v_target, **detc}
    print(f"C bpe  (vocab {len(vocab)}): {totc/8/1e6:.2f} MB  ratio {totc/base:.3f}", flush=True)

    # tokens per UNIT: how much evidence each named thing gets, against the 5.2 of a surface type
    stem_tokens = Counter()
    used, covered, F = build_signatures(types, A, best["top_suffixes"])
    for w, c in counts.items():
        for k in range(0, min(MAX_SUF, len(w) - MIN_STEM) + 1):
            stem = w if k == 0 else w[:-k]
            if stem in used and (w[len(stem):] in used[stem]):
                stem_tokens[stem] += c
                break
    out["tokens_per_stem"] = (sum(stem_tokens.values()) / len(stem_tokens)) if stem_tokens else None
    out["stems_with_tokens"] = len(stem_tokens)
    out["evidence_multiple"] = (out["tokens_per_stem"] / out["tokens_per_surface_type"]) if stem_tokens else None

    out["verdict"] = {
        "B_beats_C": bool(out["B_signatures_bits"] < out["C_bpe"]["bits"]),
        "evidence_multiple_ge_3": bool((out["evidence_multiple"] or 0) >= 3.0),
    }
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expQ.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\ntokens/surface type {out['tokens_per_surface_type']:.1f} -> tokens/stem "
          f"{out['tokens_per_stem']:.1f} ({out['evidence_multiple']:.2f}x); verdict {out['verdict']}; "
          f"{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
