"""expQ inspection -- what did the MDL signature model actually learn? (S4's interpretability check)

A description-length win means nothing if the "suffixes" are arbitrary string fragments. Latin has a known,
finite inventory of inflectional endings, so this is a rare case where an unsupervised result can be checked
against something real without a labelled dataset.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expQ_morphology import MAX_SUF, MIN_STEM, build_signatures, load_types  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 100

counts, alphabet, _ = load_types(None)
types = list(counts)
used, covered, F = build_signatures(types, len(alphabet), K)

suf_use = Counter()
sig_size = Counter()
for s, sig in used.items():
    sig_size[len(sig)] += 1
    for f in sig:
        suf_use[f] += 1

print(f"stems {len(used)}, signatures {len({frozenset(v) for v in used.values()})}, suffixes in use {len(suf_use)}")
print("\nthe 40 suffixes attached to the most stems (Latin inflectional endings, if this worked):")
for f, n in suf_use.most_common(40):
    print(f"  {'-' + f if f else 'NULL':<10} {n:>7} stems")

sigs = Counter()
for s, sig in used.items():
    sigs[sig] += 1
print("\nthe 12 largest signatures (a signature IS the pattern; the stem is its argument):")
for sig, n in sigs.most_common(12):
    shown = sorted(sig, key=lambda x: (len(x), x))
    ex = [s for s, g in used.items() if g == sig][:3]
    print(f"  {n:>6} stems | {{{', '.join('-' + x if x else 'NULL' for x in shown[:9])}"
          f"{' ...' if len(shown) > 9 else ''}}}")
    for e in ex:
        print(f"           e.g. {e} + {{{', '.join(sorted(sig, key=len)[:6])}}}")
        break

print("\nthe 10 stems with the most forms:")
for s, sig in sorted(used.items(), key=lambda kv: -len(kv[1]))[:10]:
    print(f"  {s:<14} {len(sig):>3} forms: {', '.join(s + f for f in sorted(sig, key=len)[:8])}")

out = {"K": K, "stems": len(used), "suffixes_in_use": len(suf_use),
       "top_suffixes": [[f, n] for f, n in suf_use.most_common(60)],
       "signature_size_hist": dict(sorted(sig_size.items())[:15])}
(HERE.parent / "runs" / "research" / "expQ_inspect.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
