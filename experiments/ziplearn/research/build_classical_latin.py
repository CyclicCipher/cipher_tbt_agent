"""Select the CLASSICAL subset of the Latin Library dump, using the source's own classification.

`corpora/latin_library` (cloned from github.com/cltk/lat_text_latin_library, public domain) is 97.2M characters of
Latin of every period -- archaic, classical, patristic, medieval, humanist, neo-Latin -- in a flat author dump with
no period metadata. Mixing a thousand years of language change would cost more than the extra data buys, so the
subset is taken from thelatinlibrary.com's OWN index page, which lists the classical authors and links the other
periods out to christian.html / medieval.html / neo.html / misc.html. AUTHORS is that list, transcribed from the
hrefs; nothing here is my judgement about who counts as classical, and every unmatched name on either side is
printed so the curation is auditable rather than silent.

Writes `corpora/latin_classical/<author>.txt` (one file per author, its texts concatenated) and prints the manifest.
Usage: `python experiments/ziplearn/research/build_classical_latin.py [--dry]`
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent.parent.parent / "corpora"
SRC = CORPORA / "latin_library"
DST = CORPORA / "latin_classical"

# the hrefs of thelatinlibrary.com/index.html, in order, with the repo path each one corresponds to.
# left = the site's link stem; right = the directory or file stem in the dump (None = look for the stem itself).
AUTHORS = {
    "ammianus": "ammianus", "apuleius": "apuleius", "augustus": "resgestae", "victor": "victor",
    "caesar": "caesar", "cato": "cato", "catullus": "catullus", "cicero": "cicero",
    "claudian": "claudian", "curtius": "curtius", "enn": "enn.txt", "eutropius": "eutropius",
    "florus": "florus", "frontinus": "frontinus", "gellius": "gellius", "sha": "sha",
    "hor": "horace", "justin": "justin", "juvenal": "juvenal", "liv": "livy",
    "lucan": "lucan", "lucretius": "lucretius", "martial": "martial", "nepos": "nepos",
    "ovid": "ovid", "persius": "persius", "petronius": "petronius", "phaedrus": "phaedr",
    "plautus": "plautus", "pliny": "pliny", "prop": "propertius",
    "quintilian": "quintilian", "sall": "sall", "seneca": "seneca", "sen": "sen",
    "silius": "silius", "statius": "statius", "suet": "suetonius", "sulpicia": "sulpicia",
    "tac": "tacitus", "ter": "ter.", "tib": "tibullus", "valeriusflaccus": "valeriusflaccus",
    "valmax": "valmax", "varro": "varro", "vell": "vell", "verg": "vergil",
    "vitruvius": "vitruvius", "ius": "justinian",
}
# `ius` (Roman law: the Justinianic corpus, the Twelve Tables, Gaius) is linked from the classical index but is
# 10.8M characters of legal formulae -- a register of its own. Kept OUT of the default build; --with-law includes it.
LAW = {"ius"}


def texts_for(stem: str):
    """Every .txt under a directory of this name; an exact file when the stem ends in .txt; otherwise every
    top-level file whose name starts with the stem. Exact and dotted stems exist to stop a prefix over-matching
    (`enn` would otherwise swallow `ennodius`, a 6th-century author; `ter` would need to avoid anything but
    `ter.<play>`)."""
    if stem.endswith(".txt"):
        p = SRC / stem
        return [p] if p.is_file() else []
    d = SRC / stem
    if d.is_dir():
        return sorted(d.rglob("*.txt"))
    return sorted(p for p in SRC.glob(f"{stem}*.txt"))


def main():
    dry = "--dry" in sys.argv
    with_law = "--with-law" in sys.argv
    if not SRC.is_dir():
        sys.exit(f"missing {SRC}; clone github.com/cltk/lat_text_latin_library into it first")

    manifest, missing, total = [], [], 0
    used = set()
    for key, stem in AUTHORS.items():
        if key in LAW and not with_law:
            continue
        files = texts_for(stem)
        if not files:
            missing.append(f"{key} -> {stem}")
            continue
        text = "\n".join(unicodedata.normalize("NFC", p.read_text(encoding="utf-8", errors="replace"))
                         for p in files)
        used.update(files)
        manifest.append({"author": stem, "site_key": key, "files": len(files), "chars": len(text)})
        total += len(text)
        if not dry:
            DST.mkdir(parents=True, exist_ok=True)
            (DST / f"{stem}.txt").write_text(text, encoding="utf-8")

    all_txt = set(SRC.rglob("*.txt"))
    report = {
        "source": "github.com/cltk/lat_text_latin_library (public domain)",
        "classification": "thelatinlibrary.com/index.html -- the site's own classical author list",
        "law_included": with_law,
        "authors": sorted(manifest, key=lambda m: -m["chars"]),
        "n_authors": len(manifest),
        "total_chars": total,
        "dump_total_chars": sum(len(p.read_bytes()) for p in all_txt),
        "files_used": len(used), "files_in_dump": len(all_txt),
        "unmatched_site_names": missing,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "authors"}, indent=1))
    print("\ntop authors:")
    for m in report["authors"][:15]:
        print(f"  {m['chars']:>9,}  {m['author']:<18} ({m['files']} files)")
    if missing:
        print("\nUNMATCHED (site lists them, the dump does not have that name):", ", ".join(missing))
    if not dry:
        (DST / "MANIFEST.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {DST} ({total:,} characters over {len(manifest)} authors)")


if __name__ == "__main__":
    main()
