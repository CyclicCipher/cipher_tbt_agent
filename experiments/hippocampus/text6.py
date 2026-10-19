"""H6 text: character streams in one alphabet, plus the words of each 32-character chunk for the store's index.

Latin (Cicero + Livy) is the training text, split 95/5 like Dust 2's loader (not byte-identical: spaces are collapsed after filtering); Caesar (another author) and the
Nibelungenlied (Middle High German) are the shifted test streams. Text is lower-cased, accents stripped (NFKD), æ/œ/ß
spelled out, characters outside the 38-symbol alphabet dropped, and runs of spaces collapsed.

Index: a word is a maximal run of letters; every character carries the id of the word it belongs to (-1 elsewhere). A
chunk's index is the set of word ids it touches (up to MAXW): the union code of H1, with one code per word, so overlap
is the number of shared words. Overlap is weighted by each word's rarity in the training text (idf over chunks).
"""
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

CORPORA = Path(__file__).resolve().parents[2] / "corpora"
ALPHABET = " abcdefghijklmnopqrstuvwxyz.,;:?!'\"()-"
V = len(ALPHABET)
CHUNK, MAXW = 32, 8
CACHE = Path(__file__).parent / "runs" / "h6" / "text_cache.pt"


def clean(text):
    text = text.lower().replace("æ", "ae").replace("œ", "oe").replace("ß", "ss")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text)
    text = "".join(c for c in text if c in ALPHABET)
    text = re.sub(r" +", " ", text)                     # again: dropping digits etc. leaves runs of spaces
    keep = {c: i for i, c in enumerate(ALPHABET)}
    return np.array([keep[c] for c in text], dtype=np.uint8)


def word_ids(ids, vocab):
    """Per character: the id of its word (a run of letters), -1 for spaces and punctuation. New words get new ids."""
    s = "".join(ALPHABET[i] for i in ids)
    wid = np.full(len(ids), -1, dtype=np.int32)
    for m in re.finditer(r"[a-z]+", s):
        w = vocab.setdefault(m.group(), len(vocab))
        wid[m.start():m.end()] = w
    return wid


def chunk_sets(wid):
    """(n_chunks, MAXW) word ids touched by each chunk, -1 padded."""
    n = len(wid) // CHUNK
    rows = np.sort(wid[:n * CHUNK].reshape(n, CHUNK), axis=1)
    first = np.ones_like(rows, dtype=bool)
    first[:, 1:] = rows[:, 1:] != rows[:, :-1]
    keep = first & (rows >= 0)
    out = np.full((n, MAXW), -1, dtype=np.int32)
    order = np.argsort(~keep, axis=1, kind="stable")                  # kept ids first, in sorted order
    packed = np.take_along_axis(rows, order, axis=1)[:, :MAXW]
    kept = np.take_along_axis(keep, order, axis=1)[:, :MAXW]
    out[kept] = packed[kept]
    return out


def load():
    if CACHE.exists():
        return torch.load(CACHE)
    read = lambda p: open(p, encoding="utf-8", errors="ignore").read()
    lat = clean(read(CORPORA / "latin_classical" / "cicero.txt") + read(CORPORA / "latin_classical" / "livy.txt"))
    n = int(len(lat) * 0.95)
    texts = {"train": lat[:n], "heldout": lat[n:],
             "caesar": clean(read(CORPORA / "latin_classical" / "caesar.txt")),
             "mhg": clean(read(CORPORA / "mittelhochdeutsch" / "Der Nibelunge liet..txt"))}
    vocab = {}
    out = {}
    for name, ids in texts.items():
        wid = word_ids(ids, vocab)
        out[name] = dict(ids=torch.from_numpy(ids), wid=torch.from_numpy(wid), sets=torch.from_numpy(chunk_sets(wid)))
    df = np.zeros(len(vocab), dtype=np.int64)                         # chunk frequency in the TRAINING text only
    s = out["train"]["sets"].numpy()
    u = s[s >= 0]
    np.add.at(df, u, 1)
    n_chunks = len(s)
    out["idf"] = torch.tensor(np.log(n_chunks / (1.0 + df)), dtype=torch.float32)
    out["n_words"] = len(vocab)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, CACHE)
    return out


if __name__ == "__main__":
    d = load()
    for k in ("train", "heldout", "caesar", "mhg"):
        ids = d[k]["ids"]
        print(k, len(ids), "chars", len(d[k]["sets"]), "chunks |", "".join(ALPHABET[i] for i in ids[:120].tolist()))
    print("words", d["n_words"], "idf range", float(d["idf"].min()), float(d["idf"].max()))
