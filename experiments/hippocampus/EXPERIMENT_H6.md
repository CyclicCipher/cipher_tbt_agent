# Experiment H6 — the hippocampus on real text

**Pre-registered 2026-10-18, before any H6 result run** (smoke and timing runs on a throwaway seed, disclosed). The
user: "… Then move on to H6."

## Question

Everything so far is one synthetic family of facts. Does the store help a language model on **real text**, and does it
give the defining hippocampal ability there: **learning a passage in one reading**? The streams are Latin in
distribution, another Latin author, and Middle High German (where ZipLearn found its large adaptation gains).

## Data (`text6.py`)

- **Training text:** Cicero + Livy, 10.67M characters, 95%.
- **Test streams,** each read once front to back:
  - held-out Cicero/Livy (the last 5%);
  - **Caesar** (another author, 833k characters);
  - the **Nibelungenlied** (Middle High German, 475k characters).
- **Normalisation:** one 38-symbol alphabet; lower-cased, accents stripped, æ/œ/ß spelled out, other characters
  dropped, runs of spaces collapsed.
- **Index:** every 32-character chunk is indexed by the set of words it touches (the union code, one code per word).
  Overlap is weighted by word rarity (idf over the training chunks). 118,678 distinct words.

## Model and arms (`h6.py`, `run_h6.py`)

**Input:** [MEM] + a 128-character memory passage + [SEP] + the previous 128 characters + the 64-character window
being predicted (teacher-forced, causal), giving a 321-token sequence. Loss and scores are on the window only.

**Reader:** the shared `Net` with PoPE attention (H1f–H1h's content-matching result), 4 layers, d = 128, 4 heads, MLP
512, about 0.9M parameters.

**Training:** AdamW, lr 1e-3 cosine, batch 48, **3,000 steps**, compile, bf16. The arms are identical in everything
except what the memory shows:

| arm | memory passage |
|---|---|
| **none** | padding: no store |
| **recent** | the 128 characters just before the previous context: a longer context with no retrieval (the control) |
| **retrieve** | the store chunk sharing the most idf-weighted words with **the last 64 characters before the window**, delivered **with its continuation** (RETRO-style), 128 characters. Only passages ending before the previous context qualify. |

- **Store during training:** the 8,192 characters before each window.
- **Re-reads:** on **25% of training windows**, an earlier copy of the current region is also in the store, as on
  re-reading a document. The text is identical for all arms; only `retrieve` can find the copy.
- **Store during evaluation:** every earlier chunk of the stream (no cap, no gate; compression is not the subject
  here).

## Measures

- **Bits per character** over the first 50,000 characters of each stream (heldout, caesar, mhg).
- **The second reading:** a 20,000-character held-out passage read twice. Bits per character on the first pass and on
  the second, and their ratio. Only a store that keeps the first reading can help the second.

## Disclosed before registering (throwaway seed 99; nothing saved)

1. **First design** (retrieval cued by the whole previous context, top 4 chunks without continuations; 300 and 1,000
   steps): second-reading ratio 0.999. Retrieval returned the first pass's copy of the *context*, not of what followed
   it. **Changed** to "best match plus continuation", cued by the last 64 characters, plus the re-read training windows.
2. **Retrieval checks on the final design:**
   - On the second reading, the retrieved passage covers the first pass's copy of the window **99.7%** of the time.
   - On re-read training windows it overlaps the window 87% of the time (fully only 3%, because training positions do
     not align with chunks).
   - Normal windows never retrieve text from after their context.
3. **Retrieve arm at 1,000 steps:** heldout 2.81, caesar 2.74, mhg 4.36 bits/char; second-reading ratio 0.99. Copying
   had not been learned yet.
4. **Timing:** a full training step is 26 ms. A first compile of a new sequence length costs about 30–40 s and is then
   cached, so 3,000 steps ≈ 90 s.

## Predictions

- **T1 (learning a passage in one reading):** retrieve's second-reading ratio is **≤ 0.6** in both seeds. none and
  recent stay ≥ 0.97 (they cannot see the first pass).
  - **Refuted** if retrieve's ratio is > 0.9. The disclosed 1,000-step run (0.99) makes that a real risk: the reader
    may not learn to copy within 3,000 steps.
- **T2 (natural text, in distribution):** on held-out Latin, retrieve ≤ none − 0.02 bits/char, and retrieve ≤ recent.
- **T3 (shift):** retrieve's gain over none is larger on mhg than on held-out Latin (names and formulaic lines recur in
  the epic).
- **Reported:** recent against none (the value of a longer context), caesar, and training loss.

## Cost

6 runs of about 90 s each.

---

## Results

*(to be filled in after the runs)*
