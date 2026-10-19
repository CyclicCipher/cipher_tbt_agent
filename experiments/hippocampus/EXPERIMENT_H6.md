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

## Results (2026-10-18; `runs/h6/`, log `runs/h6_log.txt`)

All runs took 92–97 s, within the 2-minute rule.

| arm | seed | **bits/char: heldout** | caesar | **mhg** | **second reading: first → second (ratio)** | final training loss |
|---|---|---|---|---|---|---|
| none | 0 | 2.128 | 2.051 | 5.401 | 2.135 → 2.135 (1.000) | 2.127 |
| none | 1 | 2.135 | 2.050 | 5.434 | 2.149 → 2.150 (1.000) | 2.141 |
| recent | 0 | 2.114 | 2.053 | 5.073 | 2.125 → 2.115 (0.996) | 2.111 |
| recent | 1 | 2.135 | 2.062 | 5.272 | 2.153 → 2.140 (0.994) | 2.151 |
| **retrieve** | 0 | 2.167 | 2.097 | **4.894** | 2.176 → **1.883 (0.865)** | 1.953 |
| **retrieve** | 1 | 2.210 | 2.131 | **5.090** | 2.223 → **1.697 (0.763)** | 1.971 |

**Means over seeds:**

| arm | heldout | caesar | mhg | second-reading ratio |
|---|---|---|---|---|
| none | 2.132 | 2.050 | 5.418 | 1.000 |
| recent | 2.125 | 2.058 | 5.173 | 0.995 |
| retrieve | 2.188 | 2.114 | 4.992 | 0.814 |

**Against the predictions:**
- **T1 (one-reading learning, ratio ≤ 0.6): not met. Not refuted either** (the line was > 0.9).
  - Retrieve's second pass is **0.29 and 0.53 bits/char lower** than its first (ratios 0.865 and 0.763).
  - none and recent are unchanged (1.000 and 0.994–0.996), as predicted: they cannot see the first pass.
  - The reader learned to copy from memory, partially, within 3,000 steps.
- **T2 (in-distribution gain): REFUTED.** On held-out Latin, retrieve is **worse** than none by 0.056 bits/char
  (2.188 vs 2.132), and worse than recent. Caesar is worse by 0.064.
- **T3 (larger gain under shift): holds.** On Middle High German, retrieve beats none by **0.43 bits/char** (4.99 vs
  5.42, about 8%) and recent by 0.18. On held-out Latin the "gain" is −0.06.
- **Reported:**
  - Recent against none: a longer context barely matters in Latin prose (−0.007 heldout, +0.008 caesar) but helps the
    epic (−0.25 mhg).
  - Retrieve's lower training loss (1.95–1.97 vs 2.11–2.15) comes from the 25% copyable re-read windows.

**What H6 establishes:**
1. **On real text, the store gives one-shot learning of a passage.** A second reading is 0.29–0.53 bits/char cheaper
   after a single first reading, with no gradient step. The weights-only and longer-context models gain nothing.
2. **It pays where text repeats:** formulaic Middle High German verse (−0.43 bits/char) and re-read passages. This is
   the shape ZipLearn found for adaptation (E37: large under shift, small in distribution).
3. **It costs on non-repeating text** (+0.06 bits/char on held-out Latin and Caesar).
   - A retrieved passage is shown for every window, relevant or not, so the reader sometimes copies from an irrelevant
     match. This is the "false retrievals inject noise" risk of `PLAN.md` §5.
   - The capacity spent learning to copy is not available for ordinary prediction.
   - **⚠ Corrected by H6b:** the cost is mostly in the *weights*, not in irrelevant passages. A gated model that shows
     memory on only 0.6% of held-out windows is still +0.05 bits/char worse. The 25% re-read training windows (trivial
     copying) take training signal away from ordinary Latin. See `EXPERIMENT_H6B.md`.
4. **The obvious next step is a retrieval-confidence gate,** as in H3's combined switch: show the memory only when
   the best match is strong, and padding otherwise. That should keep the gains on repeating text and remove most of the
   cost on prose.
5. **Caveats:** a small, short-trained model (about 0.9M parameters, 3,000 steps, about 2.1 bits/char) and two seeds.
   The direction of each effect is consistent across seeds; the sizes are not stable (second-reading ratio 0.86 vs
   0.76).
