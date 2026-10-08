# DistributionPrediction

A new line of ideas (opened 2026-10-08) from the user's handwritten notes "On cryptography and learning": learning a
generating RULE rather than a distribution needs the order of the data (time) as a first-class coordinate; the model
must find its own coordinates; going beyond the data is a choice among rules weighed by evidence; and a proposed
objective — predict the extrapolated distribution, then the next token — with its predicted payoffs.

| file | what |
|---|---|
| `TRANSCRIPT.md` | the notes, verbatim, with descriptions of the drawings |
| `IDEA.md` | my rewording of the idea, with connections and open questions |
| `refs/ctm_2505.05522.md` | the Continuous Thought Machine paper, read for what it shows about the usefulness of time |
| `refs/what_when_spiking_2026.md` | Yamada & Chao 2026: a spiking network that predicts what, when and how likely — a minimal predicted distribution with time first-class |
| `make_figures.py` | recreates the notes' figures from the rules they describe (`figures/*.png`) |
| `notes/page{1,2,3}.webp` | the original photographs (page 4's photo was not saved as a file — add it as `notes/page4.*`) |

Decision (2026-10-08): the experiment will start from a transformer and modify it. Nothing is designed or run yet
beyond the figures.
