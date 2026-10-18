# Intelligent constraints — laws built into learning that hold for every case

Started 2026-10-18 at the user's request: a folder for constraint ideas, the plans to test them, and the tests.

## The idea

A Hamiltonian neural network (HNN) builds one law into the network. It learns a scalar H (the energy), and the way
motion follows from H is fixed (the symplectic matrix J). Energy is then conserved by construction. On systems that obey
that law it generalises far better than an unconstrained network. On systems that don't, such as anything with friction,
it is simply wrong. The constraint is powerful but narrow.

The question here (the user's): **are there constraints that are laws of every case in a domain?** They would buy the
same kind of generalisation without the HNN's narrowness.

**The test of generality.** The constraint must be a theorem about the structure itself, not a property of some
instances. Two examples:
- **General:** every binary relation has a converse. "a father-of b" holds exactly when "b child-of a" does, with no
  exceptions.
- **Narrow:** every relation has an inverse. This holds only for one-to-one relations. A relation embedding that
  assumes invertible operators (rotations, Lie groups) is therefore HNN-like: right only for a subclass.

**First target: the reversal curse.** A model trained on "A r B" cannot answer "B r° ?" from its weights, even after it
has seen other facts of the same relation written both ways (Berglund et al. 2023).

## Where the ideas came from

- **The reversal-curse discussion (2026-10-18).** Three mechanisms:
  - gradient updates are one-way (Zhu et al. 2024);
  - facts are lookups keyed on the subject;
  - each entity has two unconnected codes, one for being read and one for being said (Wang & Sun 2025, "binding").
- **Gemini's five constraints for a "continuous UKS".** The screenshots are in `notes/`. My evaluation is in `IDEAS.md`:
  the diagnosis is about half right, and most of the proposed constraints are narrow or off-target.
- **The CTKG line (March 2026; now only in git history, `experiments/symbolic_ai_v2/`).**
  - Its `reasoning/frame_solver.py` (commit `2af1208`) found converse pairs of operators in the data, such as
    `succ(a) = b ↔ pred(b) = a`. It checked the categorical unit and counit conditions as coverage tests.
  - It then answered queries seen only the other way round, symbolically.
  - Experiment 1 asks for the same behaviour from weights.

## Files

| file | what |
|---|---|
| `IDEAS.md` | the catalogue of candidate constraints: origin, why each might be general, status, planned test; rejected candidates and why |
| `notes/` | `gemini_continuous_uks_1–7.png`, Gemini's answer that prompted the evaluation |
| `data.py` | synthetic relational facts: single-token entities, 6 relations with converse tokens, paired and one-way splits |
| `model.py` | tiny causal transformer; MLP = plain key-value memory or DAGGER memory (transpose read of the same weights) |
| `train.py` | trains one configuration and logs full curves; refuses to re-run an existing result |
| `run_e1.py` | Experiments 1 and 1b driver (`--frozen` = 1b): capacity pilot, then the grid; reuses existing results |
| `EXPERIMENT_1.md` | the dagger memory against the reversal curse: pre-registration, then results (did not test C1) |
| `EXPERIMENT_1B.md` | the same with frozen token codes, so facts must live in the MLPs: pre-registration, then results |
| `EXPERIMENT_1C.md` | weight decay x transport (learned attention vs fixed sum) with a per-run MLP-off check |
| `run_e1c.py` | Experiment 1c driver |
| `runs/` | results, one JSON per configuration |

## Standing rules (from the user and the project)

- **Every test run takes 2 minutes at most.**
- **Every run is pre-registered** in an `EXPERIMENT_*.md` before it runs; exploratory work is labelled as such.
- **Never re-run an unchanged experiment, controls included.** Results are reused, and full curves are logged the first
  time.
- **Use 16-bit wherever it is accurate enough.** Here: bf16 autocast for the matmuls, fp32 master weights.
- **Diverse tasks.** Conclusions are not drawn from one task family alone. Experiment 1 is one synthetic family, and is
  reported as such.
- **No compute on competitor-faithful arms.** Arms test our constraints, not reproductions of other methods.
- **One general method,** not per-case specialists.
- **No heavy training on this machine.** Runs are short and small, on the laptop GPU.
- **Commits go to `exp/mdl-vs-ntp`.**

## Where it stands (2026-10-18)

- `IDEAS.md` holds five candidate constraints (C1–C5) and four rejected ones.
- **Experiment 1 (C1, the dagger memory): did not test C1.** The facts were stored in the learned embeddings and attention, not in the MLP (a plain MLP with 8 hidden units memorised everything). Tied embeddings alone gave a small reversal (~25× chance).
- **Experiment 1b (frozen random token codes): apparatus check failed again.** Attention alone memorises the facts (picking the top answer among random codes needs only ~0.5 correlation). The baseline is fully cursed (chance). The dagger arm also stays at chance on one-way facts, but its transpose path answers up to half of the PAIRED reverse facts, as separate storage. Sharing K and V does not force sharing a fact unless an entity has one code at the MLP input and at the output (C3 at the MLP interface).
- **Experiment 1c** (pre-registered): weight decay 0.01, 0.1 or 1.0, crossed with learned attention vs a fixed sum of the token codes, plus a per-run MLP-off check of where the facts live.
