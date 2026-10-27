"""Exploratory (docs/SURROGATES.md, part 2, after the diagnostics): the two cheapest fixes the diagnostics point to.
    E1  exponent grid: add x^-3 (the stepped beam's law is a sum of h_i^-3)
    E2  search width: a beam of 32 instead of 4, inside the full pipeline (cross-validation included)
At n = 100, 3 seeds, on all four problems. symreg.py's defaults are not changed.
    python -m regression.claude.explore_fixes      (from experiments/symbolics)
"""
import json

import numpy as np

from regression.claude import symreg
from regression.claude.surrogates import OUT, PROBLEMS, nrmse


def run(label, exponents=None, beam=None):
    old_e, old_b = symreg.EXPONENTS, symreg.BEAM
    if exponents:
        symreg.EXPONENTS = exponents
    if beam:
        symreg.BEAM = beam
        symreg.ols_beam_path.__defaults__ = (beam, None)
    out = {}
    try:
        for name in PROBLEMS:
            z = np.load(f"{OUT}/sims_{name}.npz")
            rows = []
            for seed in range(3):
                m = symreg.SymbolicRegressor(PROBLEMS[name][1]).fit(z[f"X_train{seed}"][:100], z[f"y_train{seed}"][:100])
                rows.append((nrmse(m.predict(z["X_test"]), z["y_test"]), nrmse(m.predict(z["X_extrap"]), z["y_extrap"]),
                             m.n_terms(), m.formula()))
            out[name] = rows
            print(f"{label:12s} {name:8s}: NRMSE in {np.mean([r[0] for r in rows]):.4f}  ex {np.mean([r[1] for r in rows]):.4f}"
                  f"  terms {[r[2] for r in rows]}", flush=True)
    finally:
        symreg.EXPONENTS, symreg.BEAM = old_e, old_b
        symreg.ols_beam_path.__defaults__ = (old_b, None)
    return out


if __name__ == "__main__":
    res = {"E1": run("E1 x^-3", exponents=(-3,) + symreg.EXPONENTS),
           "E2": run("E2 beam 32", beam=32),
           "E1+E2": run("E1+E2", exponents=(-3,) + symreg.EXPONENTS, beam=32)}
    print("\nstepped formula, E1 (seed 0):", res["E1"]["stepped"][0][3])
    json.dump(res, open(f"{OUT}/explore_fixes.json", "w"))
