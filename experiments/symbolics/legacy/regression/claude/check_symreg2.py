"""Known-formula tests for symreg2 (run from experiments/symbolics)."""
import time
import numpy as np
from regression.claude.symreg2 import Finder
rng = np.random.default_rng(0)
def feasible(lo, hi, n, ok=None):
    out = []
    while len(out) < n:
        x = rng.uniform(lo, hi)
        if ok is None or ok(x): out.append(x)
    return np.array(out)
def run(name, names, lo, hi, f, n, mfn=None, mnames=(), ok=None):
    X = feasible(lo, hi, n, ok); Xt = feasible(lo, hi, 500, ok); D = feasible(lo, hi, 2000, ok)
    M, Mt, MD = [(mfn(a) if mfn else None) for a in (X, Xt, D)]
    t0 = time.time(); m = Finder(names, mnames).fit(X, f(X), M, {'X': D, 'M': MD if mfn else np.zeros((len(D), 0))}); dt = time.time() - t0
    yt = f(Xt); e = np.sqrt(np.mean((m.predict(Xt, Mt) - yt)**2)) / yt.std()
    print(f'{name:24s} n={n:3d}: NRMSE {e:.2e}  terms {m.n_terms()}  frontier {len(m.frontier)}  {dt:5.1f}s  {m.formula()[:150]}', flush=True)
run('4x^3 + 3.12x', ['x'], [2], [10], lambda X: 4*X[:,0]**3 + 3.12*X[:,0], 10)
run('2.7 x^-2.6', ['x'], [0.3], [2], lambda X: 2.7*X[:,0]**-2.6, 20)
run('(x + 0.3)^-2', ['x'], [0.1], [1], lambda X: (X[:,0] + 0.3)**-2, 30)
run('1/(s+0.05)', ['a','b'], [0.1, 0.1], [0.5, 0.5], lambda X: 1/(0.9 - X[:,0] - X[:,1] + 0.05), 50,
    mfn=lambda X: (0.9 - X[:,0] - X[:,1])[:, None], mnames=['s'], ok=lambda x: 0.9 - x.sum() >= 0)
run('sum c_i h_i^-3 (d=3)', ['h1','h2','h3'], [0.4]*3, [1.0]*3, lambda X: 27*X[:,0]**-3 + 8*X[:,1]**-3 + 1*X[:,2]**-3, 100)
run('sum c_i h_i^-3 (d=6)', [f'h{i}' for i in range(1,7)], [0.4]*6, [1.0]*6, lambda X: X**-3 @ np.array([108, 74, 47, 26, 11, 1.2]), 100)
run('noisy 1 + 3r^2/(1-r)', ['r'], [0.05], [0.4], lambda X: 1 + 3*X[:,0]**2/(1-X[:,0]) + 0.002*rng.normal(size=len(X)), 30)
run('noise', ['a','b'], [1,1], [2,2], lambda X: rng.normal(size=len(X)), 30)
