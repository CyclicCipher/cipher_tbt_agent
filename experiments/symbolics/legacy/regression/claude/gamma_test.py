"""The Gamma test (Stefansson, Koncar & Jones 1997): a model-free estimate of the noise variance in a dataset.
For each point, its k nearest neighbours in input space (inputs scaled to [0, 1]); delta(k) = mean squared input
distance to the k-th neighbour, gamma(k) = mean of (y_i - y_nn(i, k))^2 / 2. For a smooth function plus noise,
gamma rises linearly with delta from the noise variance: the intercept at delta = 0 estimates it.
Exploratory comparison with the sweep-measured roughness (docs/CURRICULUM.md, B2):
    python -m regression.claude.gamma_test      (from experiments/symbolics)
"""
import json
import pickle

import numpy as np


def gamma_noise(X, y, lo, hi, k_max=10):
    """sqrt of the Gamma statistic, in units of std(y) (comparable to NRMSE and to the sweep roughness)."""
    U = (np.asarray(X, float) - lo) / (hi - lo)
    D = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, 1)[:, :k_max]
    delta = np.array([D[np.arange(len(y)), order[:, k]].mean() for k in range(k_max)])
    gamma = np.array([0.5 * np.mean((y - y[order[:, k]]) ** 2) for k in range(k_max)])
    slope, icept = np.polyfit(delta, gamma, 1)
    return float(np.sqrt(max(icept, 0.0)) / np.std(y))


if __name__ == "__main__":
    from regression.claude.tasks import curriculum, heldout
    data = pickle.load(open("runs/curriculum/curriculum_data.pkl", "rb"))
    rough = json.load(open("runs/curriculum/roughness.json"))
    rows = []
    for t in curriculum():
        d = data[t.name]
        rows.append((t.name, len(t.names), rough[t.name], gamma_noise(d["X"], d["y"], t.lo, t.hi)))
    hd = pickle.load(open("runs/curriculum/heldout_data.pkl", "rb"))
    rh = json.load(open("runs/curriculum/roughness_heldout.json"))
    for t in heldout():
        d = hd[t.name]
        rows.append((t.name, len(t.names), rh[t.name], gamma_noise(d["sets"]["train0"], d["y_train0"], t.lo, t.hi)))
    print(f"{'task':34s} {'d':>2s} {'sweep':>8s} {'Gamma':>8s}")
    for name, d, r, g in rows:
        print(f"{name:34s} {d:2d} {r:8.4f} {g:8.4f}")
    r, g = np.array([x[2] for x in rows]), np.array([x[3] for x in rows])
    noisy = r > 0.003
    print(f"\ntasks with sweep roughness > 0.003: {noisy.sum()}; Gamma / sweep, median {np.median(g[noisy] / r[noisy]):.2f}, "
          f"within a factor 2 for {np.mean((g[noisy] / r[noisy] > 0.5) & (g[noisy] / r[noisy] < 2)):.0%}")
    print(f"near-noise-free tasks (sweep <= 0.003): {(~noisy).sum()}; Gamma estimate median {np.median(g[~noisy]):.4f}, max {g[~noisy].max():.4f}")


def gp_noise(X, y, lo, hi):
    """The noise level fitted by a Gaussian process with a white-noise kernel (Matern 5/2, one length scale per input),
    in units of std(y): a model-based estimate."""
    import warnings
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    U = (np.asarray(X, float) - lo) / (hi - lo)
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(U.shape[1]), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-4, (1e-12, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(U, y)
    return float(np.sqrt(g.kernel_.k2.noise_level))          # normalize_y: already in units of std(y)
