"""Two-sample drift statistics. Each takes (reference, window) 1-D arrays and returns a
score where bigger = more drift. Own implementations so they are testable; KS and
Wasserstein are cross-checked against scipy in tests."""
import numpy as np
from scipy import stats

EPS = 1e-4


def psi(ref, win, bins=10):
    """Population Stability Index with bins = reference deciles (common industry recipe)."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(win, edges)[0] / len(win)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks(ref, win):
    return float(stats.ks_2samp(ref, win).statistic)


def ks_pvalue(ref, win):
    return float(stats.ks_2samp(ref, win).pvalue)


def wasserstein(ref, win):
    return float(stats.wasserstein_distance(ref, win))


def mean_z(ref, win):
    """|difference of means| in standard errors: only sees mean shifts (baseline)."""
    se = np.sqrt(np.var(ref, ddof=1) / len(ref) + np.var(win, ddof=1) / len(win))
    return float(abs(np.mean(win) - np.mean(ref)) / se)


def mmd(ref, win, max_ref=500, seed=0):
    """Unbiased MMD^2 with RBF kernel, bandwidth = median pairwise distance of a reference
    subsample (median heuristic). The reference is subsampled to max_ref for speed."""
    rng = np.random.default_rng(seed)
    x = ref if len(ref) <= max_ref else rng.choice(ref, max_ref, replace=False)
    y = np.asarray(win)
    d = np.abs(x[:, None] - x[None, :])
    h = np.median(d[np.triu_indices(len(x), 1)])
    h = h if h > 0 else 1.0
    k = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * h * h))
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    n, m = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1))
                 + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())


DETECTORS = {"psi": psi, "ks": ks, "wasserstein": wasserstein, "mmd": mmd, "mean_z": mean_z}
