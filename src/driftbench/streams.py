"""Synthetic shifts. All shifts are applied to a standard-normal base; `mag` = 0 means no shift."""
import numpy as np

KINDS = ["mean", "scale", "bimodal", "tail"]


def sample(kind, mag, n, rng):
    z = rng.standard_normal(n)
    if mag == 0:
        return z
    if kind == "mean":      # shift by mag standard deviations
        return z + mag
    if kind == "scale":     # std multiplied by (1 + mag)
        return z * (1 + mag)
    if kind == "bimodal":   # same mean and variance as N(0,1), but two modes at +-a
        a = mag                      # a in [0,1): mixture of N(+-a, 1-a^2) has variance 1
        s = np.sqrt(1 - a * a)
        return rng.choice([-a, a], n) + s * z
    if kind == "tail":      # heavier tails (Student t, df shrinking), rescaled to unit variance
        df = 2.0 + 8.0 * (1 - mag)   # mag 1 -> df 2 (infinite var; scale by sqrt(3) as a rough match)
        t = rng.standard_t(df, n)
        return t * np.sqrt((df - 2) / df) if df > 2.5 else t / np.sqrt(3.0)
    raise ValueError(kind)
