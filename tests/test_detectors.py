import numpy as np, pytest
from scipy import stats
from driftbench.detectors import psi, ks, wasserstein, mean_z, mmd
from driftbench.streams import sample, KINDS

rng = np.random.default_rng(0)
A = rng.standard_normal(1000)

def test_ks_wasserstein_match_scipy():
    B = rng.standard_normal(300) + 0.3
    assert ks(A, B) == stats.ks_2samp(A, B).statistic
    assert wasserstein(A, B) == stats.wasserstein_distance(A, B)

def test_identical_samples_score_zero():
    assert psi(A, A) == pytest.approx(0, abs=1e-9)
    assert ks(A, A) == 0 and wasserstein(A, A) == 0

def test_psi_hand_computed():
    ref = np.arange(100.0)                      # deciles split into 10 equal bins of 10
    win = np.arange(50.0)                       # only the lower five bins, 20% each
    p = np.full(10, 0.1); q = np.array([.2]*5 + [1e-4]*5)
    expect = float(np.sum((q - p) * np.log(q / p)))
    assert psi(ref, win) == pytest.approx(expect, rel=1e-3)

def test_scores_grow_with_mean_shift():
    for f in (psi, ks, wasserstein, mean_z, mmd):
        s = [f(A, sample("mean", m, 2000, np.random.default_rng(5))) for m in (0, 0.5, 1.5)]
        assert s[0] < s[1] < s[2], f.__name__

def test_mmd_unbiased_near_zero_under_null():
    vals = [mmd(A, np.random.default_rng(i).standard_normal(200)) for i in range(40)]
    assert abs(np.mean(vals)) < 3 * np.std(vals) / np.sqrt(40) + 1e-3  # unbiased: mean ~ 0, can be negative

def test_mean_z_blind_to_bimodal_but_ks_psi_mmd_see_it():
    B = sample("bimodal", 0.95, 20000, np.random.default_rng(2))
    assert abs(B.mean()) < 0.03 and abs(B.var() - 1) < 0.05   # same first two moments
    assert mean_z(A, B) < 3
    assert psi(A, B) > 0.2 and ks(A, B) > 0.1 and mmd(A, B[:300]) > 0.01

def test_samplers_shapes_and_zero_mag():
    r = np.random.default_rng(1)
    for k in KINDS:
        assert sample(k, 0.5, 10, r).shape == (10,)
    x, y = sample("tail", 0, 5, np.random.default_rng(3)), np.random.default_rng(3).standard_normal(5)
    assert np.allclose(x, y)
    with pytest.raises(ValueError):
        sample("nope", 1, 5, r)
