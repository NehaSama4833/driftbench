import numpy as np
from driftbench import experiment as ex
from driftbench.detectors import ks

def test_calibration_hits_target_on_fresh_seed():
    thr = ex.calibrate(ks, seed=1, reps=200)
    rng = np.random.default_rng(99)
    far = np.mean([ex.run_stream(ks, thr, "mean", 0, rng)[0] for _ in range(300)])
    assert 0.01 < far < 0.11     # target 0.05; 300 runs gives SE ~0.013

def test_big_mean_shift_detected_at_once():
    thr = ex.calibrate(ks, seed=1, reps=100)
    rng = np.random.default_rng(3)
    r = [ex.run_stream(ks, thr, "mean", 1.0, rng) for _ in range(20)]
    assert all(d == 1 for _, d in r)

def test_default_rules_overalarm_on_clean_data():
    far = ex.default_threshold_far(seed=2, reps=60)
    assert far["ks_p<0.05"] > 0.5
