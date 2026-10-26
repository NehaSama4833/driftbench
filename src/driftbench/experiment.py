"""Benchmark. Reference = N_REF clean samples. A stream of batches of size BATCH; each batch is
scored against the reference. Thresholds are calibrated so that a stream of NB null batches has
a 5% chance of raising any alarm (stream-level false-alarm rate). Then detectors are compared
on power (alarm within the post-change batches) and delay (batches until first alarm)."""
import argparse, json, time
import numpy as np
from .detectors import DETECTORS, psi, ks_pvalue
from .streams import sample, KINDS

N_REF, BATCH, NB_PRE, NB_POST = 1000, 200, 30, 30
ALPHA = 0.05


def null_max_stats(det, rng, reps):
    out = []
    for _ in range(reps):
        ref = sample("mean", 0, N_REF, rng)
        out.append(max(det(ref, sample("mean", 0, BATCH, rng)) for _ in range(NB_PRE)))
    return np.array(out)


def calibrate(det, seed, reps):
    return float(np.quantile(null_max_stats(det, np.random.default_rng(seed), reps), 1 - ALPHA))


def run_stream(det, thr, kind, mag, rng):
    ref = sample("mean", 0, N_REF, rng)
    pre = [det(ref, sample("mean", 0, BATCH, rng)) > thr for _ in range(NB_PRE)]
    post = [det(ref, sample(kind, mag, BATCH, rng)) > thr for _ in range(NB_POST)]
    delay = post.index(True) + 1 if any(post) else None
    return any(pre), delay


def default_threshold_far(seed, reps):
    """Stream false-alarm rate of rules people use out of the box, on clean data."""
    rng = np.random.default_rng(seed)
    res = {"psi>0.1": 0, "psi>0.2": 0, "ks_p<0.05": 0}
    for _ in range(reps):
        ref = sample("mean", 0, N_REF, rng)
        wins = [sample("mean", 0, BATCH, rng) for _ in range(NB_PRE)]
        res["psi>0.1"] += any(psi(ref, w) > 0.1 for w in wins)
        res["psi>0.2"] += any(psi(ref, w) > 0.2 for w in wins)
        res["ks_p<0.05"] += any(ks_pvalue(ref, w) < 0.05 for w in wins)
    return {k: v / reps for k, v in res.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-reps", type=int, default=300)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/experiment.json")
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    a = ap.parse_args(argv)
    dets = {k: DETECTORS[k] for k in a.detectors.split(",")}
    t0 = time.time()
    out = {"config": dict(N_REF=N_REF, BATCH=BATCH, NB_PRE=NB_PRE, NB_POST=NB_POST, ALPHA=ALPHA,
                          cal_reps=a.cal_reps, reps=a.reps, seed=a.seed)}
    out["default_rules_far"] = default_threshold_far(a.seed, a.cal_reps)
    print("default rules, stream false-alarm rate on clean data:", out["default_rules_far"], flush=True)
    out["thresholds"], out["cells"] = {}, []
    mags = {"mean": [0.1, 0.2, 0.5], "scale": [0.1, 0.25, 0.5], "bimodal": [0.5, 0.8, 0.95], "tail": [0.5, 0.8, 1.0]}
    for name, det in dets.items():
        thr = calibrate(det, a.seed, a.cal_reps)
        out["thresholds"][name] = thr
        rng = np.random.default_rng(a.seed + 1000)
        far = np.mean([run_stream(det, thr, "mean", 0, rng)[0] for _ in range(a.reps)])
        out["cells"].append(dict(det=name, kind="null", mag=0, power=None, far=float(far), median_delay=None))
        for kind in KINDS:
            for mag in mags[kind]:
                rng = np.random.default_rng(a.seed + 2000)
                r = [run_stream(det, thr, kind, mag, rng) for _ in range(a.reps)]
                d = [x[1] for x in r if x[1] is not None]
                out["cells"].append(dict(det=name, kind=kind, mag=mag, power=len(d) / a.reps,
                                         far=float(np.mean([x[0] for x in r])),
                                         median_delay=float(np.median(d)) if d else None))
        print(name, "thr=%.4g" % thr, "done %.0fs" % (time.time() - t0), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
