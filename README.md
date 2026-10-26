# Driftbench

A small, reproducible comparison of five data-drift statistics (PSI, KS, Wasserstein, MMD, and a mean z-score baseline) on synthetic streams with a known change point. Every detector's threshold is **calibrated to the same stream-level false-alarm rate (5%)** before power and detection delay are compared, so the comparison is fair.

**Status: Day 5 MVP, synthetic data only, one sandbox run.** Nothing here was run on real model features.

## Problem and existing tools
Evidently, Alibi Detect and whylogs already ship these detectors; this repo adds nothing to them. What I wanted to check is narrower: (1) how often the common out-of-the-box rules raise false alarms when you test every batch of a stream, and (2) which detector notices which kind of shift once thresholds are equalised. A paper on the first point exists ("When Drift Detectors cry Wolf", arXiv 2607.17336, found via search; I did not read it in depth, so I make no claim about how my numbers relate to it).

## Setup
- Reference: 1000 N(0,1) samples. Each batch: 200 samples, scored against the reference.
- Stream: 30 clean batches, then 30 shifted batches.
- Calibration: threshold = 95th percentile of the max score over 30 clean batches (300 simulated streams), so a clean 30-batch stream alarms with probability ~5%.
- Shifts (magnitude in the first column): `mean` = shift in std units; `scale` = std x (1+mag); `bimodal` = mixture of N(+-a, 1-a^2), identical mean and variance to N(0,1); `tail` = Student-t, df from 10 (mag 0) to 2 (mag 1), rescaled (mag 1 is only roughly unit-variance).
- Power = fraction of 100 streams alarming within the 30 shifted batches. Delay = median batches to first alarm among detected streams.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                                   # 10 tests
.venv/bin/python -m driftbench.experiment             # about 9 min (MMD dominates)
.venv/bin/python -m driftbench.report results/experiment.json
```

## Results (100 streams per cell; see `results/experiment_table.txt`)

**1. Out-of-the-box rules on clean data, tested on every batch of a 30-batch stream (no drift present):**

| rule | share of clean streams that raised at least one alarm |
|---|---|
| PSI > 0.1 | 82% |
| KS test p < 0.05 | 74% |
| PSI > 0.2 | 0.7% |

The p-value rule is expected to over-alarm when repeated 30 times (1 - 0.95^30 = 78% if tests were independent; mine reuse one reference, so not identical, but the same order). PSI 0.1 over-alarms at these sample sizes. PSI 0.2 is the opposite: very quiet here, though I did not measure what it misses.

**2. After calibration** all five hold the stream false-alarm rate near 5% on fresh clean streams (measured: PSI 4%, KS 2%, Wasserstein 5%, MMD 5%, mean-z 6%; with 100 streams the standard error is about 2 points).

**3. Power / median delay (power, then delay in batches):**

| shift | PSI | KS | Wasserstein | MMD | mean z |
|---|---|---|---|---|---|
| mean 0.2 | 0.70 (10.5) | 0.96 (5) | 0.99 (2) | 0.89 (5) | 0.99 (2) |
| scale 0.25 | 0.92 (3.5) | 0.57 (11) | 1.00 (2) | 1.00 (3) | 0.04 (12) |
| bimodal 0.8 | 0.84 (9) | 0.53 (11) | 0.05 (14) | 0.40 (9) | 0.02 (13.5) |
| bimodal 0.95 | 1.00 (1) | 1.00 (1) | 1.00 (1) | 1.00 (1) | 0.06 (13.5) |
| tail 0.8 | 0.95 (3) | 0.65 (9) | 0.78 (7) | 0.92 (4) | 0.02 (16) |

What I take from it, for this setup only:
- No detector wins everywhere. Wasserstein and the mean z-score are best on a plain mean shift; the z-score is blind to scale, bimodal and tail changes (its power stays at the false-alarm level, as the tests also assert for bimodal).
- On the same-mean-and-variance bimodal shift, PSI was best (0.84 at mag 0.8) and Wasserstein was at chance (0.05). Wasserstein measures how far mass moves, and this shift moves little.
- KS was weaker than PSI and MMD on scale and tail shifts here.
- Tiny shifts (mean 0.1, scale 0.1, bimodal 0.5) are poorly detected by everything at 200 samples per batch.

## Limitations (read these)
- **Synthetic, 1-D, Gaussian base.** Real features are multivariate, discrete, correlated and seasonal. Nothing here says which detector to use on real data.
- **PSI uses 10 reference-decile bins and MMD uses a median-heuristic RBF kernel on a 500-point reference subsample.** Other bin counts and bandwidths would shift the ranking; I did not sweep them. Wasserstein is in the data's units, so its threshold is only valid for this unit-scale data.
- **Stream-level calibration is my design choice** (alarm on any of 30 batches). A different alarm policy (e.g. require 2 consecutive batches) would change everything.
- **Calibration uses 300 streams per detector**, so thresholds carry sampling error, visible in the 2%-6% false-alarm spread. One seed, one run; no confidence intervals on power.
- **Detectors are my own implementations** (KS and Wasserstein cross-checked against SciPy in tests; PSI against a hand-computed value; MMD only checked for sign, unbiasedness and sensitivity, not against Alibi Detect or another library).
- Abrupt shifts only; no gradual drift, and no online detectors (CUSUM, ADWIN, etc.).
- Run in a 2-vCPU sandbox; timings are not meaningful.

## Future work
Check my MMD and PSI against Evidently/Alibi Detect outputs; sweep bin counts and bandwidths; multivariate and categorical features; gradual drift; sequential detectors; real data.
