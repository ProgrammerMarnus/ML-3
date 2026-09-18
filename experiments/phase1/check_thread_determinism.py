#!/usr/bin/env python3
"""
Cross-process reproducibility check.

Observation: `check_determinism.py` is bit-reproducible *within* a process
(3 identical repeats), yet a separate process running the same AAPL h=1
evaluation produced sharpe -0.74 / pf 1.26, while the determinism process
produced sharpe -0.4876 / pf 1.3442.

Hypothesis: LightGBMWrapper hardcodes `n_jobs: -1` (models/predictors.py:98).
Histogram construction is a parallel reduction, so the number of OpenMP
threads changes floating-point summation order, which can flip near-tie
split decisions. On a shared machine the effective thread count varies
between runs -> results vary between runs.

Test: run this script 3 times with different OMP_NUM_THREADS and compare.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
for _ in range(5):
    if os.path.isdir(os.path.join(REPO, "market_predictor_ml")):
        break
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from run_horizon_matched import build_panel, evaluate_horizon  # noqa: E402

TICKER = "AAPL"


def main():
    threads = os.environ.get("OMP_NUM_THREADS", "(unset)")
    print(f"OMP_NUM_THREADS={threads}  cpu_count={os.cpu_count()}")
    for h in (1, 21):
        panel = build_panel(TICKER, f"Target_RiskAdj_{h}d")
        ev = evaluate_horizon(panel, h)
        n = ev["net"]
        print(
            f"  h={h:<3} sharpe={n['sharpe_ratio']:+.6f} pf={n['profit_factor']:.6f} "
            f"ann={n['annual_return']*100:+.6f}% n_days={ev['n_days']}"
        )


if __name__ == "__main__":
    main()