#!/usr/bin/env python3
"""
Determinism check: does the same seed produce the same walk-forward result?

Motivation: two identical Phase 1 runs produced AAPL sharpe 1.6178 with
profit_factor 1.2036 (PASS) and then 1.63 with 1.19 (FAIL). If results are not
reproducible, then any metric sitting near a Gate threshold is a coin flip.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
for _ in range(5):
    if os.path.isdir(os.path.join(REPO, "market_predictor_ml")):
        break
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from run_horizon_matched import (  # noqa: E402
    build_panel, evaluate_horizon, make_config, _new_model,
)
from market_predictor_ml.backtest.engine import WalkForwardSplit  # noqa: E402

TICKER = "AAPL"
TARGET = "Target_RiskAdj_21d"
REPEATS = 3


def main():
    panel = build_panel(TICKER, TARGET)
    X, y = panel["X"], panel["y"]
    splitter = WalkForwardSplit(n_splits=5, test_size=252, purge_size=5, embargo_size=5)
    folds = list(splitter.split(X, y))
    tr, te = folds[0]

    print(f"Determinism check on {TICKER}, fold 1 (train={len(tr)}, test={len(te)})")
    preds = []
    for i in range(REPEATS):
        m = _new_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        preds.append(p)
        print(f"  repeat {i}: best_iter={m.best_iteration_} "
              f"pred_mean={p.mean():+.6f} pred_std={p.std():.6f}")

    d01 = float(np.max(np.abs(preds[0] - preds[1])))
    d02 = float(np.max(np.abs(preds[0] - preds[2])))
    print(f"  max |pred_0 - pred_1| = {d01:.3e}")
    print(f"  max |pred_0 - pred_2| = {d02:.3e}")
    print()
    if max(d01, d02) > 1e-9:
        print("VERDICT: NON-DETERMINISTIC - identical seed/config gives different predictions.")
        print("         Any metric within ~1x this spread of a Gate threshold is noise.")
    else:
        print("VERDICT: deterministic at fold level.")

    print()
    print("Full-metric repeatability (h=1, net of costs):")
    panel_h1 = build_panel(TICKER, "Target_RiskAdj_1d")
    for i in range(REPEATS):
        ev = evaluate_horizon(panel_h1, 1)
        n = ev["net"]
        print(f"  run {i}: sharpe={n['sharpe_ratio']:+.4f} pf={n['profit_factor']:.4f} "
              f"ann={n['annual_return']*100:+.4f}% n_days={ev['n_days']}")


if __name__ == "__main__":
    main()