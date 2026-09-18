#!/usr/bin/env python3
"""
Locate the source of cross-process non-determinism: data or model?

Prints a hash of the feature matrix, the label vector and the realised return
series for each ticker/target, so two separate processes can be compared.
"""
import hashlib
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

from run_horizon_matched import build_panel  # noqa: E402


def h(a):
    return hashlib.md5(a.tobytes()).hexdigest()[:16]


def main():
    print("pid", os.getpid())
    for target in ("Target_RiskAdj_1d", "Target_RiskAdj_21d"):
        panel = build_panel("AAPL", target)
        X, y = panel["X"], panel["y"]
        r = panel["next_day_return"]
        print(f"{target}")
        print(f"  n_rows={X.shape[0]} n_features={X.shape[1]}")
        print(f"  X hash={h(X)}  y hash={h(y)}  r_next hash={h(r)}")
        print(f"  y[0]={y[0]:.12f}  y[-1]={y[-1]:.12f}")
        print(f"  r[0]={r[0]:.12f}  nan_in_r={int(sum(1 for v in r if v != v))}")
        print(f"  dates {panel['dates'][0].date()} .. {panel['dates'][-1].date()}")


if __name__ == "__main__":
    main()