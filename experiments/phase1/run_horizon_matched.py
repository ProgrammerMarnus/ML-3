#!/usr/bin/env python3
"""
Phase 1b - horizon-matched, economically valid re-measurement.

WHY THIS EXISTS
---------------
run_validation.py (Phase 1) produced numbers that are NOT economically
interpretable. Root cause is in the label/backtest pairing:

    pipeline.prepare_data(target_column='Target_RiskAdj_21d')
    backtest/engine.py:  strategy_returns[1:] = positions[:-1] * y_test[1:]

`y_test` is the *21-day forward* risk-adjusted return
(Target_RiskAdj_21d = close[t+21]/close[t] - 1, divided by annualised
21-day volatility). The engine multiplies the position by that 21-day
quantity and treats the result as a DAILY return, then compounds it.

Consequences:
  1. The resulting series is not a P&L that can be earned. Its units are
     "21-day return / annualised vol", not a daily return.
  2. Overlapping 21-day windows make the series ~95% autocorrelated at
     lag 1, which inflates the annualised Sharpe by roughly sqrt(21) ~ 4.6x.
  3. The training horizon (21d) does not match the 1-day holding return the
     engine credits.

This script re-measures the SAME signal with correct accounting: positions
are converted to P&L using the ACTUAL next-day realised return, for two
horizons that each match their own label:

    H=1   daily rebalance,  trained on Target_RiskAdj_1d
    H=21  rebalance every 21 days, trained on Target_RiskAdj_21d

Plus:
  * a buy-and-hold benchmark on the identical fold dates (context only, never
    treated as alpha), and
  * a Newey-West-style deflation of the original mismatched series, so the
    overstatement of the Phase 1 numbers is quantified rather than asserted.

Nothing here changes production code. It is a measurement instrument.

GATE 1 CRITERIA (unchanged from the plan):
    sharpe_ratio > 0.5 AND profit_factor > 1.2 AND max_drawdown > -0.25
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
for _ in range(5):
    if os.path.isdir(os.path.join(_REPO, "market_predictor_ml")):
        break
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)

from market_predictor_ml import MarketPredictorPipeline, Config
from market_predictor_ml.backtest.engine import (
    WalkForwardSplit,
    compute_economic_metrics,
)
from market_predictor_ml.config.settings import BacktestConfig
from market_predictor_ml.decision import create_positions
from market_predictor_ml.models import get_model
from market_predictor_ml.utils import (
    winsorize_features,
    remove_near_zero_variance_features,
)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
START, END = "2015-01-01", "2024-12-31"
N_SPLITS, TEST_SIZE, PURGE, EMBARGO = 5, 252, 5, 5
COST, SLIPPAGE = 0.001, 0.0005
HORIZONS = [1, 21]
MISMATCHED_TARGET = "Target_RiskAdj_21d"


def make_config() -> Config:
    cfg = Config()
    cfg.backtest = BacktestConfig(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        purge_size=PURGE,
        embargo_size=EMBARGO,
        transaction_cost=COST,
        slippage=SLIPPAGE,
        risk_free_rate=0.02,
    )
    cfg.labels.return_horizons = [1, 5, 21]
    cfg.features.variance_threshold = 1e-4
    cfg.model.lightgbm_n_estimators = 500
    cfg.model.lightgbm_early_stopping_rounds = 50
    return cfg


def _new_model():
    return get_model(
        "lightgbm",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        early_stopping_rounds=50,
    )


def build_panel(ticker: str, target_column: str) -> Dict:
    """
    Features, dates, next-day realised returns and volatility, all row-aligned
    to exactly the rows `prepare_data` would keep for `target_column`.
    """
    cfg = make_config()
    p = MarketPredictorPipeline(config=cfg)
    p.load_data(ticker=ticker, start_date=START, end_date=END)
    p.engineer_features()
    p.create_labels()

    full_features: List[str] = list(p.feature_names_)

    # Same row filter as prepare_data: drop rows with any NaN feature/label.
    panel = p.labels_[full_features + [target_column]].dropna()
    dates = panel.index

    close = p.data_["Close"].reindex(dates)
    next_day_return = close.shift(-1) / close - 1.0

    vol = p.labels_["Volatility_Realized_21d"].reindex(dates)

    X = panel[full_features].to_numpy(dtype=float)
    X = winsorize_features(
        X,
        lower_percentile=cfg.features.winsorize_lower,
        upper_percentile=cfg.features.winsorize_upper,
    )
    X, keep = remove_near_zero_variance_features(
        X, threshold=cfg.features.variance_threshold
    )
    feature_names = [full_features[i] for i in keep]

    return {
        "X": X,
        "y": panel[target_column].to_numpy(dtype=float),
        "dates": dates,
        "next_day_return": next_day_return.to_numpy(dtype=float),
        "volatility": vol.to_numpy(dtype=float),
        "feature_names": feature_names,
    }

def evaluate_horizon(panel: Dict, horizon: int) -> Dict:
    """
    Walk-forward P&L with the ACTUAL next-day return.

    Position is decided at the close of day t and earns
    close[t+1]/close[t] - 1.  With horizon > 1 the position is refreshed on
    the first day of each block and held for `horizon` days, so turnover (and
    therefore cost drag) falls by roughly the horizon factor.
    """
    X = panel["X"]
    y = panel["y"]
    r_next = panel["next_day_return"]
    vol = panel["volatility"]
    dates = panel["dates"]

    splitter = WalkForwardSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        purge_size=PURGE,
        embargo_size=EMBARGO,
    )

    net_all, pos_all, ret_all, bh_all, date_all = [], [], [], [], []

    for train_idx, test_idx in splitter.split(X, y):
        model = _new_model()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        test_vol = vol[test_idx]
        reb = np.arange(0, len(test_idx), horizon)
        raw = create_positions(
            preds[reb], volatility=test_vol[reb], method="volatility_adjusted"
        )

        pos = np.zeros(len(test_idx))
        for k, j in enumerate(reb):
            pos[j: min(j + horizon, len(test_idx))] = raw[k]

        r = r_next[test_idx]
        gross = pos * r
        changes = np.abs(np.diff(pos, prepend=0.0))
        net = gross - changes * (COST + SLIPPAGE)

        valid = np.isfinite(net)
        net_all.append(net[valid])
        pos_all.append(pos[valid])
        ret_all.append(gross[valid])
        bh_all.append(r[valid])
        date_all.extend(list(dates[test_idx][valid]))

    net = np.concatenate(net_all)
    pos = np.concatenate(pos_all)
    gross = np.concatenate(ret_all)
    bh = np.concatenate(bh_all)

    metrics = compute_economic_metrics(net, pos, transaction_cost=COST)
    gross_metrics = compute_economic_metrics(gross, pos, transaction_cost=COST)
    bh_metrics = compute_economic_metrics(bh, np.ones_like(bh), transaction_cost=0.0)
    bh_metrics["total_return"] = float(np.prod(1 + bh) - 1)

    return {
        "horizon": horizon,
        "n_days": int(len(net)),
        "net": metrics,
        "gross": gross_metrics,
        "buy_hold": bh_metrics,
        "cost_drag_ann": float((gross_metrics["annual_return"] or 0)
                               - (metrics["annual_return"] or 0)),
        "equity_curve": np.cumprod(1 + net),
        "dates": date_all,
    }


def overlap_inflation(series: np.ndarray, max_lag: int = 40) -> Dict:
    """
    Quantify how much overlapping (h-day) labels inflate a Sharpe computed on
    the mismatched series. Uses the Newey-West/Hansen-Hodrick correction:

        factor = sqrt(1 + 2 * sum_{k=1..K} (1 - k/(K+1)) * rho_k)

    A correct i.i.d. series has factor ~1.0. An h-day overlapping series has
    factor ~sqrt(h). Deflated Sharpe = naive Sharpe / factor.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max_lag + 10:
        return {"lag1_autocorr": None, "inflation_factor": None}
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return {"lag1_autocorr": None, "inflation_factor": None}
    rho = [float(np.dot(x[:-k], x[k:]) / denom) for k in range(1, max_lag + 1)]
    factor = float(
        np.sqrt(
            max(
                1e-12,
                1.0 + 2.0 * sum((1 - k / (max_lag + 1)) * r for k, r in enumerate(rho, 1)),
            )
        )
    )
    return {"lag1_autocorr": rho[0], "inflation_factor": factor}


def mismatched_series(panel_mismatch: Dict) -> np.ndarray:
    """Reproduce the Phase 1 (invalid) return series for deflation analysis."""
    X = panel_mismatch["X"]
    y = panel_mismatch["y"]
    splitter = WalkForwardSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        purge_size=PURGE,
        embargo_size=EMBARGO,
    )
    out = []
    for train_idx, test_idx in splitter.split(X, y):
        model = _new_model()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        pos = create_positions(
            preds, volatility=panel_mismatch["volatility"][test_idx],
            method="volatility_adjusted",
        )
        n = len(test_idx)
        sr = np.zeros(n)
        if n > 1:
            sr[1:] = pos[:-1] * y[test_idx][1:]
        changes = np.abs(np.diff(pos, prepend=0.0))
        out.append(sr - changes * (COST + SLIPPAGE))
    return np.concatenate(out)

def _serializable(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _serializable(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def verdict(metrics: Dict) -> str:
    sh = metrics.get("sharpe_ratio")
    pf = metrics.get("profit_factor")
    dd = metrics.get("max_drawdown")
    if sh is None or pf is None or dd is None:
        return "INCONCLUSIVE"
    if not np.isfinite(sh) or not np.isfinite(pf):
        return "FAIL"
    return "PASS" if (sh > 0.5 and pf > 1.2 and dd > -0.25) else "FAIL"


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results: Dict = {}

    for ticker in TICKERS:
        print(f"\n=== {ticker} ===", flush=True)
        per_ticker: Dict = {"horizons": {}, "diagnosis": {}}

        # Diagnosis: how much did overlapping labels inflate the Phase 1 number?
        try:
            p_mm = build_panel(ticker, MISMATCHED_TARGET)
            mm = mismatched_series(p_mm)
            mm_metrics = compute_economic_metrics(
                mm, np.zeros_like(mm), transaction_cost=COST
            )
            infl = overlap_inflation(mm)
            factor = infl["inflation_factor"] or 1.0
            per_ticker["diagnosis"] = {
                "phase1_sharpe_reproduced": float(mm_metrics.get("sharpe_ratio") or 0.0),
                "lag1_autocorr": infl["lag1_autocorr"],
                "inflation_factor": infl["inflation_factor"],
                "deflated_sharpe": float((mm_metrics.get("sharpe_ratio") or 0.0) / factor),
            }
        except Exception as exc:  # keep going, this block is diagnostic only
            per_ticker["diagnosis"] = {"error": repr(exc)}

        # Correct measurement at each horizon.
        for h in HORIZONS:
            target = f"Target_RiskAdj_{h}d"
            try:
                panel = build_panel(ticker, target)
                ev = evaluate_horizon(panel, h)
                ev_clean = {
                    "target": target,
                    "n_days": ev["n_days"],
                    "net": _serializable(ev["net"]),
                    "gross": _serializable(ev["gross"]),
                    "buy_hold": _serializable(ev["buy_hold"]),
                    "cost_drag_ann": ev["cost_drag_ann"],
                    "verdict": verdict(ev["net"]),
                }
                per_ticker["horizons"][f"h{h}"] = ev_clean
                print(
                    f"  h={h:<2} net sharpe={ev['net']['sharpe_ratio']:>6.2f} "
                    f"pf={ev['net']['profit_factor']:>5.2f} "
                    f"maxdd={ev['net']['max_drawdown']*100:>6.2f}% "
                    f"ann={ev['net']['annual_return']*100:>7.2f}% "
                    f"| gross sharpe={ev['gross']['sharpe_ratio']:>6.2f} "
                    f"| bh ann={ev['buy_hold']['annual_return']*100:>6.2f}% "
                    f"| {ev_clean['verdict']}",
                    flush=True,
                )
            except Exception as exc:
                per_ticker["horizons"][f"h{h}"] = {"error": repr(exc)}
                print(f"  h={h:<2} ERROR {exc!r}", flush=True)

        results[ticker] = per_ticker

    stamp = datetime.date.today().isoformat()
    out_path = os.path.join(out_dir, f"validation_horizon_matched_{stamp}.json")

    summary_counts = {f"h{h}": 0 for h in HORIZONS}
    for t, d in results.items():
        for h in HORIZONS:
            if d["horizons"].get(f"h{h}", {}).get("verdict") == "PASS":
                summary_counts[f"h{h}"] += 1

    payload = {
        "generated_at": stamp,
        "purpose": "Phase 1b: horizon-matched, economically valid re-measurement",
        "config": {
            "tickers": TICKERS,
            "start": START,
            "end": END,
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "purge_size": PURGE,
            "embargo_size": EMBARGO,
            "transaction_cost": COST,
            "slippage": SLIPPAGE,
            "effective_cost_per_unit_turnover": COST + SLIPPAGE,
            "position_method": "volatility_adjusted",
            "horizons": HORIZONS,
            "pnl_basis": "position[t] * (close[t+1]/close[t]-1) - |dposition|*cost",
        },
        "gate1": {
            "pass_criteria": "sharpe > 0.5 AND profit_factor > 1.2 AND max_drawdown > -0.25",
            "pass_count_by_horizon": summary_counts,
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    for h in HORIZONS:
        print(f"\nGate 1 scorecard (h={h}, net of costs):")
        print("Ticker | Sharpe | PF    | MaxDD    | AnnRet   | BH AnnRet | Verdict")
        print("-" * 78)
        for t in TICKERS:
            r = results[t]["horizons"].get(f"h{h}", {})
            if "error" in r:
                print(f"{t:<6} | ERROR: {r['error'][:50]}")
                continue
            n = r["net"]
            print(
                f"{t:<6} | {n['sharpe_ratio']:>6.2f} | {n['profit_factor']:>5.2f} | "
                f"{n['max_drawdown']*100:>7.2f}% | {n['annual_return']*100:>7.2f}% | "
                f"{r['buy_hold']['annual_return']*100:>8.2f}% | {r['verdict']}"
            )


if __name__ == "__main__":
    main()
