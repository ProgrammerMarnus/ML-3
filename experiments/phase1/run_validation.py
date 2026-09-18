#!/usr/bin/env python3
"""Phase 1 - decision-grade validation script."""
from __future__ import annotations
import datetime, json, os, sys
from typing import Dict
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
for _ in range(5):
    if os.path.isdir(os.path.join(_REPO, "market_predictor_ml")):
        break
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)
from market_predictor_ml import MarketPredictorPipeline, Config
from market_predictor_ml.config.settings import BacktestConfig
from market_predictor_ml.backtest.engine import WalkForwardSplit, run_walk_forward_backtest
from market_predictor_ml.models import get_model

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
START = "2015-01-01"
END = "2024-12-31"
BASE_BACKTEST = BacktestConfig(
    n_splits=5, test_size=252, purge_size=5, embargo_size=5,
    transaction_cost=0.001, slippage=0.0005, risk_free_rate=0.02,
)

def build_pipeline_config() -> Config:
    cfg = Config(); cfg.backtest = BASE_BACKTEST
    cfg.labels.return_horizons = [1,5,21]; cfg.labels.label_method="future_return"
    cfg.features.variance_threshold=1e-4
    cfg.model.lightgbm_n_estimators=500; cfg.model.lightgbm_early_stopping_rounds=50
    cfg.decision.default_method="volatility_adjusted"; cfg.decision.prediction_threshold=0.0
    return cfg

def run_direct_backtest_for_ticker(ticker, start_date, end_date):
    pipeline = MarketPredictorPipeline(config=build_pipeline_config())
    try:
        pipeline.load_data(ticker=ticker, start_date=start_date, end_date=end_date)
        pipeline.engineer_features(); pipeline.create_labels()
        pipeline.prepare_data(target_column="Target_RiskAdj_21d")
    except Exception as exc:
        return {"error": f"data/feature/label/prepare failed: {repr(exc)}"}
    if pipeline.X_ is None or pipeline.y_ is None:
        return {"error": "prepare_data produced no X/y"}
    X = np.asarray(pipeline.X_, dtype=np.float64)
    y = np.asarray(pipeline.y_, dtype=np.float64)
    cv = WalkForwardSplit(n_splits=BASE_BACKTEST.n_splits, test_size=BASE_BACKTEST.test_size,
                          purge_size=BASE_BACKTEST.purge_size, embargo_size=BASE_BACKTEST.embargo_size)
    def model_factory():
        return get_model("lightgbm", n_estimators=500, learning_rate=0.05, max_depth=6,
                         num_leaves=31, min_child_samples=50, subsample=0.8,
                         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, early_stopping_rounds=50)
    model = model_factory()
    try:
        results = run_walk_forward_backtest(model=model, X=X, y=y, cv_splitter=cv,
            position_method="volatility_adjusted", volatility=pipeline.volatility_,
            transaction_cost=BASE_BACKTEST.transaction_cost)
    except TypeError:
        results = run_walk_forward_backtest(model=model, X=X, y=y, cv_splitter=cv,
            position_method="volatility_adjusted", transaction_cost=BASE_BACKTEST.transaction_cost)
    if results is None:
        return {"error": "run_walk_forward_backtest returned None"}
    metrics = results.get("metrics") or {}
    n_idx = int(results.get("indices", np.array([])).size)
    metrics["_fold_count"] = n_idx // max(1, BASE_BACKTEST.test_size)
    return {"metrics": metrics, "results": results}

def summarize_metrics(metrics):
    wanted = ["total_return","annual_return","volatility","sharpe_ratio","sortino_ratio",
              "max_drawdown","calmar_ratio","win_rate","profit_factor","turnover"]
    out = {}
    for k in wanted:
        v = metrics.get(k)
        if v is None: out[k]=None
        elif isinstance(v,(int,float,np.floating,np.integer)): out[k]=float(v)
        else: out[k]=v
    return out

def verdict_for(metrics):
    sh=pf=dd=None
    sh=metrics.get("sharpe_ratio"); pf=metrics.get("profit_factor"); dd=metrics.get("max_drawdown")
    if sh is None or pf is None or dd is None: return "INCONCLUSIVE"
    if sh>0.5 and pf>1.2 and dd>-0.25: return "PASS"
    return "FAIL"

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    results={}
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")
        r=run_direct_backtest_for_ticker(ticker, START, END)
        if "error" in r: summary={"error": r["error"]}
        else: summary=summarize_metrics(r["metrics"])
        summary["verdict"]=verdict_for(r.get("metrics",{}) if "error" not in r else {})
        results[ticker]=summary
        print(json.dumps(summary, indent=2, default=str))
    timestamp=datetime.date.today().isoformat()
    out_path=os.path.join(out_dir, f"validation_{timestamp}.json")
    payload={"generated_at":timestamp,"config":{"tickers":TICKERS,"start":START,"end":END,
            "backtest":{"n_splits":BASE_BACKTEST.n_splits,"test_size":BASE_BACKTEST.test_size,
            "purge_size":BASE_BACKTEST.purge_size,"embargo_size":BASE_BACKTEST.embargo_size,
            "transaction_cost":BASE_BACKTEST.transaction_cost,"slippage":BASE_BACKTEST.slippage,
            "risk_free_rate":BASE_BACKTEST.risk_free_rate},
            "engine":"market_predictor_ml.backtest.engine (WalkForwardSplit + run_walk_forward_backtest)"},
            "measurement_warning": ("INVALID AS AN ECONOMIC MEASUREMENT. The engine multiplies "
            "the position by y_test[1:], where y is Target_RiskAdj_21d - a 21-day forward "
            "risk-adjusted return - and treats it as a daily return. Overlapping 21-day windows "
            "inflate the reported Sharpe by roughly sqrt(21). See "
            "run_horizon_matched.py / validation_horizon_matched_*.json for the corrected numbers."),
            "results":results,"gate1":{"per_ticker_pass":{t:r.get("verdict") for t,r in results.items()},
            "pass_count":sum(1 for r in results.values() if r.get("verdict")=="PASS"),
            "decision_rule":"PASS >= 3 of 5 tickers -> Phase 2; MARGINAL 1-2 -> Phase 3 first; FAIL 0 -> stop"}}
    with open(out_path,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2,default=str)
    print(f"\nSaved: {out_path}")
    print("\nGate 1 scorecard:")
    print("Ticker | Sharpe | PF    | MaxDD    | AnnRet   | Verdict")
    print("-"*66)
    for t in TICKERS:
        s=results[t]
        sh=s.get("sharpe_ratio"); pf=s.get("profit_factor"); dd=s.get("max_drawdown"); ar=s.get("annual_return")
        print(f"{t:<6} | {(sh or 0):>5.2f} | {(pf or 0):>5.2f} | {(dd or 0)*100:>8.2f}% | {(ar or 0)*100:>8.2f}% | {s.get('verdict')}")

if __name__=="__main__": main()
