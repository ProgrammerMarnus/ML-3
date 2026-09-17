#!/usr/bin/env python3
"""Live paper trader for Market Predictor ML (Alpaca paper endpoint).

Flow per cycle:
  1. Load .env (APCA_* keys, TICKER, USE_RL, USE_GNN).
  2. Download recent history -> engineer features -> optional GNN.
  3. Train LightGBM (fast live settings) and predict latest return.
  4. Prediction -> base position -> optional RL residual adjustment.
  5. Reconcile Alpaca paper position toward target (market, DAY).
  6. Append JSONL trade log consumed by the Streamlit dashboard.

Usage:
    python market_predictor_ml/live/run_paper_trader.py [--once]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

import numpy as np

from market_predictor_ml.data import download_stock_data, preprocess_data
from market_predictor_ml.features import create_all_features, get_feature_columns
from market_predictor_ml.models import get_model
from market_predictor_ml.decision import create_positions
from market_predictor_ml.utils import winsorize_features
from market_predictor_ml.live.alpaca_client import AlpacaPaperClient
from market_predictor_ml.live.rl_policy import RLPolicy, rl_enabled
from market_predictor_ml.live.gnn_features import GNNFeatureAugmenter, gnn_enabled

LOG_PATH = ROOT / "paper_trades.jsonl"


def _log_event(event: dict) -> None:
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def build_signal(ticker: str, lookback_years: int = 5) -> dict:
    """Train on recent history and produce (prediction, position)."""
    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(days=int(lookback_years * 365))).isoformat()
    df = preprocess_data(download_stock_data(ticker, str(start), end.isoformat()))
    feats = create_all_features(df)
    feature_cols = get_feature_columns(feats)
    if gnn_enabled():
        feats = GNNFeatureAugmenter().augment(feats, feature_cols)
        feature_cols = get_feature_columns(feats)
    close = feats["Close"].astype(float)
    fut = np.log(close.shift(-1) / close)
    vol = close.pct_change().rolling(21).std().replace(0, np.nan).bfill()
    label = (fut / vol).replace([np.inf, -np.inf], np.nan)
    data = feats[feature_cols].copy()
    data["__label"] = label.values
    data = data.dropna()
    if len(data) < 300:
        raise RuntimeError(f"Not enough rows ({len(data)}) for {ticker}")
    X = winsorize_features(data[feature_cols].values.astype(float))
    y = data["__label"].values.astype(float)
    vols = feats.loc[data.index, "Volatility_Realized_21d"].fillna(0.02).values
    split = int(len(X) * 0.9)
    model = get_model(
        "lightgbm", n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=31, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        early_stopping_rounds=30,
    )
    model.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
    pred = float(model.predict(X[-1:])[0])
    last_vol = float(vols[-1]) if len(vols) else 0.02
    max_pos = float(os.getenv("PAPER_TRADING_MAX_POSITION", "1.0"))
    try:
        base = float(create_positions(
            np.array([pred]), volatility=np.array([max(last_vol, 1e-4)]),
            method="volatility_adjusted", target_vol=0.02, max_position=max_pos)[0])
    except Exception:
        base = float(create_positions(
            np.array([pred]), method="fixed",
            max_position=max_pos, threshold=0.0)[0])
    final = RLPolicy().adjust(base, pred) if rl_enabled() else base
    return {
        "ticker": ticker, "price": float(close.iloc[-1]),
        "prediction": pred, "base_position": base,
        "final_position": float(np.clip(final, -1.0, 1.0)),
        "n_rows": len(data), "n_features": len(feature_cols),
        "use_rl": rl_enabled(), "use_gnn": gnn_enabled(),
    }


def reconcile_to_target(client, ticker: str, target_frac: float, price: float) -> dict:
    """Convert fractional target [-1,1] into share orders vs current pos."""
    notional_per_unit = 10_000.0
    target_qty = float(target_frac) * notional_per_unit / max(price, 1e-6)
    current_qty = client.get_position_qty(ticker)
    delta = target_qty - current_qty
    if abs(delta) * price < 50.0:  # ignore dust
        return {"action": "hold", "delta_qty": 0.0,
                "current_qty": current_qty, "target_qty": target_qty}
    side = "buy" if delta > 0 else "sell"
    res = client.submit_market_order(ticker, abs(delta), side)
    res.update({"action": side, "delta_qty": float(delta),
                "current_qty": current_qty, "target_qty": target_qty})
    return res


def run_once(ticker: str) -> dict:
    """Execute a single paper-trading cycle."""
    client = AlpacaPaperClient()
    account = client.get_account()
    signal = build_signal(ticker, int(os.getenv("PAPER_TRADING_LOOKBACK_YEARS", "5")))
    live_price = client.get_latest_price(ticker)
    if live_price:
        signal["price"] = live_price
    order = reconcile_to_target(client, ticker, signal["final_position"], signal["price"])
    event = {**signal, **order, "account": account, "dry_run": not client.live}
    _log_event(event)
    print(f"[{event['ts']}] {ticker} px={signal['price']:.2f} "
          f"pred={signal['prediction']:+.4f} pos={signal['final_position']:+.3f} "
          f"-> {order['action']} (dry_run={event['dry_run']}, "
          f"rl={signal['use_rl']}, gnn={signal['use_gnn']})")
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca paper trader (live loop).")
    parser.add_argument("--ticker", default=os.getenv("TICKER", "AAPL"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int,
                        default=int(os.getenv("PAPER_TRADING_INTERVAL_SEC", "60")))
    args = parser.parse_args()
    print("=" * 70)
    print("MARKET PREDICTOR ML - LIVE PAPER TRADER")
    print("=" * 70)
    print(f"Ticker: {args.ticker} | USE_RL={os.getenv('USE_RL')} "
          f"| USE_GNN={os.getenv('USE_GNN')} | Log: {LOG_PATH}")
    if args.once:
        run_once(args.ticker)
        return
    while True:
        try:
            run_once(args.ticker)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception:
            traceback.print_exc()
            _log_event({"ticker": args.ticker,
                        "error": traceback.format_exc(limit=3)})
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
