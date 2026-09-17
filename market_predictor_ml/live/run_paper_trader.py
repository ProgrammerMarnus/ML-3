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

import joblib
import numpy as np

from market_predictor_ml.data import download_stock_data, preprocess_data
from market_predictor_ml.features import create_all_features, get_feature_columns
from market_predictor_ml.models import get_model
from market_predictor_ml.decision import create_positions
from market_predictor_ml.utils import winsorize_features
from market_predictor_ml.live.paper_client import AlpacaPaperClient, is_alpaca_available
from market_predictor_ml.live.rl_policy import RLPolicy, is_rl_available, rl_enabled
from market_predictor_ml.live.gnn_features import (
    GNNFeatureAugmenter,
    gnn_enabled,
    is_gnn_available,
)

LOG_PATH = ROOT / "paper_trades.jsonl"
STATE_PATH = ROOT / "paper_trader_state.json"

# Fallback when the 21d realised volatility is unavailable (~20% annualised).
_DAILY_VOL_FALLBACK = 0.0125


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` when missing/invalid."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Ignoring invalid {name}={raw!r}; using {default}")
        return default


def _log_event(event: dict) -> None:
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def _model_cache_path(ticker: str, lookback_years: int) -> Path:
    """Location of the cached model for ``ticker``."""
    cache_dir = Path(os.getenv("PAPER_TRADING_MODEL_CACHE", str(ROOT / "model_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker.upper()}_{lookback_years}y.joblib"


def _train_or_load_model(ticker: str, lookback_years: int, X: np.ndarray,
                         y: np.ndarray, feature_cols: list, split: int) -> tuple:
    """Return ``(model, cache_info)``; retrain at most once per UTC day.

    Training on ~5 years of daily bars takes tens of seconds on a single-board
    machine, which is wasted work inside a short polling loop. The fitted model
    is cached per ticker/lookback and reused while it was trained today.
    """
    path = _model_cache_path(ticker, lookback_years)
    today = datetime.now(timezone.utc).date().isoformat()
    force = os.getenv("PAPER_TRADING_RETRAIN", "false").lower() == "true"

    if not force and path.exists():
        try:
            cached = joblib.load(path)
            if (cached.get("trained_on") == today
                    and cached.get("feature_cols") == list(feature_cols)):
                print(f"[model-cache] Reusing {path.name} (trained {today}, "
                      f"{len(feature_cols)} features)")
                return cached["model"], {"cached": True, "trained_on": today,
                                         "path": str(path)}
        except Exception as exc:
            print(f"[model-cache] Ignoring unusable cache {path.name}: {exc}")

    model = get_model(
        "lightgbm", n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=31, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        early_stopping_rounds=30,
    )
    model.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
    try:
        joblib.dump({"trained_on": today, "feature_cols": list(feature_cols),
                     "lookback_years": lookback_years, "model": model}, path)
        print(f"[model-cache] Trained and cached {path.name}")
    except Exception as exc:
        print(f"[model-cache] Could not write {path.name}: {exc}")
    return model, {"cached": False, "trained_on": today, "path": str(path)}


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
    # Volatility_Realized_21d is ANNUALISED (log-return std * sqrt(252)) while
    # volatility_adjusted_position(target_vol=...) expects a DAILY volatility
    # (its docstring: "target daily volatility of position"). Sizing with the
    # annualised figure made every position ~sqrt(252) ~ 16x too small.
    vols = (feats.loc[data.index, "Volatility_Realized_21d"]
            .div(np.sqrt(252)).fillna(_DAILY_VOL_FALLBACK).values)
    split = int(len(X) * 0.9)
    model, cache_info = _train_or_load_model(
        ticker, lookback_years, X, y, feature_cols, split
    )
    pred = float(model.predict(X[-1:])[0])
    last_vol = float(vols[-1]) if len(vols) else _DAILY_VOL_FALLBACK
    max_pos = _env_float("PAPER_TRADING_MAX_POSITION", 1.0)
    # Target DAILY volatility of the position (2% => ~32% annualised).
    target_vol = _env_float("PAPER_TRADING_TARGET_VOL", 0.02)
    try:
        base = float(create_positions(
            np.array([pred]), volatility=np.array([max(last_vol, 1e-4)]),
            method="volatility_adjusted", target_vol=target_vol,
            max_position=max_pos)[0])
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
        "model_cache": cache_info,
    }


def _capital_base(account: dict | None) -> float:
    """Notional backing a full (+/-1.0) position.

    Defaults to live account equity (Alpaca paper starts at $100k) and can
    be overridden with ``PAPER_TRADING_CAPITAL``. Falls back to $10k when
    the account equity is unknown (dry-run).
    """
    override = os.getenv("PAPER_TRADING_CAPITAL")
    if override:
        try:
            return max(float(override), 0.0)
        except ValueError:
            print(f"[WARN] Ignoring invalid PAPER_TRADING_CAPITAL={override!r}")
    equity = (account or {}).get("equity")
    try:
        equity = float(equity)
    except (TypeError, ValueError):
        equity = 0.0
    return equity if equity > 0 else 10_000.0


def _min_order_notional() -> float:
    """Dust threshold: orders smaller than this are skipped."""
    try:
        return float(os.getenv("PAPER_TRADING_MIN_ORDER_NOTIONAL", "50"))
    except ValueError:
        return 50.0


class RiskGuard:
    """Safety rails for the unattended paper trader.

    - **drawdown halt**: once equity falls ``PAPER_TRADING_MAX_DRAWDOWN``
      (default 10%) below its peak, no new risk is opened. Risk-reducing
      orders are always allowed, and the halt clears on a new equity high.
    - **stop-loss**: a position whose unrealised loss exceeds
      ``PAPER_TRADING_STOP_LOSS_PCT`` (default 5%) is flattened regardless
      of what the model says. Set to 0 to disable.
    - **market hours**: orders are skipped while the market is closed unless
      ``PAPER_TRADING_IGNORE_MARKET_HOURS=true``.

    Peak equity and the halt flag are persisted to ``paper_trader_state.json``
    so restarting the process does not silently reset the drawdown guard.
    """

    def __init__(self, state_path: Path = STATE_PATH) -> None:
        self.state_path = Path(state_path)
        self.max_drawdown = _env_float("PAPER_TRADING_MAX_DRAWDOWN", 0.10)
        self.stop_loss_pct = _env_float("PAPER_TRADING_STOP_LOSS_PCT", 0.05)
        self.require_open_market = (
            os.getenv("PAPER_TRADING_IGNORE_MARKET_HOURS", "false").lower() != "true"
        )
        self.peak_equity = 0.0
        self.halted = False
        self.halt_reason = ""
        self._load()

    # ------------------------------------------------------------------ state
    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except Exception as exc:
            print(f"[WARN] Could not read {self.state_path.name}: {exc}")
            return
        self.peak_equity = float(state.get("peak_equity") or 0.0)
        self.halted = bool(state.get("halted", False))
        self.halt_reason = str(state.get("halt_reason") or "")

    def _persist(self) -> None:
        try:
            self.state_path.write_text(json.dumps({
                "peak_equity": self.peak_equity,
                "halted": self.halted,
                "halt_reason": self.halt_reason,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception as exc:
            print(f"[WARN] Could not write {self.state_path.name}: {exc}")

    # ----------------------------------------------------------------- checks
    def update_equity(self, equity: float) -> float:
        """Track peak equity; return the drawdown from that peak."""
        if equity > self.peak_equity:
            self.peak_equity = equity
            if self.halted and self.halt_reason.startswith("max_drawdown"):
                print(f"[HALT-CLEARED] New equity high ${equity:,.2f}")
                self.halted, self.halt_reason = False, ""
            self._persist()
        return self.drawdown(equity)

    def drawdown(self, equity: float) -> float:
        """Current drawdown from peak equity (0 when at/above the peak)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def check_drawdown(self, equity: float) -> bool:
        """Return True when new risk must be blocked."""
        if self.max_drawdown > 0 and self.drawdown(equity) >= self.max_drawdown:
            if not self.halted:
                self.halt_reason = f"max_drawdown {self.drawdown(equity):.2%}"
                print(f"[HALT] {self.halt_reason}; new risk blocked until a new "
                      "equity high (exits still allowed)")
            self.halted = True
            self._persist()
        return self.halted

    def stop_loss_hit(self, avg_entry: float, price: float, qty: float) -> bool:
        """True when an open position has lost more than the stop-loss level."""
        if self.stop_loss_pct <= 0 or qty == 0 or avg_entry <= 0 or price <= 0:
            return False
        direction = 1.0 if qty > 0 else -1.0
        pnl_pct = direction * (price - avg_entry) / avg_entry
        return pnl_pct <= -self.stop_loss_pct

    def market_ok(self, client) -> tuple[bool, str]:
        """(allowed, reason) - whether orders may be submitted right now."""
        if not self.require_open_market:
            return True, "market hours ignored"
        clock = client.get_clock()
        if clock is None:
            return True, "clock unavailable"
        if clock["is_open"]:
            return True, "market open"
        return False, f"market closed (next open {clock['next_open']})"


def reconcile_to_target(client, ticker: str, target_frac: float, price: float,
                        capital: float, min_notional: float | None = None) -> dict:
    """Convert fractional target [-1,1] into share orders vs current pos.

    ``capital`` is the notional that backs a full (+/-1.0) position, so a
    target of 0.002 on a $100k account means a $200 target position. Orders
    below ``min_notional`` (default $50) are treated as dust and skipped.
    """
    if min_notional is None:
        min_notional = _min_order_notional()
    capital = float(capital)
    target_qty = float(target_frac) * capital / max(price, 1e-6)
    current_qty = client.get_position_qty(ticker)
    delta = target_qty - current_qty
    if abs(delta) * price < min_notional:  # ignore dust
        return {"action": "hold", "delta_qty": 0.0,
                "current_qty": current_qty, "target_qty": target_qty,
                "capital": capital, "target_notional": target_qty * price}
    side = "buy" if delta > 0 else "sell"
    res = client.submit_market_order(ticker, abs(delta), side, price=price)
    res.update({"action": side, "delta_qty": float(delta),
                "current_qty": current_qty, "target_qty": target_qty,
                "capital": capital, "target_notional": target_qty * price})
    return res


def run_once(ticker: str, guard: RiskGuard | None = None) -> dict:
    """Execute a single paper-trading cycle, subject to the risk guard."""
    guard = guard or RiskGuard()
    client = AlpacaPaperClient()
    account = client.get_account()
    equity = float(account.get("equity") or 0.0)
    drawdown = guard.update_equity(equity) if equity else 0.0
    halted = guard.check_drawdown(equity) if equity else guard.halted

    signal = build_signal(ticker, int(os.getenv("PAPER_TRADING_LOOKBACK_YEARS", "5")))
    live_price = client.get_latest_price(ticker)
    if live_price:
        signal["price"] = live_price
    price = float(signal["price"])
    capital = _capital_base(account)
    qty = client.get_position_qty(ticker)
    avg_entry = client.get_position_avg_entry(ticker) or price

    target = float(signal["final_position"])
    risk = {
        "risk_halted": halted,
        "halt_reason": guard.halt_reason if halted else "",
        "drawdown": round(drawdown, 6),
        "max_drawdown": guard.max_drawdown,
        "stop_loss_pct": guard.stop_loss_pct,
        "stop_loss": False,
        "avg_entry": avg_entry,
    }

    # A stop-loss overrides the model: flatten the position.
    if guard.stop_loss_hit(avg_entry, price, qty):
        risk["stop_loss"] = True
        print(f"[STOP-LOSS] {ticker} entry={avg_entry:.2f} px={price:.2f} "
              f"(limit {guard.stop_loss_pct:.1%}) -> flattening")
        target = 0.0
    elif halted:
        # Drawdown halt: allow risk-reducing moves, block new/increased risk.
        current_frac = (qty * price / capital) if capital else 0.0
        if abs(target) > abs(current_frac) + 1e-9:
            risk["risk_blocked"] = True
            target = current_frac

    market_ok, market_reason = guard.market_ok(client)
    if not market_ok:
        # Log the signal, but do not queue orders against a closed market.
        risk["market_closed"] = True
        order = {"action": "skipped", "reason": market_reason, "delta_qty": 0.0,
                 "current_qty": qty, "target_qty": qty, "capital": capital,
                 "target_notional": qty * price}
    else:
        order = reconcile_to_target(client, ticker, target, price, capital=capital)

    event = {**signal, **order, "account": account, "dry_run": not client.live,
             "rl_available": is_rl_available(), "risk": risk, "market": market_reason}
    _log_event(event)
    print(f"[{event['ts']}] {ticker} px={price:.2f} "
          f"pred={signal['prediction']:+.4f} pos={target:+.3f} "
          f"target=${order['target_notional']:,.2f} -> {order['action']} "
          f"(dd={drawdown:.2%}, halted={halted}, stop={risk['stop_loss']}, "
          f"market={market_reason})")
    return event


def preflight(ticker: str) -> int:
    """Validate credentials, connectivity and optional layers without trading."""
    print("-" * 70)
    print("PREFLIGHT CHECK (no orders are submitted)")
    print("-" * 70)
    problems = []
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    print(f"APCA_API_KEY_ID     : {'set' if key else 'MISSING'}")
    print(f"APCA_API_SECRET_KEY : {'set' if secret else 'MISSING'}")
    if not (key and secret):
        problems.append("Alpaca paper credentials missing (see .env.example)")
    client = AlpacaPaperClient()
    print(f"alpaca-py installed : {is_alpaca_available()}")
    print(f"live trading client : {client.live}")
    account = {}
    if client.live:
        try:
            account = client.get_account()
            print(f"account             : {account}")
            print(f"position {ticker:<9} : {client.get_position_qty(ticker)}")
            print(f"latest quote {ticker:<6} : {client.get_latest_price(ticker)}")
        except Exception as exc:  # network / auth failure
            problems.append(f"Alpaca API call failed: {exc}")
    else:
        problems.append("client fell back to dry-run (missing alpaca-py or keys)")

    use_rl = os.getenv("USE_RL", "false")
    use_gnn = os.getenv("USE_GNN", "false")
    print(f"USE_RL={use_rl:<5} enabled={str(rl_enabled()):<5} available={is_rl_available()}")
    if use_rl.lower() == "true" and not is_rl_available():
        print("  [WARN] USE_RL=true but gymnasium/stable-baselines3 are not "
              "installed -> RL residual layer is skipped.")
    print(f"USE_GNN={use_gnn:<4} enabled={str(gnn_enabled()):<5} available={is_gnn_available()}")
    print(f"capital base        : ${_capital_base(account):,.2f} "
          f"(equity, override with PAPER_TRADING_CAPITAL)")
    print(f"min order notional  : ${_min_order_notional():,.2f} "
          f"(override with PAPER_TRADING_MIN_ORDER_NOTIONAL)")
    guard = RiskGuard()
    equity = float(account.get("equity") or 0.0)
    print(f"max drawdown        : {guard.max_drawdown:.1%} "
          f"(PAPER_TRADING_MAX_DRAWDOWN)")
    print(f"stop loss           : {guard.stop_loss_pct:.1%} "
          f"(PAPER_TRADING_STOP_LOSS_PCT, 0 disables)")
    print(f"trading halted      : {guard.halted} {guard.halt_reason}".rstrip())
    print(f"peak equity         : ${guard.peak_equity:,.2f} "
          f"(current drawdown {guard.drawdown(equity):.2%})")
    if client.live:
        print(f"market              : {guard.market_ok(client)[1]}")
    if problems:
        print("-" * 70)
        for p in problems:
            print(f"[PROBLEM] {p}")
        return 1
    print("-" * 70)
    print("result              : OK - ready to paper trade")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca paper trader (live loop).")
    parser.add_argument("--ticker", default=os.getenv("TICKER", "AAPL"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Run preflight checks (env, account, data) and exit")
    parser.add_argument("--interval", type=int,
                        default=int(os.getenv("PAPER_TRADING_INTERVAL_SEC", "60")))
    args = parser.parse_args()
    print("=" * 70)
    print("MARKET PREDICTOR ML - LIVE PAPER TRADER")
    print("=" * 70)
    print(f"Ticker: {args.ticker} | USE_RL={os.getenv('USE_RL')} "
          f"| USE_GNN={os.getenv('USE_GNN')} | Log: {LOG_PATH}")
    if args.check:
        raise SystemExit(preflight(args.ticker))
    if os.getenv("USE_RL", "false").lower() == "true" and not is_rl_available():
        print("[WARN] USE_RL=true but stable-baselines3/gymnasium are not installed; "
              "the RL residual layer is skipped (see requirements.txt).")
    guard = RiskGuard()
    print(f"Risk guard: max_drawdown={guard.max_drawdown:.1%} "
          f"stop_loss={guard.stop_loss_pct:.1%} "
          f"require_open_market={guard.require_open_market}")
    if args.once:
        run_once(args.ticker, guard)
        return
    while True:
        try:
            run_once(args.ticker, guard)
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
