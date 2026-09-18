"""Streamlit live dashboard for the Alpaca paper trader.

Reads ``paper_trades.jsonl`` (written by run_paper_trader.py) and shows:
  - latest signal / position / order
  - prediction & position history
  - price history
  - account snapshot + config flags (USE_RL / USE_GNN)

The UI only runs when executed as a Streamlit script (``streamlit run``);
importing this module is side-effect free, so tooling that walks the package
can import it without a live Streamlit runtime.

Usage:
    streamlit run market_predictor_ml/dashboard/paper_monitor.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

import pandas as pd
import streamlit as st

LOG_PATH = ROOT / "paper_trades.jsonl"


def load_events() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")
    return df


def main() -> None:
    """Render the live paper-trader dashboard."""
    st.set_page_config(page_title="Market Predictor - Paper Trader", layout="wide")
    st.title("Market Predictor ML - Live Paper Trader")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    col_cfg1.metric("TICKER", os.getenv("TICKER", "AAPL"))
    col_cfg2.metric("USE_RL", os.getenv("USE_RL", "false"))
    col_cfg3.metric("USE_GNN", os.getenv("USE_GNN", "false"))

    df = load_events()
    if df.empty:
        st.warning(f"No trades logged yet at {LOG_PATH}. Start the trader: "
                   "`python market_predictor_ml/live/run_paper_trader.py`")
        return

    latest = df.iloc[-1].to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{latest.get('price', float('nan')):.2f}")
    c2.metric("Prediction", f"{latest.get('prediction', 0.0):+.4f}")
    c3.metric("Final position", f"{latest.get('final_position', 0.0):+.3f}")
    c4.metric("Last action", str(latest.get("action", "-")))

    st.subheader("Signal history")
    plot_cols = [c for c in ["prediction", "base_position", "final_position"] if c in df.columns]
    if plot_cols and "ts" in df.columns:
        st.line_chart(df.set_index("ts")[plot_cols])
    elif plot_cols:
        st.line_chart(df[plot_cols])

    st.subheader("Price history")
    if "price" in df.columns:
        st.line_chart(df.set_index("ts")["price"] if "ts" in df.columns else df["price"])

    st.subheader("Account (latest)")
    st.json(latest.get("account", {}))

    st.subheader("Recent events (newest first)")
    show_cols = [c for c in ["ts", "ticker", "price", "prediction", "base_position",
                             "final_position", "action", "delta_qty", "current_qty",
                             "target_qty", "dry_run", "use_rl", "use_gnn", "error"]
                 if c in df.columns]
    st.dataframe(df[show_cols].iloc[::-1].head(100), use_container_width=True)

    if st.button("Refresh"):
        st.rerun()


if __name__ == "__main__":
    main()
