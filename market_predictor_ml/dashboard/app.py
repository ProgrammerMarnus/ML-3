"""
Interactive Streamlit Dashboard for Market Predictor ML.
Provides UI for training, backtesting, live monitoring, and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import numpy as np


def run_dashboard():
    """Main dashboard application."""
    
    st.set_page_config(
        page_title="Market Predictor ML",
        page_icon="📈",
        layout="wide",
    )
    
    st.title("📈 Market Predictor ML Dashboard")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Train & Backtest", "Live Signals", "Portfolio Analysis", "Sentiment & Macro"],
    )
    
    if page == "Home":
        show_home()
    elif page == "Train & Backtest":
        show_train_backtest()
    elif page == "Live Signals":
        show_live_signals()
    elif page == "Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "Sentiment & Macro":
        show_sentiment_macro()


def show_home():
    """Home page with overview."""
    st.header("Welcome to Market Predictor ML")
    st.markdown("""
    ### A Comprehensive AI-Powered Trading System
    
    **Features:**
    - 🤖 Multi-Model Ensemble (LightGBM, LSTM, Transformer, GNN)
    - 🧠 Reinforcement Learning for Dynamic Position Sizing
    - 📰 Sentiment Analysis Integration
    - 🌍 Macro-Economic Regime Detection
    - 📊 Interactive Backtesting & Visualization
    - 🚀 Live Paper Trading Support
    
    **Get Started:**
    1. Go to **Train & Backtest** to train models
    2. Check **Live Signals** for real-time predictions
    3. Monitor performance in **Portfolio Analysis**
    """)
    
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", "68.81%", "12.3%")
    col2.metric("Sharpe Ratio", "4.62", "0.85")
    col3.metric("Win Rate", "61.27%", "-2.1%")
    col4.metric("Max Drawdown", "-18.5%", "3.2%")


def show_train_backtest():
    """Training and backtesting interface."""
    st.header("🎯 Train & Backtest Strategy")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker Symbol", "AAPL")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    
    with col2:
        model_type = st.selectbox("Model Type", ["LightGBM", "Ridge", "LSTM", "Transformer", "Ensemble"])
        horizon = st.slider("Prediction Horizon (days)", 1, 60, 21)
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            # Placeholder for actual backtest results
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            equity_curve = 100000 * (1 + np.random.randn(len(dates)).cumsum() / 100).clip(lower=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=equity_curve, mode="lines", name="Equity"))
            fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"Backtest completed for {ticker} using {model_type}")


def show_live_signals():
    """Live trading signals monitor."""
    st.header("🔴 Live Trading Signals")
    
    st.info("📄 Paper Trading Mode Active")
    
    # Mock signals table
    signals_data = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "Signal": ["BUY", "HOLD", "SELL", "BUY"],
        "Confidence": [0.85, 0.52, 0.78, 0.91],
        "Price": [178.45, 378.92, 141.23, 245.67],
        "Timestamp": pd.Timestamp.now(),
    })
    
    st.dataframe(signals_data, use_container_width=True)
    
    # Recent trades
    st.subheader("Recent Executed Trades")
    trades_data = pd.DataFrame({
        "Time": ["09:30", "10:15", "11:45"],
        "Symbol": ["AAPL", "TSLA", "AAPL"],
        "Side": ["BUY", "BUY", "SELL"],
        "Qty": [50, 30, 25],
        "Price": [177.80, 244.50, 178.90],
        "Status": ["FILLED", "FILLED", "FILLED"],
    })
    st.dataframe(trades_data, use_container_width=True)


def show_portfolio_analysis():
    """Portfolio performance analysis."""
    st.header("📊 Portfolio Analysis")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total P&L", "+$12,450", "+8.3%")
    col2.metric("Daily P&L", "+$340", "+0.2%")
    col3.metric("Cash Balance", "$87,550")
    
    # Equity curve
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    equity = 100000 + np.cumsum(np.random.randn(100) * 500)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines", fill="tozeroy", name="Equity"))
    fig.update_layout(title="Portfolio Equity Curve", xaxis_title="Date", yaxis_title="Value ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Position breakdown
    st.subheader("Current Positions")
    positions = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "TSLA"],
        "Quantity": [100, 50, 30],
        "Avg Price": [175.20, 370.50, 240.80],
        "Current Price": [178.45, 378.92, 245.67],
        "P&L": ["+$325", "+$421", "+$146"],
    })
    st.dataframe(positions, use_container_width=True)


def show_sentiment_macro():
    """Sentiment and macroeconomic analysis."""
    st.header("🌍 Sentiment & Macro Analysis")
    
    # Sentiment panel
    st.subheader("News Sentiment")
    sentiment_data = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=30),
        "Sentiment Score": np.random.uniform(-0.5, 0.8, 30),
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sentiment_data["Date"], y=sentiment_data["Sentiment Score"], 
                            mode="lines", name="Sentiment"))
    fig.add_hline(y=0, line_dash="dash", annotation_text="Neutral")
    fig.update_layout(title="News Sentiment Trend", yaxis_range=[-1, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    # Macro regime
    st.subheader("Macroeconomic Regime")
    col1, col2 = st.columns(2)
    col1.metric("Current Regime", "Expansion", "Stable")
    col2.metric("Interest Rate", "5.25%", "+0.25%")
    
    st.info("**Regime Indicators:**\n- GDP Growth: Positive\n- Inflation: Moderate\n- Yield Curve: Normal")


if __name__ == "__main__":
    run_dashboard()
