"""
Example: Run Paper Trading with Live Signals.
Demonstrates the paper trading workflow with real-time data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from market_predictor_ml.data.loader import load_stock_data
from market_predictor_ml.features.factory import FeatureFactory
from market_predictor_ml.models.lightgbm_model import LightGBMWrapper
from market_predictor_ml.live.trader import PaperTrader
from market_predictor_ml.pipeline import MarketPredictorPipeline


def main():
    print("=" * 60)
    print("Paper Trading Example")
    print("=" * 60)
    
    # Create pipeline
    print("\n1. Initializing pipeline...")
    pipeline = MarketPredictorPipeline(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2024-01-01",
        prediction_horizon=21,
        model_type="lightgbm",
    )
    
    # Run training
    print("\n2. Training model...")
    results = pipeline.run()
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Total Return: {results['total_return']*100:.1f}%")
    
    # Create paper trader
    print("\n3. Setting up paper trader...")
    paper_trader = PaperTrader(
        model_pipeline=pipeline,
        broker_client=None,  # Not needed for paper mode
        symbols=["AAPL"],
        max_position_pct=0.1,
    )
    
    # Fetch latest data
    print("\n4. Fetching latest market data...")
    data = paper_trader.fetch_latest_data()
    
    if len(data) == 0:
        print("   ⚠ No data available. Using historical snapshot.")
        # Use last known data
        df = load_stock_data("AAPL", start_date="2023-01-01", end_date="2024-01-01")
        data = {"AAPL": df}
    
    # Generate signals
    print("\n5. Generating trading signals...")
    signals = paper_trader.generate_signals(data)
    
    for symbol, sig in signals.items():
        signal_str = "BUY" if sig["signal"] > 0 else ("SELL" if sig["signal"] < 0 else "HOLD")
        print(f"   {symbol}: {signal_str} (confidence: {sig.get('prediction', 0):.4f})")
    
    # Execute trades (simulated)
    print("\n6. Executing simulated trades...")
    executed = paper_trader.execute_trades(signals)
    
    for trade in executed:
        print(f"   {trade['symbol']}: {trade['side']} @ ${trade['price']:.2f} ({trade['status']})")
    
    # Show portfolio
    print("\n7. Paper Portfolio Status:")
    print(f"   Cash: ${paper_trader.paper_portfolio['cash']:.2f}")
    print(f"   Positions: {len(paper_trader.paper_portfolio['positions'])}")
    
    for pos_symbol, pos_data in paper_trader.paper_portfolio['positions'].items():
        print(f"      - {pos_symbol}: {pos_data['qty']} shares @ ${pos_data['avg_price']:.2f}")
    
    print("\n" + "=" * 60)
    print("Paper trading example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
