#!/usr/bin/env python3
"""
Example usage of Market Predictor ML.

This script demonstrates how to use the pipeline to:
1. Load stock data
2. Engineer features
3. Create labels
4. Train a model
5. Run a walk-forward backtest
"""

from market_predictor_ml import MarketPredictorPipeline, Config


def run_example(ticker: str = "AAPL"):
    """
    Run a complete example with default settings.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    """
    print("=" * 60)
    print("MARKET PREDICTOR ML - EXAMPLE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = MarketPredictorPipeline()
    
    # Run complete workflow
    pipeline.load_data(
        ticker=ticker,
        start_date="2018-01-01",
        end_date="2023-12-31"
    )
    
    pipeline.engineer_features()
    pipeline.create_labels()
    pipeline.prepare_data(target_column='Target_RiskAdj_21d')
    pipeline.train_model(model_type='lightgbm')
    results = pipeline.run_backtest(
        position_method='volatility_adjusted',
        n_splits=3,  # Fewer splits for faster demo
        test_size=126  # ~6 months
    )
    
    # Show feature importance
    print("\nTop 15 Features by Importance:")
    print("-" * 40)
    importance = pipeline.get_feature_importance(top_n=15)
    print(importance.to_string(index=False))
    
    # Get equity curve
    equity = pipeline.get_equity_curve()
    print(f"\nFinal Equity: {equity.iloc[-1]:.2f}")
    print(f"Number of trades: {len(equity)}")
    
    return results


if __name__ == "__main__":
    run_example()
