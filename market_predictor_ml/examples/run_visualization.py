"""
Example demonstrating the visualization dashboard with realistic transaction costs.

This example runs a complete backtest with:
- Transaction cost modeling (commission + slippage)
- Advanced position sizing
- Comprehensive visualization dashboard
- Performance report generation
"""

import sys
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np
from market_predictor_ml.pipeline import MarketPredictorPipeline
from market_predictor_ml.config.settings import Config, BacktestConfig
from market_predictor_ml.visualization.dashboard import BacktestDashboard
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt


def run_visualization_example():
    """Run complete pipeline with visualization dashboard."""
    
    print("=" * 70)
    print("MARKET PREDICTOR ML - VISUALIZATION DASHBOARD EXAMPLE")
    print("=" * 70)
    print()
    
    # Configure with custom settings
    config = Config()
    
    # Override some defaults via direct attribute access
    config.data.default_start_date = '2018-01-01'
    config.data.default_end_date = '2024-12-31'
    config.labels.return_horizons = [21]  # Use 21-day horizon
    
    # Enhanced backtest config with transaction costs
    backtest_config = BacktestConfig(
        commission_rate=0.001,  # 0.1% commission
        slippage=0.0005,        # 0.05% slippage
        min_trade_size=100,
        initial_capital=100000,
        allow_shorting=True,
        risk_free_rate=0.02,
        n_splits=5,
        test_size=63,  # ~3 months per split
        purge_size=5,
        embargo_size=5
    )
    
    print("Configuration:")
    print(f"  Ticker: AAPL")
    print(f"  Period: {config.data.default_start_date} to {config.data.default_end_date}")
    print(f"  Prediction Horizon: {config.labels.return_horizons[0]} days")
    print(f"  Model: lightgbm")
    print(f"  Position Sizing: volatility_adjusted")
    print(f"  Commission: {backtest_config.commission_rate:.2%}")
    print(f"  Slippage: {backtest_config.slippage:.2%}")
    print()
    
    # Run pipeline
    print("Running backtest pipeline...")
    pipeline = MarketPredictorPipeline(config)
    
    # Update config with backtest settings
    config.backtest.commission_rate = backtest_config.commission_rate
    config.backtest.slippage = backtest_config.slippage
    config.backtest.n_splits = backtest_config.n_splits
    config.backtest.test_size = backtest_config.test_size
    config.backtest.purge_size = backtest_config.purge_size
    config.backtest.embargo_size = backtest_config.embargo_size
    
    # Execute pipeline steps
    results = (pipeline
               .load_data(ticker='AAPL')
               .engineer_features()
               .create_labels()
               .prepare_data()
               .train_model()
               .run_backtest()
              )
    
    if results is None:
        print("ERROR: Pipeline execution failed!")
        return
    
    print("\nBacktest completed successfully!")
    print()
    
    # Extract and display metrics
    metrics = results.get('metrics', {})
    print("PERFORMANCE METRICS (with transaction costs):")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'ratio' in key or 'sharpe' in key or 'sortino' in key or 'calmar' in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            elif 'return' in key or 'drawdown' in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    print()
    
    # Create dashboard
    print("Generating visualization dashboard...")
    dashboard = BacktestDashboard(results)
    
    # Save dashboard plot
    dashboard_path = '/workspace/backtest_dashboard.png'
    try:
        dashboard.plot_all(save_path=dashboard_path, show=False)
        print(f"Dashboard saved to: {dashboard_path}")
    except Exception as e:
        print(f"Warning: Could not save dashboard plot: {e}")
    
    # Generate text report
    report_path = '/workspace/backtest_report.txt'
    report = dashboard.generate_report(save_path=report_path)
    print(f"\nReport saved to: {report_path}")
    print()
    print(report)
    
    # Plot individual components
    print("\nGenerating individual plots...")
    
    # Equity curve
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    dashboard.plot_equity_curve(ax=ax1)
    plt.savefig('/workspace/equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  - Equity curve saved to: /workspace/equity_curve.png")
    
    # Drawdown
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    dashboard.plot_drawdown(ax=ax2)
    plt.savefig('/workspace/drawdown_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  - Drawdown analysis saved to: /workspace/drawdown_analysis.png")
    
    # Trade distribution
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    dashboard.plot_trade_distribution(ax=ax3)
    plt.savefig('/workspace/trade_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("  - Trade distribution saved to: /workspace/trade_distribution.png")
    
    # Feature importance (if available)
    if results.get('feature_importance') is not None:
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        dashboard.plot_feature_importance(top_n=15, ax=ax4)
        plt.savefig('/workspace/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print("  - Feature importance saved to: /workspace/feature_importance.png")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. backtest_dashboard.png - Complete dashboard with all plots")
    print("  2. equity_curve.png - Cumulative returns over time")
    print("  3. drawdown_analysis.png - Drawdown periods and severity")
    print("  4. trade_distribution.png - Distribution of individual trade returns")
    print("  5. feature_importance.png - Top predictive features")
    print("  6. backtest_report.txt - Text summary report")
    print()
    
    return results


if __name__ == "__main__":
    results = run_visualization_example()
