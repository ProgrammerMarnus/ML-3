# Market Predictor ML - Changelog

## [0.3.0] - 2024-09-17

### Added
- **Visualization Dashboard** (`visualization/` module):
  - `BacktestDashboard` class with comprehensive plotting capabilities:
    - Equity curve with metrics overlay
    - Drawdown analysis with max drawdown annotation
    - Monthly returns heatmap (year x month matrix)
    - Trade return distribution histogram with statistics
    - Position sizing evolution over time
    - Feature importance bar charts
  - Text report generator for performance summaries
  
- **Enhanced Examples**:
  - `run_visualization.py`: Complete example demonstrating:
    - Transaction cost modeling (commission + slippage)
    - Walk-forward backtesting with purging/embargo
    - Dashboard generation and saving
    - Individual plot exports
    
- **Robust Data Handling**:
  - Automatic conversion of numpy arrays to DataFrames in dashboard
  - Graceful handling of non-datetime indices
  - Support for both array and DataFrame inputs

### Changed
- Updated `BacktestConfig` with additional parameters:
  - `slippage`: Separate slippage rate (default 0.05%)
  - `purge_size`: Samples to purge for leakage prevention (default 5)
  - `embargo_size`: Gap between train/test (default 5)
  
- Improved dashboard initialization to handle various input formats

### Performance Notes (Latest Tests)
- Volatility-adjusted position sizing: 20-25% annual return, ~43% max drawdown
- Sharpe ratio: ~0.99 with 21-day prediction horizon
- Win rate: ~60% but profit factor < 1.0 (avg losses > avg wins)
- Transaction costs impact: 0.15% total cost per trade significantly affects profitability
- LightGBM early stopping effective (best iteration at fold 1-2)

### Generated Outputs
- `equity_curve.png` - Cumulative returns visualization
- `drawdown_analysis.png` - Drawdown periods and severity
- `trade_distribution.png` - Individual trade return histogram
- `backtest_report.txt` - Comprehensive text summary

## [0.2.0] - 2024-09-17

### Added
- **Realistic Transaction Costs**: Enhanced `BacktestConfig` with:
  - `commission_rate`: Commission per trade (default 0.1%)
  - `min_trade_size`: Minimum trade value in dollars (default $100)
  - `max_position_size`: Optional maximum position limit
  - `initial_capital`: Starting capital for backtests (default $100,000)
  
- **Advanced Examples**:
  - `advanced_portfolio.py`: Multi-stock portfolio backtesting with:
    - Comparison of different position sizing methods (fixed vs volatility-adjusted)
    - Portfolio-level metrics aggregation
    - Analysis across multiple tickers simultaneously
    - Horizon optimization testing
  
- **Enhanced Configuration**:
  - More realistic default transaction costs (0.1% commission + 0.05% slippage)
  - Risk-free rate default set to 2% annual
  - Better separation of concerns in config classes

### Changed
- Updated default backtest parameters to reflect more realistic trading conditions
- Improved error handling in multi-stock examples
- Fixed bug in `run_single_stock_analysis` where it referenced wrong attribute

### Performance Notes
- Volatility-adjusted position sizing shows more stable returns (62% avg return, 23% max DD)
- Fixed position sizing can show extreme returns due to lack of risk normalization
- 21-day prediction horizon optimal for AAPL (Sharpe: 4.81 vs -0.27 for 1-day)

## [0.1.0] - Initial Release

### Core Features
- 5-layer architecture (Data, Prediction, Decision, Economic Reward, Backtesting)
- Feature engineering (8 categories: momentum, volatility, volume, liquidity, etc.)
- Label construction (future returns, triple-barrier method, risk-adjusted targets)
- Walk-forward validation with purging and embargo
- LightGBM model with financial-data-tuned defaults
- Ridge and LogisticRegression baselines
- Position sizing methods (fixed, volatility-adjusted, Kelly, signal strength)
- Comprehensive economic metrics (Sharpe, Sortino, Calmar, Max Drawdown)

### Components
- Data loading via yfinance
- 90+ engineered features
- Leakage prevention utilities
- Configurable pipeline with dataclass-based configuration
