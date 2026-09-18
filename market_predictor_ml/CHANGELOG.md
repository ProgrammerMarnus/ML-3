# Market Predictor ML - Changelog

## [0.4.0] - 2026-09-18

Audit remediation release: all 51 findings from `Deep-Audit-ML3-1.txt`
(10 Critical, 10 High, 12 Medium, 13 Low) are implemented, plus two
reliability fixes found during verification. 40 tests pass
(`python -m pytest market_predictor_ml/tests -q`).

### Critical fixes (wrong numbers / crashes)
- **Alerting** (`monitoring/alerting.py`): operator strings are normalised,
  `AlertRule` gained a `threshold` field, and the four factory rules construct
  correctly (previously every pre-built alert raised on creation).
- **Data providers** (`data/providers.py`): fixed the CSV kwarg
  (`data_dir=`) and defined the module logger used by `create_data_provider`.
- **Live predictor** (`live/predictor.py`): removed the bogus
  `EnhancedSettings` import; `create_live_predictor` now requires a feature
  pipeline and model registry instead of silently building empty ones.
- **Hyperopt** (`optimization/hyperopt.py`): sizing kwargs match the real
  signatures (`target_vol`, `fraction`), and train-fit winsorize/standardise
  parameters are applied to the test fold - Optuna no longer selects on
  test-set statistics or silently failed trials.
- **Engine-harness persistence** (`run_paper_trader.py`): restarts restore
  cash, initial capital, positions (including shorts), risk-halt state and
  the active session id; the restore now runs after components exist.
- **Portfolio backtester** (`backtest/enhanced_engine.py`): positions close on
  zero/NaN signals and flip on sign changes; `Trade.return_pct` is side-aware
  for shorts.
- **Dashboards**: chart configs are emitted as JSON
  (`monitoring/dashboard.py`), and the monthly-returns heatmap degrades
  gracefully on non-datetime indices (`visualization/dashboard.py`).
- **Dependencies** (`requirements.txt`): added `pydantic>=2.0.0` and
  `optuna>=3.0.0`; migrated `config/enhanced_settings.py` to the Pydantic v2
  API (`field_validator`, `model_validator`, `ConfigDict`, `model_dump`).

### High-severity fixes (wrong math / dead features)
- Canonical Sortino downside deviation in all three copies
  (`monitoring/metrics.py`, `risk/metrics.py`, `backtest/enhanced_engine.py`).
- Alpha uses arithmetic annualization, so negative cumulative excess returns
  no longer produce NaN (`monitoring/metrics.py`).
- Volatility sizing clips symmetrically, so short signals survive
  (`decision/sizers.py`).
- Kelly sizing uses per-asset volatility (`f* = mu / sigma^2`) instead of the
  cross-sectional prediction variance; signal-strength sizing applies its
  power curve once (`decision/sizing.py`).
- Break-even backtests report metrics instead of returning `{}`.
- Live risk monitor: concentration checks, volatility CRITICAL escalation,
  historical VaR/ES wired into `check_all_risks`, and the engine feeds
  portfolio returns so those checks have data (`live/risk.py`, `live/engine.py`).
- Signals split `target_quantity` (absolute) from `delta_quantity` (signed
  trade); orders follow the delta and weight checks use the target
  (`live/signals.py`, `live/portfolio.py`).
- MAE/MFE are computed from the recorded price path
  (`backtest/enhanced_engine.py`).
- Metrics collector converts daily dollar P&L to fractional returns via the
  `equity_before` snapshot (`monitoring/metrics.py`).

### Medium fixes (design / safety / performance)
- `OrderManager` mutations are `RLock`-guarded; the engine consumes
  `queue.Queue` events with a blocking timeout (no busy-wait, no lost events).
- Broker price updates go through `BrokerAdapter.update_price`; the risk
  monitor exposes a public `current_value` property.
- Sentiment features aggregate news once (O(n) instead of O(n^2)); mock news
  seeding is process-stable (`zlib.crc32`); GDP YoY growth infers quarterly
  vs monthly frequency.
- Walk-forward backtest applies the next-bar convention
  (`positions[:-1] * y_test[1:]`), matching README §27.
- Portfolio backtester gained a gross-exposure cap (`max_gross_exposure`) and
  uses `max_portfolio_turnover` as a daily notional throttle.
- `TradingEngine.stop()` joins the loop thread; `IBacktestEngine.run` now
  matches the concrete engine signature; SQL in the reference persistence
  store is parameterised; `BacktestConfig.train_size` is `Optional[int]`;
  data-layer preprocessing lives in one place (`data/providers.py`,
  re-exported by `data/loader.py`); README's prescribed methodology is
  now implemented.

### Low / consistency fixes
- Half-spread cost model documented and applied per one-way trade; arithmetic
  vs CAGR annualization conventions documented where they differ.
- `LabelGenerator` honours separate triple-barrier profit-target and
  stop-loss barriers (`features/pipeline.py`).
- Market-session detection uses `ZoneInfo("America/New_York")`
  (`live/predictor.py`).
- MAPE guards the empty-denominator case; `override_from_env` supports
  top-level scalars, nested sections and JSON list values.
- Kelly position sizer uses its recorded trade history to estimate win
  rate/payoff (`decision/sizers.py`).
- Rebalancing closes holdings whose target weight falls below
  `min_position_weight` (`live/portfolio.py`).
- Hyperopt config fields it referenced were added to the dataclass configs,
  and `run_optimization_example` constructs a valid `Config`.

### Reliability fixes (found during verification)
- `dashboard/paper_monitor.py` no longer executes its UI at import time
  (script body wrapped in `main()` with the standard `__name__` guard, and
  the empty-log path returns instead of relying on `st.stop()`), so the
  module imports cleanly and `streamlit run` still works (verified with
  Streamlit's `AppTest` harness).
- `rl/` treats `gymnasium` as optional (mirroring the stable-baselines3
  guard): the package imports without it and `TradingEnv` raises a clear,
  actionable ImportError; `gymnasium` installed for local use.

### Documentation
- Merged `ARCHITECTURE_IMPROVEMENTS.md` +
  `ARCHITECTURE_IMPLEMENTATION_SUMMARY.md` into `ARCHITECTURE.md` (all
  originally pending items are complete; usage examples verified against the
  current APIs). `LIVE_TRADING_IMPLEMENTATION.md` refreshed (line counts,
  thread safety, risk features, accurate next steps).
- Removed stale artifacts: `repomix-output.xml` (regenerable packing export),
  `backtest_report.txt` (generated output of `examples/run_visualization.py`),
  and `examples/run_paper_trader.py` (dead example importing modules that do
  not exist; superseded by the two real entry points in README §44).

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
