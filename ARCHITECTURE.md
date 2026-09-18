# Market Predictor ML — Architecture

Single reference for the system's architecture, module map, and usage.
Supersedes the former `ARCHITECTURE_IMPROVEMENTS.md` and
`ARCHITECTURE_IMPLEMENTATION_SUMMARY.md` (merged; see `CHANGELOG.md` 0.4.0).

The original design rationale lives in `README.md` (the numbered spec). The
release history lives in `market_predictor_ml/CHANGELOG.md`. The code-level
audit that drove the 0.4.0 hardening pass is `Deep-Audit-ML3-1.txt`.

## Layer Flow

```
Historical market data
    |  data/            providers (yfinance/CSV), preprocessing, sentiment, macro
    |  features/        transformers + labels
    v
Model prediction  ->  models/   (LightGBM, Ridge, Logistic) + registry
    |
Position sizing   ->  decision/ (functional API + class-based sizers)
    |
Simulated execution with costs -> backtest/
    |
Reward / evaluation -> monitoring/ (metrics, alerting) + visualization/
```

## 1. Core Interfaces (`core/__init__.py`)

Ten abstract base classes: `IDataLoader`, `IFeatureTransformer`,
`ILabelGenerator`, `IPredictionModel`, `IPositionSizer`, `IBacktestEngine`,
`IRiskMonitor`, `IDataProvider`, `IModelRegistry`, `IEventSubscriber`.

`IBacktestEngine.run` is aligned with the concrete portfolio engine:
`run(signals, prices, volumes=None, **kwargs)` (date x symbol frames).
Vectorised numpy engines use their own interface.

## 2. Configuration

- `config/settings.py` — dataclass-based `Config` aggregating
  `DataConfig`, `FeatureConfig`, `LabelConfig`, `ModelConfig`,
  `DecisionConfig`, `BacktestConfig`. Includes the fields consumed by the
  hyperparameter search (`label_method`, `direction_threshold`,
  `fixed_position_size`, `signal_threshold`, `volatility_target`).
- `config/enhanced_settings.py` — Pydantic v2 model (`field_validator`,
  `model_validator`, `ConfigDict`) with `MPML_*` environment overrides.
  Top-level scalars and nested sections are supported, and values are parsed
  as JSON so list fields (e.g. `MPML_FEATURES_MOMENTUM_PERIODS=[5,10,20]`)
  can be overridden. Entry point: `load_config(environment=..., from_env=True)`.

```python
from market_predictor_ml.config.enhanced_settings import load_config

config = load_config(environment="production")
config.override_from_env()          # MPML_* variables
config.save_yaml("config.yaml")     # Config.from_yaml also available
```

## 3. Data Layer

- `data/providers.py` — `YFinanceDataProvider`, `CSVDataProvider`,
  `MarketDataLoader` (validation pipeline), plus two factories:
  - `create_data_provider(provider_type, config)` → `IDataProvider`
  - `create_data_loader(provider_type, **kwargs)` → `MarketDataLoader`
- `data/loader.py` — download helpers; `preprocess_data` / `compute_returns`
  are re-exported from `providers.py` (single source of truth).
- `data/sentiment.py` — news sentiment; per-day aggregation is O(n) over the
  price index, and mock news is seeded with `zlib.crc32` (process-stable).
- `data/macro.py` — CPI/GDP series; YoY growth auto-selects 4 periods for
  quarterly data and 12 for monthly.

## 4. Features (`features/pipeline.py`)

Scikit-learn style transformers chained by `FeaturePipeline`:
`MomentumTransformer`, `VolatilityTransformer`, `VolumeTransformer`,
`TechnicalIndicatorTransformer`, `TimeFeatureTransformer`, plus
`LabelGenerator` (direction / return / triple-barrier with independent
`profit_target` and `stop_loss` barriers) and `create_default_pipeline`.

```python
from market_predictor_ml.features import create_default_pipeline

pipeline = create_default_pipeline(horizon=5, label_type="direction")
X = pipeline.fit_transform(price_data)
pipeline.check_lookahead_bias(X)
pipeline.save("models/feature_pipeline_v1.pkl")
```

## 5. Models

- `models/predictors.py` — `LightGBMWrapper`, Ridge and Logistic baselines.
- `models/model_registry.py` — `InMemoryModelRegistry` plus module-level
  `get_registry()`, `register_model(...)`, `get_model(...)` helpers with
  metadata persistence and best-model selection by metric.

## 6. Decision Layer

- `decision/sizing.py` — functional API used by the optimizer:
  `create_positions(predictions, volatility, method, **kwargs)`. Keyword
  arguments are filtered per target sizer, so a shared kwargs dict is safe.
  `kelly_criterion_position` uses per-asset volatility (f* = mu / sigma^2),
  and `signal_strength_position` applies its `power` curve once.
- `decision/sizers.py` — class-based `IPositionSizer` hierarchy
  (`VolatilityAdjustedPositionSizer`, `KellyPositionSizer`,
  `FixedFractionPositionSizer`, `SignalStrengthPositionSizer`,
  `RiskParityPositionSizer`) via `create_position_sizer(method, **kwargs)`.
  Volatility sizing clips symmetrically, so short signals are honoured.

## 7. Backtesting

- `backtest/engine.py` — purged/embargoed walk-forward validation. Strategy
  returns use the next-bar convention prescribed by README §27
  (`positions[:-1] * y_test[1:]`), avoiding trade-at-close look-ahead.
- `backtest/enhanced_engine.py` — portfolio engine with:
  - `TransactionCostModel`: commission, half-spread, volume-based slippage
  - Gross-exposure cap (`max_gross_exposure`) and daily turnover throttle
    (`max_portfolio_turnover`)
  - Positions close on NaN/zero signals or sign flips; `Trade.return_pct`
    is side-aware for shorts
  - MAE/MFE captured from the recorded price path
  - `Benchmark.calculate_alpha_beta` (arithmetic annualization) and
    `Sortino` computed with canonical downside deviation


## 8. Live Trading (`live/`)

| Module | Lines | Role |
| --- | --- | --- |
| `oms.py` | 392 | Order lifecycle, fills, positions, callbacks (`RLock`-guarded) |
| `broker_base.py` | 211 | `BrokerAdapter` ABC + `update_price` hook + validation |
| `brokers.py` | 549 | `PaperBroker` (price-book for fills), Alpaca, IBKR adapters |
| `signals.py` | 263 | Prediction → signal; `target_quantity` (absolute) vs `delta_quantity` (signed trade) |
| `portfolio.py` | 328 | Weights, rebalancing (min/max position weight), signal orders |
| `risk.py` | 406 | Drawdown, daily loss, volatility (CRITICAL escalation), concentration, VaR/ES |
| `engine.py` | 372 | Event loop on `queue.Queue`, risk checks, clean `stop()` with thread join |
| `state_store.py` | 458 | SQLite persistence (cash/positions/risk/session restore) |
| `predictor.py` | 510 | Live prediction service; market hours via `ZoneInfo("America/New_York")` |
| `run_paper_trader.py` | 499 | Alpaca paper trader (JSONL trade log for the dashboard) |

Key behaviours:

- Producers push onto `queue.Queue`; the loop blocks with a timeout instead of
  busy-polling, and OMS mutations are lock-protected.
- `generate_signal` sets `target_quantity` (desired absolute position) and
  `delta_quantity` (signed trade needed); `generate_signal_orders` sizes orders
  from the delta and checks weights against the absolute target.
- The engine feeds portfolio returns to `LiveRiskMonitor`, enabling the
  volatility and VaR checks; `var_limit` is live via `calculate_var` /
  `calculate_expected_shortfall`.
- Price updates go through `BrokerAdapter.update_price` (no isinstance checks);
  `PaperBroker` overrides it for fill simulation.
- `TradingEngine.stop()` joins the loop thread (2 s timeout).

Two paper-trading entry points (see README §44):

1. Alpaca trader: `python market_predictor_ml/live/run_paper_trader.py`
   (`--check` / `--once`), logging to `paper_trades.jsonl`, visualised by
   `streamlit run market_predictor_ml/dashboard/paper_monitor.py`.
2. Engine harness: `python run_paper_trader.py --config paper_trading.yaml`
   (event-driven engine + OMS + `PaperBroker`, `--dry-run` supported).
   On restart it restores cash, positions, risk-halt state and session id.

Known caveat: `TradingEngine` instantiates its own internal `OrderManager`
while `PortfolioManager` holds another, so orders created by the portfolio
manager are submitted through the engine's OMS but do not appear in the
portfolio manager's book. Unify these before multi-account use.

## 9. Monitoring

- `monitoring/metrics.py` — `PerformanceMetrics` (Sharpe, canonical Sortino,
  arithmetic alpha/beta, Calmar, drawdown) and `MetricsCollector`, which
  converts daily dollar P&L to fractional returns using the equity snapshot
  (`equity_before`) before computing ratios.
- `monitoring/alerting.py` — `AlertManager.threshold_condition` accepts spaced
  operators; `AlertRule` carries a `threshold` field so message templates
  render real values. Factories: sharpe, drawdown, prediction-error, volume.
- `monitoring/dashboard.py` — HTML dashboard; chart configs are emitted as
  JSON (`json.dumps` + `html.escape`) for reliable `JSON.parse`.
- `monitoring/logger.py` — structured logging helpers.

## 10. Visualization

`visualization/dashboard.py` — `BacktestDashboard` with equity curve, drawdown,
monthly-returns heatmap (guarded for non-datetime indices), trade
distribution, position-sizing evolution and feature importance.

## 11. Reinforcement Learning (`rl/`)

- `environment.py` — `TradingEnv` (gymnasium optional; clear ImportError when
  absent), continuous `[0, 1]` action space, regime read from feature rows.
- `agents.py` — PPO/DQN factories (stable-baselines3 optional).
- `hybrid_strategy.py` — direction + RL position size with volatility and
  regime adjustments.

## 12. Testing

`market_predictor_ml/tests/` — 40 tests across four files:
`test_core.py` (10), `test_live_trading.py` (6), `test_monitoring.py` (17),
`test_rl_and_dashboard.py` (7). Run with `python -m pytest market_predictor_ml/tests -q`.

## Migration Guide

```python
# Config (Pydantic, env-overridable)
from market_predictor_ml.config.enhanced_settings import load_config
config = load_config(environment="development")

# Data loading (provider pattern)
from market_predictor_ml.data.providers import create_data_loader
loader = create_data_loader("csv", data_dir="./my_data")
data = loader.load("SPY", "2020-01-01", "2024-12-31")

# Position sizing (class-based, interface-driven)
from market_predictor_ml.decision.sizers import create_position_sizer
sizer = create_position_sizer("volatility_adjusted", target_volatility=0.02)
size = sizer.calculate_position(signal=0.8, volatility=0.025, capital=100_000)

# Portfolio backtest
from market_predictor_ml.backtest import create_backtest_engine
engine = create_backtest_engine(initial_capital=1_000_000, commission=0.001,
                                benchmark_returns=benchmark_ret)
results = engine.run(signals_df, prices_df, volumes_df)
```

## Duplicated Subsystems (deliberate, documented)

Two parallel implementations exist and are documented in their module
docstrings (audit M-9): `decision/sizing.py` (functional, used by hyperopt) vs
`decision/sizers.py` (class-based, interface-driven), and
`persistence/state_store.py` (unused reference schema) vs
`live/state_store.py` (runtime store used by the engine harness). Prefer one
per concern and reconcile before building on both.
