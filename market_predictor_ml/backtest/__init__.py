"""Backtest module initialization."""

from .engine import (
    WalkForwardSplit,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_economic_metrics,
    apply_transaction_costs,
    run_walk_forward_backtest,
)
from .enhanced_engine import (
    OrderType,
    Trade,
    Position,
    TransactionCostModel,
    Benchmark,
    EnhancedBacktestEngine,
    create_backtest_engine,
)

__all__ = [
    # Original engine
    "WalkForwardSplit",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    "compute_calmar_ratio",
    "compute_economic_metrics",
    "apply_transaction_costs",
    "run_walk_forward_backtest",
    # Enhanced engine
    "OrderType",
    "Trade",
    "Position",
    "TransactionCostModel",
    "Benchmark",
    "EnhancedBacktestEngine",
    "create_backtest_engine",
]
