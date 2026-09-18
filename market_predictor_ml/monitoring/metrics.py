"""
Metrics Collection Module

Provides metrics collection for performance monitoring including:
- Prediction accuracy metrics
- Portfolio performance metrics (Sharpe, Sortino, etc.)
- Trading metrics (turnover, win rate, etc.)
- System health metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.
    
    Metrics include:
    - Returns: total, annualized, cumulative
    - Risk-adjusted: Sharpe, Sortino, Calmar ratios
    - Drawdown: max, average, duration
    - Trading: win rate, profit factor, turnover
    - Alpha/Beta vs benchmark
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate periodic returns from price series."""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns from periodic returns."""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_annualized_return(
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        total_return = (1 + returns).prod()
        years = len(returns) / periods_per_year

        if years <= 0:
            return 0.0

        return (total_return ** (1 / years)) - 1
    
    @staticmethod
    def calculate_volatility(
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0

        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Periodic returns series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Sharpe ratio (annualized)
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)

        if excess_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() * periods_per_year) / (
            excess_returns.std() * np.sqrt(periods_per_year)
        )
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        target_return: float = 0.0,
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation instead of total volatility).
        """
        returns = pd.Series(returns).dropna()
        if len(returns) < 2:
            return 0.0

        rf_per = risk_free_rate / periods_per_year
        excess = returns - rf_per
        downside = np.minimum(excess - target_return, 0.0)
        downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)
        if downside_dev == 0:
            return 0.0 if excess.mean() <= 0 else float("inf")
        return (excess.mean() * periods_per_year) / downside_dev

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.

        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        if len(cumulative_returns) == 0:
            return 0.0, 0, 0

        # Convert to wealth index
        wealth = 1 + cumulative_returns

        # Running maximum
        running_max = wealth.cummax()

        # Drawdown series
        drawdown = (wealth - running_max) / running_max

        # Max drawdown
        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()

        # Find corresponding peak
        peak_idx = wealth[:trough_idx].idxmax()

        return max_dd, peak_idx, trough_idx
    
    @staticmethod
    def calculate_calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(returns) == 0:
            return 0.0

        annualized_return = PerformanceMetrics.calculate_annualized_return(
            returns, periods_per_year
        )
        cumulative = PerformanceMetrics.calculate_cumulative_returns(returns)
        max_dd, _, _ = PerformanceMetrics.calculate_max_drawdown(cumulative)

        if max_dd == 0:
            return 0.0

        return annualized_return / abs(max_dd)
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trade list."""
        if not trades:
            return 0.0

        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        if not trades:
            return 0.0

        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_turnover(
        positions: pd.DataFrame,
        portfolio_value: float,
    ) -> float:
        """
        Calculate portfolio turnover.

        Args:
            positions: DataFrame with position changes
            portfolio_value: Average portfolio value

        Returns:
            Turnover ratio
        """
        if portfolio_value == 0:
            return 0.0

        total_turnover = positions["change"].abs().sum()
        return total_turnover / portfolio_value
    
    @staticmethod
    def calculate_alpha_beta(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Tuple[float, float]:
        """
        Calculate alpha and beta relative to benchmark.

        Returns:
            Tuple of (alpha, beta) - both annualized
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0

        # Align series
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0, 0.0

        r_strat = aligned.iloc[:, 0]
        r_bench = aligned.iloc[:, 1]

        # Excess returns
        rf_daily = risk_free_rate / periods_per_year
        excess_strat = r_strat - rf_daily
        excess_bench = r_bench - rf_daily

        # Beta calculation
        covariance = excess_strat.cov(excess_bench)
        variance = excess_bench.var()

        if variance == 0:
            beta = 0.0
        else:
            beta = covariance / variance

        # Alpha (arithmetic annualization, matching Benchmark.calculate_alpha_beta;
        # avoids NaN from CAGR on negative cumulative excess returns).
        alpha = (excess_strat.mean() - beta * excess_bench.mean()) * periods_per_year

        return alpha, beta
    
    @staticmethod
    def calculate_information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate information ratio (active return / tracking error)."""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        # Align series
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0

        active_return = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        tracking_error = active_return.std() * np.sqrt(periods_per_year)

        if tracking_error == 0:
            return 0.0

        return (active_return.mean() * periods_per_year) / tracking_error


class MetricsCollector:
    """
    Collects and stores metrics over time for monitoring and analysis.
    
    Features:
    - Time-series metric storage
    - Aggregation functions
    - Export to JSON/DataFrame
    - Alert threshold checking
    """
    
    def __init__(self, window_size: int = 252):
        """
        Initialize metrics collector.

        Args:
            window_size: Rolling window size for calculations (default: 1 year)
        """
        self.window_size = window_size
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.trade_log: List[Dict[str, Any]] = []
        self.prediction_log: List[Dict[str, Any]] = []
    
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = datetime.now()

        self.metrics_history[name].append((timestamp, value))

        # Trim old data
        if len(self.metrics_history[name]) > self.window_size * 2:
            cutoff = datetime.now() - timedelta(days=self.window_size * 2)
            self.metrics_history[name] = [
                (ts, val) for ts, val in self.metrics_history[name] if ts > cutoff
            ]
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a trade for performance analysis."""
        trade["timestamp"] = trade.get("timestamp", datetime.now())
        self.trade_log.append(trade)

        # Update metrics
        if "pnl" in trade:
            self.record_metric("cumulative_pnl", self.get_total_pnl())
    
    def record_prediction(
        self,
        symbol: str,
        prediction: float,
        actual: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a prediction for accuracy tracking."""
        pred_record = {
            "symbol": symbol,
            "prediction": prediction,
            "actual": actual,
            "timestamp": timestamp or datetime.now(),
        }
        self.prediction_log.append(pred_record)

        # Update accuracy metrics if actual is available
        if actual is not None:
            error = prediction - actual
            self.record_metric("prediction_error", error)
            self.record_metric("absolute_error", abs(error))
    
    def get_metric_series(self, name: str) -> pd.Series:
        """Get time series for a metric."""
        if name not in self.metrics_history or not self.metrics_history[name]:
            return pd.Series(dtype=float)

        df = pd.DataFrame(self.metrics_history[name], columns=["timestamp", "value"])
        df.set_index("timestamp", inplace=True)
        return df["value"]
    
    def get_total_pnl(self) -> float:
        """Get total P&L from trade log."""
        return sum(t.get("pnl", 0) for t in self.trade_log)
    
    def calculate_recent_metrics(
        self,
        days: int = 30,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculate metrics for recent period."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trade_log if t.get("timestamp", datetime.now()) > cutoff]

        if not recent_trades:
            return {}

        # Calculate daily returns from trades: convert dollar P&L to
        # fractional returns using the equity available at day start.
        # Sharpe/Sortino expect returns, not dollar amounts (H-4).
        daily_pnl = defaultdict(float)
        daily_equity = {}
        for trade in recent_trades:
            ts = trade.get("timestamp", datetime.now())
            if isinstance(ts, datetime):
                date_key = ts.date()
                daily_pnl[date_key] += trade.get("pnl", 0)
                daily_equity.setdefault(date_key, trade.get("equity_before"))

        if not daily_pnl:
            return {}

        dates = sorted(daily_pnl)
        rets = []
        for d in dates:
            base = daily_equity.get(d) or 1.0
            try:
                base_f = float(base)
            except (TypeError, ValueError):
                base_f = 1.0
            if base_f == 0:
                base_f = 1.0
            rets.append(daily_pnl[d] / base_f)
        returns_series = pd.Series(rets, index=dates)

        metrics = {
            "total_pnl": self.get_total_pnl(),
            "num_trades": len(recent_trades),
            "win_rate": PerformanceMetrics.calculate_win_rate(recent_trades),
            "profit_factor": PerformanceMetrics.calculate_profit_factor(recent_trades),
            "sharpe_ratio": PerformanceMetrics.calculate_sharpe_ratio(returns_series),
            "sortino_ratio": PerformanceMetrics.calculate_sortino_ratio(returns_series),
            "max_drawdown": PerformanceMetrics.calculate_max_drawdown(
                PerformanceMetrics.calculate_cumulative_returns(returns_series)
            )[0],
        }

        if benchmark_returns is not None:
            alpha, beta = PerformanceMetrics.calculate_alpha_beta(returns_series, benchmark_returns)
            metrics["alpha"] = alpha
            metrics["beta"] = beta

        return metrics
    
    def get_prediction_accuracy(self, days: int = 30) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_preds = [
            p for p in self.prediction_log
            if p.get("actual") is not None and p.get("timestamp", datetime.now()) > cutoff
        ]

        if not recent_preds:
            return {}

        errors = [abs(p["prediction"] - p["actual"]) for p in recent_preds]
        squared_errors = [(p["prediction"] - p["actual"]) ** 2 for p in recent_preds]
        
        mape_values = [
            abs(e / p["actual"])
            for e, p in zip(errors, recent_preds)
            if p["actual"] != 0
        ]

        return {
            "mae": np.mean(errors),
            "rmse": np.sqrt(np.mean(squared_errors)),
            "mape": float(np.mean(mape_values) * 100) if mape_values else 0.0,
            "num_predictions": len(recent_preds),
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all metrics to dictionary."""
        return {
            "metrics_history": {
                name: [(ts.isoformat(), val) for ts, val in values]
                for name, values in self.metrics_history.items()
            },
            "trade_count": len(self.trade_log),
            "prediction_count": len(self.prediction_log),
            "total_pnl": self.get_total_pnl(),
        }
    
    def export_to_json(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.export_to_dict(), f, indent=2)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to DataFrame."""
        dfs = []
        for name, values in self.metrics_history.items():
            if values:
                df = pd.DataFrame(values, columns=["timestamp", "value"])
                df["metric"] = name
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)
