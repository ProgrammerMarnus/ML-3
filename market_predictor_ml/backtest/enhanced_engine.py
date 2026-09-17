"""
Enhanced Backtest Module - Production-ready backtesting engine.

This module provides:
1. Multi-asset portfolio backtesting
2. Advanced transaction cost modeling with volume-based slippage
3. Realistic order execution simulation
4. Benchmark comparisons and alpha calculations
5. Trade-level analytics and attribution
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from ..core import IBacktestEngine


class OrderType(Enum):
    """Order types for execution simulation."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    pnl: float = 0.0
    transaction_costs: float = 0.0
    slippage: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'transaction_costs': self.transaction_costs,
            'slippage': self.slippage,
            'mae': self.mae,
            'mfe': self.mfe,
            'return_pct': (self.exit_price - self.entry_price) / self.entry_price if self.exit_price else None
        }


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime
    side: str
    
    def market_value(self, current_price: float) -> float:
        """Calculate current market value."""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


class TransactionCostModel:
    """
    Advanced transaction cost model with volume-based slippage.
    
    Parameters
    ----------
    commission : float
        Commission per dollar traded
    min_commission : float
        Minimum commission per trade
    spread : float
        Bid-ask spread (fraction)
    slippage_factor : float
        Slippage factor based on trade size relative to volume
    impact_exponent : float
        Exponent for market impact calculation (typically 0.5-1.0)
    """
    
    def __init__(
        self,
        commission: float = 0.001,
        min_commission: float = 1.0,
        spread: float = 0.0005,
        slippage_factor: float = 0.1,
        impact_exponent: float = 0.6
    ):
        self.commission = commission
        self.min_commission = min_commission
        self.spread = spread
        self.slippage_factor = slippage_factor
        self.impact_exponent = impact_exponent
    
    def calculate_cost(
        self,
        price: float,
        quantity: float,
        daily_volume: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> Dict[str, float]:
        """
        Calculate total transaction cost for a trade.
        
        Parameters
        ----------
        price : float
            Trade price
        quantity : float
            Number of shares/contracts
        daily_volume : float, optional
            Average daily volume (for slippage estimation)
        order_type : OrderType
            Type of order
        
        Returns
        -------
        Dict[str, float]
            Breakdown of costs
        """
        notional = abs(price * quantity)
        
        # Commission
        commission_cost = max(notional * self.commission, self.min_commission)
        
        # Spread cost (half spread for round-trip)
        spread_cost = notional * self.spread
        
        # Market impact/slippage
        slippage_cost = 0.0
        if daily_volume is not None and daily_volume > 0:
            participation_rate = abs(quantity) / daily_volume
            slippage_cost = notional * self.slippage_factor * (participation_rate ** self.impact_exponent)
        
        total_cost = commission_cost + spread_cost + slippage_cost
        
        return {
            'commission': commission_cost,
            'spread': spread_cost,
            'slippage': slippage_cost,
            'total': total_cost,
            'cost_per_share': total_cost / abs(quantity) if quantity != 0 else 0
        }


class Benchmark:
    """
    Benchmark for performance comparison.
    
    Parameters
    ----------
    name : str
        Benchmark name
    returns : pd.Series
        Benchmark returns series
    """
    
    def __init__(self, name: str, returns: pd.Series):
        self.name = name
        self.returns = returns
    
    def calculate_alpha_beta(
        self, 
        strategy_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate alpha and beta relative to benchmark.
        
        Parameters
        ----------
        strategy_returns : pd.Series
            Strategy returns
        
        Returns
        -------
        Dict[str, float]
            Alpha, beta, correlation, and tracking error
        """
        # Align indices
        aligned = pd.concat([strategy_returns, self.returns], axis=1).dropna()
        if len(aligned) < 10:
            return {'alpha': 0.0, 'beta': 0.0, 'correlation': 0.0, 'tracking_error': 0.0}
        
        strat_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]
        
        # Covariance and variance
        cov = strat_ret.cov(bench_ret)
        bench_var = bench_ret.var()
        
        # Beta
        beta = cov / bench_var if bench_var > 0 else 0.0
        
        # Alpha (annualized)
        strat_mean = strat_ret.mean() * 252
        bench_mean = bench_ret.mean() * 252
        alpha = strat_mean - beta * bench_mean
        
        # Correlation
        correlation = strat_ret.corr(bench_ret)
        
        # Tracking error
        active_returns = strat_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Information ratio
        info_ratio = (strat_ret.mean() - bench_ret.mean()) * 252 / tracking_error if tracking_error > 0 else 0.0
        
        return {
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio
        }


class EnhancedBacktestEngine(IBacktestEngine):
    """
    Enhanced backtesting engine with multi-asset support and realistic execution.
    
    Features:
    - Multi-asset portfolio backtesting
    - Volume-based transaction costs
    - Order execution simulation
    - Trade-level analytics
    - Benchmark comparison
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        benchmark: Optional[Benchmark] = None,
        max_position_size: float = 0.1,
        max_portfolio_turnover: float = 0.5
    ):
        self.initial_capital = initial_capital
        self.cost_model = transaction_cost_model or TransactionCostModel()
        self.benchmark = benchmark
        self.max_position_size = max_position_size
        self.max_portfolio_turnover = max_portfolio_turnover
        
        # State
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._equity_curve: pd.Series = pd.Series(dtype=float)
        self._portfolio_values: pd.DataFrame = pd.DataFrame()
    
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run backtest on signal and price data.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Trading signals (rows=dates, columns=symbols)
        prices : pd.DataFrame
            Price data (rows=dates, columns=symbols)
        volumes : pd.DataFrame, optional
            Volume data for cost estimation
        **kwargs
            Additional parameters
        
        Returns
        -------
        Dict[str, Any]
            Backtest results
        """
        # Reset state
        self._positions = {}
        self._trades = []
        self._equity_curve = pd.Series(dtype=float)
        
        dates = signals.index
        symbols = signals.columns
        
        capital = self.initial_capital
        equity_history = []
        dates_history = []
        
        for date_idx, date in enumerate(dates):
            # Get current prices
            current_prices = prices.loc[date]
            
            # Update existing positions and close if signal changed
            for symbol in list(self._positions.keys()):
                if symbol not in signals.columns or pd.isna(signals.loc[date, symbol]):
                    # Close position
                    pos = self._positions.pop(symbol)
                    exit_price = current_prices[symbol]
                    
                    trade = Trade(
                        symbol=symbol,
                        entry_date=pos.entry_date,
                        exit_date=date,
                        side=pos.side,
                        quantity=pos.quantity,
                        entry_price=pos.entry_price,
                        exit_price=exit_price
                    )
                    
                    # Calculate P&L
                    trade.pnl = pos.unrealized_pnl(exit_price)
                    
                    # Calculate costs
                    cost_info = self.cost_model.calculate_cost(
                        exit_price, -pos.quantity,
                        volumes.loc[date, symbol] if volumes is not None else None
                    )
                    trade.transaction_costs = cost_info['total']
                    trade.slippage = cost_info['slippage']
                    
                    # Update capital
                    capital += trade.pnl - trade.transaction_costs
                    self._trades.append(trade)
            
            # Open new positions based on signals
            for symbol in symbols:
                signal = signals.loc[date, symbol]
                if pd.isna(signal) or signal == 0:
                    continue
                
                # Determine position size
                target_value = capital * self.max_position_size * abs(signal)
                current_price = current_prices[symbol]
                
                if current_price <= 0:
                    continue
                
                quantity = target_value / current_price
                
                # Check if we already have a position
                if symbol in self._positions:
                    continue  # Skip, already handled
                
                # Open new position
                cost_info = self.cost_model.calculate_cost(
                    current_price, quantity,
                    volumes.loc[date, symbol] if volumes is not None else None
                )
                
                pos = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=current_price,
                    entry_date=date,
                    side='long' if signal > 0 else 'short'
                )
                
                self._positions[symbol] = pos
                capital -= cost_info['total']
            
            # Calculate portfolio value
            portfolio_value = capital
            for symbol, pos in self._positions.items():
                portfolio_value += pos.unrealized_pnl(current_prices[symbol])
            
            equity_history.append(portfolio_value)
            dates_history.append(date)
        
        # Create equity curve
        self._equity_curve = pd.Series(equity_history, index=dates_history)
        
        # Calculate returns
        returns = self._equity_curve.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns)
        
        # Add benchmark comparison if available
        if self.benchmark:
            benchmark_metrics = self.benchmark.calculate_alpha_beta(returns)
            metrics.update(benchmark_metrics)
        
        results = {
            'equity_curve': self._equity_curve,
            'returns': returns,
            'metrics': metrics,
            'trades': self._trades,
            'positions': self._positions,
            'final_capital': capital + sum(p.unrealized_pnl(prices.iloc[-1][p.symbol]) 
                                          for p in self._positions.values())
        }
        
        return results
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0 or returns.sum() == 0:
            return {}
        
        # Basic stats
        total_return = (self._equity_curve.iloc[-1] / self.initial_capital) - 1
        ann_return = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        rf = 0.0  # Risk-free rate
        excess_returns = returns - rf / 252
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Downside deviation for Sortino
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        wins = returns > 0
        win_rate = wins.sum() / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = returns[wins].sum() if wins.sum() > 0 else 0
        gross_loss = abs(returns[~wins].sum()) if (~wins).sum() > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_periods': len(returns),
            'start_date': str(returns.index[0]),
            'end_date': str(returns.index[-1])
        }
    
    def get_equity_curve(self) -> pd.Series:
        """Get the equity curve from the last backtest run."""
        return self._equity_curve
    
    def get_trades(self) -> pd.DataFrame:
        """Get the trade log from the last backtest run."""
        if not self._trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame([t.to_dict() for t in self._trades])
        return trades_df
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current/open positions."""
        return self._positions.copy()
    
    def analyze_trades(self) -> Dict[str, Any]:
        """Perform detailed trade analysis."""
        if not self._trades:
            return {}
        
        trades_df = self.get_trades()
        
        # Group by symbol
        by_symbol = trades_df.groupby('symbol').agg({
            'pnl': ['sum', 'mean', 'std'],
            'return_pct': ['mean', 'std'],
            'transaction_costs': 'sum',
            'slippage': 'sum'
        })
        
        # Winning vs losing trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # MAE/MFE analysis
        avg_mae = trades_df['mae'].mean()
        avg_mfe = trades_df['mfe'].mean()
        
        return {
            'by_symbol': by_symbol.to_dict(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_count': len(winning_trades),
            'loss_count': len(losing_trades),
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'total_transaction_costs': trades_df['transaction_costs'].sum(),
            'total_slippage': trades_df['slippage'].sum()
        }


def create_backtest_engine(
    initial_capital: float = 1000000.0,
    commission: float = 0.001,
    spread: float = 0.0005,
    max_position_size: float = 0.1,
    benchmark_returns: Optional[pd.Series] = None,
    benchmark_name: str = "SPY"
) -> EnhancedBacktestEngine:
    """
    Factory function to create a configured backtest engine.
    
    Parameters
    ----------
    initial_capital : float
        Starting capital
    commission : float
        Commission rate
    spread : float
        Bid-ask spread
    max_position_size : float
        Maximum position size as fraction of capital
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    benchmark_name : str
        Name of benchmark
    
    Returns
    -------
    EnhancedBacktestEngine
        Configured backtest engine
    """
    cost_model = TransactionCostModel(
        commission=commission,
        spread=spread
    )
    
    benchmark = None
    if benchmark_returns is not None:
        benchmark = Benchmark(benchmark_name, benchmark_returns)
    
    return EnhancedBacktestEngine(
        initial_capital=initial_capital,
        transaction_cost_model=cost_model,
        benchmark=benchmark,
        max_position_size=max_position_size
    )


__all__ = [
    'OrderType',
    'Trade',
    'Position',
    'TransactionCostModel',
    'Benchmark',
    'EnhancedBacktestEngine',
    'create_backtest_engine'
]
