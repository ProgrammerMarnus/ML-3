"""
Backtest module - Walk-forward validation and performance evaluation.

Implements:
1. Walk-forward cross-validation with purging and embargo
2. Performance metrics (Sharpe, Sortino, Max Drawdown, etc.)
3. Economic reward calculation with transaction costs
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Generator
from sklearn.model_selection import BaseCrossValidator


class WalkForwardSplit(BaseCrossValidator):
    """
    Walk-forward cross-validation splitter with purging and embargo.
    
    This prevents look-ahead bias by:
    1. Training only on past data
    2. Purging: Removing training samples that overlap with test period labels
    3. Embargo: Adding gap between train and test to prevent leakage
    
    Parameters
    ----------
    n_splits : int
        Number of walk-forward splits
    test_size : int or float
        Size of test set (int=samples, float=fraction)
    train_size : int or float, optional
        Size of train set (if None, uses all available past data)
    purge_size : int
        Number of samples to purge from end of train set
    embargo_size : int
        Number of samples to skip between train and test
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,
        train_size: Optional[int] = None,
        purge_size: int = 0,
        embargo_size: int = 0,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.purge_size = purge_size
        self.embargo_size = embargo_size
    
    def split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into train/test sets.
        
        Yields
        ------
        train_idx : np.ndarray
            Indices for training set
        test_idx : np.ndarray
            Indices for test set
        """
        n_samples = len(X)
        
        # Calculate step size between splits
        if self.train_size is not None:
            fixed_train = self.train_size
        else:
            fixed_train = n_samples // 2
        
        step = (n_samples - fixed_train - self.test_size) // self.n_splits
        
        if step <= 0:
            raise ValueError("Not enough samples for specified splits and sizes")
        
        for i in range(self.n_splits):
            # Test set indices
            test_start = fixed_train + i * step
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            test_idx = np.arange(test_start, test_end)
            
            # Train set indices (before test start)
            train_end = test_start - self.purge_size - self.embargo_size
            train_idx = np.arange(0, train_end)
            
            if len(train_idx) == 0:
                continue
            
            yield train_idx, test_idx
    
    def get_n_splits(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return number of splits."""
        return self.n_splits


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    risk_free_rate : float
        Annual risk-free rate
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    return sharpe


def compute_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sortino ratio (uses downside deviation).
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    risk_free_rate : float
        Annual risk-free rate
    
    Returns
    -------
    float
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.std(downside_returns)
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    return sortino


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Compute maximum drawdown from equity curve.
    
    Parameters
    ----------
    equity_curve : np.ndarray
        Cumulative equity curve
    
    Returns
    -------
    float
        Maximum drawdown (as positive fraction)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-10)
    
    return np.max(drawdown)


def compute_calmar_ratio(returns: np.ndarray) -> float:
    """
    Compute Calmar ratio (annual return / max drawdown).
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    
    Returns
    -------
    float
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Annualized return
    ann_return = np.mean(returns) * 252
    
    # Equity curve
    equity = np.cumprod(1 + returns)
    max_dd = compute_max_drawdown(equity)
    
    if max_dd == 0:
        return np.inf if ann_return > 0 else 0.0
    
    return ann_return / max_dd


def compute_economic_metrics(
    returns: np.ndarray,
    positions: Optional[np.ndarray] = None,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005
) -> Dict[str, float]:
    """
    Compute comprehensive economic performance metrics.
    
    Parameters
    ----------
    returns : np.ndarray
        Strategy returns (after costs if already computed)
    positions : np.ndarray, optional
        Position sizes (for turnover calculation)
    transaction_cost : float
        Transaction cost per unit traded
    slippage : float
        Slippage cost per unit traded
    
    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {}
    
    # Basic statistics
    total_return = np.prod(1 + returns) - 1
    ann_return = np.mean(returns) * 252
    vol = np.std(returns) * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    
    # Drawdown metrics
    equity = np.cumprod(1 + returns)
    max_dd = compute_max_drawdown(equity)
    calmar = compute_calmar_ratio(returns)
    
    # Win/loss statistics
    wins = returns > 0
    win_rate = np.sum(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(returns[wins]) if np.sum(wins) > 0 else 0
    avg_loss = np.mean(returns[~wins]) if np.sum(~wins) > 0 else 0
    profit_factor = -avg_win / avg_loss if avg_loss != 0 else np.inf
    
    # Turnover (if positions provided)
    turnover = None
    if positions is not None:
        position_changes = np.abs(np.diff(positions, prepend=0))
        turnover = np.mean(position_changes)
    
    metrics = {
        'total_return': total_return,
        'annual_return': ann_return,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'turnover': turnover,
        'n_trades': len(returns),
    }
    
    return metrics


def apply_transaction_costs(
    returns: np.ndarray,
    positions: np.ndarray,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005
) -> np.ndarray:
    """
    Apply transaction costs and slippage to returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Gross returns
    positions : np.ndarray
        Position sizes at start of period
    transaction_cost : float
        Transaction cost per unit traded
    slippage : float
        Slippage cost per unit traded
    
    Returns
    -------
    np.ndarray
        Net returns after costs
    """
    # Position changes (turnover)
    position_changes = np.abs(np.diff(positions, prepend=0))
    
    # Total trading cost
    trading_cost = position_changes * (transaction_cost + slippage)
    
    # Net returns
    net_returns = returns - trading_cost
    
    return net_returns


def run_walk_forward_backtest(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_splitter: WalkForwardSplit,
    position_method: str = 'fixed',
    volatility: Optional[np.ndarray] = None,
    transaction_cost: float = 0.001,
) -> Dict:
    """
    Run complete walk-forward backtest.
    
    Parameters
    ----------
    model : BaseEstimator
        ML model to train
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    cv_splitter : WalkForwardSplit
        Cross-validation splitter
    position_method : str
        Position sizing method
    volatility : np.ndarray, optional
        Volatility estimates for sizing
    transaction_cost : float
        Transaction cost per unit traded
    
    Returns
    -------
    Dict
        Backtest results including predictions, positions, returns, and metrics
    """
    all_predictions = []
    all_positions = []
    all_returns = []
    all_indices = []
    
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        print(f"Fold {fold + 1}/{cv_splitter.n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Create positions
        if position_method == 'volatility_adjusted' and volatility is not None:
            vol_test = volatility[test_idx]
            positions = create_positions(predictions, volatility=vol_test, method=position_method)
        else:
            positions = create_positions(predictions, method=position_method)
        
        # Compute returns: the position decided at close t earns the NEXT
        # bar return (position.shift(1) * next_day_return), matching the
        # README methodology and avoiding trade-at-close look-ahead.
        n = len(y_test)
        strategy_returns = np.zeros(n)
        if n > 1:
            strategy_returns[1:] = positions[:-1] * y_test[1:]
        
        # Apply transaction costs
        strategy_returns = apply_transaction_costs(
            strategy_returns, positions, transaction_cost=transaction_cost
        )
        
        # Store results
        all_predictions.extend(predictions)
        all_positions.extend(positions)
        all_returns.extend(strategy_returns)
        all_indices.extend(test_idx)
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_positions = np.array(all_positions)
    all_returns = np.array(all_returns)
    all_indices = np.array(all_indices)
    
    # Compute metrics
    metrics = compute_economic_metrics(all_returns, all_positions, transaction_cost=transaction_cost)
    
    results = {
        'predictions': all_predictions,
        'positions': all_positions,
        'returns': all_returns,
        'indices': all_indices,
        'metrics': metrics,
        'equity_curve': np.cumprod(1 + all_returns),
    }
    
    return results


# Import here to avoid circular dependency
from market_predictor_ml.decision import create_positions
