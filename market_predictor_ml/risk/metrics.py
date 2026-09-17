"""
Advanced Risk Metrics for portfolio analysis.
Implements VaR, CVaR, Sortino, Calmar, Omega, and other institutional measures.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple


def calculate_var(
    returns: Union[np.ndarray, pd.Series],
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: 'historical' or 'gaussian'
        
    Returns:
        VaR as a positive number (loss)
    """
    returns_array = np.asarray(returns)
    
    if method == "historical":
        return -np.percentile(returns_array, (1 - confidence_level) * 100)
    
    elif method == "gaussian":
        mu = np.mean(returns_array)
        sigma = np.std(returns_array)
        return -(mu + sigma * stats.norm.ppf(1 - confidence_level))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_cvar(
    returns: Union[np.ndarray, pd.Series],
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).
    
    Average loss beyond the VaR threshold.
    """
    returns_array = np.asarray(returns)
    var = calculate_var(returns_array, confidence_level, method="historical")
    
    # Get returns worse than VaR
    tail_returns = returns_array[returns_array <= -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -np.mean(tail_returns)


def calculate_sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Calculate Sortino Ratio.
    
    Like Sharpe but only penalizes downside volatility.
    """
    returns_array = np.asarray(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Downside deviation
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.inf
    
    annualized_return = np.mean(excess_returns) * annualization_factor
    annualized_downside = downside_std * np.sqrt(annualization_factor)
    
    return annualized_return / annualized_downside


def calculate_calmar_ratio(
    returns: Union[np.ndarray, pd.Series],
    annualization_factor: int = 252,
) -> float:
    """
    Calculate Calmar Ratio.
    
    Annualized return divided by maximum drawdown.
    """
    returns_array = np.asarray(returns)
    
    # Annualized return
    total_return = np.prod(1 + returns_array) - 1
    n_years = len(returns_array) / annualization_factor
    if n_years <= 0:
        return 0.0
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Maximum drawdown
    cum_returns = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    max_dd = np.max(drawdown)
    
    if max_dd == 0:
        return np.inf
    
    return annualized_return / max_dd


def calculate_omega_ratio(
    returns: Union[np.ndarray, pd.Series],
    threshold: float = 0.0,
) -> float:
    """
    Calculate Omega Ratio.
    
    Ratio of gains to losses relative to a threshold.
    """
    returns_array = np.asarray(returns)
    
    gains = returns_array[returns_array > threshold] - threshold
    losses = threshold - returns_array[returns_array <= threshold]
    
    if np.sum(losses) == 0:
        return np.inf
    
    return np.sum(gains) / np.sum(losses)


def calculate_max_drawdown(
    returns: Union[np.ndarray, pd.Series],
) -> Tuple[float, int, int]:
    """
    Calculate Maximum Drawdown and its start/end indices.
    
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    returns_array = np.asarray(returns)
    cum_returns = np.cumprod(1 + returns_array)
    
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    
    max_dd = np.max(drawdown)
    end_idx = np.argmax(drawdown)
    
    # Find start of drawdown
    peak_before_end = np.argmax(cum_returns[:end_idx + 1])
    
    return max_dd, peak_before_end, end_idx


def calculate_tail_ratio(
    returns: Union[np.ndarray, pd.Series],
    percentile: float = 5,
) -> float:
    """
    Calculate Tail Ratio.
    
    Ratio of right tail to left tail returns.
    """
    returns_array = np.asarray(returns)
    
    right_tail = np.percentile(returns_array, 100 - percentile)
    left_tail = np.percentile(returns_array, percentile)
    
    if abs(left_tail) < 1e-8:
        return np.inf
    
    return abs(right_tail / left_tail)
