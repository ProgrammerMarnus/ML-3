"""
Label construction module.

Implements multiple label construction methods:
1. Future returns (continuous)
2. Direction with thresholds (discrete)
3. Triple-barrier method
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def compute_future_returns(
    df: pd.DataFrame,
    horizons: list[int] = [1, 5, 21],
    return_type: str = 'simple'
) -> pd.DataFrame:
    """
    Compute future returns for different horizons.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Close' column
    horizons : list[int]
        List of future horizons (in days)
    return_type : str
        'simple' for simple returns, 'log' for log returns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added future return columns
    """
    df = df.copy()
    
    for h in horizons:
        if return_type == 'simple':
            df[f'FutureReturn_{h}d'] = df['Close'].shift(-h) / df['Close'] - 1
        elif return_type == 'log':
            df[f'FutureReturn_{h}d'] = np.log(df['Close'].shift(-h) / df['Close'])
        else:
            raise ValueError("return_type must be 'simple' or 'log'")
    
    return df


def compute_direction_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
    include_neutral: bool = True
) -> pd.DataFrame:
    """
    Create directional labels based on future returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Close' column
    horizon : int
        Prediction horizon in days
    threshold : float
        Threshold for up/down classification (e.g., 0.02 for 2%)
    include_neutral : bool
        Whether to include neutral class (returns within threshold)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with direction labels (1=up, 0=neutral/-1=down)
    """
    df = df.copy()
    
    # Compute future return
    future_return = df['Close'].shift(-horizon) / df['Close'] - 1
    
    if include_neutral:
        # Three-class: up (1), neutral (0), down (-1)
        conditions = [
            future_return > threshold,
            future_return < -threshold,
            (future_return >= -threshold) & (future_return <= threshold)
        ]
        choices = [1, -1, 0]
        df[f'Direction_{h}d'] = np.select(conditions, choices, default=0)
    else:
        # Binary: up (1), down (0)
        df[f'Direction_{horizon}d'] = (future_return > threshold).astype(int)
    
    return df


def triple_barrier_labeling(
    df: pd.DataFrame,
    horizon: int = 21,
    profit_target: float = 0.05,
    stop_loss: float = 0.03,
    vertical_barrier: Optional[int] = None
) -> pd.DataFrame:
    """
    Implement the triple-barrier labeling method.
    
    This method labels each observation based on which barrier is hit first:
    - Upper barrier (profit target): label = 1
    - Lower barrier (stop loss): label = -1
    - Vertical barrier (time limit): label based on return at that time
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns and DatetimeIndex
    horizon : int
        Maximum holding period in days
    profit_target : float
        Profit target as a fraction (e.g., 0.05 for 5%)
    stop_loss : float
        Stop loss as a fraction (e.g., 0.03 for 3%)
    vertical_barrier : int, optional
        Time limit in days. If None, uses horizon.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with triple-barrier labels and metadata
    """
    df = df.copy()
    
    if vertical_barrier is None:
        vertical_barrier = horizon
    
    n_samples = len(df)
    labels = np.zeros(n_samples)
    reasons = np.zeros(n_samples)  # 1=profit target, -1=stop loss, 0=time limit
    t_touch = np.full(n_samples, np.nan)  # Time to touch
    
    close_prices = df['Close'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    
    for i in range(n_samples - horizon):
        entry_price = close_prices[i]
        
        # Define barriers
        upper_barrier = entry_price * (1 + profit_target)
        lower_barrier = entry_price * (1 - stop_loss)
        
        # Get price path for the holding period
        end_idx = min(i + vertical_barrier, n_samples - 1)
        path_high = high_prices[i+1:end_idx+1]
        path_low = low_prices[i+1:end_idx+1]
        
        if len(path_high) == 0:
            continue
        
        # Check which barrier is hit first
        upper_hit = np.where(path_high >= upper_barrier)[0]
        lower_hit = np.where(path_low <= lower_barrier)[0]
        
        if len(upper_hit) > 0 and len(lower_hit) > 0:
            # Both barriers hit - check which came first
            if upper_hit[0] < lower_hit[0]:
                labels[i] = 1
                reasons[i] = 1
                t_touch[i] = upper_hit[0]
            else:
                labels[i] = -1
                reasons[i] = -1
                t_touch[i] = lower_hit[0]
        elif len(upper_hit) > 0:
            # Only upper barrier hit
            labels[i] = 1
            reasons[i] = 1
            t_touch[i] = upper_hit[0]
        elif len(lower_hit) > 0:
            # Only lower barrier hit
            labels[i] = -1
            reasons[i] = -1
            t_touch[i] = lower_hit[0]
        else:
            # No barrier hit - use return at vertical barrier
            exit_price = close_prices[end_idx]
            ret = exit_price / entry_price - 1
            labels[i] = np.sign(ret)
            reasons[i] = 0
            t_touch[i] = vertical_barrier
    
    df['TripleBarrier_Label'] = labels
    df['TripleBarrier_Reason'] = reasons
    df['TripleBarrier_TouchTime'] = t_touch
    
    return df


def create_regression_target(
    df: pd.DataFrame,
    horizon: int = 21,
    risk_adjust: bool = False,
    volatility_window: int = 21
) -> pd.DataFrame:
    """
    Create regression target: risk-adjusted future returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    horizon : int
        Return horizon in days
    risk_adjust : bool
        Whether to divide by volatility (Sharpe-like adjustment)
    volatility_window : int
        Window for volatility calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with target column
    """
    df = df.copy()
    
    # Future return
    future_ret = df['Close'].shift(-horizon) / df['Close'] - 1
    
    if risk_adjust:
        # Annualized volatility
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        vol = log_ret.rolling(window=volatility_window).std() * np.sqrt(252)
        
        # Risk-adjusted return (like Sharpe ratio but for single period)
        target = future_ret / (vol + 1e-10)
        df[f'Target_RiskAdj_{horizon}d'] = target
    else:
        df[f'Target_Return_{horizon}d'] = future_ret
    
    return df


def create_all_labels(
    df: pd.DataFrame,
    return_horizons: list[int] = [1, 5, 21],
    tb_horizon: int = 21,
    tb_profit: float = 0.05,
    tb_stop: float = 0.03
) -> pd.DataFrame:
    """
    Create all label types in one call.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns
    return_horizons : list[int]
        Horizons for future return calculation
    tb_horizon : int
        Horizon for triple-barrier method
    tb_profit : float
        Profit target for triple-barrier
    tb_stop : float
        Stop loss for triple-barrier
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all labels
    """
    df = df.copy()
    
    # Future returns
    df = compute_future_returns(df, horizons=return_horizons)
    
    # Directional labels
    for h in return_horizons:
        df = compute_direction_labels(df, horizon=h, threshold=0.0, include_neutral=False)
    
    # Triple-barrier labels
    df = triple_barrier_labeling(
        df,
        horizon=tb_horizon,
        profit_target=tb_profit,
        stop_loss=tb_stop
    )
    
    # Risk-adjusted targets
    for h in return_horizons:
        df = create_regression_target(df, horizon=h, risk_adjust=True)
    
    return df
