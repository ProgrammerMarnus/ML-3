"""
Decision layer - Position sizing based on predictions.

Converts model predictions into trading positions using:
1. Fixed fractional sizing
2. Volatility-adjusted sizing
3. Kelly criterion (fractional)
4. Risk parity approaches
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def fixed_fractional_position(
    predictions: np.ndarray,
    max_position: float = 1.0,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Simple fixed fractional position sizing.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (expected returns)
    max_position : float
        Maximum absolute position size (e.g., 1.0 = 100%)
    threshold : float
        Minimum prediction magnitude to take a position
    
    Returns
    -------
    np.ndarray
        Position sizes (-max_position to +max_position)
    """
    positions = np.zeros_like(predictions)
    
    # Apply threshold
    mask = np.abs(predictions) > threshold
    
    # Scale predictions to position size
    positions[mask] = np.sign(predictions[mask]) * max_position
    
    return positions


def volatility_adjusted_position(
    predictions: np.ndarray,
    volatility: np.ndarray,
    target_vol: float = 0.02,
    max_position: float = 1.0,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Volatility-adjusted position sizing.
    
    Positions are scaled inversely to volatility to maintain
    roughly constant risk contribution.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (expected returns)
    volatility : np.ndarray
        Estimated volatility (e.g., annualized)
    target_vol : float
        Target daily volatility of position
    max_position : float
        Maximum absolute position size
    threshold : float
        Minimum prediction magnitude to take a position
    
    Returns
    -------
    np.ndarray
        Position sizes
    """
    positions = np.zeros_like(predictions)
    
    # Avoid division by zero
    vol_safe = np.maximum(volatility, 1e-6)
    
    # Calculate raw position size (inverse vol scaling)
    # Assuming predictions are in same units as vol
    raw_positions = predictions / vol_safe * target_vol
    
    # Apply threshold
    mask = np.abs(raw_positions) > threshold
    
    # Clip to max position
    positions[mask] = np.clip(raw_positions[mask], -max_position, max_position)
    
    return positions


def kelly_criterion_position(
    predictions: np.ndarray,
    win_rate: Optional[float] = None,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    fraction: float = 0.25,
    max_position: float = 1.0
) -> np.ndarray:
    """
    Fractional Kelly criterion position sizing.
    
    The Kelly criterion maximizes log wealth growth but can be
    very aggressive. Using a fraction (e.g., 0.25 = quarter-Kelly)
    provides more conservative sizing.
    
    Parameters
    ----------
    predictions : np.ndarray
        Expected returns
    win_rate : float, optional
        Historical win rate (if None, estimated from predictions)
    avg_win : float, optional
        Average winning return
    avg_loss : float, optional
        Average losing return (as positive number)
    fraction : float
        Fraction of full Kelly (0.25 = quarter-Kelly)
    max_position : float
        Maximum absolute position size
    
    Returns
    -------
    np.ndarray
        Position sizes
    """
    # Simplified Kelly for continuous returns: f* = E[r] / Var(r)
    # We use a simplified version based on prediction magnitude
    
    # Estimate variance from prediction spread if not provided
    pred_var = np.var(predictions) + 1e-10
    
    # Kelly fraction for each prediction
    kelly_f = predictions / pred_var
    
    # Apply fractional Kelly
    positions = fraction * kelly_f
    
    # Clip to max position
    positions = np.clip(positions, -max_position, max_position)
    
    return positions


def signal_strength_position(
    predictions: np.ndarray,
    lookback_window: int = 252,
    max_position: float = 1.0,
    power: float = 1.0
) -> np.ndarray:
    """
    Position sizing based on signal strength relative to history.
    
    Stronger signals (relative to historical distribution) get
    larger positions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    lookback_window : int
        Window for computing historical statistics
    max_position : float
        Maximum absolute position size
    power : float
        Power to apply to normalized signal (1=linear, 2=quadratic)
    
    Returns
    -------
    np.ndarray
        Position sizes
    """
    positions = np.zeros_like(predictions)
    
    # Compute rolling z-score of predictions
    pred_series = pd.Series(predictions)
    rolling_mean = pred_series.rolling(window=lookback_window, min_periods=1).mean()
    rolling_std = pred_series.rolling(window=lookback_window, min_periods=1).std().replace(0, 1e-10)
    
    z_scores = (pred_series - rolling_mean) / rolling_std
    
    # Convert z-score to position using sigmoid-like function
    # This naturally bounds positions while allowing gradation
    positions = np.tanh(z_scores.values) * max_position
    
    # Apply power for more/less aggressive sizing
    if power != 1.0:
        positions = np.sign(positions) * (np.abs(positions) ** power) * max_position
    
    return positions


def create_positions(
    predictions: np.ndarray,
    volatility: Optional[np.ndarray] = None,
    method: str = 'volatility_adjusted',
    **kwargs
) -> np.ndarray:
    """
    Factory function to create positions using various methods.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (expected returns)
    volatility : np.ndarray, optional
        Volatility estimates for vol-adjusted sizing
    method : str
        Sizing method: 'fixed', 'volatility_adjusted', 'kelly', 'signal_strength'
    **kwargs
        Additional arguments passed to specific method
    
    Returns
    -------
    np.ndarray
        Position sizes
    """
    if method == 'fixed':
        return fixed_fractional_position(predictions, **kwargs)
    elif method == 'volatility_adjusted':
        if volatility is None:
            raise ValueError("Volatility required for volatility_adjusted method")
        return volatility_adjusted_position(predictions, volatility, **kwargs)
    elif method == 'kelly':
        return kelly_criterion_position(predictions, **kwargs)
    elif method == 'signal_strength':
        return signal_strength_position(predictions, **kwargs)
    else:
        raise ValueError(f"Unknown position sizing method: {method}")
