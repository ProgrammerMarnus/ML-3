"""
Position sizing strategies with multiple implementations.

Implements:
- Volatility-adjusted position sizing
- Kelly criterion
- Fixed fraction
- Signal strength based
- Risk parity approaches

NOTE (M-9): this class-based IPositionSizer hierarchy overlaps with the
functional API in decision/sizing.py. decision/sizing.py is the implementation
used by optimization/hyperopt.py; prefer create_position_sizer() here for
interface-driven code so the two implementations do not drift apart.
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..core import IPositionSizer


class VolatilityAdjustedPositionSizer(IPositionSizer):
    """
    Position sizing based on target volatility.
    
    Adjusts position size inversely proportional to volatility
    to maintain consistent risk exposure.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.02,
        max_position: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        min_position: float = 0.0,
        lookback_period: int = 21,
        max_total_exposure: Optional[float] = None,  # Ignored but accepted for config compatibility
        **kwargs  # Accept any extra kwargs for flexibility
    ):
        self.target_volatility = target_volatility
        # Support both max_position (old) and max_position_pct (config file)
        self.max_position = max_position_pct if max_position_pct is not None else (max_position if max_position is not None else 1.0)
        self.min_position = min_position
        self.lookback_period = lookback_period
        self.max_total_exposure = max_total_exposure if max_total_exposure is not None else 1.0
    
    def calculate_position(
        self,
        signal: float,
        volatility: float,
        capital: float,
        **kwargs
    ) -> float:
        """
        Calculate position size based on signal and volatility.
        
        Parameters
        ----------
        signal : float
            Prediction signal (typically -1 to 1 or raw prediction)
        volatility : float
            Current volatility estimate
        capital : float
            Available capital
            
        Returns
        -------
        float
            Position size as fraction of capital
        """
        if volatility <= 0:
            return 0.0
        
        # Base position from volatility targeting
        base_position = self.target_volatility / volatility
        
        # Apply signal direction and strength
        position = base_position * np.clip(signal, -1, 1)
        
        # Apply symmetric limits (honor the -1..1 signal contract; long-only is enforced at the strategy layer)
        position = np.clip(position, -self.max_position, self.max_position)
        
        return position
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'target_volatility': self.target_volatility,
            'max_position': self.max_position,
            'min_position': self.min_position,
            'lookback_period': self.lookback_period
        }


class KellyPositionSizer(IPositionSizer):
    """
    Kelly criterion position sizing.
    
    Calculates optimal bet size based on win probability and payoff ratio.
    Uses fractional Kelly to reduce risk.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position: float = 1.0,
        min_win_rate: float = 0.4,
        rolling_window: int = 252
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_win_rate = min_win_rate
        self.rolling_window = rolling_window
        self._trade_history: list = []
    
    def update_history(self, returns: np.ndarray):
        """Update trade history for win rate calculation."""
        self._trade_history.extend(returns.tolist())
        if len(self._trade_history) > self.rolling_window:
            self._trade_history = self._trade_history[-self.rolling_window:]
    
    def calculate_position(
        self,
        signal: float,
        volatility: float,
        capital: float,
        **kwargs
    ) -> float:
        """
        Calculate Kelly-optimal position size.
        
        Parameters
        ----------
        signal : float
            Expected return prediction
        volatility : float
            Volatility estimate
        capital : float
            Available capital
            
        Returns
        -------
        float
            Position size as fraction of capital
        """
        # Prefer historical win rate / payoff ratio when history is available
        # (L-11: update_history() populated _trade_history but it was unused);
        # fall back to the signal-derived estimate otherwise.
        history = np.asarray(self._trade_history, dtype=float)
        history = history[np.isfinite(history)]
        if history.size >= 10:
            wins = history[history > 0]
            losses = history[history < 0]
            if wins.size and losses.size:
                win_prob = float(wins.size / history.size)
                payoff_ratio = float(np.mean(wins) / abs(np.mean(losses)))
            else:
                win_prob = float(history.mean() > 0)
                payoff_ratio = 1.0
        else:
            # Estimate win probability from signal (higher = higher win prob)
            win_prob = 0.5 + signal * 0.25  # Maps signal to ~0.25-0.75 range
            # Assume symmetric payoff without history
            payoff_ratio = 1.0

        win_prob = float(np.clip(win_prob, self.min_win_rate, 1 - self.min_win_rate))
        
        # Kelly formula: f* = (p * b - q) / b
        # where p = win prob, q = loss prob, b = payoff ratio
        q = 1 - win_prob
        kelly_size = (win_prob * payoff_ratio - q) / payoff_ratio
        
        # Apply fractional Kelly
        kelly_size *= self.kelly_fraction
        
        # Apply signal direction
        position = kelly_size * np.sign(signal)
        
        # Apply limits
        position = np.clip(position, -self.max_position, self.max_position)
        
        return position
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'kelly_fraction': self.kelly_fraction,
            'max_position': self.max_position,
            'min_win_rate': self.min_win_rate,
            'rolling_window': self.rolling_window
        }


class FixedFractionPositionSizer(IPositionSizer):
    """
    Fixed fraction position sizing.
    
    Simple approach: always invest a fixed fraction of capital.
    """
    
    def __init__(
        self,
        fixed_fraction: float = 0.1,
        max_position: float = 1.0
    ):
        self.fixed_fraction = fixed_fraction
        self.max_position = max_position
    
    def calculate_position(
        self,
        signal: float,
        volatility: float,
        capital: float,
        **kwargs
    ) -> float:
        """Calculate fixed fraction position."""
        # Only take positions when signal is positive (or above threshold)
        threshold = kwargs.get('threshold', 0.0)
        
        if signal > threshold:
            position = self.fixed_fraction
        elif signal < -threshold:
            position = -self.fixed_fraction
        else:
            position = 0.0
        
        return np.clip(position, -self.max_position, self.max_position)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'fixed_fraction': self.fixed_fraction,
            'max_position': self.max_position
        }


class SignalStrengthPositionSizer(IPositionSizer):
    """
    Position sizing based on signal strength.
    
    Scales position linearly or non-linearly with prediction confidence.
    """
    
    def __init__(
        self,
        max_position: float = 1.0,
        power: float = 1.0,
        threshold: float = 0.0
    ):
        self.max_position = max_position
        self.power = power  # >1 for convex, <1 for concave
        self.threshold = threshold
    
    def calculate_position(
        self,
        signal: float,
        volatility: float,
        capital: float,
        **kwargs
    ) -> float:
        """
        Calculate position based on signal strength.
        
        Parameters
        ----------
        signal : float
            Prediction signal
        volatility : float
            Volatility (unused in this method)
        capital : float
            Available capital (unused in this method)
            
        Returns
        -------
        float
            Position size
        """
        # Apply threshold
        if abs(signal) < self.threshold:
            return 0.0
        
        # Scale by signal strength raised to power
        sign = np.sign(signal)
        strength = abs(signal) ** self.power
        
        # Normalize to max position
        position = sign * strength * self.max_position
        
        return np.clip(position, -self.max_position, self.max_position)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'max_position': self.max_position,
            'power': self.power,
            'threshold': self.threshold
        }


class RiskParityPositionSizer(IPositionSizer):
    """
    Risk parity approach to position sizing.
    
    Allocates capital such that each position contributes equally to portfolio risk.
    """
    
    def __init__(
        self,
        target_risk: float = 0.02,
        max_position: float = 1.0,
        correlation_estimate: float = 0.3
    ):
        self.target_risk = target_risk
        self.max_position = max_position
        self.correlation_estimate = correlation_estimate
    
    def calculate_position(
        self,
        signal: float,
        volatility: float,
        capital: float,
        n_assets: int = 1,
        **kwargs
    ) -> float:
        """
        Calculate risk-parity position size.
        
        Parameters
        ----------
        signal : float
            Prediction signal
        volatility : float
            Asset volatility
        capital : float
            Available capital
        n_assets : int
            Number of assets in portfolio
            
        Returns
        -------
        float
            Position size
        """
        if volatility <= 0:
            return 0.0
        
        # Risk parity weight inversely proportional to volatility
        # Adjusted for number of assets and correlation
        if n_assets == 1:
            risk_weight = self.target_risk / volatility
        else:
            # Simplified multi-asset adjustment
            avg_correlation = self.correlation_estimate
            diversification_factor = np.sqrt(1 + (n_assets - 1) * avg_correlation)
            risk_weight = (self.target_risk / volatility) / diversification_factor
        
        # Apply signal direction
        position = risk_weight * np.clip(signal, -1, 1)
        
        return np.clip(position, -self.max_position, self.max_position)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'target_risk': self.target_risk,
            'max_position': self.max_position,
            'correlation_estimate': self.correlation_estimate
        }


# Factory function
def create_position_sizer(
    method: str = "volatility_adjusted",
    **kwargs
) -> IPositionSizer:
    """
    Create a position sizer with the specified method.
    
    Parameters
    ----------
    method : str
        Sizing method: 'volatility_adjusted', 'kelly', 'fixed', 
                       'signal_strength', 'risk_parity'
    **kwargs
        Additional arguments passed to sizer constructor
    
    Returns
    -------
    IPositionSizer
        Configured position sizer instance
    """
    methods = {
        'volatility_adjusted': VolatilityAdjustedPositionSizer,
        'kelly': KellyPositionSizer,
        'fixed': FixedFractionPositionSizer,
        'signal_strength': SignalStrengthPositionSizer,
        'risk_parity': RiskParityPositionSizer,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown position sizing method: {method}")
    
    return methods[method](**kwargs)
