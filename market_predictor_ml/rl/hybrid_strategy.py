"""
Hybrid Strategy combining Ensemble signals with RL-based position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class HybridRLStrategy:
    """
    Combines directional signals from ensemble models with RL-based position sizing.
    
    - Direction (Long/Short/Hold): From ensemble of LightGBM/LSTM/Transformer
    - Position Size (0-100%): From trained RL agent (PPO)
    """
    
    def __init__(
        self,
        rl_agent: Any,
        max_position_pct: float = 0.5,
        risk_adjustment: bool = True,
    ):
        self.rl_agent = rl_agent
        self.max_position_pct = max_position_pct
        self.risk_adjustment = risk_adjustment
        
    def get_position_size(
        self,
        observation: np.ndarray,
        signal_direction: int,  # -1, 0, 1
        volatility: float,
        regime: int = 0,
    ) -> float:
        """
        Determine position size based on RL agent output and signal direction.
        
        Args:
            observation: Current state vector for RL agent
            signal_direction: -1 (short), 0 (hold), 1 (long)
            volatility: Current market volatility
            regime: Market regime indicator
            
        Returns:
            Position size as fraction of portfolio (0 to max_position_pct)
        """
        if signal_direction == 0:
            return 0.0
        
        # Get raw position size from RL agent
        action, _ = self.rl_agent.predict(observation, deterministic=True)
        raw_size = float(np.clip(action[0], 0, 1))
        
        # Apply direction
        position = raw_size * signal_direction
        
        # Risk adjustment based on volatility
        if self.risk_adjustment and volatility > 0:
            vol_scale = np.clip(0.02 / (volatility + 1e-8), 0.5, 1.5)
            position *= vol_scale
        
        # Regime adjustment (reduce in recession)
        if regime == 2:  # Recession regime
            position *= 0.5
        
        # Clip to max
        position = np.clip(position, -self.max_position_pct, self.max_position_pct)
        
        return position
    
    def generate_trades(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        initial_observation: np.ndarray,
    ) -> pd.DataFrame:
        """Generate full trade sequence using hybrid strategy."""
        trades = []
        current_obs = initial_observation
        
        for i in range(len(df)):
            row = df.iloc[i]
            signal = signals[i] if i < len(signals) else 0
            
            # Extract features for observation update
            volatility = row.get("volatility_21d", 0.02)
            regime = row.get("regime", 0)
            
            # Get position size
            position = self.get_position_size(current_obs, signal, volatility, regime)
            
            trades.append({
                "date": row.get("Date", row.name),
                "signal": signal,
                "position": position,
                "price": row["Close"],
                "volatility": volatility,
                "regime": regime,
            })
            
            # Update observation (simplified - in real env this happens in step())
            # Here we just shift the balance/portfolio value part
            if i < len(df) - 1:
                current_obs = current_obs.copy()
                # In production, this would be updated from the env step
        
        return pd.DataFrame(trades)
