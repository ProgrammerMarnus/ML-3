"""
Custom Gymnasium Environment for Trading Simulation.
Supports discrete and continuous action spaces for RL agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning.
    
    State: [Account Balance, Position Size, Market Features, Regime Indicator]
    Action: Discrete (Buy/Sell/Hold) or Continuous (Position Size 0-100%)
    Reward: Risk-adjusted return (Sharpe, Sortino, or Drawdown-penalized PnL)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position_size: float = 1.0,
        reward_type: str = "sharpe",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.reward_type = reward_type
        self.render_mode = render_mode
        
        # Feature columns (exclude date and target if present)
        feature_cols = [c for c in df.columns if c not in ["Date", "date", "target", "Returns"]]
        self.n_features = len(feature_cols)
        
        # Action space: Continuous [0, 1] for position sizing
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [balance_norm, position, features..., regime]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + self.n_features + 1,), dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.shares = 0
        self.trade_history = []
        self.portfolio_values = []
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        row = self.df.iloc[self.current_step]
        features = row[[c for c in self.df.columns if c not in ["Date", "date", "target", "Returns"]]].values.astype(np.float32)
        
        # Normalize balance and position
        balance_norm = self.balance / self.initial_balance
        position_norm = self.position
        
        # Regime indicator (default 0 if not present)
        regime = row.get("regime", 0) if isinstance(row, dict) else 0
        
        obs = np.concatenate([[balance_norm, position_norm], features, [regime]])
        return obs.astype(np.float32)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.shares = 0
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        row = self.df.iloc[self.current_step]
        price = row["Close"]
        
        # Parse action (position size 0-1)
        target_position = np.clip(action[0], 0, self.max_position_size)
        
        # Calculate trade
        current_value = self.balance + self.shares * price
        target_shares = int((target_position * current_value) / price) if price > 0 else 0
        trade_shares = target_shares - self.shares
        
        # Execute trade with costs
        cost = abs(trade_shares * price) * (self.commission_rate + self.slippage_rate)
        if trade_shares > 0:  # Buy
            required_cash = trade_shares * price + cost
            if required_cash <= self.balance:
                self.shares = target_shares
                self.balance -= required_cash
        elif trade_shares < 0:  # Sell
            self.shares = target_shares
            self.balance += abs(trade_shares * price) - cost
        
        # Update position (-1 to 1)
        portfolio_value = self.balance + self.shares * price
        self.position = (self.shares * price) / portfolio_value if portfolio_value > 0 else 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate reward
        self.portfolio_values.append(portfolio_value)
        reward = self._calculate_reward()
        
        # Store trade info
        self.trade_history.append({
            "step": self.current_step,
            "price": price,
            "shares": self.shares,
            "balance": self.balance,
            "action": float(action[0]),
        })
        
        info = {
            "portfolio_value": portfolio_value,
            "return": (portfolio_value - self.initial_balance) / self.initial_balance,
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on configured metric."""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
        
        if self.reward_type == "sharpe":
            if len(returns) < 2 or np.std(returns) == 0:
                return np.mean(returns)
            return np.mean(returns) / (np.std(returns) + 1e-8)
        
        elif self.reward_type == "sortino":
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                return np.mean(returns)
            return np.mean(returns) / (np.std(downside_returns) + 1e-8)
        
        elif self.reward_type == "drawdown":
            peak = np.maximum.accumulate(self.portfolio_values)
            drawdown = (peak - np.array(self.portfolio_values)) / peak
            penalty = np.max(drawdown) if len(drawdown) > 0 else 0
            return np.mean(returns) - penalty
        
        else:  # Simple return
            return np.mean(returns)
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Shares: {self.shares}, Value: ${self.portfolio_values[-1]:.2f}")
