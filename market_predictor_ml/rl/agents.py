"""
RL Agents using Stable Baselines3.
Implements PPO and DQN for trading strategies.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np

# gymnasium is optional here: it is only referenced in type hints, and
# `from __future__ import annotations` keeps those unevaluated. The real
# requirement is enforced by the factory functions (via stable-baselines3).
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    DQN = None
    BaseCallback = object  # Fallback for type hints


class RewardLoggingCallback(BaseCallback):
    """Callback to log rewards during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
    
    def _on_step(self) -> bool:
        if len(self.locals["rewards"]) > 0:
            self.rewards.append(self.locals["rewards"][0])
        return True


def create_ppo_agent(
    env: gym.Env,
    policy: str = "MlpPolicy",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 0,
    tensorboard_log: Optional[str] = None,
) -> Any:
    """Create a PPO agent for continuous action spaces."""
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
    
    return PPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )


def create_dqn_agent(
    env: gym.Env,
    policy: str = "MlpPolicy",
    learning_rate: float = 1e-4,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    tau: float = 1.0,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    exploration_fraction: float = 0.1,
    exploration_final_eps: float = 0.02,
    verbose: int = 0,
    tensorboard_log: Optional[str] = None,
) -> Any:
    """Create a DQN agent for discrete action spaces."""
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
    
    # Note: For discrete actions, env.action_space should be Discrete
    return DQN(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
