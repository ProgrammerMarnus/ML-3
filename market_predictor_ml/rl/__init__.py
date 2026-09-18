"""
Reinforcement Learning Module for Market Predictor ML.
Implements PPO and DQN agents for dynamic position sizing and execution optimization.

Optional dependencies: ``gymnasium`` (environment) and ``stable-baselines3``
(agents). The submodules import cleanly without them; a missing package is
reported only when the corresponding class/agent is actually instantiated.
"""

from market_predictor_ml.rl.environment import TradingEnv
from market_predictor_ml.rl.agents import create_ppo_agent, create_dqn_agent
from market_predictor_ml.rl.hybrid_strategy import HybridRLStrategy

__all__ = [
    "TradingEnv",
    "create_ppo_agent",
    "create_dqn_agent",
    "HybridRLStrategy",
]
