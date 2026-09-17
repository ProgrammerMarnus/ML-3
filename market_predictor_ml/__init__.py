"""
Market Predictor ML - A system for predicting risk-adjusted returns.

This package implements a 5-layer architecture:
1. Data Layer - Data loading and preprocessing
2. Prediction Layer - Feature engineering and ML models
3. Decision Layer - Position sizing based on predictions
4. Economic Reward Layer - Reward functions with costs and penalties
5. Backtest Layer - Walk-forward validation and performance metrics
"""

__version__ = "0.1.0"

from .pipeline import MarketPredictorPipeline
from .config import Config, DEFAULT_CONFIG

__all__ = [
    "MarketPredictorPipeline",
    "Config",
    "DEFAULT_CONFIG",
]
