"""Live trading module for Market Predictor ML.

Provides Alpaca trading clients (``AlpacaClient`` for paper/live via
the PR, ``AlpacaPaperClient`` for paper-only dry-run), optional RL
policy refinement, optional GNN feature augmentation, plus the
LiveTrader / PaperTrader orchestrators.
"""

from .alpaca_client import AlpacaClient
from .paper_client import AlpacaPaperClient, is_alpaca_available
from .rl_policy import RLPolicy, is_rl_available
from .gnn_features import GNNFeatureAugmenter, is_gnn_available
from .trader import LiveTrader, PaperTrader

__all__ = [
    "AlpacaClient",
    "AlpacaPaperClient",
    "is_alpaca_available",
    "RLPolicy",
    "is_rl_available",
    "GNNFeatureAugmenter",
    "is_gnn_available",
    "LiveTrader",
    "PaperTrader",
]
