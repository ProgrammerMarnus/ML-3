"""
Graph Neural Network Module for cross-asset correlation modeling.
Uses PyTorch Geometric for graph-based learning.
"""

from market_predictor_ml.gnn.model import AssetGNN, CorrelationGraphBuilder
from market_predictor_ml.gnn.utils import build_correlation_graph, get_graph_features

__all__ = [
    "AssetGNN",
    "CorrelationGraphBuilder",
    "build_correlation_graph",
    "get_graph_features",
]
