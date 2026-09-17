"""
GNN Utilities for graph construction and feature extraction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


def build_correlation_graph(
    returns_df: pd.DataFrame,
    window_size: int = 60,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Build a correlation-based adjacency matrix from asset returns.
    
    Args:
        returns_df: DataFrame with assets as columns and dates as index
        window_size: Rolling window for correlation calculation
        threshold: Correlation threshold for edge creation
        
    Returns:
        Binary adjacency matrix
    """
    # Calculate rolling correlation
    rolling_corr = returns_df.tail(window_size).corr()
    
    # Threshold to create edges
    adj_matrix = (np.abs(rolling_corr.values) > threshold).astype(float)
    
    # Self-connections
    np.fill_diagonal(adj_matrix, 1.0)
    
    return adj_matrix


def get_graph_features(
    price_data: Dict[str, pd.DataFrame],
    feature_cols: List[str],
) -> tuple:
    """
    Prepare node features and adjacency matrix for GNN.
    
    Args:
        price_data: Dictionary of ticker -> DataFrame with OHLCV data
        feature_cols: List of feature columns to use
        
    Returns:
        Tuple of (feature_tensor, adjacency_matrix, tickers)
    """
    import torch
    
    tickers = list(price_data.keys())
    n_assets = len(tickers)
    
    # Extract features for each asset
    features_list = []
    returns_list = []
    
    for ticker in tickers:
        df = price_data[ticker]
        if len(df) == 0:
            continue
            
        # Get features
        feat = df[feature_cols].values
        features_list.append(feat[-1])  # Use latest features
        
        # Get returns for correlation
        if "Returns" in df.columns:
            returns_list.append(df["Returns"].values)
    
    if len(features_list) == 0:
        raise ValueError("No valid features found")
    
    # Stack features [num_assets, feature_dim]
    X = np.array(features_list)
    
    # Build returns DataFrame for correlation
    if len(returns_list) > 0:
        min_len = min(len(r) for r in returns_list)
        returns_array = np.array([r[-min_len:] for r in returns_list])
        returns_df = pd.DataFrame(returns_array.T, columns=tickers)
        
        # Build adjacency matrix
        adj = build_correlation_graph(returns_df)
    else:
        # Default: fully connected
        adj = np.ones((n_assets, n_assets))
    
    return X, adj, tickers
