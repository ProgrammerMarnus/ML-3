"""
GNN Model for Cross-Asset Correlation Learning.
Implements a Graph Convolutional Network (GCN) for asset relationship modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class AssetGNN(nn.Module):
    """
    Graph Neural Network for learning cross-asset relationships.
    
    Uses Graph Convolutional layers to propagate information between 
    correlated assets in a dynamic correlation graph.
    """
    
    def __init__(
        self,
        num_assets: int,
        input_features: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_assets = num_assets
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_features, hidden_dim)
        
        # Graph convolution layers (simplified - using MLP + aggregation)
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gcn_layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [batch_size, num_assets, input_features]
            adj_matrix: Adjacency matrix [num_assets, num_assets]
            
        Returns:
            Predictions [batch_size, num_assets, output_dim]
        """
        # Project input
        h = self.input_proj(x)
        h = self.dropout(h)
        
        # Graph convolutions with message passing
        for gcn_layer in self.gcn_layers:
            # Message passing: aggregate neighbor features
            # h_new = adj @ h (simplified GCN without normalization)
            h_agg = torch.matmul(adj_matrix, h)
            h = h_agg + h  # Residual connection
            h = gcn_layer(h)
        
        # Output
        out = self.output_layer(h)
        return out
    
    def predict(self, x: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
        """Make predictions with numpy arrays."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            adj_tensor = torch.FloatTensor(adj_matrix)
            pred = self.forward(x_tensor, adj_tensor)
            return pred.numpy()


class CorrelationGraphBuilder:
    """
    Builds dynamic correlation graphs from asset returns.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        threshold: float = 0.5,
        method: str = "pearson",
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.method = method
        
    def build_adjacency_matrix(
        self,
        returns_df: 'pd.DataFrame',
    ) -> np.ndarray:
        """
        Build adjacency matrix from rolling correlations.
        
        Args:
            returns_df: DataFrame of asset returns [time, assets]
            
        Returns:
            Binary adjacency matrix [num_assets, num_assets]
        """
        import pandas as pd
        
        # Calculate rolling correlation
        rolling_corr = returns_df.tail(self.window_size).corr(method=self.method)
        
        # Threshold to create binary adjacency
        adj_matrix = (np.abs(rolling_corr.values) > self.threshold).astype(float)
        
        # Ensure diagonal is 1 (self-connection)
        np.fill_diagonal(adj_matrix, 1.0)
        
        return adj_matrix
    
    def build_weighted_adjacency(
        self,
        returns_df: 'pd.DataFrame',
    ) -> np.ndarray:
        """Build weighted adjacency matrix using correlation values."""
        import pandas as pd
        
        rolling_corr = returns_df.tail(self.window_size).corr(method=self.method)
        return rolling_corr.values
