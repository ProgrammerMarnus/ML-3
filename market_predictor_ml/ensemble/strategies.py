"""
Ensemble Strategies for combining multiple model predictions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression, Ridge


class VotingEnsemble:
    """
    Combines predictions from multiple models using voting.
    
    Supports:
    - Hard voting (majority class)
    - Soft voting (probability-weighted)
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        voting_type: str = "soft",
    ):
        self.models = models
        self.voting_type = voting_type
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions[name] = pred
        
        if self.voting_type == "hard":
            # Majority vote
            pred_array = np.array(list(predictions.values()))
            return np.sign(np.median(pred_array, axis=0))
        
        else:  # soft voting
            # Weighted average (assuming predictions are signed confidence)
            pred_array = np.array(list(predictions.values()))
            return np.mean(pred_array, axis=0)


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    
    Uses base model predictions as features for a meta-model.
    """
    
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Optional[Any] = None,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit base models and meta-learner."""
        # Fit base models
        base_preds = []
        for name, model in self.base_models.items():
            model.fit(X, y)
            pred = model.predict(X).reshape(-1, 1)
            base_preds.append(pred)
        
        # Stack predictions
        X_meta = np.hstack(base_preds)
        
        # Fit meta-model
        self.meta_model.fit(X_meta, y)
        self.fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacked predictions."""
        if not self.fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get base predictions
        base_preds = []
        for name, model in self.base_models.items():
            pred = model.predict(X).reshape(-1, 1)
            base_preds.append(pred)
        
        X_meta = np.hstack(base_preds)
        return self.meta_model.predict(X_meta)


class DynamicWeightEnsemble:
    """
    Ensemble with dynamic weights based on recent performance.
    
    Adjusts model weights based on rolling Sharpe ratio or accuracy.
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        window_size: int = 60,
        performance_metric: str = "sharpe",
    ):
        self.models = models
        self.window_size = window_size
        self.performance_metric = performance_metric
        self.model_weights = {name: 1.0 / len(models) for name in models}
        self.model_performance = {name: [] for name in models}
        
    def update_weights(self, predictions: Dict[str, np.ndarray], actual: np.ndarray):
        """Update model weights based on recent performance."""
        for name, pred in predictions.items():
            # Calculate performance metric
            if self.performance_metric == "accuracy":
                perf = np.mean(np.sign(pred) == np.sign(actual))
            elif self.performance_metric == "sharpe":
                returns = pred * np.sign(actual)  # Simplified
                if len(returns) > 1 and np.std(returns) > 0:
                    perf = np.mean(returns) / np.std(returns)
                else:
                    perf = 0
            else:
                perf = np.corrcoef(pred, actual)[0, 1]
            
            self.model_performance[name].append(perf)
            
            # Keep only recent window
            if len(self.model_performance[name]) > self.window_size:
                self.model_performance[name] = self.model_performance[name][-self.window_size:]
        
        # Calculate new weights based on average recent performance
        perfs = np.array([np.mean(self.model_performance[name]) for name in self.models])
        perfs = np.maximum(perfs, 0)  # Negative performance gets zero weight
        
        if np.sum(perfs) > 0:
            self.model_weights = {
                name: perf / np.sum(perfs)
                for name, perf in zip(self.models.keys(), perfs)
            }
        else:
            # Equal weights if all negative
            self.model_weights = {name: 1.0 / len(self.models) for name in self.models}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted ensemble prediction."""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.model_weights[name])
        
        predictions_array = np.array(predictions)
        weights_array = np.array(weights)
        
        return np.average(predictions_array, axis=0, weights=weights_array)
