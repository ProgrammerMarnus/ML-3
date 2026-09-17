"""
Models module - ML prediction models.

Implements:
1. LightGBM model (recommended baseline)
2. Linear regression baseline
3. Model wrapper with proper cross-validation handling
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings


class LightGBMWrapper(BaseEstimator, RegressorMixin):
    """
    LightGBM regressor wrapper with sensible defaults for financial data.
    
    Parameters tuned for:
    - Robustness to noise
    - Prevention of overfitting
    - Handling of feature correlations
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
        
        self.model_ = None
        self.best_iteration_ = None
        self.feature_importances_ = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit the LightGBM model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        eval_set : tuple, optional
            Validation set (X_val, y_val) for early stopping
        sample_weight : np.ndarray, optional
            Sample weights
        """
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.colsample_bytree,
            'bagging_fraction': self.subsample,
            'bagging_freq': 1,
            'min_child_samples': self.min_child_samples,
            'lambda_l1': self.reg_alpha,
            'lambda_l2': self.reg_lambda,
            'verbose': -1 if not self.verbose else 0,
            'seed': self.random_state,
            'n_jobs': -1,
        }
        
        train_data = lgb.Dataset(X, label=y)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if eval_set is not None:
            X_val, y_val = eval_set
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        self.model_ = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
            ],
        )
        
        self.best_iteration_ = self.model_.best_iteration
        self.feature_importances_ = self.model_.feature_importance(importance_type='gain')
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.predict(X, num_iteration=self.best_iteration_)
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Parameters
        ----------
        feature_names : List[str]
            Names of features
            
        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class RidgeBaseline(BaseEstimator, RegressorMixin):
    """
    Ridge regression baseline with automatic scaling.
    
    Simple but effective baseline that often performs well
    on noisy financial data.
    """
    
    def __init__(self, alpha: float = 1.0, normalize: bool = True):
        self.alpha = alpha
        self.normalize = normalize
        
        self.scaler_ = None
        self.model_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit the model with optional scaling."""
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
        
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X_scaled, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict(X)
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get coefficients as feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted yet.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(self.model_.coef_),
            'coefficient': self.model_.coef_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LogisticBaseline(BaseEstimator):
    """
    Logistic regression for classification tasks.
    
    Used for directional prediction (up/down).
    """
    
    def __init__(
        self, 
        C: float = 1.0, 
        normalize: bool = True,
        class_weight: str = 'balanced'
    ):
        self.C = C
        self.normalize = normalize
        self.class_weight = class_weight
        
        self.scaler_ = None
        self.model_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit the logistic regression model."""
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
        
        self.model_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        self.model_.fit(X_scaled, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict_proba(X)


def get_model(model_type: str = 'lightgbm', **kwargs):
    """
    Factory function to create models.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'lightgbm', 'ridge', 'logistic'
    **kwargs
        Additional arguments passed to model constructor
    
    Returns
    -------
    BaseEstimator
        Initialized model
    """
    if model_type == 'lightgbm':
        return LightGBMWrapper(**kwargs)
    elif model_type == 'ridge':
        return RidgeBaseline(**kwargs)
    elif model_type == 'logistic':
        return LogisticBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
