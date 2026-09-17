"""Utility functions for data leakage prevention and preprocessing."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def check_data_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    correlation_threshold: float = 0.95
) -> List[str]:
    """
    Check for potential data leakage between train and test sets.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features
    feature_names : List[str]
        Names of features
    correlation_threshold : float
        Threshold for flagging high correlation
    
    Returns
    -------
    List[str]
        List of features with potential leakage
    """
    leaky_features = []
    
    for i, name in enumerate(feature_names):
        train_col = X_train[:, i]
        test_col = X_test[:, i]
        
        # Check for identical values (direct leakage)
        if np.array_equal(train_col[:len(test_col)], test_col):
            leaky_features.append(f"{name} (identical)")
            continue
        
        # Check for very high correlation
        if len(train_col) > 1 and len(test_col) > 1:
            corr = np.corrcoef(train_col[:min(len(train_col), len(test_col))], 
                              test_col[:min(len(train_col), len(test_col))])[0, 1]
            if abs(corr) > correlation_threshold and not np.isnan(corr):
                leaky_features.append(f"{name} (corr={corr:.3f})")
    
    return leaky_features


def remove_near_zero_variance_features(
    X: np.ndarray,
    threshold: float = 1e-4,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Remove features with near-zero variance.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    threshold : float
        Variance threshold below which to remove features
    feature_names : List[str], optional
        Names of features
    
    Returns
    -------
    Tuple[np.ndarray, List[int]]
        Filtered feature matrix and indices of kept features
    """
    variances = np.var(X, axis=0)
    keep_mask = variances > threshold
    keep_indices = np.where(keep_mask)[0]
    
    return X[:, keep_mask], keep_indices.tolist()


def winsorize_features(
    X: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    Winsorize features to limit outlier impact.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (can be numpy array or pandas DataFrame)
    lower_percentile : float
        Lower percentile for winsorization
    upper_percentile : float
        Upper percentile for winsorization
    
    Returns
    -------
    np.ndarray
        Winsorized feature matrix
    """
    # Convert DataFrame to numpy array if needed
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X
    
    X_winsorized = X_array.copy()
    
    for i in range(X_winsorized.shape[1]):
        lower = np.percentile(X_winsorized[:, i], lower_percentile)
        upper = np.percentile(X_winsorized[:, i], upper_percentile)
        X_winsorized[:, i] = np.clip(X_winsorized[:, i], lower, upper)
    
    return X_winsorized


def standardize_features(
    X: np.ndarray,
    fit_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using only training data statistics.
    
    IMPORTANT: This prevents look-ahead bias by computing mean/std
    only from the training set.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    fit_indices : np.ndarray, optional
        Indices of samples to use for computing mean/std
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Standardized features, means, and stds
    """
    if fit_indices is not None:
        mean = np.mean(X[fit_indices], axis=0)
        std = np.std(X[fit_indices], axis=0)
    else:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    X_standardized = (X - mean) / std
    
    return X_standardized, mean, std


def validate_feature_matrix(
    X: np.ndarray,
    feature_names: List[str]
) -> dict:
    """
    Comprehensive validation of feature matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of features
    
    Returns
    -------
    dict
        Validation report
    """
    report = {
        'shape': X.shape,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'has_nan': np.any(np.isnan(X)),
        'has_inf': np.any(np.isinf(X)),
        'nan_count_per_feature': {},
        'variance_per_feature': {},
        'issues': []
    }
    
    # Check each feature
    for i, name in enumerate(feature_names):
        col = X[:, i]
        
        # NaN count
        nan_count = np.sum(np.isnan(col))
        if nan_count > 0:
            report['nan_count_per_feature'][name] = nan_count
            report['issues'].append(f"{name}: {nan_count} NaN values")
        
        # Variance
        var = np.var(col)
        report['variance_per_feature'][name] = var
        
        if var < 1e-10:
            report['issues'].append(f"{name}: Near-zero variance ({var:.2e})")
        
        # Check for inf
        if np.any(np.isinf(col)):
            report['issues'].append(f"{name}: Contains infinite values")
    
    # Overall checks
    if report['has_nan']:
        report['issues'].append("Matrix contains NaN values")
    
    if report['has_inf']:
        report['issues'].append("Matrix contains infinite values")
    
    # Check for duplicate features
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if np.array_equal(X[:, i], X[:, j]):
                report['issues'].append(
                    f"Duplicate features: {feature_names[i]} and {feature_names[j]}"
                )
    
    return report
