"""
Market Predictor ML - Main pipeline module.

This module provides the main Pipeline class that orchestrates
all components of the system:
1. Data loading and preprocessing
2. Feature engineering
3. Label construction
4. Model training
5. Position sizing
6. Backtesting and evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import warnings

from .data import (
    download_stock_data,
    preprocess_data,
    compute_returns,
)
from .features import (
    create_all_features,
    create_all_labels,
    get_feature_columns,
)
from .models import get_model, LightGBMWrapper, RidgeBaseline
from .decision import create_positions
from .backtest import (
    WalkForwardSplit,
    run_walk_forward_backtest,
    compute_economic_metrics,
)
from .utils import (
    winsorize_features,
    remove_near_zero_variance_features,
    standardize_features,
    validate_feature_matrix,
)
from .config import Config, DEFAULT_CONFIG


class MarketPredictorPipeline:
    """
    Main pipeline for market prediction and backtesting.
    
    This class orchestrates the complete workflow from raw price data
    to economic performance metrics.
    
    Parameters
    ----------
    config : Config, optional
        Configuration object. Uses DEFAULT_CONFIG if None.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
        
        # State variables
        self.data_ = None
        self.features_ = None
        self.labels_ = None
        self.feature_names_ = []
        self.X_ = None
        self.y_ = None
        self.volatility_ = None
        self.model_ = None
        self.backtest_results_ = None
    
    def load_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> 'MarketPredictorPipeline':
        """
        Load and preprocess stock data.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_date : str, optional
            Start date (uses config default if None)
        end_date : str, optional
            End date (uses config default if None)
        interval : str
            Data interval
        
        Returns
        -------
        MarketPredictorPipeline
            Self for method chaining
        """
        start = start_date or self.config.data.default_start_date
        end = end_date or self.config.data.default_end_date
        
        print(f"Loading data for {ticker} from {start} to {end}...")
        self.data_ = download_stock_data(ticker, start, end, interval)
        self.data_ = preprocess_data(self.data_)
        
        print(f"Loaded {len(self.data_)} rows")
        return self
    
    def engineer_features(self) -> 'MarketPredictorPipeline':
        """
        Create all engineered features.
        
        Returns
        -------
        MarketPredictorPipeline
            Self for method chaining
        """
        if self.data_ is None:
            raise ValueError("Must call load_data() first")
        
        print("Engineering features...")
        self.features_ = create_all_features(self.data_)
        self.feature_names_ = get_feature_columns(self.features_)
        
        print(f"Created {len(self.feature_names_)} features")
        return self
    
    def create_labels(self) -> 'MarketPredictorPipeline':
        """
        Create all label types.
        
        Returns
        -------
        MarketPredictorPipeline
            Self for method chaining
        """
        if self.features_ is None:
            raise ValueError("Must call engineer_features() first")
        
        print("Creating labels...")
        self.labels_ = create_all_labels(
            self.features_,
            return_horizons=self.config.labels.return_horizons,
            tb_horizon=self.config.labels.triple_barrier_horizon,
            tb_profit=self.config.labels.triple_barrier_profit_target,
            tb_stop=self.config.labels.triple_barrier_stop_loss,
        )
        
        return self
    
    def prepare_data(
        self,
        target_column: str = 'Target_RiskAdj_21d',
        drop_na: bool = True
    ) -> 'MarketPredictorPipeline':
        """
        Prepare feature matrix X and target vector y.
        
        Parameters
        ----------
        target_column : str
            Column name to use as target
        drop_na : bool
            Whether to drop rows with NaN values
        
        Returns
        -------
        MarketPredictorPipeline
            Self for method chaining
        """
        if self.labels_ is None:
            raise ValueError("Must call create_labels() first")
        
        print("Preparing data matrix...")
        
        # Get feature columns
        df = self.labels_[self.feature_names_ + [target_column]].copy()
        
        # Drop NaN rows
        if drop_na:
            initial_len = len(df)
            df = df.dropna()
            print(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        # Extract volatility for position sizing
        vol_col = 'Volatility_Realized_21d'
        if vol_col in df.columns:
            self.volatility_ = df[vol_col].values
        else:
            self.volatility_ = None
        
        # Convert to arrays
        self.X_ = df[self.feature_names_].values
        self.y_ = df[target_column].values
        
        # Winsorize features
        print("Winsorizing features...")
        self.X_ = winsorize_features(
            self.X_,
            lower_percentile=self.config.features.winsorize_lower,
            upper_percentile=self.config.features.winsorize_upper,
        )
        
        # Remove near-zero variance features
        print("Removing low-variance features...")
        self.X_, keep_indices = remove_near_zero_variance_features(
            self.X_,
            threshold=self.config.features.variance_threshold,
        )
        self.feature_names_ = [self.feature_names_[i] for i in keep_indices]
        
        print(f"Final feature matrix shape: {self.X_.shape}")
        
        # Validate
        validation = validate_feature_matrix(self.X_, self.feature_names_)
        if validation['issues']:
            warnings.warn(f"Data quality issues: {validation['issues']}")
        
        return self
    
    def train_model(
        self,
        model_type: str = 'lightgbm',
        use_validation: bool = True,
        val_fraction: float = 0.2
    ) -> 'MarketPredictorPipeline':
        """
        Train the ML model.
        
        Parameters
        ----------
        model_type : str
            Type of model ('lightgbm', 'ridge')
        use_validation : bool
            Whether to use validation set for early stopping
        val_fraction : float
            Fraction of data to use for validation
        
        Returns
        -------
        MarketPredictorPipeline
            Self for method chaining
        """
        if self.X_ is None or self.y_ is None:
            raise ValueError("Must call prepare_data() first")
        
        print(f"Training {model_type} model...")
        
        # Split for validation
        n_samples = len(self.X_)
        val_size = int(n_samples * val_fraction)
        
        if use_validation and val_size > 0:
            X_train = self.X_[:-val_size]
            y_train = self.y_[:-val_size]
            X_val = self.X_[-val_size:]
            y_val = self.y_[-val_size:]
            eval_set = (X_val, y_val)
        else:
            X_train = self.X_
            y_train = self.y_
            eval_set = None
        
        # Create and train model
        if model_type == 'lightgbm':
            self.model_ = get_model(
                'lightgbm',
                n_estimators=self.config.model.lightgbm_n_estimators,
                learning_rate=self.config.model.lightgbm_learning_rate,
                max_depth=self.config.model.lightgbm_max_depth,
                num_leaves=self.config.model.lightgbm_num_leaves,
                min_child_samples=self.config.model.lightgbm_min_child_samples,
                subsample=self.config.model.lightgbm_subsample,
                colsample_bytree=self.config.model.lightgbm_colsample_bytree,
                reg_alpha=self.config.model.lightgbm_reg_alpha,
                reg_lambda=self.config.model.lightgbm_reg_lambda,
                early_stopping_rounds=self.config.model.lightgbm_early_stopping_rounds,
            )
        elif model_type == 'ridge':
            self.model_ = get_model(
                'ridge',
                alpha=self.config.model.ridge_alpha,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_.fit(X_train, y_train, eval_set=eval_set)
        
        print("Model training complete")
        return self
    
    def run_backtest(
        self,
        position_method: Optional[str] = None,
        n_splits: Optional[int] = None,
        test_size: Optional[int] = None,
    ) -> Dict:
        """
        Run walk-forward backtest.
        
        Parameters
        ----------
        position_method : str, optional
            Position sizing method
        n_splits : int, optional
            Number of CV splits
        test_size : int, optional
            Test set size
        
        Returns
        -------
        Dict
            Backtest results
        """
        if self.X_ is None or self.y_ is None or self.model_ is None:
            raise ValueError("Must prepare data and train model first")
        
        pos_method = position_method or self.config.decision.default_method
        n_split = n_splits or self.config.backtest.n_splits
        t_size = test_size or self.config.backtest.test_size
        
        print(f"Running walk-forward backtest with {n_split} splits...")
        
        # Create CV splitter
        cv_splitter = WalkForwardSplit(
            n_splits=n_split,
            test_size=t_size,
            purge_size=self.config.backtest.purge_size,
            embargo_size=self.config.backtest.embargo_size,
        )
        
        # Run backtest
        self.backtest_results_ = run_walk_forward_backtest(
            model=self.model_,
            X=self.X_,
            y=self.y_,
            cv_splitter=cv_splitter,
            position_method=pos_method,
            volatility=self.volatility_,
            transaction_cost=self.config.backtest.transaction_cost,
        )
        
        # Print results
        self._print_results()
        
        return self.backtest_results_
    
    def _print_results(self):
        """Print backtest results summary."""
        if self.backtest_results_ is None:
            return
        
        metrics = self.backtest_results_['metrics']
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return:      {metrics.get('total_return', 0):.2%}")
        print(f"Annual Return:     {metrics.get('annual_return', 0):.2%}")
        print(f"Volatility:        {metrics.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):.2f}")
        print(f"Max Drawdown:      {metrics.get('max_drawdown', 0):.2%}")
        print(f"Calmar Ratio:      {metrics.get('calmar_ratio', 0):.2f}")
        print(f"Win Rate:          {metrics.get('win_rate', 0):.2%}")
        print(f"Profit Factor:     {metrics.get('profit_factor', 0):.2f}")
        if metrics.get('turnover') is not None:
            print(f"Avg Turnover:      {metrics.get('turnover', 0):.2f}")
        print("="*50)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
        
        Returns
        -------
        pd.DataFrame
            Feature importance table
        """
        if self.model_ is None:
            raise ValueError("Must train model first")
        
        if hasattr(self.model_, 'get_feature_importance'):
            importance_df = self.model_.get_feature_importance(self.feature_names_)
            return importance_df.head(top_n)
        else:
            return pd.DataFrame({'feature': self.feature_names_, 'importance': 0})
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve from backtest results.
        
        Returns
        -------
        pd.Series
            Equity curve
        """
        if self.backtest_results_ is None:
            raise ValueError("Must run backtest first")
        
        equity = self.backtest_results_['equity_curve']
        indices = self.backtest_results_['indices']
        
        # Map back to original index if available
        if hasattr(self, 'labels_') and self.labels_ is not None:
            original_index = self.labels_.index[indices]
            return pd.Series(equity, index=original_index, name='Equity')
        else:
            return pd.Series(equity, name='Equity')
