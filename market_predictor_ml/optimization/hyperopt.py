"""
Hyperparameter Optimization Module for Market Predictor ML.

Implements Optuna-based optimization to find the best combination of model 
and strategy parameters that maximize economic metrics (Sharpe Ratio, Total Return)
during walk-forward validation.
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any, Optional, Callable, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import warnings

from ..config.settings import Config, BacktestConfig, ModelConfig, FeatureConfig, LabelConfig
from ..pipeline import MarketPredictorPipeline
from ..backtest.engine import WalkForwardSplit as WalkForwardValidator


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optimization target
    objective_metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, sortino_ratio, calmar_ratio
    minimize: bool = False  # True if minimizing the metric
    
    # Search space bounds
    n_trials: int = 50
    timeout: Optional[int] = None  # Seconds, if None uses n_trials
    n_startup_trials: int = 10
    
    # Walk-forward settings for optimization
    n_splits: int = 3
    purge_size: int = 5
    embargo_size: int = 5
    
    # Pruning
    enable_pruning: bool = True
    pruning_interval: int = 1  # Check every N splits
    
    # Parallelization
    n_jobs: int = 1  # Number of parallel jobs (-1 for all CPUs)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Study name
    study_name: str = "market_predictor_optimization"
    
    # Storage (for distributed optimization)
    storage: Optional[str] = None
    
    def create_study(self) -> optuna.Study:
        """Create an Optuna study with configured sampler and pruner."""
        sampler = TPESampler(seed=self.seed, n_startup_trials=self.n_startup_trials)
        
        pruner = None
        if self.enable_pruning:
            pruner = MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.pruning_interval
            )
        
        return optuna.create_study(
            direction="minimize" if self.minimize else "maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True
        )


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna for Market Predictor ML.
    
    Optimizes model hyperparameters, feature selection thresholds, and 
    decision layer parameters to maximize economic performance.
    """
    
    def __init__(
        self,
        base_config: Config,
        backtest_config: BacktestConfig,
        opt_config: Optional[OptimizationConfig] = None
    ):
        self.base_config = base_config
        self.backtest_config = backtest_config
        self.opt_config = opt_config or OptimizationConfig()
        
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.trials_history: List[pd.DataFrame] = []
        
    def _suggest_model_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest model hyperparameters for the trial."""
        # Default to lightgbm for optimization
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        }
    
    def _suggest_feature_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest feature engineering parameters."""
        return {
            "momentum_windows": trial.suggest_categorical(
                "momentum_windows",
                [[5, 10, 20], [3, 7, 14, 21], [5, 10, 21, 60]]
            ),
            "volatility_windows": trial.suggest_categorical(
                "volatility_windows",
                [[5, 10, 20], [10, 20, 60]]
            ),
            "use_volume_features": trial.suggest_categorical("use_volume_features", [True, False]),
            "use_liquidity_features": trial.suggest_categorical("use_liquidity_features", [True, False]),
            "winsorize_threshold": trial.suggest_float("winsorize_threshold", 2.5, 4.0),
        }
    
    def _suggest_decision_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest decision layer parameters."""
        return {
            "position_method": trial.suggest_categorical(
                "position_method",
                ["fixed", "volatility_adjusted", "kelly"]
            ),
            "fixed_position_size": trial.suggest_float("fixed_position_size", 0.01, 0.10),
            "volatility_target": trial.suggest_float("volatility_target", 0.10, 0.30),
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.25, 1.0),
            "signal_threshold": trial.suggest_float("signal_threshold", 0.0, 0.3),
        }
    
    def _suggest_label_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest label construction parameters."""
        return {
            "horizon": trial.suggest_int("horizon", 5, 60),
            "label_method": trial.suggest_categorical(
                "label_method",
                ["future_return", "direction", "triple_barrier"]
            ),
            "threshold": trial.suggest_float("threshold", 0.0, 0.05),
        }
    
    def _build_trial_config(self, trial: optuna.Trial) -> Tuple[Config, BacktestConfig]:
        """Build complete configuration from trial suggestions."""
        # Get parameter suggestions
        model_params = self._suggest_model_params(trial)
        feature_params = self._suggest_feature_params(trial)
        decision_params = self._suggest_decision_params(trial)
        label_params = self._suggest_label_params(trial)

        # Build model config with LightGBM params
        model_config = ModelConfig(
            lightgbm_n_estimators=model_params["n_estimators"],
            lightgbm_max_depth=model_params["max_depth"],
            lightgbm_learning_rate=model_params["learning_rate"],
            lightgbm_num_leaves=model_params["num_leaves"],
            lightgbm_min_child_samples=model_params["min_child_samples"],
            lightgbm_subsample=model_params["subsample"],
            lightgbm_colsample_bytree=model_params["colsample_bytree"],
            lightgbm_reg_alpha=model_params["reg_alpha"],
            lightgbm_reg_lambda=model_params["reg_lambda"],
        )

        # Build updated base config - map to actual FeatureConfig signature
        base_config = Config(
            data=self.base_config.data,
            model=model_config,
            features=FeatureConfig(
                momentum_periods=feature_params["momentum_windows"],
                volatility_windows=feature_params["volatility_windows"],
                volume_windows=[5, 10, 20] if feature_params.get("use_volume_features", False) else [],
                liquidity_windows=[5, 10, 20] if feature_params.get("use_liquidity_features", False) else [],
                ma_periods=[5, 10, 20, 60],
                winsorize_lower=1.0,
                winsorize_upper=min(feature_params.get("winsorize_threshold", 3.0) * 100, 99.0),
                variance_threshold=0.0001
            ),
            labels=LabelConfig(
                return_horizons=[label_params["horizon"]],
                triple_barrier_horizon=label_params["horizon"],
                triple_barrier_profit_target=label_params["threshold"] * 2,
                triple_barrier_stop_loss=label_params["threshold"],
                risk_adjust_window=min(label_params["horizon"], 21)
            )
        )

        # Build updated backtest config
        backtest_config = BacktestConfig(
            n_splits=self.opt_config.n_splits,
            purge_size=self.opt_config.purge_size,
            embargo_size=self.opt_config.embargo_size,
            commission_rate=self.backtest_config.commission_rate,
            slippage=self.backtest_config.slippage,
            initial_capital=self.backtest_config.initial_capital,
        )

        return base_config, backtest_config
    
    def _run_single_fold(
        self,
        config: Config,
        backtest_config: BacktestConfig,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        data: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """Run a single fold of walk-forward validation."""
        try:
            # Prepare data - drop Close and Returns columns for features
            X = data.drop(columns=["Close", "Returns"])
            y_raw = data["Returns"]
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_raw = y_raw.iloc[train_idx]
            y_test_actual = y_raw.iloc[test_idx].values
            
            # Create labeler and fit on training data
            from ..features import create_all_labels
            label_method = config.labels.label_method if hasattr(config.labels, 'label_method') else 'future_return'
            horizon = config.labels.return_horizons[0] if config.labels.return_horizons else 5
            
            # Create labels based on method
            if label_method == 'direction':
                threshold = getattr(config.labels, 'direction_threshold', 0.0)
                y_train = (y_train_raw > threshold).astype(int)
            elif label_method == 'triple_barrier':
                # Simplified: just use sign of returns for now
                y_train = np.sign(y_train_raw).astype(int)
            else:
                # future_return or risk_adjusted - use raw returns
                y_train = y_train_raw.values
            
            # Create preprocessor and fit
            from ..utils import winsorize_features, remove_near_zero_variance_features, standardize_features
            
            # Ensure X_train is numpy array before processing
            if hasattr(X_train, 'values'):
                X_train_arr = X_train.values
            else:
                X_train_arr = np.array(X_train)
                
            X_train_processed = winsorize_features(
                X_train_arr,
                lower_percentile=config.features.winsorize_lower,
                upper_percentile=config.features.winsorize_upper,
            )
            X_train_processed, keep_indices = remove_near_zero_variance_features(
                X_train_processed,
                threshold=config.features.variance_threshold,
            )
            feature_names = [X.columns[i] for i in keep_indices]
            X_train_processed, _, _ = standardize_features(X_train_processed)
            
            # Create and train model
            from ..models import get_model
            model = get_model(
                'lightgbm',
                n_estimators=config.model.lightgbm_n_estimators,
                learning_rate=config.model.lightgbm_learning_rate,
                max_depth=config.model.lightgbm_max_depth,
                num_leaves=config.model.lightgbm_num_leaves,
                min_child_samples=config.model.lightgbm_min_child_samples,
                subsample=config.model.lightgbm_subsample,
                colsample_bytree=config.model.lightgbm_colsample_bytree,
                reg_alpha=config.model.lightgbm_reg_alpha,
                reg_lambda=config.model.lightgbm_reg_lambda,
            )
            model.fit(X_train_processed, y_train)
            
            # Process test data with same transformations
            if hasattr(X_test, 'values'):
                X_test_arr = X_test.values
            else:
                X_test_arr = np.array(X_test)
            X_test_processed = X_test_arr[:, keep_indices]
            X_test_processed = winsorize_features(
                X_test_processed,
                lower_percentile=config.features.winsorize_lower,
                upper_percentile=config.features.winsorize_upper,
            )
            X_test_processed, _, _ = standardize_features(X_test_processed)
            
            # Predict
            y_pred = model.predict(X_test_processed)
            
            # Create decision layer and get positions
            from ..decision import create_positions
            position_method = config.decision.default_method if hasattr(config.decision, 'default_method') else 'volatility_adjusted'
            
            # Need volatility for vol-adjusted method - use rolling std of test returns
            if position_method == 'volatility_adjusted':
                volatility = np.std(y_test_actual, ddof=1) * np.ones_like(y_test_actual)
            else:
                volatility = None
                
            positions = create_positions(
                predictions=y_pred,
                volatility=volatility,
                method=position_method,
                max_position=getattr(config.decision, 'fixed_position_size', 0.02) * 50,  # Scale to reasonable position size
                threshold=getattr(config.decision, 'signal_threshold', 0.0),
                volatility_target=getattr(config.decision, 'volatility_target', 0.15),
                kelly_fraction=getattr(config.decision, 'kelly_fraction', 0.5),
            )
            
            # Calculate portfolio returns
            portfolio_returns = positions * y_test_actual
            
            # Apply transaction costs
            trades = np.abs(np.diff(positions, prepend=positions[0]))
            transaction_costs = trades * backtest_config.commission_rate
            portfolio_returns -= transaction_costs
            
            return {
                "returns": portfolio_returns,
                "positions": positions,
                "predictions": y_pred,
                "actual": y_test_actual
            }
            
        except Exception as e:
            import traceback
            warnings.warn(f"Fold failed: {str(e)}\n{traceback.format_exc()}")
            return None
    
    def _calculate_objective_metric(
        self,
        results: List[Dict[str, float]]
    ) -> float:
        """Calculate the objective metric from fold results."""
        if not results:
            return -np.inf if not self.opt_config.minimize else np.inf
        
        # Concatenate all returns
        all_returns = np.concatenate([r["returns"] for r in results])
        
        if len(all_returns) == 0:
            return -np.inf if not self.opt_config.minimize else np.inf
        
        # Calculate metrics
        total_return = np.prod(1 + all_returns) - 1
        
        if len(all_returns) < 2:
            return total_return
        
        mean_return = np.mean(all_returns)
        std_return = np.std(all_returns)
        
        # Sharpe ratio (annualized, assuming daily data)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino ratio
        downside_returns = all_returns[all_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else sharpe_ratio
        
        # Drawdown
        cumulative = np.cumprod(1 + all_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else sharpe_ratio
        
        # Select metric
        metrics = {
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }
        
        return metrics.get(self.opt_config.objective_metric, sharpe_ratio)
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        try:
            # Build configuration from trial
            config, backtest_config = self._build_trial_config(trial)
            
            # Load data - use hardcoded AAPL for now since Config doesn't have ticker attribute
            from ..data.loader import download_stock_data
            data = download_stock_data(
                ticker="AAPL",
                start_date=config.data.default_start_date,
                end_date=config.data.default_end_date
            )
            
            if data is None or len(data) < 100:
                raise ValueError("Insufficient data")
            
            # Add basic returns column
            data["Returns"] = data["Close"].pct_change().fillna(0)
            
            # Create walk-forward splits
            validator = WalkForwardValidator(n_splits=self.opt_config.n_splits, purge_size=self.opt_config.purge_size, embargo_size=self.opt_config.embargo_size)
            splits = list(validator.split(data))
            
            # Run each fold
            fold_results = []
            for fold_num, (train_idx, test_idx) in enumerate(splits):
                result = self._run_single_fold(
                    config, backtest_config, train_idx, test_idx, data
                )
                if result is not None:
                    fold_results.append(result)
                    
                # Report intermediate value for pruning
                if len(fold_results) > 0 and len(fold_results) % self.opt_config.pruning_interval == 0:
                    intermediate_value = self._calculate_objective_metric(fold_results)
                    trial.report(intermediate_value, fold_num)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            # Calculate final objective
            objective_value = self._calculate_objective_metric(fold_results)
            
            return objective_value
            
        except Exception as e:
            warnings.warn(f"Trial failed: {str(e)}")
            return -np.inf if not self.opt_config.minimize else np.inf
    
    def optimize(
        self,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with optimization results
        """
        # Create study
        self.study = self.opt_config.create_study()
        
        # Run optimization
        print(f"Starting optimization with {self.opt_config.n_trials} trials...")
        print(f"Objective: {'Maximize' if not self.opt_config.minimize else 'Minimize'} {self.opt_config.objective_metric}")
        print("-" * 60)
        
        self.study.optimize(
            self.objective,
            n_trials=self.opt_config.n_trials,
            timeout=self.opt_config.timeout,
            n_jobs=self.opt_config.n_jobs,
            show_progress_bar=show_progress
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        # Convert to DataFrame
        self.trials_history = self.study.trials_dataframe()
        
        # Print results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best {self.opt_config.objective_metric}: {self.best_value:.4f}")
        print("\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return {
            "best_value": self.best_value,
            "best_params": self.best_params,
            "trials_dataframe": self.trials_history,
            "study": self.study
        }
    
    def get_optimization_results(self) -> Optional[pd.DataFrame]:
        """Get optimization results as DataFrame."""
        return self.trials_history
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Trial history
            ax1 = axes[0]
            values = [t.value for t in self.study.trials if t.value is not None]
            ax1.plot(range(len(values)), values, marker='o', linestyle='-', alpha=0.7)
            ax1.axhline(y=self.best_value, color='r', linestyle='--', label=f'Best: {self.best_value:.4f}')
            ax1.set_xlabel("Trial")
            ax1.set_ylabel(self.opt_config.objective_metric.replace("_", " ").title())
            ax1.set_title("Optimization History")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Parameter importance
            ax2 = axes[1]
            try:
                importance = optuna.importance.get_param_importances(self.study)
                params = list(importance.keys())[:10]  # Top 10
                values = [importance[p] for p in params]
                ax2.barh(params, values)
                ax2.set_xlabel("Importance")
                ax2.set_title("Top 10 Parameter Importances")
                ax2.invert_yaxis()
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                ax2.text(0.5, 0.5, f"Importance calculation failed:\n{str(e)}", 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")


def run_optimization_example():
    """Example usage of the hyperparameter optimizer."""
    from ..config.settings import Config, BacktestConfig
    
    # Base configuration
    config = Config(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date="2023-12-31",
        model={"model_type": "lightgbm"},
        features={},
        labels={}
    )
    
    backtest_config = BacktestConfig(
        n_splits=3,
        commission_rate=0.001,
        slippage_bps=5
    )
    
    # Optimization configuration
    opt_config = OptimizationConfig(
        objective_metric="sharpe_ratio",
        n_trials=20,  # Reduced for demo
        n_splits=3,
        enable_pruning=True,
        seed=42
    )
    
    # Run optimization
    optimizer = HyperparameterOptimizer(config, backtest_config, opt_config)
    results = optimizer.optimize(show_progress=True)
    
    # Plot results
    try:
        optimizer.plot_optimization_history(save_path="optimization_history.png")
    except Exception as e:
        print(f"Could not plot: {e}")
    
    return results


if __name__ == "__main__":
    results = run_optimization_example()
