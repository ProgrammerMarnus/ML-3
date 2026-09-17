"""
Example: Hyperparameter Optimization with Market Predictor ML.

This script demonstrates how to use Optuna to find optimal hyperparameters
for the market prediction pipeline, maximizing Sharpe Ratio.
"""

import sys
sys.path.insert(0, '/workspace')

from market_predictor_ml.config.settings import Config, BacktestConfig
from market_predictor_ml.optimization.hyperopt import HyperparameterOptimizer, OptimizationConfig

def main():
    print("=" * 70)
    print("MARKET PREDICTOR ML - HYPERPARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 70)
    
    # Base configuration using dataclass structure
    from market_predictor_ml.config.settings import DataConfig, FeatureConfig, LabelConfig, ModelConfig
    
    config = Config(
        data=DataConfig(
            default_start_date="2019-01-01",
            default_end_date="2023-12-31"
        ),
        model=ModelConfig(),
        features=FeatureConfig(),
        labels=LabelConfig()
    )
    
    backtest_config = BacktestConfig(
        n_splits=3,
        commission_rate=0.001,
        slippage=0.0005,
        initial_capital=100000
    )
    
    # Optimization configuration (reduced trials for demo)
    opt_config = OptimizationConfig(
        objective_metric="sharpe_ratio",
        n_trials=15,  # Small number for quick demo
        n_splits=3,
        n_startup_trials=5,
        enable_pruning=True,
        seed=42
    )
    
    # Run optimization
    optimizer = HyperparameterOptimizer(config, backtest_config, opt_config)
    results = optimizer.optimize(show_progress=True)
    
    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBest Sharpe Ratio: {results['best_value']:.4f}")
    print("\nTop 10 Best Parameters:")
    
    # Sort trials by value
    if results['trials_dataframe'] is not None:
        df = results['trials_dataframe'].sort_values('value', ascending=False).head(10)
        print(df[['number', 'value']].to_string(index=False))
    
    # Plot if matplotlib available
    try:
        optimizer.plot_optimization_history(save_path="/workspace/optimization_history.png")
        print("\nOptimization history plot saved to: /workspace/optimization_history.png")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
    
    return results

if __name__ == "__main__":
    results = main()
