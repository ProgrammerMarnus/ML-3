#!/usr/bin/env python3
"""
Advanced example: Multi-stock portfolio backtest with realistic transaction costs.

This script demonstrates:
1. Running backtest on multiple stocks
2. Using realistic transaction costs and constraints
3. Comparing different position sizing methods
4. Analyzing portfolio-level metrics
"""

from market_predictor_ml import MarketPredictorPipeline, Config
from market_predictor_ml.config.settings import (
    DataConfig, FeatureConfig, LabelConfig, 
    ModelConfig, DecisionConfig, BacktestConfig
)
import pandas as pd
import numpy as np


def run_multi_stock_backtest(
    tickers: list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
    position_methods: list = None,
):
    """
    Run backtest on multiple stocks with different position sizing methods.
    
    Parameters
    ----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    position_methods : list
        List of position sizing methods to compare
    """
    if position_methods is None:
        position_methods = ['fixed', 'volatility_adjusted', 'kelly']
    
    print("=" * 70)
    print("MARKET PREDICTOR ML - ADVANCED MULTI-STOCK BACKTEST")
    print("=" * 70)
    print(f"\nTickers: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Position Methods: {position_methods}")
    print("=" * 70)
    
    # Store results for each method
    all_results = {}
    
    for method in position_methods:
        print(f"\n{'='*70}")
        print(f"Testing Position Sizing Method: {method.upper()}")
        print("=" * 70)
        
        # Create custom config with realistic transaction costs
        config = Config(
            data=DataConfig(
                default_start_date=start_date,
                default_end_date=end_date,
            ),
            features=FeatureConfig(
                winsorize_lower=1.0,
                winsorize_upper=99.0,
                variance_threshold=1e-4,
            ),
            labels=LabelConfig(
                return_horizons=[1, 5, 21],
                triple_barrier_horizon=21,
            ),
            model=ModelConfig(
                lightgbm_n_estimators=500,
                lightgbm_learning_rate=0.05,
                lightgbm_max_depth=6,
            ),
            decision=DecisionConfig(
                default_method=method,
                target_volatility=0.02,
                kelly_fraction=0.25,
            ),
            backtest=BacktestConfig(
                n_splits=3,
                test_size=126,
                transaction_cost=0.001,  # 0.1%
                slippage=0.0005,          # 0.05%
                commission_rate=0.001,    # 0.1%
                min_trade_size=100.0,
                risk_free_rate=0.02,
            )
        )
        
        pipeline = MarketPredictorPipeline(config=config)
        
        # Aggregate results across stocks
        stock_metrics = []
        
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            
            try:
                pipeline.load_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                
                pipeline.engineer_features()
                pipeline.create_labels()
                pipeline.prepare_data(target_column='Target_RiskAdj_21d')
                pipeline.train_model(model_type='lightgbm')
                
                results = pipeline.run_backtest(
                    position_method=method,
                    n_splits=3,
                    test_size=126
                )
                
                metrics = results['metrics']
                metrics['ticker'] = ticker
                stock_metrics.append(metrics)
                
                print(f"  {ticker}: Sharpe={metrics['sharpe_ratio']:.2f}, "
                      f"Return={metrics['total_return']*100:.1f}%, "
                      f"MaxDD={metrics['max_drawdown']*100:.1f}%")
                
            except Exception as e:
                print(f"  Error processing {ticker}: {str(e)}")
                continue
        
        # Aggregate portfolio metrics (simple average)
        if stock_metrics:
            df_metrics = pd.DataFrame(stock_metrics)
            
            portfolio_metrics = {
                'method': method,
                'avg_sharpe': df_metrics['sharpe_ratio'].mean(),
                'avg_return': df_metrics['total_return'].mean(),
                'avg_volatility': df_metrics['volatility'].mean(),
                'avg_max_drawdown': df_metrics['max_drawdown'].mean(),
                'avg_win_rate': df_metrics['win_rate'].mean(),
                'num_stocks': len(stock_metrics),
            }
            
            all_results[method] = {
                'portfolio': portfolio_metrics,
                'stocks': df_metrics,
            }
            
            print(f"\n--- Portfolio Summary ({method}) ---")
            print(f"Average Sharpe:     {portfolio_metrics['avg_sharpe']:.2f}")
            print(f"Average Return:     {portfolio_metrics['avg_return']*100:.1f}%")
            print(f"Average Volatility: {portfolio_metrics['avg_volatility']*100:.2f}%")
            print(f"Average Max DD:     {portfolio_metrics['avg_max_drawdown']*100:.1f}%")
            print(f"Average Win Rate:   {portfolio_metrics['avg_win_rate']*100:.1f}%")
    
    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON OF POSITION SIZING METHODS")
    print("=" * 70)
    
    comparison_data = []
    for method, results in all_results.items():
        p = results['portfolio']
        comparison_data.append({
            'Method': method,
            'Avg Sharpe': f"{p['avg_sharpe']:.2f}",
            'Avg Return': f"{p['avg_return']*100:.1f}%",
            'Avg Vol': f"{p['avg_volatility']*100:.2f}%",
            'Avg MaxDD': f"{p['avg_max_drawdown']*100:.1f}%",
            'Avg Win%': f"{p['avg_win_rate']*100:.1f}%",
            'Stocks': p['num_stocks'],
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Best method
    best_method = max(all_results.keys(), 
                     key=lambda m: all_results[m]['portfolio']['avg_sharpe'])
    print(f"\nBest Method by Sharpe Ratio: {best_method.upper()}")
    
    return all_results


def run_single_stock_analysis(ticker: str = "AAPL"):
    """
    Detailed analysis for a single stock with multiple configurations.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    """
    print("=" * 70)
    print(f"DETAILED ANALYSIS: {ticker}")
    print("=" * 70)
    
    # Test different label horizons
    horizons = [1, 5, 21]
    results_by_horizon = {}
    
    for h in horizons:
        print(f"\nTesting {h}-day return horizon...")
        
        config = Config(
            labels=LabelConfig(
                return_horizons=[h],
            ),
            backtest=BacktestConfig(
                n_splits=3,
                test_size=126,
                transaction_cost=0.001,
                slippage=0.0005,
            )
        )
        
        pipeline = MarketPredictorPipeline(config=config)
        
        try:
            pipeline.load_data(ticker=ticker, start_date="2018-01-01", end_date="2023-12-31")
            pipeline.engineer_features()
            pipeline.create_labels()
            
            target_col = f'Target_RiskAdj_{h}d'
            if target_col not in pipeline.labels_.columns:
                print(f"  Target column {target_col} not available, skipping...")
                continue
            
            pipeline.prepare_data(target_column=target_col)
            pipeline.train_model(model_type='lightgbm')
            
            results = pipeline.run_backtest(
                position_method='volatility_adjusted',
                n_splits=3,
                test_size=126
            )
            
            metrics = results['metrics']
            results_by_horizon[h] = metrics
            
            print(f"  Horizon {h}d: Sharpe={metrics['sharpe_ratio']:.2f}, "
                  f"Return={metrics['total_return']*100:.1f}%")
            
        except Exception as e:
            print(f"  Error with horizon {h}: {str(e)}")
    
    # Show best horizon
    if results_by_horizon:
        best_h = max(results_by_horizon.keys(), 
                    key=lambda h: results_by_horizon[h]['sharpe_ratio'])
        print(f"\nBest Horizon: {best_h} days (Sharpe: {results_by_horizon[best_h]['sharpe_ratio']:.2f})")
    
    return results_by_horizon


if __name__ == "__main__":
    # Run multi-stock comparison
    multi_results = run_multi_stock_backtest(
        tickers=["AAPL", "MSFT", "GOOGL"],
        start_date="2019-01-01",
        end_date="2023-12-31",
        position_methods=['fixed', 'volatility_adjusted']
    )
    
    # Run detailed single stock analysis
    single_results = run_single_stock_analysis(ticker="AAPL")
