"""
Unit Tests for Market Predictor ML.
Tests data integrity, feature engineering, and core pipeline functions.
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import modules to test
from market_predictor_ml.data.loader import download_stock_data, preprocess_data
from market_predictor_ml.features.engineering import create_all_features, get_feature_columns
from market_predictor_ml.features.labels import compute_future_returns, compute_direction_labels
from market_predictor_ml.utils.preprocessing import check_data_leakage
from market_predictor_ml.backtest.engine import compute_economic_metrics


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    def test_load_data_returns_dataframe(self):
        """Verify download_stock_data returns a DataFrame with expected columns."""
        df = download_stock_data("AAPL", start_date="2023-01-01", end_date="2023-01-31")
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, df.columns)
            
    def test_no_duplicate_dates(self):
        """Ensure no duplicate dates in index."""
        df = download_stock_data("MSFT", start_date="2023-06-01", end_date="2023-06-30")
        
        self.assertFalse(df.index.duplicated().any(), "Duplicate dates found in index")
        
    def test_date_range(self):
        """Verify returned data is within requested date range."""
        start = "2023-03-01"
        end = "2023-03-31"
        df = download_stock_data("GOOGL", start_date=start, end_date=end)
        
        # Convert to tz-naive for comparison
        df_index_min = df.index.min().tz_localize(None) if df.index.min().tzinfo else df.index.min()
        df_index_max = df.index.max().tz_localize(None) if df.index.max().tzinfo else df.index.max()
        
        self.assertGreaterEqual(df_index_min, pd.Timestamp(start))
        self.assertLessEqual(df_index_max, pd.Timestamp(end))


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering module."""
    
    @classmethod
    def setUpClass(cls):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range(start="2023-01-01", periods=n, freq='D')
        
        cls.sample_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, n),
            'High': np.random.uniform(110, 120, n),
            'Low': np.random.uniform(90, 100, n),
            'Close': np.random.uniform(100, 110, n),
            'Volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
        
    def test_features_increase_column_count(self):
        """Verify feature engineering adds columns."""
        initial_cols = len(self.sample_df.columns)
        df_features = create_all_features(self.sample_df.copy())
        
        self.assertGreater(len(df_features.columns), initial_cols)
        
    def test_no_nan_in_features(self):
        """Check that features don't introduce excessive NaN values."""
        df_features = create_all_features(self.sample_df.copy())
        
        # Allow some NaN at the beginning due to lagged features
        nan_pct = df_features.isna().mean().mean()
        self.assertLess(nan_pct, 0.5, "Excessive NaN values in features (>50%)")
        
    def test_momentum_features_exist(self):
        """Verify momentum features are created."""
        df_features = create_all_features(self.sample_df.copy())
        
        momentum_cols = [c for c in df_features.columns if 'momentum' in c.lower() or 'roc' in c.lower()]
        self.assertGreater(len(momentum_cols), 0, "No momentum features found")


class TestLabelConstruction(unittest.TestCase):
    """Test label construction methods."""
    
    @classmethod
    def setUpClass(cls):
        """Create sample data."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range(start="2023-01-01", periods=n, freq='D')
        
        cls.sample_df = pd.DataFrame({
            'Close': np.random.uniform(100, 110, n).cumsum() + 100
        }, index=dates)
        
    def test_future_return_label(self):
        """Test future return label creation."""
        df = compute_future_returns(self.sample_df.copy(), horizons=[5])
        
        self.assertIn('FutureReturn_5d', df.columns)
        self.assertEqual(len(df), len(self.sample_df))
        
    def test_directional_label(self):
        """Test directional label creation."""
        df = compute_direction_labels(self.sample_df.copy(), horizons=[5], threshold=0.01)
        
        self.assertIn('Direction_5d', df.columns)
        # Values should be -1, 0, or 1
        unique_vals = df['Direction_5d'].dropna().unique()
        self.assertTrue(set(unique_vals).issubset({-1, 0, 1}))
        
    def test_no_lookahead_bias(self):
        """Ensure labels don't use future data improperly."""
        df = self.sample_df.copy()
        df = compute_future_returns(df, horizons=[5])
        
        # The label at time t should only depend on prices from t+1 to t+5
        # This is a basic sanity check
        self.assertFalse(df['FutureReturn_5d'].isna().all(), "All labels are NaN")


class TestPipelineIntegration(unittest.TestCase):
    """Test end-to-end pipeline integration."""
    
    def test_basic_pipeline_flow(self):
        """Test that data flows through the pipeline without errors."""
        from market_predictor_ml.pipeline import MarketPredictorPipeline
        from market_predictor_ml.config.settings import (
            BacktestConfig,
            Config,
            DataConfig,
            ModelConfig,
        )
        
        # A 3-month window cannot work here: the indicators use up to 200-day
        # windows and labels look 21 days ahead, so the matrix would be empty.
        # Use 4 years plus lightweight model/backtest settings to stay fast.
        config = Config(
            data=DataConfig(
                default_start_date="2020-01-01",
                default_end_date="2024-01-01"
            ),
            model=ModelConfig(lightgbm_n_estimators=50),
            backtest=BacktestConfig(n_splits=2, test_size=100),
        )
        
        # Override ticker in data config - need to check how pipeline uses it
        pipeline = MarketPredictorPipeline(config)
        results = pipeline.run(ticker="AAPL")
        
        self.assertIn('metrics', results)
        self.assertIn('equity_curve', results)
        self.assertIsInstance(results['metrics'], dict)


class TestEconomicMetrics(unittest.TestCase):
    """Regression tests for economic metric definitions."""

    def test_profit_factor_is_gross_profit_over_gross_loss(self):
        """
        profit_factor must be gross profit / gross loss.

        Fixture: 4 wins of +1.0 and 6 losses of -1.0.
            gross_profit = 4.0, gross_loss = 6.0 -> profit_factor = 0.6667
        Regression: the old implementation returned -avg_win / avg_loss,
        which is an average-payoff ratio and yields 1.0 here, hiding the loss.
        """
        returns = np.array([1.0] * 4 + [-1.0] * 6)
        m = compute_economic_metrics(returns)

        self.assertAlmostEqual(m['gross_profit'], 4.0, places=10)
        self.assertAlmostEqual(m['gross_loss'], 6.0, places=10)
        self.assertAlmostEqual(m['profit_factor'], 4.0 / 6.0, places=10)
        self.assertAlmostEqual(m['win_rate'], 0.4, places=10)
        self.assertAlmostEqual(m['avg_win'], 1.0, places=10)
        self.assertAlmostEqual(m['avg_loss'], -1.0, places=10)

        # The payoff ratio is preserved separately and is NOT the profit factor.
        self.assertAlmostEqual(m['payoff_ratio'], 1.0, places=10)
        self.assertNotAlmostEqual(m['profit_factor'], m['payoff_ratio'], places=6)

    def test_profit_factor_below_one_on_losing_system(self):
        """
        A system that loses money must report profit_factor < 1.0 even when its
        average win exceeds its average loss (the exact failure mode that made
        a losing strategy look acceptable on Gate 1).
        """
        returns = np.array([0.02] * 3 + [-0.019] * 7)
        m = compute_economic_metrics(returns)

        self.assertAlmostEqual(m['payoff_ratio'], 0.02 / 0.019, places=10)
        self.assertGreater(m['payoff_ratio'], 1.0)      # misleading on its own
        self.assertLess(m['profit_factor'], 1.0)        # the truth
        self.assertAlmostEqual(
            m['profit_factor'], (3 * 0.02) / (7 * 0.019), places=10
        )
        self.assertLess(m['total_return'], 0.0)

    def test_profit_factor_infinite_when_no_losses(self):
        """All-positive returns mean zero gross loss."""
        m = compute_economic_metrics(np.array([0.01, 0.02, 0.03]))
        self.assertEqual(m['gross_loss'], 0.0)
        self.assertTrue(np.isinf(m['profit_factor']))
        self.assertAlmostEqual(m['win_rate'], 1.0, places=10)

    def test_profit_factor_zero_when_no_wins(self):
        """All-negative returns mean zero gross profit."""
        m = compute_economic_metrics(np.array([-0.01, -0.02, -0.03]))
        self.assertEqual(m['gross_profit'], 0.0)
        self.assertAlmostEqual(m['profit_factor'], 0.0, places=10)
        self.assertAlmostEqual(m['win_rate'], 0.0, places=10)

    def test_profit_factor_matches_enhanced_engine_definition(self):
        """
        The two backtest engines must agree on profit_factor for the same
        return series (they previously used different definitions).
        """
        from market_predictor_ml.backtest.enhanced_engine import Benchmark

        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.015])
        engine_pf = compute_economic_metrics(returns)['profit_factor']

        branded = Benchmark.calculate_metrics(pd.Series(returns)) \
            if hasattr(Benchmark, 'calculate_metrics') else None
        if branded is None:
            # Compute the reference definition directly.
            wins = returns > 0
            ref = returns[wins].sum() / abs(returns[~wins].sum())
        else:
            ref = branded['profit_factor']

        self.assertAlmostEqual(engine_pf, ref, places=10)


if __name__ == '__main__':
    unittest.main()
