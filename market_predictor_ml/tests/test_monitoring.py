"""
Comprehensive Test Suite for Market Predictor ML

This module provides unit, integration, and property-based tests
for all major components of the system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_predictor_ml.tests.fixtures import (
    generate_sample_prices,
    generate_multi_asset_data,
    generate_sample_features,
    MockDataLoader,
    MockModel,
    MockBrokerClient,
    FIXTURE_PRICES_AAPL,
)


class TestMonitoringLogger:
    """Tests for structured logging module."""
    
    def test_logger_creation(self):
        """Test logger can be created."""
        from market_predictor_ml.monitoring.logger import get_logger, StructuredLogger
        
        logger = get_logger("test_logger")
        assert isinstance(logger, StructuredLogger)
    
    def test_correlation_id(self):
        """Test correlation ID tracking."""
        from market_predictor_ml.monitoring.logger import (
            set_correlation_id,
            get_correlation_id,
            clear_correlation_id,
        )
        
        # Set correlation ID
        corr_id = set_correlation_id("test-123")
        assert corr_id == "test-123"
        assert get_correlation_id() == "test-123"
        
        # Auto-generate
        auto_id = set_correlation_id()
        assert auto_id is not None
        assert len(auto_id) > 0
        
        # Clear
        clear_correlation_id()
        assert get_correlation_id() is None
    
    def test_logger_context(self):
        """Test logger context enrichment."""
        from market_predictor_ml.monitoring.logger import get_logger
        
        logger = get_logger("test_context")
        logger.set_context(symbol="AAPL", strategy="momentum")
        
        # Should not raise
        logger.info("Test message with context")
        
        logger.clear_context()
        logger.info("Test message without context")


class TestMonitoringMetrics:
    """Tests for metrics collection module."""
    
    def test_performance_metrics_calculations(self):
        """Test performance metric calculations."""
        from market_predictor_ml.monitoring.metrics import PerformanceMetrics
        
        # Generate sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # Test Sharpe ratio
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        
        # Test Sortino ratio
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        
        # Test max drawdown
        cumulative = PerformanceMetrics.calculate_cumulative_returns(returns)
        max_dd, peak_idx, trough_idx = PerformanceMetrics.calculate_max_drawdown(cumulative)
        assert isinstance(max_dd, float)
        assert max_dd <= 0
    
    def test_metrics_collector(self):
        """Test metrics collector functionality."""
        from market_predictor_ml.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector(window_size=30)
        
        # Record metrics
        collector.record_metric("sharpe_ratio", 1.5)
        collector.record_metric("sharpe_ratio", 1.6)
        
        # Get series
        series = collector.get_metric_series("sharpe_ratio")
        assert len(series) == 2
        
        # Export
        data = collector.export_to_dict()
        assert "metrics_history" in data


class TestMonitoringAlerting:
    """Tests for alerting module."""
    
    def test_alert_rule_threshold(self):
        """Test threshold-based alert rules."""
        from market_predictor_ml.monitoring.alerting import (
            AlertManager,
            AlertRule,
            AlertSeverity,
        )
        
        manager = AlertManager()
        
        # Create threshold rule
        rule = AlertRule(
            name="test_threshold",
            metric_name="sharpe_ratio",
            condition=AlertManager.threshold_condition(1.0, "<"),
            severity=AlertSeverity.WARNING,
        )
        
        manager.add_rule(rule)
        
        # Should trigger
        alerts = manager.check_metric("sharpe_ratio", 0.5)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        
        # Should not trigger (cooldown)
        alerts2 = manager.check_metric("sharpe_ratio", 0.3)
        assert len(alerts2) == 0
    
    def test_prebuilt_alerts(self):
        """Test pre-built alert rules."""
        from market_predictor_ml.monitoring.alerting import (
            create_drawdown_alert,
            create_sharpe_ratio_alert,
        )
        
        dd_alert = create_drawdown_alert(-0.1)
        assert dd_alert.name == "max_drawdown_breach"
        
        sharpe_alert = create_sharpe_ratio_alert(-1.0)
        assert sharpe_alert.name == "low_sharpe_ratio"


class TestMonitoringDashboard:
    """Tests for dashboard module."""
    
    def test_dashboard_creation(self):
        """Test dashboard creation and panels."""
        from market_predictor_ml.monitoring.dashboard import Dashboard
        
        dashboard = Dashboard(title="Test Dashboard")
        
        # Add metric panel
        dashboard.add_metric_panel(
            title="Sharpe Ratio",
            value=1.5,
            previous_value=1.3,
            suffix="",
        )
        
        # Add table panel
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        dashboard.add_table_panel("Sample Table", df)
        
        # Export
        data = dashboard.to_dict()
        assert data["title"] == "Test Dashboard"
        assert len(data["panels"]) == 2
    
    def test_dashboard_html_export(self):
        """Test HTML export."""
        from market_predictor_ml.monitoring.dashboard import Dashboard
        
        dashboard = Dashboard()
        dashboard.add_metric_panel("Metric", 100.0)
        
        html = dashboard.to_html()
        assert "<!DOCTYPE html>" in html
        assert "Metric" in html


class TestDataFixtures:
    """Tests for test fixtures."""
    
    def test_generate_sample_prices(self):
        """Test price data generation."""
        df = generate_sample_prices("AAPL", "2020-01-01", "2020-12-31")
        
        assert len(df) > 0
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.index.is_monotonic_increasing
    
    def test_generate_multi_asset_data(self):
        """Test multi-asset data generation."""
        data = generate_multi_asset_data(["AAPL", "GOOGL"], "2020-01-01", "2020-06-30")
        
        assert "AAPL" in data
        assert "GOOGL" in data
        assert len(data["AAPL"]) > 0
    
    def test_mock_classes(self):
        """Test mock classes."""
        mock_loader = MockDataLoader()
        data = mock_loader.load_data("AAPL")
        assert len(data) > 0
        
        mock_model = MockModel(prediction_value=0.5)
        X = pd.DataFrame({"a": [1, 2, 3]})
        pred = mock_model.predict(X)
        assert len(pred) == 3
        assert all(pred == 0.5)
        
        mock_broker = MockBrokerClient()
        account = mock_broker.get_account()
        assert account["cash"] == 100000.0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        from market_predictor_ml.monitoring.logger import get_logger, set_correlation_id
        from market_predictor_ml.monitoring.metrics import MetricsCollector, PerformanceMetrics
        from market_predictor_ml.monitoring.alerting import AlertManager, create_drawdown_alert
        from market_predictor_ml.monitoring.dashboard import Dashboard
        
        # Set up correlation ID
        corr_id = set_correlation_id("integration-test-1")
        
        # Create logger
        logger = get_logger("integration_test")
        logger.set_context(test="monitoring_workflow")
        logger.info("Starting integration test")
        
        # Create metrics collector
        collector = MetricsCollector()
        
        # Generate sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Calculate and record metrics
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        collector.record_metric("sharpe_ratio", sharpe)
        
        # Set up alerting
        alert_manager = AlertManager()
        alert_manager.add_rule(create_drawdown_alert(-0.1))
        
        # Check alerts
        cumulative = PerformanceMetrics.calculate_cumulative_returns(returns)
        max_dd, _, _ = PerformanceMetrics.calculate_max_drawdown(cumulative)
        alerts = alert_manager.check_metric("max_drawdown", max_dd)
        
        # Create dashboard
        dashboard = Dashboard(title="Integration Test Dashboard")
        dashboard.add_metric_panel("Sharpe Ratio", sharpe)
        dashboard.add_metric_panel("Max Drawdown", max_dd, suffix="%")
        
        if alerts:
            dashboard.add_alert_panel("Active Alerts", [a.to_dict() for a in alerts])
        
        # Export
        dashboard_data = dashboard.to_dict()
        assert dashboard_data["title"] == "Integration Test Dashboard"
        
        logger.info("Integration test completed", extra_fields={"sharpe": sharpe})
        
        assert True  # If we got here, the workflow succeeded
    
    def test_end_to_end_data_pipeline(self):
        """Test end-to-end data processing pipeline."""
        # Generate data
        prices = generate_sample_prices("TEST", "2020-01-01", "2021-12-31")
        features = generate_sample_features(prices)
        
        # Validate
        assert len(features) > 0
        assert not features.isnull().all().any()
        
        # Use mock model
        mock_model = MockModel()
        X = features.dropna()
        
        if len(X) > 0:
            predictions = mock_model.predict(X)
            assert len(predictions) == len(X)


class TestPropertyBased:
    """Property-based tests for invariants."""
    
    def test_returns_sum_property(self):
        """Test that cumulative returns match product of periodic returns."""
        from market_predictor_ml.monitoring.metrics import PerformanceMetrics
        
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        cumulative = PerformanceMetrics.calculate_cumulative_returns(returns)
        
        # Final cumulative should equal product of (1 + returns) - 1
        expected = (1 + returns).prod() - 1
        actual = cumulative.iloc[-1]
        
        assert abs(expected - actual) < 1e-10
    
    def test_drawdown_bounds(self):
        """Test that drawdown is always between -1 and 0."""
        from market_predictor_ml.monitoring.metrics import PerformanceMetrics
        
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        cumulative = PerformanceMetrics.calculate_cumulative_returns(returns)
        
        max_dd, _, _ = PerformanceMetrics.calculate_max_drawdown(cumulative)
        
        assert -1 <= max_dd <= 0
    
    def test_sharpe_ratio_symmetry(self):
        """Test Sharpe ratio sign flips with negative returns."""
        from market_predictor_ml.monitoring.metrics import PerformanceMetrics
        
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        sharpe_positive = PerformanceMetrics.calculate_sharpe_ratio(returns)
        sharpe_negative = PerformanceMetrics.calculate_sharpe_ratio(-returns)
        
        # Signs should be opposite
        assert sharpe_positive * sharpe_negative <= 0


def run_tests():
    """Run all tests and return results."""
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-q",
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
