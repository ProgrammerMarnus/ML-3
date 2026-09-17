"""Test fixtures module."""

from .data import (
    generate_sample_prices,
    generate_multi_asset_data,
    generate_sample_features,
    generate_sample_predictions,
    generate_sample_trades,
    MockDataLoader,
    MockModel,
    MockBrokerClient,
    MockFeatureTransformer,
    FIXTURE_PRICES_AAPL,
    FIXTURE_FEATURES,
    FIXTURE_PREDICTIONS,
    FIXTURE_TRADES,
)

__all__ = [
    "generate_sample_prices",
    "generate_multi_asset_data",
    "generate_sample_features",
    "generate_sample_predictions",
    "generate_sample_trades",
    "MockDataLoader",
    "MockModel",
    "MockBrokerClient",
    "MockFeatureTransformer",
    "FIXTURE_PRICES_AAPL",
    "FIXTURE_FEATURES",
    "FIXTURE_PREDICTIONS",
    "FIXTURE_TRADES",
]
