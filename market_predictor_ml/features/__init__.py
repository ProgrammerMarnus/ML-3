"""Features module initialization."""

from .engineering import (
    compute_momentum_features,
    compute_volatility_features,
    compute_volume_features,
    compute_liquidity_features,
    compute_price_pattern_features,
    compute_technical_indicators,
    compute_time_features,
    create_all_features,
    get_feature_columns,
)
from .labels import (
    compute_future_returns,
    compute_direction_labels,
    triple_barrier_labeling,
    create_regression_target,
    create_all_labels,
)
from .pipeline import (
    MomentumTransformer,
    VolatilityTransformer,
    VolumeTransformer,
    TechnicalIndicatorTransformer,
    TimeFeatureTransformer,
    LabelGenerator,
    FeaturePipeline,
    create_default_pipeline,
)

__all__ = [
    # Functional API
    "compute_momentum_features",
    "compute_volatility_features",
    "compute_volume_features",
    "compute_liquidity_features",
    "compute_price_pattern_features",
    "compute_technical_indicators",
    "compute_time_features",
    "create_all_features",
    "get_feature_columns",
    "compute_future_returns",
    "compute_direction_labels",
    "triple_barrier_labeling",
    "create_regression_target",
    "create_all_labels",
    # Transformer API
    "MomentumTransformer",
    "VolatilityTransformer",
    "VolumeTransformer",
    "TechnicalIndicatorTransformer",
    "TimeFeatureTransformer",
    "LabelGenerator",
    "FeaturePipeline",
    "create_default_pipeline",
]
