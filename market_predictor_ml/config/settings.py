"""
Configuration settings for Market Predictor ML.

Default parameters for each component of the system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """Configuration for data loading."""
    default_start_date: str = "2015-01-01"
    default_end_date: str = "2024-12-31"
    default_interval: str = "1d"
    return_horizons: List[int] = field(default_factory=lambda: [1, 5, 21])


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 21, 63])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63])
    volume_windows: List[int] = field(default_factory=lambda: [5, 10, 21])
    liquidity_windows: List[int] = field(default_factory=lambda: [5, 10, 21])
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    
    # Preprocessing
    winsorize_lower: float = 1.0
    winsorize_upper: float = 99.0
    variance_threshold: float = 1e-4


@dataclass
class LabelConfig:
    """Configuration for label construction."""
    return_horizons: List[int] = field(default_factory=lambda: [1, 5, 21])
    triple_barrier_horizon: int = 21
    triple_barrier_profit_target: float = 0.05
    triple_barrier_stop_loss: float = 0.03
    risk_adjust_window: int = 21


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # LightGBM defaults
    lightgbm_n_estimators: int = 500
    lightgbm_learning_rate: float = 0.05
    lightgbm_max_depth: int = 6
    lightgbm_num_leaves: int = 31
    lightgbm_min_child_samples: int = 50
    lightgbm_subsample: float = 0.8
    lightgbm_colsample_bytree: float = 0.8
    lightgbm_reg_alpha: float = 0.1
    lightgbm_reg_lambda: float = 0.1
    lightgbm_early_stopping_rounds: int = 50
    
    # Ridge baseline
    ridge_alpha: float = 1.0
    
    # Logistic baseline
    logistic_C: float = 1.0


@dataclass
class DecisionConfig:
    """Configuration for position sizing."""
    default_method: str = "volatility_adjusted"
    max_position: float = 1.0
    target_volatility: float = 0.02
    kelly_fraction: float = 0.25
    signal_strength_power: float = 1.0
    prediction_threshold: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    n_splits: int = 5
    test_size: int = 252  # ~1 trading year
    train_size: int = None  # Use all available history
    purge_size: int = 0
    embargo_size: int = 0
    
    # Transaction costs
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    
    # Risk-free rate (annual)
    risk_free_rate: float = 0.02

    # Additional realistic constraints
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1% per trade
    min_trade_size: float = 100.0   # Minimum trade value in dollars
    max_position_size: float = None  # Optional max position limit
    allow_shorting: bool = True


@dataclass
class Config:
    """Master configuration containing all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        # Simple implementation - can be extended for nested dicts
        return cls()


# Default global configuration
DEFAULT_CONFIG = Config()
