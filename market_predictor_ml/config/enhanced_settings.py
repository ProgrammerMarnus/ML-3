"""
Enhanced configuration management with Pydantic validation.

Provides:
- Type-safe configuration with validation
- Environment variable overrides
- Hierarchical config loading (base, dev, prod)
- Config versioning for experiment tracking
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
from datetime import datetime


class DataConfig(BaseModel):
    """Configuration for data loading."""
    
    default_start_date: str = "2015-01-01"
    default_end_date: str = "2024-12-31"
    default_interval: str = "1d"
    return_horizons: List[int] = Field(default_factory=lambda: [1, 5, 21])
    
    @field_validator('default_start_date', 'default_end_date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got {v}")


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""
    
    momentum_periods: List[int] = Field(default_factory=lambda: [5, 10, 21, 63])
    volatility_windows: List[int] = Field(default_factory=lambda: [5, 10, 21, 63])
    volume_windows: List[int] = Field(default_factory=lambda: [5, 10, 21])
    liquidity_windows: List[int] = Field(default_factory=lambda: [5, 10, 21])
    ma_periods: List[int] = Field(default_factory=lambda: [5, 10, 20, 50, 200])
    
    # Preprocessing
    winsorize_lower: float = 1.0
    winsorize_upper: float = 99.0
    variance_threshold: float = 1e-4
    
    @field_validator('winsorize_lower', 'winsorize_upper')
    @classmethod
    def validate_percentiles(cls, v, info):
        if info.field_name == 'winsorize_lower' and not (0 <= v <= 100):
            raise ValueError("winsorize_lower must be between 0 and 100")
        if info.field_name == 'winsorize_upper' and not (0 <= v <= 100):
            raise ValueError("winsorize_upper must be between 0 and 100")
        return v
    
    @model_validator(mode='after')
    def validate_upper_greater_than_lower(self):
        if self.winsorize_upper <= self.winsorize_lower:
            raise ValueError("winsorize_upper must be greater than winsorize_lower")
        return self


class LabelConfig(BaseModel):
    """Configuration for label construction."""
    
    return_horizons: List[int] = Field(default_factory=lambda: [1, 5, 21])
    triple_barrier_horizon: int = 21
    triple_barrier_profit_target: float = 0.05
    triple_barrier_stop_loss: float = 0.03
    risk_adjust_window: int = 21
    
    @field_validator('triple_barrier_profit_target', 'triple_barrier_stop_loss')
    @classmethod
    def validate_thresholds(cls, v):
        if not (0 < v < 1):
            raise ValueError("Thresholds must be between 0 and 1")
        return v


class ModelConfig(BaseModel):
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
    
    @field_validator('lightgbm_learning_rate')
    @classmethod
    def validate_learning_rate(cls, v):
        if not (0 < v <= 1):
            raise ValueError("Learning rate must be between 0 and 1")
        return v
    
    @field_validator('lightgbm_subsample', 'lightgbm_colsample_bytree')
    @classmethod
    def validate_fractions(cls, v):
        if not (0 < v <= 1):
            raise ValueError("Fraction must be between 0 and 1")
        return v


class DecisionConfig(BaseModel):
    """Configuration for position sizing."""
    
    default_method: str = "volatility_adjusted"
    max_position: float = 1.0
    target_volatility: float = 0.02
    kelly_fraction: float = 0.25
    signal_strength_power: float = 1.0
    prediction_threshold: float = 0.0
    
    @field_validator('max_position')
    @classmethod
    def validate_max_position(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("max_position must be between 0 and 1")
        return v
    
    @field_validator('kelly_fraction')
    @classmethod
    def validate_kelly_fraction(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("kelly_fraction must be between 0 and 1")
        return v


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    
    n_splits: int = 5
    test_size: int = 252  # ~1 trading year
    train_size: Optional[int] = None
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
    max_position_size: Optional[float] = None
    allow_shorting: bool = True
    
    @field_validator('transaction_cost', 'slippage', 'commission_rate')
    @classmethod
    def validate_costs(cls, v):
        if v < 0:
            raise ValueError("Costs cannot be negative")
        return v
    
    @field_validator('n_splits')
    @classmethod
    def validate_n_splits(cls, v):
        if v < 2:
            raise ValueError("n_splits must be at least 2")
        return v


class Config(BaseModel):
    """Master configuration containing all sub-configs."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Metadata
    version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    environment: str = "development"
    
    model_config = ConfigDict(validate_assignment=True, extra='allow')
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def override_from_env(self, prefix: str = "MPML_"):
        """
        Override configuration values from environment variables.

        Environment variables should be named like:
        MPML_DATA_DEFAULT_START_DATE=2020-01-01
        MPML_MODEL_LIGHTGBM_LEARNING_RATE=0.01
        MPML_FEATURES_MOMENTUM_PERIODS=[5,10,20]

        Top-level scalar fields (e.g. MPML_ENVIRONMENT) and one level of
        nesting are supported. Values are parsed as JSON so list fields such
        as momentum_periods can be overridden (L-7).
        """
        import json

        def _parse(raw_value):
            try:
                return json.loads(raw_value)
            except (json.JSONDecodeError, TypeError):
                return raw_value

        for field_name, field_value in self.model_dump().items():
            # Top-level scalar field (e.g. environment)
            if not isinstance(field_value, dict):
                env_value = os.environ.get(f"{prefix}{field_name.upper()}")
                if env_value is not None:
                    try:
                        setattr(self, field_name, _parse(env_value))
                    except (ValueError, TypeError):
                        pass
                continue

            # Nested config section (one level below the root)
            for subfield in field_value:
                env_var = f"{prefix}{field_name.upper()}_{subfield.upper()}"
                env_value = os.environ.get(env_var)
                if env_value is None:
                    continue
                try:
                    setattr(getattr(self, field_name), subfield, _parse(env_value))
                except (ValueError, TypeError):
                    pass

        return self


# Default configurations for different environments
DEFAULT_CONFIG = Config()

DEVELOPMENT_CONFIG = Config(
    environment="development",
    model=ModelConfig(
        lightgbm_n_estimators=100,  # Faster for development
        lightgbm_early_stopping_rounds=10,
    ),
    backtest=BacktestConfig(
        n_splits=3,  # Fewer splits for faster iteration
    ),
)

PRODUCTION_CONFIG = Config(
    environment="production",
    model=ModelConfig(
        lightgbm_n_estimators=1000,
        lightgbm_early_stopping_rounds=100,
    ),
    backtest=BacktestConfig(
        n_splits=10,  # More splits for robust evaluation
    ),
)


def load_config(
    environment: Optional[str] = None,
    config_path: Optional[str] = None,
    from_env: bool = True
) -> Config:
    """
    Load configuration with priority:
    1. Environment variable overrides (if from_env=True)
    2. Config file (if config_path provided)
    3. Environment-specific defaults
    4. Default configuration
    
    Parameters
    ----------
    environment : str, optional
        Environment name ('development', 'production', etc.)
    config_path : str, optional
        Path to config file (YAML or JSON)
    from_env : bool
        Whether to apply environment variable overrides
    
    Returns
    -------
    Config
        Loaded configuration object
    """
    # Start with environment-specific defaults
    if environment == "production":
        config = PRODUCTION_CONFIG.model_copy()
    elif environment == "development":
        config = DEVELOPMENT_CONFIG.model_copy()
    else:
        config = DEFAULT_CONFIG.model_copy()
    
    # Override with file config if provided
    if config_path:
        path = Path(config_path)
        if path.suffix in ['.yaml', '.yml']:
            config = Config.from_yaml(config_path)
        elif path.suffix == '.json':
            config = Config.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    # Apply environment variable overrides
    if from_env:
        config.override_from_env()
    
    return config
