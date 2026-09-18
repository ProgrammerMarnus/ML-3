"""
Core module - Abstract base classes and interfaces for all major components.

This module defines the contracts that all implementations must follow,
enabling dependency injection, testing, and modular architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np


class IDataLoader(ABC):
    """Interface for data loading components."""
    
    @abstractmethod
    def load(self, ticker: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Load market data for a given ticker and date range."""
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the loaded data meets quality standards."""
        pass


class IFeatureTransformer(ABC):
    """Interface for feature engineering components (scikit-learn style)."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IFeatureTransformer':
        """Fit the transformer to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by creating features."""
        pass
    
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        pass
    
    @abstractmethod
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names after transformation."""
        pass


class ILabelGenerator(ABC):
    """Interface for label/target generation."""
    
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels from price data."""
        pass
    
    @abstractmethod
    def get_label_columns(self) -> List[str]:
        """Get the names of generated label columns."""
        pass


class IPredictionModel(ABC):
    """Interface for ML prediction models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'IPredictionModel':
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IPredictionModel':
        """Set model parameters."""
        pass
    
    @abstractmethod
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance scores."""
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        pass


class IPositionSizer(ABC):
    """Interface for position sizing strategies."""
    
    @abstractmethod
    def calculate_position(
        self, 
        signal: float, 
        volatility: float, 
        capital: float,
        **kwargs
    ) -> float:
        """Calculate position size based on signal and risk."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get position sizer parameters."""
        pass


class IBacktestEngine(ABC):
    """Interface for backtesting engines."""
    
    @abstractmethod
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a portfolio backtest over signals and prices (M-10).

        The signature matches the concrete portfolio engine (see
        backtest/enhanced_engine.py::EnhancedBacktestEngine.run): `signals`,
        `prices` and `volumes` are date x symbol frames. Vectorised engines
        that operate on numpy arrays should define their own interface.
        """
        pass
    
    @abstractmethod
    def get_equity_curve(self) -> pd.Series:
        """Get the equity curve from the last backtest run."""
        pass
    
    @abstractmethod
    def get_trades(self) -> pd.DataFrame:
        """Get the trade log from the last backtest run."""
        pass


class IRiskMonitor(ABC):
    """Interface for risk monitoring components."""
    
    @abstractmethod
    def check_limits(self, portfolio: Dict[str, float]) -> bool:
        """Check if portfolio is within risk limits."""
        pass
    
    @abstractmethod
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        pass


class IDataProvider(ABC):
    """Interface for pluggable market data providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the data provider."""
        pass
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        start: str, 
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical market data."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get the current/latest price for a symbol."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data provider is available/configured."""
        pass


class IModelRegistry(ABC):
    """Interface for model version tracking and storage."""
    
    @abstractmethod
    def register_model(
        self,
        model: IPredictionModel,
        metadata: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a trained model and return its ID."""
        pass
    
    @abstractmethod
    def get_model(self, model_id: str) -> IPredictionModel:
        """Retrieve a model by ID."""
        pass
    
    @abstractmethod
    def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List registered models with optional filtering."""
        pass
    
    @abstractmethod
    def get_best_model(self, metric: str = 'sharpe_ratio') -> Optional[Dict[str, Any]]:
        """Get the best performing model based on a metric."""
        pass


class IEventSubscriber(ABC):
    """Interface for event-driven architecture components."""
    
    @abstractmethod
    def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle an incoming event."""
        pass
    
    @abstractmethod
    def get_subscribed_events(self) -> List[str]:
        """Get list of event types this subscriber handles."""
        pass
