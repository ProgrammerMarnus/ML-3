"""
Live Prediction Adapter

Connects ML models to the trading engine by:
- Fetching live market data
- Running feature engineering pipeline
- Generating model predictions
- Emitting trading signals
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..core import IDataLoader, IFeatureTransformer, IPredictionModel
from ..data.providers import MarketDataLoader
from ..features.pipeline import FeaturePipeline
from ..models.model_registry import InMemoryModelRegistry
from .signals import TradingSignal, SignalType


logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Market status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    HOLIDAY = "holiday"


@dataclass
class PredictionResult:
    """Result of a prediction run."""
    timestamp: datetime
    symbol: str
    prediction: float
    confidence: float
    features: Dict[str, float]
    raw_data: Dict[str, Any]
    model_version: str
    latency_ms: float


class LivePredictor:
    """
    Live prediction service that connects ML models to the trading engine.
    
    Responsibilities:
    - Fetch live market data for configured symbols
    - Apply feature engineering transformations
    - Generate predictions using loaded models
    - Convert predictions to trading signals
    - Handle market hours and data latency
    """
    
    def __init__(
        self,
        data_loader: IDataLoader,
        feature_pipeline: FeaturePipeline,
        model_registry: InMemoryModelRegistry,
        symbols: List[str],
        model_ids: Dict[str, str],  # symbol -> model_id mapping
        min_confidence: float = 0.6,
        prediction_threshold: float = 0.02,  # 2% move threshold
        lookback_days: int = 60,
        check_market_hours: bool = True,
    ):
        """
        Initialize the live predictor.
        
        Args:
            data_loader: Data provider for fetching live prices
            feature_pipeline: Feature transformation pipeline
            model_registry: Registry containing trained models
            symbols: List of symbols to generate predictions for
            model_ids: Mapping of symbol to model ID to use
            min_confidence: Minimum confidence threshold for signals
            prediction_threshold: Minimum prediction magnitude to generate signal
            lookback_days: Days of historical data needed for features
            check_market_hours: Whether to validate market hours before predicting
        """
        self.data_loader = data_loader
        self.feature_pipeline = feature_pipeline
        self.model_registry = model_registry
        self.symbols = symbols
        self.model_ids = model_ids
        self.min_confidence = min_confidence
        self.prediction_threshold = prediction_threshold
        self.lookback_days = lookback_days
        self.check_market_hours = check_market_hours
        
        # Cache for recent data to avoid repeated fetches
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch_time: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 60  # Cache valid for 1 minute
        
        logger.info(f"LivePredictor initialized for {len(symbols)} symbols")
    
    def get_market_status(self, symbol: str) -> MarketStatus:
        """
        Determine current market status for a symbol.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            Current market status
        """
        now = datetime.now()
        weekday = now.weekday()
        
        # Check for weekend
        if weekday >= 5:
            return MarketStatus.CLOSED
        
        current_time = now.time()
        
        # US Market hours (simplified - assumes EST)
        market_open = time(9, 30)
        market_close = time(16, 0)
        pre_market_start = time(4, 0)
        after_hours_end = time(20, 0)
        
        if current_time < market_open:
            if current_time >= pre_market_start:
                return MarketStatus.PRE_MARKET
            else:
                return MarketStatus.CLOSED
        elif current_time >= market_close:
            if current_time < after_hours_end:
                return MarketStatus.AFTER_HOURS
            else:
                return MarketStatus.CLOSED
        else:
            return MarketStatus.OPEN
    
    def _fetch_live_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch live/historical data for a symbol with caching.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with OHLCV data, or None if fetch failed
        """
        now = datetime.now()
        
        # Check cache
        if symbol in self._data_cache:
            last_fetch = self._last_fetch_time.get(symbol)
            if last_fetch and (now - last_fetch).total_seconds() < self._cache_ttl_seconds:
                logger.debug(f"Using cached data for {symbol}")
                return self._data_cache[symbol]
        
        try:
            # Fetch data (end_date=None gets latest data)
            end_date = now
            start_date = end_date - timedelta(days=self.lookback_days + 10)  # Buffer for feature calc
            
            logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
            
            df = self.data_loader.load(
                ticker=symbol,
                start_date=start_date.date().isoformat(),
                end_date=end_date.date().isoformat(),
                interval='1d'
            )

            if df is None or len(df) == 0:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Validate data quality. OHLCV column names are capitalised
            # throughout the data and feature layers.
            if df['Close'].isnull().any():
                logger.warning(f"Missing close prices for {symbol}, forward filling")
                df['Close'] = df['Close'].ffill()
            
            # Update cache
            self._data_cache[symbol] = df
            self._last_fetch_time[symbol] = now
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            return None
    
    def _calculate_features(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Calculate features for the latest bar.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            DataFrame with features for the latest bar, or None
        """
        try:
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for {symbol}: {missing_cols}")
                return None
            
            # Apply feature pipeline.
            # fit_transform is used because the transformers are stateless and
            # the pipeline may not have been fitted in this process yet.
            # Note: We transform all data but only use the last row for prediction
            df_with_features = self.feature_pipeline.fit_transform(df)
            
            if df_with_features is None or len(df_with_features) == 0:
                logger.warning(f"Feature transformation returned empty result for {symbol}")
                return None
            
            # Check for NaN/Inf in features
            feature_cols = [col for col in df_with_features.columns if col not in required_cols]
            if df_with_features[feature_cols].isnull().any().any():
                logger.warning(f"NaN values in features for {symbol}, filling with 0")
                df_with_features[feature_cols] = df_with_features[feature_cols].fillna(0)
            
            if np.isinf(df_with_features[feature_cols]).any().any():
                logger.warning(f"Inf values in features for {symbol}, replacing with large numbers")
                df_with_features[feature_cols] = df_with_features[feature_cols].replace([np.inf, -np.inf], [1e6, -1e6])
            
            return df_with_features
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}", exc_info=True)
            return None
    
    def _generate_prediction(
        self, 
        features_df: pd.DataFrame, 
        symbol: str, 
        model_id: str
    ) -> Optional[PredictionResult]:
        """
        Generate prediction for the latest bar.
        
        Args:
            features_df: DataFrame with calculated features
            symbol: Stock symbol
            model_id: ID of model to use
            
        Returns:
            PredictionResult or None if prediction failed
        """
        try:
            # Get model from registry
            try:
                model_wrapper = self.model_registry.get_model(model_id)
            except KeyError:
                logger.warning(
                    f"Model '{model_id}' is not registered for {symbol}; "
                    "skipping prediction (train and register a model first)"
                )
                return None
            if model_wrapper is None:
                logger.error(f"Model {model_id} not found in registry")
                return None
            
            model = model_wrapper.model
            metadata = model_wrapper.metadata
            
            # Get latest bar features
            latest_row = features_df.iloc[-1:]
            feature_dict = latest_row.to_dict('records')[0]
            
            # Extract feature array (exclude non-feature columns)
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'datetime', 'date', 'symbol']
            feature_names = [k for k in feature_dict.keys() if k not in exclude_cols]
            X = latest_row[feature_names].values
            
            # Generate prediction
            start_time = datetime.now()
            prediction = model.predict(X)[0]
            
            # Try to get confidence/probability if available
            confidence = 0.5  # Default
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)[0]
                    confidence = float(np.max(proba))
                except:
                    pass
            
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build result
            result = PredictionResult(
                timestamp=datetime.now(),
                symbol=symbol,
                prediction=float(prediction),
                confidence=confidence,
                features={k: float(v) for k, v in feature_dict.items() if k not in exclude_cols},
                raw_data={
                    'close': float(features_df['Close'].iloc[-1]),
                    'volume': float(features_df['Volume'].iloc[-1]),
                },
                model_version=model_id,
                latency_ms=latency_ms
            )
            
            logger.info(
                f"Prediction for {symbol}: {prediction:.4f} (confidence: {confidence:.2f}, "
                f"latency: {latency_ms:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}", exc_info=True)
            return None
    
    def _prediction_to_signal(self, pred: PredictionResult) -> Optional[TradingSignal]:
        """
        Convert prediction to trading signal.
        
        Args:
            pred: Prediction result
            
        Returns:
            Signal if thresholds met, None otherwise
        """
        # Check confidence threshold
        if pred.confidence < self.min_confidence:
            logger.debug(f"Prediction confidence {pred.confidence:.2f} below threshold {self.min_confidence}")
            return None
        
        # Check prediction magnitude threshold
        if abs(pred.prediction) < self.prediction_threshold:
            logger.debug(f"Prediction magnitude {pred.prediction:.4f} below threshold {self.prediction_threshold}")
            return None
        
        # Determine signal type
        if pred.prediction > 0:
            signal_type = SignalType.LONG
        else:
            signal_type = SignalType.SHORT
        
        # Create signal
        signal = TradingSignal(
            symbol=pred.symbol,
            signal_type=signal_type,
            strength=min(abs(pred.prediction), 1.0),  # Cap at 1.0
            target_quantity=0,  # Will be calculated by position sizer
            entry_price=pred.raw_data.get('close'),
            timestamp=pred.timestamp,
            metadata={
                'prediction_value': pred.prediction,
                'model_version': pred.model_version,
                'latency_ms': pred.latency_ms,
                'confidence': pred.confidence,
            }
        )
        
        logger.info(f"Generated signal for {pred.symbol}: type={signal_type.value}, strength={signal.strength:.2f}")
        return signal
    
    def generate_signals(self) -> List[TradingSignal]:
        """
        Generate trading signals for all configured symbols.
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Check market hours if enabled
        if self.check_market_hours:
            # Use first symbol to determine market status (simplified)
            status = self.get_market_status(self.symbols[0] if self.symbols else 'SPY')
            
            if status == MarketStatus.CLOSED:
                logger.warning("Market is closed, skipping signal generation")
                return signals
            elif status == MarketStatus.HOLIDAY:
                logger.warning("Market is closed for holiday, skipping signal generation")
                return signals
            elif status in [MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS]:
                logger.info(f"Market is in {status.value} session, proceeding with caution")
        
        # Generate predictions for each symbol
        for symbol in self.symbols:
            model_id = self.model_ids.get(symbol)
            if not model_id:
                logger.warning(f"No model ID configured for {symbol}, skipping")
                continue
            
            # Fetch data
            df = self._fetch_live_data(symbol)
            if df is None:
                continue
            
            # Calculate features
            features_df = self._calculate_features(df, symbol)
            if features_df is None:
                continue
            
            # Generate prediction
            pred = self._generate_prediction(features_df, symbol, model_id)
            if pred is None:
                continue
            
            # Convert to signal
            signal = self._prediction_to_signal(pred)
            if signal is not None:
                signals.append(signal)
        
        logger.info(f"Generated {len(signals)} signals from {len(self.symbols)} symbols")
        return signals
    
    def get_latest_predictions(self) -> Dict[str, PredictionResult]:
        """
        Get latest predictions for all symbols (even if below signal thresholds).
        
        Returns:
            Dictionary mapping symbol to PredictionResult
        """
        predictions = {}
        
        for symbol in self.symbols:
            model_id = self.model_ids.get(symbol)
            if not model_id:
                continue
            
            df = self._fetch_live_data(symbol)
            if df is None:
                continue
            
            features_df = self._calculate_features(df, symbol)
            if features_df is None:
                continue
            
            pred = self._generate_prediction(features_df, symbol, model_id)
            if pred is not None:
                predictions[symbol] = pred
        
        return predictions


def create_live_predictor(
    config: Dict[str, Any],
    data_loader: Optional[IDataLoader] = None,
    feature_pipeline: Optional[FeaturePipeline] = None,
    model_registry: Optional[InMemoryModelRegistry] = None,
) -> LivePredictor:
    """
    Factory function to create a LivePredictor from configuration.
    
    Args:
        config: Configuration dictionary
        data_loader: Optional pre-initialized data loader
        feature_pipeline: Optional pre-initialized feature pipeline
        model_registry: Optional pre-initialized model registry
        
    Returns:
        Configured LivePredictor instance
    """
    from ..data.providers import create_data_provider
    from ..config.enhanced_settings import EnhancedSettings
    
    # Load or create components
    if data_loader is None:
        provider_type = config.get('data_provider', 'yfinance')
        data_loader = create_data_provider(provider_type)
    
    if feature_pipeline is None:
        # TODO: Load feature pipeline from config/artifact
        feature_pipeline = FeaturePipeline([])
        logger.warning("Using empty feature pipeline - configure and load from artifact")
    
    if model_registry is None:
        # TODO: Load model registry from config/artifact
        model_registry = InMemoryModelRegistry()
        logger.warning("Using empty model registry - configure and load models")
    
    # Extract configuration
    symbols = config.get('symbols', [])
    model_ids = config.get('model_ids', {})
    min_confidence = config.get('min_confidence', 0.6)
    prediction_threshold = config.get('prediction_threshold', 0.02)
    lookback_days = config.get('lookback_days', 60)
    check_market_hours = config.get('check_market_hours', True)
    
    return LivePredictor(
        data_loader=data_loader,
        feature_pipeline=feature_pipeline,
        model_registry=model_registry,
        symbols=symbols,
        model_ids=model_ids,
        min_confidence=min_confidence,
        prediction_threshold=prediction_threshold,
        lookback_days=lookback_days,
        check_market_hours=check_market_hours,
    )
