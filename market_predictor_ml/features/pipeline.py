"""
Feature Pipeline Module - Scikit-learn style transformers and pipeline.

This module provides:
1. Individual feature transformers that follow sklearn API
2. FeaturePipeline class that chains transformations
3. Feature store pattern for versioning and consistency
4. Automated look-ahead bias detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path

from ..core import IFeatureTransformer, ILabelGenerator


class MomentumTransformer(IFeatureTransformer):
    """
    Transformer for momentum features.
    
    Parameters
    ----------
    periods : List[int]
        Lookback periods for momentum calculation
    """
    
    def __init__(self, periods: List[int] = [5, 10, 21, 63]):
        self.periods = periods
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MomentumTransformer':
        """Fit the transformer (no-op for stateless transformations)."""
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by creating momentum features."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")
        
        df = X.copy()
        
        for period in self.periods:
            # Rate of Change
            df[f'Momentum_ROC_{period}d'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
            
            # Simple momentum
            df[f'Momentum_{period}d'] = df['Close'] / df['Close'].shift(period)
            
            # Normalized momentum (z-score relative to recent history)
            rolling_mean = df['Close'].rolling(window=period).mean()
            rolling_std = df['Close'].rolling(window=period).std()
            df[f'Momentum_ZScore_{period}d'] = (df['Close'] - rolling_mean) / (rolling_std + 1e-10)
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names after transformation."""
        feature_names = []
        for period in self.periods:
            feature_names.extend([
                f'Momentum_ROC_{period}d',
                f'Momentum_{period}d',
                f'Momentum_ZScore_{period}d'
            ])
        return feature_names
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {'periods': self.periods}
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class VolatilityTransformer(IFeatureTransformer):
    """
    Transformer for volatility features.
    
    Parameters
    ----------
    windows : List[int]
        Rolling windows for volatility calculation
    """
    
    def __init__(self, windows: List[int] = [5, 10, 21, 63]):
        self.windows = windows
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VolatilityTransformer':
        """Fit the transformer (no-op for stateless transformations)."""
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by creating volatility features."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")
        
        df = X.copy()
        
        # Log returns for volatility calculation
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in self.windows:
            # Realized volatility (standard deviation of log returns)
            df[f'Volatility_Realized_{window}d'] = df['LogReturn'].rolling(window=window).std() * np.sqrt(252)
            
            # Parkinson volatility (uses high-low range)
            hl_ratio = np.log(df['High'] / df['Low'])
            df[f'Volatility_Parkinson_{window}d'] = np.sqrt(
                (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(window=window).mean()
            ) * np.sqrt(252)
            
            # Garman-Klass volatility
            log_ho = np.log(df['High'] / df['Open'])
            log_lo = np.log(df['Low'] / df['Open'])
            log_co = np.log(df['Close'] / df['Open'])
            
            gk = 0.5 * (log_ho - log_lo) ** 2 - (2 * np.log(2) - 1) * log_co ** 2
            df[f'Volatility_GK_{window}d'] = np.sqrt(gk.rolling(window=window).mean()) * np.sqrt(252)
            
            # Volatility trend (ratio of short-term to long-term vol)
            if window > 5:
                short_vol = df['LogReturn'].rolling(window=5).std()
                long_vol = df['LogReturn'].rolling(window=window).std()
                df[f'Volatility_Ratio_{window}d'] = short_vol / (long_vol + 1e-10)
        
        # Remove temporary column
        df = df.drop(columns=['LogReturn'], errors='ignore')
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names after transformation."""
        feature_names = []
        for window in self.windows:
            feature_names.extend([
                f'Volatility_Realized_{window}d',
                f'Volatility_Parkinson_{window}d',
                f'Volatility_GK_{window}d'
            ])
            if window > 5:
                feature_names.append(f'Volatility_Ratio_{window}d')
        return feature_names
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {'windows': self.windows}
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class VolumeTransformer(IFeatureTransformer):
    """
    Transformer for volume features.
    
    Parameters
    ----------
    windows : List[int]
        Rolling windows for volume analysis
    """
    
    def __init__(self, windows: List[int] = [5, 10, 21]):
        self.windows = windows
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VolumeTransformer':
        """Fit the transformer (no-op for stateless transformations)."""
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by creating volume features."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")
        
        df = X.copy()
        
        for window in self.windows:
            # Volume ratio (current vs average)
            avg_volume = df['Volume'].rolling(window=window).mean()
            df[f'Volume_Ratio_{window}d'] = df['Volume'] / (avg_volume + 1e-10)
            
            # Volume trend
            df[f'Volume_Trend_{window}d'] = df['Volume'].rolling(window=window).sum() / \
                                             (df['Volume'].shift(window).rolling(window=window).sum() + 1e-10)
            
            # On-Balance Volume (OBV) change
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df[f'OBV_Change_{window}d'] = obv.diff(periods=window)
            
            # Volume-weighted price change
            vwap = (df['Close'] * df['Volume']).rolling(window=window).sum() / \
                   (df['Volume'].rolling(window=window).sum() + 1e-10)
            df[f'VWAP_Deviation_{window}d'] = (df['Close'] - vwap) / (vwap + 1e-10)
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names after transformation."""
        feature_names = []
        for window in self.windows:
            feature_names.extend([
                f'Volume_Ratio_{window}d',
                f'Volume_Trend_{window}d',
                f'OBV_Change_{window}d',
                f'VWAP_Deviation_{window}d'
            ])
        return feature_names
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {'windows': self.windows}
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class TechnicalIndicatorTransformer(IFeatureTransformer):
    """
    Transformer for technical indicators.
    
    Parameters
    ----------
    ma_periods : List[int]
        Periods for moving averages
    rsi_period : int
        Period for RSI calculation
    bb_period : int
        Period for Bollinger Bands
    """
    
    def __init__(self, ma_periods: List[int] = [5, 10, 20, 50, 200], 
                 rsi_period: int = 14, bb_period: int = 20):
        self.ma_periods = ma_periods
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TechnicalIndicatorTransformer':
        """Fit the transformer (no-op for stateless transformations)."""
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by creating technical indicator features."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")
        
        df = X.copy()
        
        # Moving averages
        for period in self.ma_periods:
            df[f'Tech_SMA_{period}d'] = df['Close'].rolling(window=period).mean()
            df[f'Tech_EMA_{period}d'] = df['Close'].ewm(span=period, adjust=False).mean()
            
            # Price relative to MA
            df[f'Tech_PriceToSMA_{period}d'] = df['Close'] / (df[f'Tech_SMA_{period}d'] + 1e-10)
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['Tech_MACD'] = ema12 - ema26
        df['Tech_MACD_Signal'] = df['Tech_MACD'].ewm(span=9, adjust=False).mean()
        df['Tech_MACD_Hist'] = df['Tech_MACD'] - df['Tech_MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        df['Tech_RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_bb = df['Close'].rolling(window=self.bb_period).mean()
        std_bb = df['Close'].rolling(window=self.bb_period).std()
        df['Tech_BB_Upper'] = sma_bb + 2 * std_bb
        df['Tech_BB_Lower'] = sma_bb - 2 * std_bb
        df['Tech_BB_Width'] = (df['Tech_BB_Upper'] - df['Tech_BB_Lower']) / (sma_bb + 1e-10)
        df['Tech_BB_Position'] = (df['Close'] - df['Tech_BB_Lower']) / \
                                 (df['Tech_BB_Upper'] - df['Tech_BB_Lower'] + 1e-10)
        
        # ATR (Average True Range)
        tr1 = df['High'] - df['Low']
        tr2 = np.abs(df['High'] - df['Close'].shift(1))
        tr3 = np.abs(df['Low'] - df['Close'].shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        df['Tech_ATR'] = tr.rolling(window=14).mean()
        df['Tech_ATR_Normalized'] = df['Tech_ATR'] / (df['Close'] + 1e-10)
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names after transformation."""
        feature_names = []
        for period in self.ma_periods:
            feature_names.extend([
                f'Tech_SMA_{period}d',
                f'Tech_EMA_{period}d',
                f'Tech_PriceToSMA_{period}d'
            ])
        
        feature_names.extend([
            'Tech_MACD', 'Tech_MACD_Signal', 'Tech_MACD_Hist',
            'Tech_RSI', 'Tech_BB_Upper', 'Tech_BB_Lower', 
            'Tech_BB_Width', 'Tech_BB_Position',
            'Tech_ATR', 'Tech_ATR_Normalized'
        ])
        
        return feature_names
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {
            'ma_periods': self.ma_periods,
            'rsi_period': self.rsi_period,
            'bb_period': self.bb_period
        }
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class TimeFeatureTransformer(IFeatureTransformer):
    """
    Transformer for time-based features.
    """
    
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeFeatureTransformer':
        """Fit the transformer (no-op for stateless transformations)."""
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by creating time features."""
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")
        
        df = X.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Day of week (0=Monday, 4=Friday)
        df['Time_DayOfWeek'] = df.index.dayofweek
        
        # Month
        df['Time_Month'] = df.index.month
        
        # Quarter
        df['Time_Quarter'] = df.index.quarter
        
        # Day of month
        df['Time_DayOfMonth'] = df.index.day
        
        # Week of year
        df['Time_WeekOfYear'] = df.index.isocalendar().week.astype(int)
        
        # Is month end
        df['Time_IsMonthEnd'] = df.index.is_month_end.astype(int)
        
        # Is month start
        df['Time_IsMonthStart'] = df.index.is_month_start.astype(int)
        
        # Is quarter end
        df['Time_IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names after transformation."""
        return [
            'Time_DayOfWeek', 'Time_Month', 'Time_Quarter', 'Time_DayOfMonth',
            'Time_WeekOfYear', 'Time_IsMonthEnd', 'Time_IsMonthStart', 'Time_IsQuarterEnd'
        ]
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {}
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class LabelGenerator(ILabelGenerator):
    """
    Generator for labels/target variables.
    
    Parameters
    ----------
    horizon : int
        Forecast horizon in days
    label_type : str
        Type of label: 'direction', 'return', 'triple_barrier'
    threshold : float
        Threshold for direction classification (if applicable)
    """
    
    def __init__(self, horizon: int = 5, label_type: str = 'direction', threshold: float = 0.0):
        self.horizon = horizon
        self.label_type = label_type
        self.threshold = threshold
        self._label_columns = []
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels from price data."""
        df = df.copy()
        
        if self.label_type == 'direction':
            future_return = df['Close'].shift(-self.horizon) / df['Close'] - 1
            df['Label_Direction'] = (future_return > self.threshold).astype(int)
            self._label_columns = ['Label_Direction']
        
        elif self.label_type == 'return':
            df['Label_Return'] = df['Close'].shift(-self.horizon) / df['Close'] - 1
            self._label_columns = ['Label_Return']
        
        elif self.label_type == 'triple_barrier':
            # Triple barrier labeling
            df = self._triple_barrier_method(df)
        
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")
        
        # Drop rows with NaN labels
        df = df.dropna(subset=self._label_columns)
        
        return df
    
    def _triple_barrier_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implement triple barrier labeling method."""
        # Simplified implementation
        pt = self.threshold  # Profit target
        sl = -self.threshold  # Stop loss
        
        labels = []
        for i in range(len(df) - self.horizon):
            entry_price = df['Close'].iloc[i]
            future_prices = df['Close'].iloc[i+1:i+1+self.horizon]
            
            if len(future_prices) == 0:
                labels.append(np.nan)
                continue
            
            max_price = future_prices.max()
            min_price = future_prices.min()
            final_price = future_prices.iloc[-1]
            
            # Check barriers
            if (max_price / entry_price - 1) >= pt:
                labels.append(1)  # Hit profit target
            elif (min_price / entry_price - 1) <= sl:
                labels.append(-1)  # Hit stop loss
            else:
                labels.append(np.sign(final_price / entry_price - 1))  # Final return sign
        
        labels.extend([np.nan] * self.horizon)
        df['Label_TB'] = labels
        self._label_columns = ['Label_TB']
        
        return df
    
    def get_label_columns(self) -> List[str]:
        """Get the names of generated label columns."""
        return self._label_columns
    
    def get_params(self) -> Dict[str, Any]:
        """Get label generator parameters."""
        return {
            'horizon': self.horizon,
            'label_type': self.label_type,
            'threshold': self.threshold
        }


class FeaturePipeline:
    """
    Pipeline that chains multiple feature transformers.
    
    This class provides:
    - Sequential application of transformers
    - Look-ahead bias detection
    - Feature tracking and metadata
    - Serialization/deserialization
    """
    
    def __init__(self, transformers: List[IFeatureTransformer], 
                 label_generator: Optional[ILabelGenerator] = None,
                 name: str = "feature_pipeline"):
        self.transformers = transformers
        self.label_generator = label_generator
        self.name = name
        self._is_fitted = False
        self._feature_metadata = {
            'created_at': None,
            'input_columns': [],
            'output_columns': [],
            'label_columns': [],
            'transformer_params': []
        }
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeaturePipeline':
        """Fit all transformers in the pipeline."""
        for transformer in self.transformers:
            transformer.fit(X, y)
        
        if self.label_generator:
            # Label generator doesn't need fitting, just track columns
            pass
        
        self._is_fitted = True
        self._feature_metadata['created_at'] = datetime.now().isoformat()
        self._feature_metadata['input_columns'] = list(X.columns)
        
        # Collect transformer params
        self._feature_metadata['transformer_params'] = [
            t.get_params() for t in self.transformers
        ]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all pipeline stages."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        df = X.copy()
        
        # Apply each transformer
        for transformer in self.transformers:
            df = transformer.transform(df)
        
        # Generate labels if configured
        if self.label_generator:
            df = self.label_generator.generate(df)
            self._feature_metadata['label_columns'] = self.label_generator.get_label_columns()
        
        # Track output columns
        all_columns = set(df.columns)
        base_columns = set(X.columns)
        new_columns = all_columns - base_columns
        self._feature_metadata['output_columns'] = sorted(list(new_columns))
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def check_lookahead_bias(self, X: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """
        Check for potential look-ahead bias in features.
        
        Returns a report of suspicious patterns.
        """
        report = {
            'has_bias': False,
            'suspicious_features': [],
            'details': []
        }
        
        df = X.copy()
        
        # Apply transformations
        for transformer in self.transformers:
            if not transformer.is_fitted:
                transformer.fit(df)
            df = transformer.transform(df)
        
        # Check for features that use future information
        for col in df.columns:
            if col in X.columns:
                continue
            
            # Check if feature has perfect correlation with future returns
            if 'Close' in df.columns:
                for horizon in [1, 5, 10]:
                    if len(df) > horizon:
                        future_return = df['Close'].shift(-horizon) / df['Close'] - 1
                        valid_mask = ~(df[col].isna() | future_return.isna())
                        
                        if valid_mask.sum() > 10:
                            corr = df.loc[valid_mask, col].corr(future_return.loc[valid_mask])
                            
                            if abs(corr) > 0.5:  # Suspiciously high correlation
                                report['has_bias'] = True
                                report['suspicious_features'].append(col)
                                report['details'].append({
                                    'feature': col,
                                    'horizon': horizon,
                                    'correlation': corr,
                                    'warning': f"High correlation ({corr:.3f}) with {horizon}-day future return"
                                })
        
        if verbose and report['has_bias']:
            print("⚠️  WARNING: Potential look-ahead bias detected!")
            for detail in report['details']:
                print(f"  - {detail['warning']}")
        
        return report
    
    def get_feature_names(self) -> List[str]:
        """Get all output feature names."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted to get feature names")
        
        feature_names = []
        for transformer in self.transformers:
            feature_names.extend(transformer.get_feature_names_out())
        
        return feature_names
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get pipeline metadata."""
        return self._feature_metadata.copy()
    
    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save transformers
        transformers_data = []
        for transformer in self.transformers:
            transformers_data.append({
                'class': transformer.__class__.__name__,
                'params': transformer.get_params(),
                'is_fitted': transformer.is_fitted
            })
        
        # Save full pipeline as pickle
        with open(save_path, 'wb') as f:
            pickle.dump({
                'transformers': self.transformers,
                'label_generator': self.label_generator,
                'name': self.name,
                'metadata': self._feature_metadata
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        """Load pipeline from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        pipeline = cls(
            transformers=data['transformers'],
            label_generator=data['label_generator'],
            name=data['name']
        )
        pipeline._is_fitted = True
        pipeline._feature_metadata = data['metadata']
        
        return pipeline
    
    def get_pipeline_hash(self) -> str:
        """Get a unique hash for this pipeline configuration."""
        config_str = json.dumps(self._feature_metadata, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


def create_default_pipeline(horizon: int = 5, label_type: str = 'direction') -> FeaturePipeline:
    """
    Create a default feature pipeline with common transformers.
    
    Parameters
    ----------
    horizon : int
        Forecast horizon for labels
    label_type : str
        Type of label to generate
    
    Returns
    -------
    FeaturePipeline
        Configured feature pipeline
    """
    transformers = [
        MomentumTransformer(periods=[5, 10, 21, 63]),
        VolatilityTransformer(windows=[5, 10, 21, 63]),
        VolumeTransformer(windows=[5, 10, 21]),
        TechnicalIndicatorTransformer(ma_periods=[5, 10, 20, 50, 200]),
        TimeFeatureTransformer()
    ]
    
    label_generator = LabelGenerator(horizon=horizon, label_type=label_type)
    
    return FeaturePipeline(
        transformers=transformers,
        label_generator=label_generator,
        name="default_pipeline"
    )


__all__ = [
    'MomentumTransformer',
    'VolatilityTransformer',
    'VolumeTransformer',
    'TechnicalIndicatorTransformer',
    'TimeFeatureTransformer',
    'LabelGenerator',
    'FeaturePipeline',
    'create_default_pipeline'
]
