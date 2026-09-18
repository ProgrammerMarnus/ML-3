"""
Data layer implementations with pluggable providers.

Implements:
- Abstract base class for data providers
- YFinance provider implementation
- CSV file provider for local data
- Data validation pipeline
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path
import yfinance as yf
import logging

from ..core import IDataLoader, IDataProvider

logger = logging.getLogger(__name__)


class YFinanceDataProvider(IDataProvider):
    """Yahoo Finance data provider implementation."""
    
    @property
    def name(self) -> str:
        return "YFinance"
    
    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical market data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for {symbol} in date range")
            
            # Clean column names
            df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
            
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from YFinance: {e}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get the current/latest price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return float(info['lastPrice'])
        except Exception as e:
            raise RuntimeError(f"Failed to get current price from YFinance: {e}")
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        try:
            ticker = yf.Ticker("SPY")
            _ = ticker.fast_info
            return True
        except:
            return False


class CSVDataProvider(IDataProvider):
    """CSV file-based data provider for local data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    @property
    def name(self) -> str:
        return "CSV"
    
    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Load historical data from CSV file."""
        # Try different filename patterns
        patterns = [
            f"{symbol}.csv",
            f"{symbol.lower()}.csv",
            f"{symbol}_{interval}.csv",
        ]
        
        for pattern in patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Filter by date range
                df = df[(df.index >= start) & (df.index <= end)]
                
                if df.empty:
                    raise ValueError(f"No data found for {symbol} in date range")
                
                return df
        
        raise FileNotFoundError(f"No CSV file found for symbol {symbol} in {self.data_dir}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get the latest price from CSV data."""
        # Load all available data
        df = self.get_historical_data(symbol, "1900-01-01", "2100-12-31")
        return float(df['Close'].iloc[-1])
    
    def is_available(self) -> bool:
        """Check if data directory exists and has files."""
        return self.data_dir.exists() and any(self.data_dir.glob("*.csv"))


class MarketDataLoader(IDataLoader):
    """
    Main data loader with validation and preprocessing.
    
    Uses dependency injection to support multiple data providers.
    """
    
    def __init__(self, provider: Optional[IDataProvider] = None):
        self.provider = provider or YFinanceDataProvider()
        self._validation_results: Dict[str, Any] = {}
    
    def load(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """Load and validate market data."""
        interval = kwargs.get('interval', '1d')
        
        # Fetch data from provider
        df = self.provider.get_historical_data(ticker, start_date, end_date, interval)
        
        # Validate data
        if not self.validate(df):
            raise ValueError(f"Data validation failed for {ticker}")
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate data quality."""
        issues = []
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for empty dataframe
        if df.empty:
            issues.append("DataFrame is empty")
        
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for zero volume
        if 'Volume' in df.columns and (df['Volume'] == 0).any():
            issues.append("Zero volume entries found")
        
        # Check date ordering
        if not df.index.is_monotonic_increasing:
            issues.append("Index is not monotonically increasing")
        
        self._validation_results = {
            'valid': len(issues) == 0,
            'issues': issues,
            'row_count': len(df),
            'date_range': (str(df.index.min()), str(df.index.max())) if not df.empty else None
        }
        
        return len(issues) == 0
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get the last validation report."""
        return self._validation_results


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing: handle missing values, ensure proper types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw price data
    
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    df = df.copy()
    
    # Forward fill then backward fill for any remaining NaNs
    df = df.ffill().bfill()
    
    # Remove any rows that still have NaNs
    df = df.dropna()
    
    # Ensure numeric types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def compute_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 21]) -> pd.DataFrame:
    """
    Compute simple and log returns for multiple periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'Close' column
    periods : List[int]
        List of periods for return calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added return columns
    """
    df = df.copy()
    
    for period in periods:
        # Simple returns
        df[f'Return_{period}d'] = df['Close'].pct_change(periods=period)
        
        # Log returns
        df[f'LogReturn_{period}d'] = np.log(df['Close'] / df['Close'].shift(period))
    
    return df


# Factory function for creating data loaders
def create_data_loader(
    provider_type: str = "yfinance",
    **kwargs
) -> MarketDataLoader:
    """
    Create a data loader with the specified provider.
    
    Parameters
    ----------
    provider_type : str
        Type of provider: 'yfinance', 'csv'
    **kwargs
        Additional arguments passed to provider constructor
    
    Returns
    -------
    MarketDataLoader
        Configured data loader instance
    """
    if provider_type == "yfinance":
        provider = YFinanceDataProvider()
    elif provider_type == "csv":
        data_dir = kwargs.get('data_dir', './data')
        provider = CSVDataProvider(data_dir)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return MarketDataLoader(provider)


def create_data_provider(provider_type: str = "yfinance", config: Optional[Dict] = None) -> IDataProvider:
    """
    Factory function to create data provider instances.
    
    Args:
        provider_type: Type of provider ("yfinance", "alpaca", "polygon", "csv")
        config: Configuration dictionary
    
    Returns:
        IDataProvider instance
    """
    config = config or {}
    
    if provider_type == "yfinance":
        return YFinanceDataProvider()
    elif provider_type == "csv":
        return CSVDataProvider(data_dir=config.get("data_dir", "./data"))
    else:
        logger.warning(f"Unknown provider type '{provider_type}', defaulting to yfinance")
        return YFinanceDataProvider()
