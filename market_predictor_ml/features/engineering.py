"""
Feature engineering module.

Implements 8 categories of features as described in the README:
1. Momentum features
2. Volatility features
3. Volume features
4. Liquidity features
5. Price pattern features
6. Technical indicators
7. Time-based features
8. Cross-asset features (for future extension)
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def compute_momentum_features(df: pd.DataFrame, periods: List[int] = [5, 10, 21, 63]) -> pd.DataFrame:
    """
    Compute momentum features including ROC and rate of change.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'Close' column
    periods : List[int]
        Lookback periods for momentum calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with momentum features
    """
    df = df.copy()
    
    for period in periods:
        # Rate of Change
        df[f'Momentum_ROC_{period}d'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
        
        # Simple momentum
        df[f'Momentum_{period}d'] = df['Close'] / df['Close'].shift(period)
        
        # Normalized momentum (z-score relative to recent history)
        rolling_mean = df['Close'].rolling(window=period).mean()
        rolling_std = df['Close'].rolling(window=period).std()
        df[f'Momentum_ZScore_{period}d'] = (df['Close'] - rolling_mean) / rolling_std
    
    return df


def compute_volatility_features(
    df: pd.DataFrame, 
    windows: List[int] = [5, 10, 21, 63]
) -> pd.DataFrame:
    """
    Compute volatility features including realized volatility and range-based measures.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with OHLC columns
    windows : List[int]
        Rolling windows for volatility calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with volatility features
    """
    df = df.copy()
    
    # Log returns for volatility calculation
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    
    for window in windows:
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
        log_oc = np.log(df['Open'] / df['Close'].shift(1))
        
        gk = 0.5 * (log_ho - log_lo) ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        df[f'Volatility_GK_{window}d'] = np.sqrt(gk.rolling(window=window).mean()) * np.sqrt(252)
        
        # Volatility trend (ratio of short-term to long-term vol)
        if window > 5:
            short_vol = df['LogReturn'].rolling(window=5).std()
            long_vol = df['LogReturn'].rolling(window=window).std()
            df[f'Volatility_Ratio_{window}d'] = short_vol / long_vol
    
    return df


def compute_volume_features(
    df: pd.DataFrame, 
    windows: List[int] = [5, 10, 21]
) -> pd.DataFrame:
    """
    Compute volume-based features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'Volume' column
    windows : List[int]
        Rolling windows for volume analysis
    
    Returns
    -------
    pd.DataFrame
        DataFrame with volume features
    """
    df = df.copy()
    
    for window in windows:
        # Volume ratio (current vs average)
        avg_volume = df['Volume'].rolling(window=window).mean()
        df[f'Volume_Ratio_{window}d'] = df['Volume'] / avg_volume
        
        # Volume trend
        df[f'Volume_Trend_{window}d'] = df['Volume'].rolling(window=window).sum() / \
                                         df['Volume'].shift(window).rolling(window=window).sum()
        
        # On-Balance Volume (OBV) change
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df[f'OBV_Change_{window}d'] = obv.diff(periods=window)
        
        # Volume-weighted price change
        vwap = (df['Close'] * df['Volume']).rolling(window=window).sum() / \
               df['Volume'].rolling(window=window).sum()
        df[f'VWAP_Deviation_{window}d'] = (df['Close'] - vwap) / vwap
    
    return df


def compute_liquidity_features(
    df: pd.DataFrame, 
    windows: List[int] = [5, 10, 21]
) -> pd.DataFrame:
    """
    Compute liquidity proxies using available data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with OHLCV columns
    windows : List[int]
        Rolling windows for liquidity calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with liquidity features
    """
    df = df.copy()
    
    for window in windows:
        # Amihud illiquidity ratio (absolute return / volume)
        abs_return = np.abs(df['Close'].pct_change())
        df[f'Liquidity_Amihud_{window}d'] = (abs_return / df['Volume']).rolling(window=window).mean()
        
        # Turnover ratio (volume / market cap proxy using close price)
        df[f'Liquidity_Turnover_{window}d'] = (df['Volume'] * df['Close']).rolling(window=window).mean()
        
        # Bid-ask spread proxy using high-low
        spread_proxy = (df['High'] - df['Low']) / df['Close']
        df[f'Liquidity_SpreadProxy_{window}d'] = spread_proxy.rolling(window=window).mean()
    
    return df


def compute_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price pattern and structure features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with OHLC columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with price pattern features
    """
    df = df.copy()
    
    # Candlestick patterns
    df['Pattern_Doji'] = (np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)).astype(float)
    
    # Upper and lower shadows
    df['Pattern_UpperShadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['Pattern_LowerShadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # Price position in range
    df['Pattern_PositionInRange'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # Higher highs and higher lows (trend structure)
    df['Pattern_HigherHigh'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['Pattern_LowerLow'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    
    # Consecutive up/down days
    df['Pattern_ConsecutiveUp'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df['Pattern_ConsecutiveDown'] = (df['Close'] < df['Close'].shift(1)).astype(int)
    
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common technical indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with OHLCV columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with technical indicators
    """
    df = df.copy()
    
    # Moving averages
    for period in [5, 10, 20, 50, 200]:
        df[f'Tech_SMA_{period}d'] = df['Close'].rolling(window=period).mean()
        df[f'Tech_EMA_{period}d'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Price relative to MA
        df[f'Tech_PriceToSMA_{period}d'] = df['Close'] / df[f'Tech_SMA_{period}d']
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['Tech_MACD'] = ema12 - ema26
    df['Tech_MACD_Signal'] = df['Tech_MACD'].ewm(span=9, adjust=False).mean()
    df['Tech_MACD_Hist'] = df['Tech_MACD'] - df['Tech_MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['Tech_RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Tech_BB_Upper'] = sma20 + 2 * std20
    df['Tech_BB_Lower'] = sma20 - 2 * std20
    df['Tech_BB_Width'] = (df['Tech_BB_Upper'] - df['Tech_BB_Lower']) / sma20
    df['Tech_BB_Position'] = (df['Close'] - df['Tech_BB_Lower']) / (df['Tech_BB_Upper'] - df['Tech_BB_Lower'] + 1e-10)
    
    # ATR (Average True Range)
    tr1 = df['High'] - df['Low']
    tr2 = np.abs(df['High'] - df['Close'].shift(1))
    tr3 = np.abs(df['Low'] - df['Close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    df['Tech_ATR'] = tr.rolling(window=14).mean()
    df['Tech_ATR_Normalized'] = df['Tech_ATR'] / df['Close']
    
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based features (day of week, month, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    
    Returns
    -------
    pd.DataFrame
        DataFrame with time features
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    
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


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all feature categories in one call.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data with OHLCV columns and DatetimeIndex
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Ensure proper index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Compute all feature categories
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_liquidity_features(df)
    df = compute_price_pattern_features(df)
    df = compute_technical_indicators(df)
    df = compute_time_features(df)
    
    # Remove temporary columns
    cols_to_drop = ['LogReturn']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding raw price/volume columns).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    
    Returns
    -------
    List[str]
        List of feature column names
    """
    exclude_patterns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close', 'Dividends']
    feature_cols = []
    
    for col in df.columns:
        is_feature = True
        for pattern in exclude_patterns:
            if pattern.lower() in col.lower() and not any(prefix in col for prefix in ['Momentum_', 'Volatility_', 'Volume_', 'Liquidity_', 'Pattern_', 'Tech_', 'Time_', 'Return_', 'LogReturn_']):
                is_feature = False
                break
        if is_feature:
            feature_cols.append(col)
    
    return feature_cols
