"""Data loading and preprocessing utilities."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Union


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str, default '1d'
        Data interval (e.g., '1d', '1h', '5m')
    
    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range")
    
    # Clean column names
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
    
    return df


def download_multiple_stocks(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> dict:
    """
    Download data for multiple stocks.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str, default '1d'
        Data interval
    
    Returns
    -------
    dict
        Dictionary mapping ticker to DataFrame
    """
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = download_stock_data(ticker, start_date, end_date, interval)
            print(f"Downloaded {ticker}: {len(data[ticker])} rows")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
    
    return data

# Single source of truth for preprocessing lives in providers.py
# (M-9: these two functions were duplicated verbatim here).
from .providers import compute_returns, preprocess_data  # noqa: E402,F401
