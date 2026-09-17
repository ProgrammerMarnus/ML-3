"""Data layer initialization."""

from .loader import (
    download_stock_data,
    download_multiple_stocks,
    preprocess_data,
    compute_returns,
)

__all__ = [
    "download_stock_data",
    "download_multiple_stocks",
    "preprocess_data",
    "compute_returns",
]
