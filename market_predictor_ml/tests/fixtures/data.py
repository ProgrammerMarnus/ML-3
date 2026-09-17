"""
Test Fixtures for Market Predictor ML

Provides reusable test data, mocks, and fixtures for comprehensive testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import random


def generate_sample_prices(
    symbol: str = "AAPL",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005,
) -> pd.DataFrame:
    """
    Generate synthetic price data using geometric Brownian motion.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        initial_price: Starting price
        volatility: Daily volatility
        drift: Daily drift
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    n_days = len(dates)
    
    # Generate returns using GBM
    returns = np.random.normal(drift, volatility, n_days)
    
    # Calculate cumulative returns and prices
    cumulative = (1 + returns).cumprod()
    prices = initial_price * cumulative
    
    # Generate OHLCV
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        "high": prices * (1 + np.random.uniform(0, 0.03, n_days)),
        "low": prices * (1 - np.random.uniform(0, 0.03, n_days)),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n_days),
    })
    
    # Ensure high >= close, open, low and low <= close, open, high
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    
    df.set_index("date", inplace=True)
    df.index.name = None
    df.attrs["symbol"] = symbol
    
    return df


def generate_multi_asset_data(
    symbols: List[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    correlation_matrix: Optional[np.ndarray] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate correlated price data for multiple assets.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        correlation_matrix: Optional correlation matrix for returns
    
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    n_assets = len(symbols)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)
    
    # Default correlation structure
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_assets) * 0.5 + 0.5
    
    # Cholesky decomposition for correlated returns
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, use nearest
        correlation_matrix = np.eye(n_assets)
        L = np.eye(n_assets)
    
    # Generate uncorrelated returns
    uncorr_returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
    
    # Apply correlation
    corr_returns = uncorr_returns @ L.T
    
    # Generate prices for each asset
    data = {}
    for i, symbol in enumerate(symbols):
        initial_price = 100.0 + random.uniform(-20, 20)
        cumulative = (1 + corr_returns[:, i]).cumprod()
        prices = initial_price * cumulative
        
        df = pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "high": prices * (1 + np.random.uniform(0, 0.03, n_days)),
            "low": prices * (1 - np.random.uniform(0, 0.03, n_days)),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n_days),
        }, index=dates)
        
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        
        data[symbol] = df
    
    return data


def generate_sample_features(
    prices: pd.DataFrame,
    n_features: int = 20,
) -> pd.DataFrame:
    """
    Generate sample feature data from prices.
    
    Args:
        prices: Price DataFrame
        n_features: Number of features to generate
    
    Returns:
        DataFrame with features
    """
    features = pd.DataFrame(index=prices.index)
    
    # Basic features
    features["returns_1d"] = prices["close"].pct_change(1)
    features["returns_5d"] = prices["close"].pct_change(5)
    features["returns_10d"] = prices["close"].pct_change(10)
    features["returns_20d"] = prices["close"].pct_change(20)
    
    # Volatility features
    features["volatility_5d"] = features["returns_1d"].rolling(5).std()
    features["volatility_20d"] = features["returns_1d"].rolling(20).std()
    
    # Volume features
    features["volume_ratio_5d"] = prices["volume"] / prices["volume"].rolling(5).mean()
    features["volume_ratio_20d"] = prices["volume"] / prices["volume"].rolling(20).mean()
    
    # Price relative features
    features["price_vs_ma_20"] = prices["close"] / prices["close"].rolling(20).mean()
    features["price_vs_ma_50"] = prices["close"] / prices["close"].rolling(50).mean()
    
    # Additional synthetic features
    for i in range(n_features - len(features.columns)):
        features[f"synthetic_{i}"] = np.random.randn(len(features)) * 0.1
    
    return features.dropna()


def generate_sample_predictions(
    features: pd.DataFrame,
    n_models: int = 3,
    true_return_col: str = "returns_1d",
) -> Dict[str, pd.Series]:
    """
    Generate sample model predictions.
    
    Args:
        features: Features DataFrame
        n_models: Number of models
        true_return_col: Column name for actual returns
    
    Returns:
        Dictionary mapping model names to prediction series
    """
    predictions = {}
    
    # True signal (with noise)
    true_signal = features[true_return_col] if true_return_col in features.columns else np.random.randn(len(features)) * 0.01
    
    for i in range(n_models):
        # Each model has different skill level
        skill = 0.3 + random.uniform(0, 0.4)
        noise = random.uniform(0.1, 0.3)
        
        pred = skill * true_signal + np.random.randn(len(features)) * noise
        predictions[f"model_{i+1}"] = pd.Series(pred, index=features.index)
    
    return predictions


def generate_sample_trades(
    symbols: List[str],
    n_trades: int = 100,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
) -> List[Dict[str, Any]]:
    """
    Generate sample trade records.
    
    Args:
        symbols: List of symbols
        n_trades: Number of trades
        start_date: Start date
        end_date: End date
    
    Returns:
        List of trade dictionaries
    """
    trades = []
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    
    for _ in range(n_trades):
        symbol = random.choice(symbols)
        entry_date = random.choice(dates)
        holding_period = random.randint(1, 20)
        exit_date = entry_date + timedelta(days=holding_period)
        
        entry_price = 100.0 + random.uniform(-30, 30)
        pnl_pct = random.normalvariate(0.01, 0.05)
        exit_price = entry_price * (1 + pnl_pct)
        
        qty = random.randint(10, 100)
        side = random.choice(["buy", "sell"])
        
        trade = {
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": (exit_price - entry_price) * qty if side == "buy" else (entry_price - exit_price) * qty,
            "pnl_pct": pnl_pct,
            "status": "closed",
        }
        
        trades.append(trade)
    
    return trades


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data: Optional[Dict[str, pd.DataFrame]] = None):
        self.data = data or generate_multi_asset_data()
        self.call_count = 0
    
    def load_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data for a symbol."""
        self.call_count += 1
        if symbol in self.data:
            return self.data[symbol].copy()
        raise ValueError(f"No data for symbol: {symbol}")
    
    def get_symbols(self) -> List[str]:
        """Get available symbols."""
        return list(self.data.keys())


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, prediction_value: float = 0.01):
        self.prediction_value = prediction_value
        self.fit_count = 0
        self.predict_count = 0
    
    def fit(self, X, y, **kwargs):
        """Mock fit."""
        self.fit_count += 1
        return self
    
    def predict(self, X, **kwargs) -> np.ndarray:
        """Mock predict."""
        self.predict_count += 1
        if hasattr(X, "__len__"):
            return np.full(len(X), self.prediction_value)
        return np.array([self.prediction_value])
    
    def predict_proba(self, X, **kwargs) -> np.ndarray:
        """Mock predict_proba."""
        self.predict_count += 1
        n_samples = len(X) if hasattr(X, "__len__") else 1
        return np.column_stack([
            np.full(n_samples, 0.3),
            np.full(n_samples, 0.4),
            np.full(n_samples, 0.3),
        ])


class MockBrokerClient:
    """Mock broker client for testing live trading components."""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}
        self.orders = []
        self.order_id_counter = 0
    
    def get_account(self) -> Dict[str, Any]:
        """Get account info."""
        portfolio_value = self.cash + sum(
            self.positions.get(symbol, 0) * 100
            for symbol in self.positions
        )
        return {
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "positions": self.positions,
        }
    
    def get_historical_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Get historical data."""
        return generate_sample_prices(symbol)
    
    def submit_market_order(self, symbol: str, qty: int, side: str) -> Dict[str, Any]:
        """Submit market order."""
        self.order_id_counter += 1
        order = {
            "id": f"order_{self.order_id_counter}",
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "status": "filled",
            "filled_qty": qty,
            "filled_price": 100.0 + random.uniform(-2, 2),
        }
        self.orders.append(order)
        
        # Update positions
        current_qty = self.positions.get(symbol, 0)
        if side == "buy":
            self.positions[symbol] = current_qty + qty
        else:
            self.positions[symbol] = max(0, current_qty - qty)
        
        return order
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders."""
        return [o for o in self.orders if o["status"] == "open"]
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        for order in self.orders:
            if order["id"] == order_id:
                order["status"] = "canceled"
                return True
        return False


class MockFeatureTransformer:
    """Mock feature transformer for testing."""
    
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
        self.feature_names = [f"feature_{i}" for i in range(n_features)]
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        result = pd.DataFrame(
            np.random.randn(len(X), self.n_features),
            columns=self.feature_names,
            index=X.index,
        )
        return result
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform."""
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names


# Pre-generated fixture data
FIXTURE_PRICES_AAPL = generate_sample_prices("AAPL", "2020-01-01", "2023-12-31")
FIXTURE_FEATURES = generate_sample_features(FIXTURE_PRICES_AAPL)
FIXTURE_PREDICTIONS = generate_sample_predictions(FIXTURE_FEATURES)
FIXTURE_TRADES = generate_sample_trades(["AAPL", "GOOGL", "MSFT"], n_trades=50)
