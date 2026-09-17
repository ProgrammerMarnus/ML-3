"""
Alpaca Markets API Client for live trading.
Supports both paper and live trading modes.
"""

import os
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    TradingClient = None


class AlpacaClient:
    """
    Client for interacting with Alpaca Markets API.
    
    Supports:
    - Real-time market data
    - Order execution (market/limit)
    - Portfolio management
    - Paper trading mode
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required. Install with: pip install alpaca-py"
            )
        
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars."
            )
        
        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=paper,
        )
        
        # Initialize data client
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )
        
    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]
    
    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "buy" or "sell"
    ) -> Dict[str, Any]:
        """Submit a market order."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        
        order = self.trading_client.submit_order(order_request)
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "status": order.status,
        }
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: str = "2020-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get historical bar data."""
        from datetime import datetime
        
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        
        bars = self.data_client.get_stock_bars(request_params)
        data = bars.df
        
        if len(data) == 0:
            return pd.DataFrame()
        
        # Reset index to get date column
        data = data.reset_index()
        data.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        data["Date"] = pd.to_datetime(data["timestamp"]).dt.date
        data = data.set_index("Date")
        
        return data[["open", "high", "low", "close", "volume"]]
    
    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self.trading_client.cancel_orders()
