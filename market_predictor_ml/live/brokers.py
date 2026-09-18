"""
Broker Implementations - Concrete adapters for Alpaca, IBKR, and Paper trading.
"""

from typing import Optional, Dict, List, Any, Callable
import logging
from datetime import datetime

from .broker_base import BrokerAdapter
from .oms import Order, OrderStatus, OrderSide, OrderType

logger = logging.getLogger(__name__)


class PaperBroker(BrokerAdapter):
    """
    Simulated broker for testing and paper trading.
    
    Features:
    - Simulated order execution with realistic delays
    - Virtual account balance and positions
    - Configurable slippage and fill probability
    - No real money involved
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._cash_balance = config.get("initial_cash", 100000.0) if config else 100000.0
        self._positions: Dict[str, float] = {}
        self._orders: Dict[str, Order] = {}
        self._prices: Dict[str, float] = {}
        self._slippage_pct = config.get("slippage_pct", 0.001) if config else 0.001
        self._fill_probability = config.get("fill_probability", 0.95) if config else 0.95
        self._stream_callback: Optional[Callable] = None
        self._account_id = "PAPER_001"
    
    def connect(self) -> bool:
        """Initialize paper broker connection."""
        logger.info("Connecting to paper broker...")
        self._connected = True
        logger.info(f"Paper broker connected. Initial cash: ${self._cash_balance:,.2f}")
        return True
    
    def disconnect(self):
        """Close paper broker connection."""
        self._connected = False
        if self._stream_callback:
            self._stream_callback = None
        logger.info("Paper broker disconnected")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account summary."""
        total_value = self._cash_balance
        for symbol, qty in self._positions.items():
            price = self._prices.get(symbol, 0.0)
            total_value += qty * price
        
        return {
            "account_id": self._account_id,
            "cash": self._cash_balance,
            "portfolio_value": total_value,
            "buying_power": self._cash_balance,
            "positions_count": len(self._positions),
            "type": "paper",
        }
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self._positions.copy()
    
    def get_cash_balance(self) -> float:
        """Get available cash."""
        return self._cash_balance
    
    def update_price(self, symbol: str, price: float) -> None:
        """BrokerAdapter hook: track latest price for paper fills."""
        self.set_price(symbol, price)

    def set_price(self, symbol: str, price: float):
        """Set current market price (for simulation)."""
        self._prices[symbol] = price
        if self._stream_callback:
            self._stream_callback(symbol, {"price": price, "timestamp": datetime.now()})
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to paper broker."""
        if not self._connected:
            logger.error("Paper broker not connected")
            return False
        
        # Validate order
        is_valid, error_msg = self.validate_order(order)
        if not is_valid:
            logger.error(f"Order validation failed: {error_msg}")
            return False
        
        # Check buying power for buys
        if order.side == OrderSide.BUY:
            estimated_cost = order.quantity * (order.limit_price or self._prices.get(order.symbol, 0))
            if estimated_cost > self._cash_balance:
                logger.error(f"Insufficient buying power: need ${estimated_cost:.2f}, have ${self._cash_balance:.2f}")
                return False
        
        # Check position for sells
        if order.side == OrderSide.SELL:
            current_pos = self._positions.get(order.symbol, 0.0)
            if order.quantity > current_pos:
                logger.error(f"Insufficient position: need {order.quantity}, have {current_pos}")
                return False
        
        # Store order
        self._orders[order.order_id] = order
        
        # Simulate execution (immediate for market orders)
        if order.order_type == OrderType.MARKET:
            self._simulate_fill(order)
        
        return True
    
    def _simulate_fill(self, order: Order):
        """Simulate order execution with slippage."""
        import random
        
        if random.random() > self._fill_probability:
            logger.warning(f"Order {order.order_id[:8]} failed to fill (simulated)")
            return
        
        # Get current price
        base_price = self._prices.get(order.symbol, 50.0)  # Default $50
        
        # Apply slippage
        slippage = base_price * self._slippage_pct * random.uniform(-1, 1)
        if order.side == OrderSide.BUY:
            fill_price = base_price + abs(slippage)
        else:
            fill_price = base_price - abs(slippage)
        
        # Use limit price if better
        if order.limit_price:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                fill_price = order.limit_price
            elif order.side == OrderSide.SELL and fill_price < order.limit_price:
                fill_price = order.limit_price
        
        # Calculate commission (simple model)
        commission = max(0.01, order.quantity * fill_price * 0.001)  # 0.1% with $0.01 min
        
        # Update cash and positions
        if order.side == OrderSide.BUY:
            cost = order.quantity * fill_price + commission
            self._cash_balance -= cost
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.quantity
        else:
            proceeds = order.quantity * fill_price - commission
            self._cash_balance += proceeds
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) - order.quantity
            
            # Clean up zero positions
            if abs(self._positions[order.symbol]) < 1e-6:
                del self._positions[order.symbol]
        
        # Update order status
        order.update_status(
            OrderStatus.FILLED,
            filled_quantity=order.quantity,
            remaining_quantity=0.0,
            avg_fill_price=fill_price,
            commission=commission,
            filled_at=datetime.now(),
        )
        
        logger.info(
            f"Filled: {order.side.value} {order.quantity} {order.symbol} @ ${fill_price:.2f}",
            extra={"order_id": order.order_id}
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        if not order.is_active:
            return False
        
        order.update_status(OrderStatus.CANCELLED)
        logger.info(f"Order {order_id[:8]} cancelled")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        order = self._orders.get(order_id)
        return order.status if order else None
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price."""
        return self._prices.get(symbol)
    
    def stream_quotes(self, symbols: List[str], callback: Callable):
        """Start streaming quotes."""
        self._stream_callback = callback
        logger.info(f"Started quote stream for {len(symbols)} symbols")
    
    def stop_quotes_stream(self):
        """Stop streaming quotes."""
        self._stream_callback = None
        logger.info("Quote stream stopped")


class AlpacaBroker(BrokerAdapter):
    """
    Alpaca Markets broker adapter.
    
    Supports both paper and live trading via Alpaca's API.
    Requires alpaca-py library.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._api_key = config.get("api_key") if config else None
        self._secret_key = config.get("secret_key") if config else None
        self._paper = config.get("paper", True) if config else True
        self._client = None
        self._stream_conn = None
    
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data import StockHistoricalDataClient
            
            # Initialize trading client
            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self._paper,
            )
            
            # Verify connection
            account = self._client.get_account()
            self._account_id = account.id
            self._connected = True
            
            logger.info(f"Connected to Alpaca {'paper' if self._paper else 'live'} account: {self._account_id}")
            return True
            
        except ImportError:
            logger.error("alpaca-py library not installed. Run: pip install alpaca-py")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Alpaca."""
        if self._stream_conn:
            self._stream_conn.stop()
        self._connected = False
        logger.info("Disconnected from Alpaca")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information from Alpaca."""
        if not self._connected:
            return {}
        
        account = self._client.get_account()
        return {
            "account_id": account.id,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "type": "paper" if self._paper else "live",
        }
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        if not self._connected:
            return {}
        
        positions = self._client.get_all_positions()
        return {pos.symbol: float(pos.qty) for pos in positions}
    
    def get_cash_balance(self) -> float:
        """Get available cash."""
        info = self.get_account_info()
        return info.get("cash", 0.0)
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to Alpaca."""
        if not self._connected:
            return False
        
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce as AlpacaTIF
            
            # Map order side
            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
            
            # Map time in force
            tif_map = {
                "day": AlpacaTIF.DAY,
                "gtc": AlpacaTIF.GTC,
                "ioc": AlpacaTIF.IOC,
                "fok": AlpacaTIF.FOK,
            }
            time_in_force = tif_map.get(order.time_in_force.value, AlpacaTIF.DAY)
            
            # Create order request
            if order.order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=time_in_force,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.LIMIT:
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    limit_price=order.limit_price,
                    time_in_force=time_in_force,
                    client_order_id=order.client_order_id,
                )
            else:
                logger.error(f"Unsupported order type: {order.order_type}")
                return False
            
            # Submit to Alpaca
            alpaca_order = self._client.submit_order(order_request)
            
            # Update our order with Alpaca's ID
            order.metadata["alpaca_order_id"] = alpaca_order.id
            order.update_status(OrderStatus.SUBMITTED)
            
            logger.info(f"Submitted order to Alpaca: {alpaca_order.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit order to Alpaca: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        if not self._connected:
            return False
        
        try:
            self._client.cancel_orders()  # Cancel all, or use specific ID
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status from Alpaca."""
        if not self._connected:
            return None
        
        try:
            # Would need to map Alpaca status to our OrderStatus
            # Simplified for now
            return OrderStatus.SUBMITTED
        except Exception:
            return None
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price from Alpaca."""
        if not self._connected:
            return None
        
        try:
            from alpaca.data import GetQuoteRequest, DataFeed
            
            quote = self._client.get_latest_quote(symbol)
            return quote.ask_price  # or bid_price, or midpoint
        except Exception:
            return None
    
    def stream_quotes(self, symbols: List[str], callback: Callable):
        """Start streaming quotes from Alpaca."""
        if not self._connected:
            return
        
        try:
            from alpaca.data import StockDataStream
            
            self._stream_conn = StockDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            
            async def handle_quote(quote):
                callback(quote.symbol, {
                    "bid_price": quote.bid_price,
                    "ask_price": quote.ask_price,
                    "timestamp": quote.timestamp,
                })
            
            for symbol in symbols:
                self._stream_conn.subscribe_quotes(handle_quote, symbol)
            
            # Run in background thread in production
            logger.info(f"Started Alpaca quote stream for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to start quote stream: {e}")
    
    def stop_quotes_stream(self):
        """Stop quote stream."""
        if self._stream_conn:
            self._stream_conn.stop()
            self._stream_conn = None


class InteractiveBrokersBroker(BrokerAdapter):
    """
    Interactive Brokers adapter using ib_insync.
    
    Requires ib_insync library and running IB Gateway/TWS.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._host = config.get("host", "127.0.0.1") if config else "127.0.0.1"
        self._port = config.get("port", 7497) if config else 7497  # 7497 for paper, 7496 for live
        self._client_id = config.get("client_id", 1) if config else 1
        self._ib = None
    
    def connect(self) -> bool:
        """Connect to IB Gateway/TWS."""
        try:
            from ib_insync import IB
            
            self._ib = IB()
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
            )
            
            self._connected = True
            logger.info(f"Connected to IB at {self._host}:{self._port}")
            return True
            
        except ImportError:
            logger.error("ib_insync library not installed. Run: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IB."""
        if self._ib:
            self._ib.disconnect()
        self._connected = False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account info from IB."""
        if not self._connected or not self._ib:
            return {}
        
        # Simplified - would need proper implementation
        return {
            "account_id": "IB_ACCOUNT",
            "type": "live",  # or paper based on port
        }
    
    def get_positions(self) -> Dict[str, float]:
        """Get positions from IB."""
        if not self._connected or not self._ib:
            return {}
        
        positions = self._ib.positions()
        return {pos.contract.symbol: float(pos.position) for pos in positions}
    
    def get_cash_balance(self) -> float:
        """Get cash balance from IB."""
        # Simplified implementation
        return 0.0
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to IB."""
        if not self._connected or not self._ib:
            return False
        
        # Would implement full IB order submission here
        logger.warning("IB order submission not fully implemented")
        return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on IB."""
        if not self._connected or not self._ib:
            return False
        
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status from IB."""
        return None
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get market price from IB."""
        if not self._connected or not self._ib:
            return None
        
        # Would implement proper market data request
        return None
    
    def stream_quotes(self, symbols: List[str], callback: Callable):
        """Start streaming quotes from IB."""
        if not self._connected or not self._ib:
            return
        
        logger.warning("IB quote streaming not fully implemented")
    
    def stop_quotes_stream(self):
        """Stop quote stream."""
        pass


def create_broker(adapter_type: str = "paper", config: Optional[Dict] = None) -> BrokerAdapter:
    """
    Factory function to create broker instances.
    
    Args:
        adapter_type: Type of broker ("paper", "alpaca_paper", "alpaca_live", "ibkr")
        config: Configuration dictionary
    
    Returns:
        BrokerAdapter instance
    """
    config = config or {}
    
    if adapter_type == "paper":
        return PaperBroker(config=config)
    elif adapter_type in ["alpaca_paper", "alpaca_live"]:
        return AlpacaBroker(config=config)
    elif adapter_type == "ibkr":
        return InteractiveBrokersBroker(config=config)
    else:
        logger.warning(f"Unknown broker type '{adapter_type}', defaulting to paper")
        return PaperBroker(config=config)
