"""
Broker Adapter Base Class - Abstract interface for broker implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

from .oms import Order, OrderStatus

logger = logging.getLogger(__name__)


class BrokerAdapter(ABC):
    """
    Abstract base class for broker integrations.
    
    Defines the interface that all broker implementations must follow,
    enabling easy switching between brokers (Alpaca, IBKR, etc.).
    
    Responsibilities:
    - Account information retrieval
    - Market data streaming
    - Order submission and management
    - Position tracking
    - Connection health monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._connected = False
        self._account_id: Optional[str] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if broker connection is active."""
        return self._connected
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to broker."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieve account information.
        
        Returns:
            Dictionary with account details (balance, buying power, etc.)
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping symbol to quantity
        """
        pass
    
    @abstractmethod
    def get_cash_balance(self) -> float:
        """
        Get available cash balance.
        
        Returns:
            Available cash amount
        """
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """
        Submit order to broker for execution.
        
        Args:
            order: Order object to submit
            
        Returns:
            True if submission successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Broker-specific order ID
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current status of an order.
        
        Args:
            order_id: Broker-specific order ID
            
        Returns:
            OrderStatus or None if not found
        """
        pass
    
    @abstractmethod
    def get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Current price or None if unavailable
        """
        pass
    
    @abstractmethod
    def stream_quotes(self, symbols: List[str], callback):
        """
        Start streaming real-time quotes for symbols.
        
        Args:
            symbols: List of ticker symbols
            callback: Function to call with quote updates
        """
        pass
    
    @abstractmethod
    def stop_quotes_stream(self):
        """Stop streaming quotes."""
        pass
    
    def validate_order(self, order: Order) -> tuple[bool, str]:
        """
        Validate order before submission.
        
        Default implementation checks basic constraints.
        Subclasses can add broker-specific validation.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if order.quantity <= 0:
            return False, "Quantity must be positive"
        
        if order.limit_price is not None and order.limit_price <= 0:
            return False, "Limit price must be positive"
        
        if order.stop_price is not None and order.stop_price <= 0:
            return False, "Stop price must be positive"
        
        return True, ""
    
    def get_buying_power(self) -> float:
        """
        Get available buying power.
        
        Default implementation returns cash balance.
        Subclasses can override for margin accounts.
        """
        return self.get_cash_balance()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform connection health check.
        
        Returns:
            Dictionary with health status details
        """
        try:
            account_info = self.get_account_info()
            return {
                "connected": self._connected,
                "latency_ms": None,  # Could measure actual latency
                "account_status": "active" if account_info else "unknown",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
