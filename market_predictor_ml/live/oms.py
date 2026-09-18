"""
Order Management System (OMS) - Handles order lifecycle, status tracking, and execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Callable
import uuid
import threading
import logging

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Buy or sell."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order execution type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Order duration."""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


@dataclass
class Order:
    """
    Represents a trading order with full lifecycle tracking.
    
    Attributes:
        symbol: Ticker symbol (e.g., 'AAPL')
        side: Buy or sell
        quantity: Number of shares/units
        order_type: Market, limit, stop, etc.
        limit_price: Price for limit orders
        stop_price: Price for stop orders
        time_in_force: Order duration
        status: Current order status
        submitted_at: When order was submitted
        filled_at: When order was fully filled
        avg_fill_price: Average price of fills
        filled_quantity: Quantity actually filled
        commission: Transaction fees
        metadata: Additional order information
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    avg_fill_price: Optional[float] = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    commission: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity
        if self.client_order_id is None:
            self.client_order_id = f"mpml_{self.order_id[:8]}"
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_terminal(self) -> bool:
        """Check if order reached final state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def fill_ratio(self) -> float:
        """Return ratio of filled quantity to total."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity
    
    def update_status(self, new_status: OrderStatus, **kwargs):
        """Update order status with optional metadata."""
        old_status = self.status
        self.status = new_status
        
        if new_status == OrderStatus.SUBMITTED and self.submitted_at is None:
            self.submitted_at = datetime.now()
        
        if new_status == OrderStatus.FILLED:
            self.filled_at = datetime.now()
        
        # Update from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(
            f"Order {self.order_id[:8]}: {old_status.value} -> {new_status.value}",
            extra={"order_id": self.order_id, "symbol": self.symbol}
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "avg_fill_price": self.avg_fill_price,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "commission": self.commission,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Order":
        """Create Order from dictionary."""
        data = data.copy()
        data["side"] = OrderSide(data["side"])
        data["order_type"] = OrderType(data["order_type"])
        data["time_in_force"] = TimeInForce(data["time_in_force"])
        data["status"] = OrderStatus(data["status"])
        
        if data.get("submitted_at"):
            data["submitted_at"] = datetime.fromisoformat(data["submitted_at"])
        if data.get("filled_at"):
            data["filled_at"] = datetime.fromisoformat(data["filled_at"])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class OrderManager:
    """
    Manages order lifecycle, tracking, and execution coordination.
    
    Features:
    - Track all orders by ID and symbol
    - Status updates and notifications
    - Order cancellation and modification
    - Fill reconciliation
    - Position tracking from fills
    """
    
    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._orders_by_symbol: Dict[str, List[str]] = {}
        self._active_orders: set = set()
        self._callbacks: List[Callable[[Order], None]] = []
        self._positions: Dict[str, float] = {}  # symbol -> net position
        self._lock = threading.RLock()  # Reentrant: callbacks may re-enter OMS methods
    
    def register_callback(self, callback: Callable[[Order], None]):
        """Register callback for order status changes."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, order: Order):
        """Notify all registered callbacks of order update."""
        for callback in self._callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        metadata: Optional[Dict] = None,
    ) -> Order:
        """Create and register a new order."""
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            metadata=metadata or {},
        )
        with self._lock:
            self._orders[order.order_id] = order

            if symbol not in self._orders_by_symbol:
                self._orders_by_symbol[symbol] = []
    
        self._orders_by_symbol[symbol].append(order.order_id)
        
        logger.info(
            f"Created order {order.order_id[:8]}: {side.value} {quantity} {symbol}",
            extra={"order_id": order.order_id, "symbol": symbol}
        )
        
        return order
    
    def submit_order(self, order_id: str) -> bool:
        """Mark order as submitted (ready for broker execution)."""
        if order_id not in self._orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self._orders[order_id]
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active")
        with self._lock:
            order.update_status(OrderStatus.SUBMITTED)
            self._active_orders.add(order_id)

        self._notify_callbacks(order)
        return True
    
    def accept_order(self, order_id: str):
        """Mark order as accepted by broker."""
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        order.update_status(OrderStatus.ACCEPTED)
        self._notify_callbacks(order)
    
    def update_fill(
        self,
        order_id: str,
        filled_quantity: float,
        fill_price: float,
        commission: float = 0.0,
    ):
        """Update order with partial or full fill."""
        if order_id not in self._orders:
            logger.error(f"Order {order_id} not found")
            return
        
        order = self._orders[order_id]
        
        # Update filled quantities, average price, position and status atomically
        with self._lock:
            old_filled = order.filled_quantity
            order.filled_quantity += filled_quantity
            order.remaining_quantity -= filled_quantity
            order.commission += commission

            # Update average fill price
            if order.avg_fill_price is None:
                order.avg_fill_price = fill_price
            else:
                total_value = (order.avg_fill_price * old_filled) + (fill_price * filled_quantity)
                order.avg_fill_price = total_value / order.filled_quantity

            # Update position
            position_change = filled_quantity if order.side == OrderSide.BUY else -filled_quantity
            current_pos = self._positions.get(order.symbol, 0.0)
            self._positions[order.symbol] = current_pos + position_change

            # Update status
            if order.remaining_quantity <= 0:
                order.update_status(
                    OrderStatus.FILLED,
                    filled_at=datetime.now(),
                )
                self._active_orders.discard(order_id)
            else:
                order.update_status(OrderStatus.PARTIALLY_FILLED)

        
        logger.info(
            f"Fill update: {filled_quantity} @ {fill_price} (pos: {self._positions[order.symbol]})",
            extra={"order_id": order_id, "symbol": order.symbol}
        )
        
        self._notify_callbacks(order)
    
    def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """Cancel an active order."""
        if order_id not in self._orders:
            return False
        with self._lock:
            order.update_status(
                OrderStatus.CANCELLED,
                metadata={**order.metadata, "cancel_reason": reason},
            )
            self._active_orders.discard(order_id)
        self._notify_callbacks(order)
        return True
    def reject_order(self, order_id: str, reason: str = ""):
        """Mark order as rejected."""
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        with self._lock:
            order.update_status(
                OrderStatus.REJECTED,
                metadata={**order.metadata, "reject_reason": reason},
            )
            self._active_orders.discard(order_id)
        self._notify_callbacks(order)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [self._orders[oid] for oid in self._active_orders if oid in self._orders]
    
    def get_position(self, symbol: str) -> float:
        """Get current net position for symbol."""
        return self._positions.get(symbol, 0.0)
    
    def get_all_positions(self) -> Dict[str, float]:
        """Get all positions."""
        return self._positions.copy()
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending/submitted orders awaiting execution."""
        return [
            order for order in self._orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
        ]
    
    def get_summary(self) -> Dict:
        """Get OMS summary statistics."""
        total_orders = len(self._orders)
        active_orders = len(self._active_orders)
        filled_orders = sum(1 for o in self._orders.values() if o.status == OrderStatus.FILLED)
        
        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "symbols_traded": len(self._orders_by_symbol),
            "positions": len(self._positions),
        }
