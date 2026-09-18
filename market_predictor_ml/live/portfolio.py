"""
Portfolio Manager - Manages portfolio allocation, rebalancing, and position tracking.
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
import logging

from .oms import OrderManager, Order, OrderSide, OrderType
from .signals import TradingSignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """Represents a portfolio position."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    
    @classmethod
    def create(
        cls,
        symbol: str,
        quantity: float,
        avg_cost: float,
        current_price: float,
        portfolio_value: float,
    ) -> "PortfolioPosition":
        """Create position from raw data."""
        market_value = quantity * current_price
        cost_basis = quantity * avg_cost
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0.0
        weight = (market_value / portfolio_value * 100) if portfolio_value != 0 else 0.0
        
        return cls(
            symbol=symbol,
            quantity=quantity,
            avg_cost=avg_cost,
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            weight=weight,
        )


class PortfolioManager:
    """
    Manages portfolio allocation and rebalancing.
    
    Features:
    - Track positions and weights
    - Generate rebalance orders
    - Enforce concentration limits
    - Calculate portfolio metrics
    """
    
    def __init__(
        self,
        order_manager: OrderManager,
        max_position_weight: float = 0.25,
        min_position_weight: float = 0.02,
        max_sector_weight: float = 0.40,
        cash_buffer: float = 0.05,
    ):
        """
        Initialize portfolio manager.
        
        Args:
            order_manager: OMS for order creation
            max_position_weight: Maximum weight per position
            min_position_weight: Minimum weight to maintain position
            max_sector_weight: Maximum weight per sector
            cash_buffer: Target cash buffer percentage
        """
        self.order_manager = order_manager
        self.max_position_weight = max_position_weight
        self.min_position_weight = min_position_weight
        self.max_sector_weight = max_sector_weight
        self.cash_buffer = cash_buffer
        
        self._positions: Dict[str, Dict] = {}  # symbol -> {qty, avg_cost}
        self._sector_map: Dict[str, str] = {}  # symbol -> sector
        self._cash_balance: float = 0.0
        self._portfolio_value: float = 0.0
        self._prices: Dict[str, float] = {}
    
    def update_price(self, symbol: str, price: float):
        """Update price for symbol."""
        self._prices[symbol] = price
        self._recalculate_portfolio()
    
    def update_cash(self, cash: float):
        """Update cash balance."""
        self._cash_balance = cash
        self._recalculate_portfolio()
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        avg_cost: float,
        sector: Optional[str] = None,
    ):
        """Update or add position."""
        if quantity == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {
                "quantity": quantity,
                "avg_cost": avg_cost,
            }
            if sector:
                self._sector_map[symbol] = sector
        
        self._recalculate_portfolio()
    
    def _recalculate_portfolio(self):
        """Recalculate portfolio value and weights."""
        positions_value = sum(
            pos["quantity"] * self._prices.get(symbol, 0.0)
            for symbol, pos in self._positions.items()
        )
        self._portfolio_value = positions_value + self._cash_balance
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        return self._portfolio_value
    
    def get_positions(self) -> List[PortfolioPosition]:
        """Get all positions."""
        positions = []
        for symbol, pos_data in self._positions.items():
            price = self._prices.get(symbol, pos_data["avg_cost"])
            position = PortfolioPosition.create(
                symbol=symbol,
                quantity=pos_data["quantity"],
                avg_cost=pos_data["avg_cost"],
                current_price=price,
                portfolio_value=self._portfolio_value,
            )
            positions.append(position)
        return positions
    
    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Get single position."""
        if symbol not in self._positions:
            return None
        
        pos_data = self._positions[symbol]
        price = self._prices.get(symbol, pos_data["avg_cost"])
        
        return PortfolioPosition.create(
            symbol=symbol,
            quantity=pos_data["quantity"],
            avg_cost=pos_data["avg_cost"],
            current_price=price,
            portfolio_value=self._portfolio_value,
        )
    
    def get_weights(self) -> Dict[str, float]:
        """Get position weights."""
        weights = {}
        for symbol, pos_data in self._positions.items():
            price = self._prices.get(symbol, 0.0)
            value = pos_data["quantity"] * price
            weights[symbol] = value / self._portfolio_value if self._portfolio_value > 0 else 0.0
        return weights
    
    def get_cash_weight(self) -> float:
        """Get cash weight."""
        return self._cash_balance / self._portfolio_value if self._portfolio_value > 0 else 0.0
    
    def generate_rebalance_orders(
        self,
        target_weights: Dict[str, float],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[Order]:
        """
        Generate orders to rebalance portfolio to target weights.
        
        Args:
            target_weights: Dictionary of symbol -> target weight
            current_prices: Current prices (uses cached if not provided)
        
        Returns:
            List of orders to execute
        """
        prices = current_prices or self._prices
        orders = []
        
        available_cash = self._cash_balance
        portfolio_value = self._portfolio_value
        
        # Enforce min_position_weight: any held symbol whose target weight is
        # below the minimum is closed (target 0) (L-12).
        targets = dict(target_weights)
        for symbol in self._positions:
            if targets.get(symbol, 0.0) < self.min_position_weight:
                targets[symbol] = 0.0

        # Calculate target values
        for symbol, target_weight in targets.items():
            # Enforce constraints
            target_weight = min(target_weight, self.max_position_weight)
            
            target_value = portfolio_value * target_weight
            current_pos = self._positions.get(symbol, {"quantity": 0, "avg_cost": 0})
            current_qty = current_pos.get("quantity", 0.0)
            price = prices.get(symbol, 0.0)
            
            if price <= 0:
                continue
            
            current_value = current_qty * price
            target_qty = target_value / price
            qty_diff = target_qty - current_qty
            
            # Skip small trades
            if abs(qty_diff) < 1:
                continue
            
            # Create order
            if qty_diff > 0:
                # Buy
                estimated_cost = qty_diff * price
                if estimated_cost <= available_cash:
                    order = self.order_manager.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=qty_diff,
                        order_type=OrderType.MARKET,
                        metadata={"reason": "rebalance"},
                    )
                    orders.append(order)
                    available_cash -= estimated_cost
            else:
                # Sell
                order = self.order_manager.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=abs(qty_diff),
                    order_type=OrderType.MARKET,
                    metadata={"reason": "rebalance"},
                )
                orders.append(order)
                available_cash += abs(qty_diff) * price
        
        return orders
    
    def generate_signal_orders(
        self,
        signal: TradingSignal,
    ) -> Optional[Order]:
        """
        Generate order from trading signal.
        
        Args:
            signal: Trading signal
        
        Returns:
            Order or None
        """
        if not signal.is_actionable:
            return None
        
        # Weight check uses the ABSOLUTE desired position (target_quantity)
        entry_price = signal.entry_price or 0.0
        if self._portfolio_value > 0 and entry_price > 0:
            desired_weight = abs(signal.target_quantity) * entry_price / self._portfolio_value
            if desired_weight > self.max_position_weight:
                logger.warning(
                    f"Signal would exceed max weight for {signal.symbol} "
                    f"(desired {desired_weight:.2%} > limit {self.max_position_weight:.2%})"
                )
                return None
        
        # Create order from the SIGNED delta (side and size both follow the delta)
        if signal.delta_quantity > 0:
            side = OrderSide.BUY
        elif signal.delta_quantity < 0:
            side = OrderSide.SELL
        else:
            return None  # nothing to trade
        
        order = self.order_manager.create_order(
            symbol=signal.symbol,
            side=side,
            quantity=abs(signal.delta_quantity),
            order_type=OrderType.MARKET,
            metadata={
                "reason": "signal",
                "signal_type": signal.signal_type.value,
                "signal_strength": signal.strength,
                "target_quantity": signal.target_quantity,
                "delta_quantity": signal.delta_quantity,
            },
        )
        
        return order
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        positions = self.get_positions()
        long_positions = [p for p in positions if p.quantity > 0]
        
        total_market_value = sum(p.market_value for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        return {
            "portfolio_value": self._portfolio_value,
            "cash": self._cash_balance,
            "cash_weight": self.get_cash_weight(),
            "positions_count": len(positions),
            "long_positions": len(long_positions),
            "total_market_value": total_market_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "top_holdings": sorted(positions, key=lambda p: p.market_value, reverse=True)[:5],
        }
