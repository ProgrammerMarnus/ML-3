"""
Trading Engine - Event-driven orchestration layer for live trading.
"""

from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from enum import Enum
import logging
import time
import threading
import queue

from .oms import OrderManager, Order, OrderStatus
from .signals import SignalGenerator, TradingSignal
from .portfolio import PortfolioManager
from .risk import LiveRiskMonitor
from .broker_base import BrokerAdapter
from .brokers import PaperBroker

logger = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading engine states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    LIQUIDATING = "liquidating"
    ERROR = "error"


class MarketEvent:
    """Represents a market data event."""
    def __init__(self, symbol: str, price: float, timestamp: datetime, volume: float = 0):
        self.symbol = symbol
        self.price = price
        self.timestamp = timestamp
        self.volume = volume


class SignalEvent:
    """Represents a trading signal event."""
    def __init__(self, signal: TradingSignal):
        self.signal = signal
        self.timestamp = datetime.now()


class OrderEvent:
    """Represents an order status change event."""
    def __init__(self, order: Order, old_status: OrderStatus, new_status: OrderStatus):
        self.order = order
        self.old_status = old_status
        self.new_status = new_status
        self.timestamp = datetime.now()


class TradingEngine:
    """
    Event-driven trading engine orchestrating all components.
    
    Features:
    - Event loop for processing market data
    - Signal generation from predictions
    - Order management and execution
    - Risk monitoring with circuit breakers
    - State management (start/stop/pause)
    - Callback system for extensibility
    """
    
    def __init__(
        self,
        broker: BrokerAdapter,
        signal_generator: SignalGenerator,
        portfolio_manager: PortfolioManager,
        risk_monitor: LiveRiskMonitor,
        symbols: List[str],
        config: Optional[Dict] = None,
    ):
        """
        Initialize trading engine.
        
        Args:
            broker: Broker adapter for order execution
            signal_generator: Signal generator from predictions
            portfolio_manager: Portfolio allocation manager
            risk_monitor: Real-time risk monitor
            symbols: List of symbols to trade
            config: Additional configuration
        """
        self.broker = broker
        self.signal_generator = signal_generator
        self.portfolio_manager = portfolio_manager
        self.risk_monitor = risk_monitor
        self.symbols = symbols
        self.config = config or {}
        
        # Order management
        self.order_manager = OrderManager()
        self.order_manager.register_callback(self._on_order_update)
        
        # State
        self._state = TradingState.STOPPED
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        
        # Event queues (thread-safe: producers run on caller threads)
        self._market_events: "queue.Queue" = queue.Queue()
        self._signal_events: "queue.Queue" = queue.Queue()
        
        # Callbacks
        self._event_callbacks: List[Callable] = []
        
        # Metrics
        self._events_processed = 0
        self._orders_generated = 0
        self._last_heartbeat: Optional[datetime] = None
        self._last_portfolio_value: float = 0.0
    
    def register_callback(self, callback: Callable):
        """Register callback for events."""
        self._event_callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """Notify all callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def start(self):
        """Start the trading engine."""
        if self._state != TradingState.STOPPED:
            logger.warning(f"Cannot start from state: {self._state}")
            return False
        
        logger.info("Starting trading engine...")
        self._state = TradingState.INITIALIZING
        
        # Connect to broker
        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            self._state = TradingState.ERROR
            return False
        
        # Initialize portfolio with current positions
        self._sync_positions()
        
        # Reset daily risk metrics
        self.risk_monitor.reset_daily()
        
        self._running = True
        self._state = TradingState.RUNNING
        self._last_heartbeat = datetime.now()
        
        # Start event loop in background thread
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        
        logger.info("Trading engine started")
        return True
    
    def stop(self):
        """Stop the trading engine."""
        logger.info("Stopping trading engine...")
        self._running = False
        self._state = TradingState.STOPPED
        
        # Join the event-loop thread (it blocks on the queue with a 0.1 s
        # timeout, so it exits promptly once _running is False)
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)
        
        # Disconnect from broker
        self.broker.disconnect()
        
        logger.info("Trading engine stopped")
    
    def pause(self):
        """Pause trading (no new orders)."""
        self._state = TradingState.PAUSED
        logger.info("Trading engine paused")
    
    def resume(self):
        """Resume trading after pause."""
        if self._state == TradingState.PAUSED:
            self._state = TradingState.RUNNING
            logger.info("Trading engine resumed")
    
    def _sync_positions(self):
        """Sync internal state with broker positions."""
        try:
            positions = self.broker.get_positions()
            account_info = self.broker.get_account_info()
            
            cash = account_info.get("cash", 0.0)
            self.portfolio_manager.update_cash(cash)
            
            for symbol, qty in positions.items():
                # Would need avg cost from broker
                self.portfolio_manager.update_position(symbol, qty, 0.0)
            
            logger.info(f"Synced {len(positions)} positions, cash: ${cash:,.2f}")
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
    
    def _run_event_loop(self):
        """Main event loop running in background thread."""
        logger.info("Event loop started")
        risk_check_interval = 1.0  # seconds between periodic risk checks
        last_risk_check = 0.0

        while self._running:
            try:
                # Block until a market event arrives (or timeout so risk checks
                # and shutdown stay responsive). No busy polling.
                try:
                    event = self._market_events.get(timeout=0.1)
                except queue.Empty:
                    event = None

                if event is not None:
                    self._process_market_event(event)

                # Drain any signal events that arrived
                while True:
                    try:
                        signal_event = self._signal_events.get_nowait()
                    except queue.Empty:
                        break
                    self._process_signal_event(signal_event)

                # Check risk metrics periodically
                now = time.monotonic()
                if now - last_risk_check >= risk_check_interval:
                    self._check_risk()
                    last_risk_check = now

                # Update heartbeat
                self._last_heartbeat = datetime.now()

            except Exception as e:
                logger.error(f"Event loop error: {e}", exc_info=True)

        logger.info("Event loop stopped")
    
    def on_market_data(self, symbol: str, price: float, timestamp: datetime, volume: float = 0):
        """Handle incoming market data."""
        if self._state != TradingState.RUNNING:
            return
        
        event = MarketEvent(symbol, price, timestamp, volume)
        self._market_events.put(event)
    
    def on_prediction(self, symbol: str, prediction: float):
        """Handle new model prediction."""
        if self._state != TradingState.RUNNING:
            return
        
        # Get current data
        current_price = self.broker.get_market_price(symbol)
        if current_price is None:
            return
        
        current_position = self.order_manager.get_position(symbol)
        account_value = self.risk_monitor.current_value or 100000.0
        
        # Generate signal
        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            prediction=prediction,
            current_price=current_price,
            volatility=0.02,  # Would get from real data
            current_position=current_position,
            account_value=account_value,
        )
        
        if signal.is_actionable:
            event = SignalEvent(signal)
            self._signal_events.put(event)
    
    def _process_market_event(self, event: MarketEvent):
        """Process market data event."""
        self._events_processed += 1
        
        # Update portfolio prices
        self.portfolio_manager.update_price(event.symbol, event.price)
        
        # Forward the price to the broker via the adapter interface
        # (PaperBroker tracks it for fill simulation; others ignore it)
        try:
            self.broker.update_price(event.symbol, event.price)
        except Exception:  # never let a price-book update break the loop
            logger.debug("broker.update_price failed", exc_info=True)
        
        # Update risk monitor (feed returns so volatility/VaR checks have data)
        portfolio_value = self.portfolio_manager.get_portfolio_value()
        prev_value = self._last_portfolio_value
        if prev_value > 0:
            self.risk_monitor.add_return((portfolio_value - prev_value) / prev_value)
        self.risk_monitor.update_portfolio_value(portfolio_value)
        self._last_portfolio_value = portfolio_value
        
        self._notify_callbacks("market_data", event)
    
    def _process_signal_event(self, event: SignalEvent):
        """Process trading signal event."""
        signal = event.signal
        
        # Check if trading allowed
        if not self.risk_monitor.is_trading_allowed():
            logger.warning(f"Trading halted, ignoring signal for {signal.symbol}")
            return
        
        # Generate order from signal
        order = self.portfolio_manager.generate_signal_orders(signal)
        
        if order:
            self._submit_order(order)
            self._orders_generated += 1
    
    def _submit_order(self, order: Order):
        """Submit order to broker."""
        # Validate
        is_valid, error_msg = self.broker.validate_order(order)
        if not is_valid:
            logger.error(f"Order validation failed: {error_msg}")
            self.order_manager.reject_order(order.order_id, error_msg)
            return
        
        # Submit to OMS
        self.order_manager.submit_order(order.order_id)
        
        # Submit to broker
        success = self.broker.submit_order(order)
        
        if not success:
            logger.error("Broker rejected order")
            self.order_manager.reject_order(order.order_id, "Broker rejection")
    
    def _on_order_update(self, order: Order):
        """Handle order status updates."""
        self._notify_callbacks("order_update", order)
        
        if order.is_terminal:
            logger.info(f"Order {order.order_id[:8]} reached terminal state: {order.status.value}")
    
    def _check_risk(self):
        """Periodic risk check."""
        breaches = self.risk_monitor.check_all_risks()
        
        if breaches:
            critical = [b for b in breaches if b.level.value in ["critical", "halt"]]
            if critical:
                logger.warning(f"Risk breaches detected: {len(critical)}")
                self._notify_callbacks("risk_alert", critical)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status summary."""
        return {
            "state": self._state.value,
            "running": self._running,
            "symbols": self.symbols,
            "events_processed": self._events_processed,
            "orders_generated": self._orders_generated,
            "active_orders": len(self.order_manager.get_active_orders()),
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "risk_status": self.risk_monitor.get_risk_status(),
            "portfolio_summary": self.portfolio_manager.get_summary(),
            "oms_summary": self.order_manager.get_summary(),
        }
