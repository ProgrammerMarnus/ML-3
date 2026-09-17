"""Live Trading Module - Event-driven trading engine, OMS, and broker adapters."""

from .engine import TradingEngine, TradingState
from .oms import Order, OrderManager, OrderStatus
from .signals import SignalGenerator
from .portfolio import PortfolioManager
from .risk import LiveRiskMonitor
from .broker_base import BrokerAdapter
from .brokers import AlpacaBroker, InteractiveBrokersBroker, PaperBroker

# Keep existing imports for backward compatibility
try:
    from .alpaca_client import AlpacaClient
    from .paper_client import AlpacaPaperClient, is_alpaca_available
    from .rl_policy import RLPolicy, is_rl_available
    from .gnn_features import GNNFeatureAugmenter, is_gnn_available
    from .trader import LiveTrader, PaperTrader
except ImportError:
    pass

__all__ = [
    # New architecture components
    "TradingEngine",
    "TradingState",
    "Order",
    "OrderManager",
    "OrderStatus",
    "SignalGenerator",
    "PortfolioManager",
    "LiveRiskMonitor",
    "BrokerAdapter",
    "AlpacaBroker",
    "InteractiveBrokersBroker",
    "PaperBroker",
    # Legacy compatibility
    "AlpacaClient",
    "AlpacaPaperClient",
    "is_alpaca_available",
    "RLPolicy",
    "is_rl_available",
    "GNNFeatureAugmenter",
    "is_gnn_available",
    "LiveTrader",
    "PaperTrader",
]
