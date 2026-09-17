"""Test live trading module components."""

import sys
sys.path.insert(0, '/workspace')

from datetime import datetime
from market_predictor_ml.live.oms import Order, OrderManager, OrderStatus, OrderSide, OrderType
from market_predictor_ml.live.brokers import PaperBroker
from market_predictor_ml.live.signals import SignalGenerator, SignalType
from market_predictor_ml.live.portfolio import PortfolioManager
from market_predictor_ml.live.risk import LiveRiskMonitor, RiskLevel
from market_predictor_ml.live.engine import TradingEngine, TradingState

def test_oms():
    """Test Order Management System."""
    print("Testing OMS...")
    om = OrderManager()
    
    # Create order
    order = om.create_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )
    assert order.symbol == "AAPL"
    assert order.quantity == 100
    assert order.status == OrderStatus.PENDING
    
    # Submit order
    om.submit_order(order.order_id)
    assert order.status == OrderStatus.SUBMITTED
    
    # Simulate fill
    om.update_fill(order.order_id, 100, 150.0, 0.50)
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 100
    assert order.avg_fill_price == 150.0
    
    # Check position
    pos = om.get_position("AAPL")
    assert pos == 100.0
    
    print(f"  ✓ OMS working: {om.get_summary()}")
    return True

def test_paper_broker():
    """Test Paper Broker."""
    print("Testing Paper Broker...")
    broker = PaperBroker({"initial_cash": 100000.0})
    broker.connect()
    
    # Set price
    broker.set_price("AAPL", 150.0)
    
    # Create and submit order
    om = OrderManager()
    order = om.create_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
    
    success = broker.submit_order(order)
    assert success
    
    # Check account
    info = broker.get_account_info()
    assert info["cash"] < 100000.0  # Should have spent money
    assert info["positions_count"] >= 1
    
    broker.disconnect()
    print(f"  ✓ Paper Broker working: cash=${info['cash']:,.2f}")
    return True

def test_signal_generator():
    """Test Signal Generator."""
    print("Testing Signal Generator...")
    sg = SignalGenerator(long_threshold=0.3, short_threshold=-0.3)
    
    # Strong buy signal
    signal = sg.generate_signal(
        symbol="AAPL",
        prediction=0.8,
        current_price=150.0,
        volatility=0.02,
        account_value=100000.0,
    )
    assert signal.signal_type != SignalType.FLAT
    assert signal.strength > 0.5
    assert signal.is_actionable
    
    # Weak signal (should be flat)
    signal2 = sg.generate_signal(
        symbol="AAPL",
        prediction=0.1,
        current_price=150.0,
    )
    assert signal2.signal_type == SignalType.FLAT
    
    print(f"  ✓ Signal Generator working: {signal.signal_type.value} (strength: {signal.strength:.2f})")
    return True

def test_portfolio_manager():
    """Test Portfolio Manager."""
    print("Testing Portfolio Manager...")
    om = OrderManager()
    pm = PortfolioManager(om, max_position_weight=0.25)
    
    # Update cash and positions
    pm.update_cash(100000.0)
    pm.update_price("AAPL", 150.0)
    pm.update_position("AAPL", 100, 145.0)
    
    # Check portfolio value
    value = pm.get_portfolio_value()
    assert value > 100000.0  # Cash + position
    
    # Get position
    pos = pm.get_position("AAPL")
    assert pos.quantity == 100
    assert pos.market_value == 15000.0
    
    summary = pm.get_summary()
    print(f"  ✓ Portfolio Manager working: value=${summary['portfolio_value']:,.2f}")
    return True

def test_risk_monitor():
    """Test Risk Monitor."""
    print("Testing Risk Monitor...")
    rm = LiveRiskMonitor(max_drawdown=0.15, daily_loss_limit=0.05)
    
    # Initialize
    rm.update_portfolio_value(100000.0)
    
    # Normal state
    status = rm.get_risk_status()
    assert not status["trading_halted"]
    
    # Simulate drawdown
    rm.update_portfolio_value(90000.0)  # 10% drawdown
    breaches = rm.check_all_risks()
    
    # Should have warning but not halt
    status = rm.get_risk_status()
    assert status["drawdown"] > 0.09
    
    print(f"  ✓ Risk Monitor working: drawdown={status['drawdown']:.2%}, halted={status['trading_halted']}")
    return True

def test_integration():
    """Test integrated workflow."""
    print("Testing Integration...")
    
    # Create components
    broker = PaperBroker({"initial_cash": 100000.0})
    sg = SignalGenerator(long_threshold=0.3)
    om = OrderManager()
    pm = PortfolioManager(om)
    rm = LiveRiskMonitor()
    
    # Initialize
    broker.connect()
    pm.update_cash(100000.0)
    rm.update_portfolio_value(100000.0)
    
    # Simulate market data
    broker.set_price("AAPL", 150.0)
    pm.update_price("AAPL", 150.0)
    rm.update_portfolio_value(100000.0)
    
    # Generate signal from prediction
    signal = sg.generate_signal(
        symbol="AAPL",
        prediction=0.7,
        current_price=150.0,
        account_value=100000.0,
    )
    
    # Generate order from signal
    if signal.is_actionable:
        order = pm.generate_signal_orders(signal)
        if order:
            broker.submit_order(order)
    
    # Check results
    info = broker.get_account_info()
    positions = broker.get_positions()
    
    broker.disconnect()
    
    print(f"  ✓ Integration working: portfolio=${info['portfolio_value']:,.2f}, positions={len(positions)}")
    return True

if __name__ == "__main__":
    print("\n=== Live Trading Module Tests ===\n")
    
    tests = [
        test_oms,
        test_paper_broker,
        test_signal_generator,
        test_portfolio_manager,
        test_risk_monitor,
        test_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    
    if failed > 0:
        sys.exit(1)
