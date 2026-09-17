"""
Live Trader orchestrator for real-time signal generation and execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime


class LiveTrader:
    """
    Orchestrates live trading workflow.
    
    - Fetches real-time data
    - Generates features
    - Runs model inference
    - Executes trades via broker API
    """
    
    def __init__(
        self,
        model_pipeline: Any,
        broker_client: Any,
        symbols: List[str],
        max_position_pct: float = 0.1,
        rebalance_frequency: str = "daily",
    ):
        self.model_pipeline = model_pipeline
        self.broker_client = broker_client
        self.symbols = symbols
        self.max_position_pct = max_position_pct
        self.rebalance_frequency = rebalance_frequency
        
        self.last_rebalance = None
        
    def fetch_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch latest market data for all symbols."""
        data = {}
        for symbol in self.symbols:
            df = self.broker_client.get_historical_data(
                symbol,
                timeframe="1Day",
                start="2020-01-01",
            )
            if len(df) > 0:
                data[symbol] = df
        return data
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals for all symbols."""
        signals = {}
        
        for symbol, df in data.items():
            # Run pipeline prediction
            try:
                # Get latest features
                features = self.model_pipeline.feature_factory.transform(df)
                
                # Get model prediction
                pred = self.model_pipeline.model.predict(features.iloc[[-1]])[0]
                
                # Determine signal
                signal = 1 if pred > 0 else (-1 if pred < 0 else 0)
                
                signals[symbol] = {
                    "symbol": symbol,
                    "signal": signal,
                    "prediction": float(pred),
                    "timestamp": datetime.now(),
                    "price": df["close"].iloc[-1] if "close" in df.columns else df["Close"].iloc[-1],
                }
            except Exception as e:
                signals[symbol] = {
                    "symbol": symbol,
                    "signal": 0,
                    "error": str(e),
                    "timestamp": datetime.now(),
                }
        
        return signals
    
    def execute_trades(self, signals: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trades based on signals."""
        executed = []
        
        # Get account info for position sizing
        account = self.broker_client.get_account()
        portfolio_value = account["portfolio_value"]
        
        for symbol, signal_info in signals.items():
            signal = signal_info.get("signal", 0)
            
            if signal == 0:
                continue
            
            # Calculate position size
            target_value = portfolio_value * self.max_position_pct
            price = signal_info.get("price", 0)
            
            if price <= 0:
                continue
            
            qty = int(target_value / price)
            if qty <= 0:
                continue
            
            # Submit order
            side = "buy" if signal > 0 else "sell"
            try:
                order = self.broker_client.submit_market_order(symbol, qty, side)
                executed.append({
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": order.get("status"),
                    "order_id": order.get("id"),
                })
            except Exception as e:
                executed.append({
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "error": str(e),
                })
        
        return executed
    
    def run_cycle(self) -> Dict[str, Any]:
        """Run one complete trading cycle."""
        # Check if rebalance is needed
        now = datetime.now()
        if self.last_rebalance and (now - self.last_rebalance).days < 1:
            return {"status": "skipped", "reason": "rebalance frequency not met"}
        
        # Fetch data
        data = self.fetch_latest_data()
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Execute trades
        executed = self.execute_trades(signals)
        
        self.last_rebalance = now
        
        return {
            "status": "success",
            "timestamp": now,
            "signals": signals,
            "executed_trades": executed,
        }


class PaperTrader(LiveTrader):
    """
    Paper trading mode - simulates execution without real orders.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paper_portfolio = {"cash": 100000.0, "positions": {}}
        self.trade_log = []
    
    def execute_trades(self, signals: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate trade execution in paper mode."""
        executed = []
        
        for symbol, signal_info in signals.items():
            signal = signal_info.get("signal", 0)
            
            if signal == 0:
                continue
            
            price = signal_info.get("price", 0)
            if price <= 0:
                continue
            
            # Simulate position update
            if signal > 0:  # Buy
                self.paper_portfolio["positions"][symbol] = {
                    "qty": self.paper_portfolio["positions"].get(symbol, {}).get("qty", 0) + 10,
                    "avg_price": price,
                }
            else:  # Sell
                if symbol in self.paper_portfolio["positions"]:
                    del self.paper_portfolio["positions"][symbol]
            
            executed.append({
                "symbol": symbol,
                "side": "buy" if signal > 0 else "sell",
                "price": price,
                "status": "simulated",
            })
            
            # Log trade
            self.trade_log.append({
                "timestamp": signal_info.get("timestamp"),
                "symbol": symbol,
                "side": "buy" if signal > 0 else "sell",
                "price": price,
                "signal": signal,
            })
        
        return executed
