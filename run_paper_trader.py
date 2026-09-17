#!/usr/bin/env python3
"""
Paper Trading Entry Point

Main script to start the paper trading system.
Loads configuration, initializes all components, and runs the trading loop.

Usage:
    python run_paper_trader.py --config paper_trading.yaml
    python run_paper_trader.py --config paper_trading.yaml --dry-run
"""

import argparse
import logging
import signal
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from market_predictor_ml.live.predictor import LivePredictor
from market_predictor_ml.live.engine import TradingEngine
from market_predictor_ml.live.brokers import create_broker
from market_predictor_ml.live.oms import OrderManager
from market_predictor_ml.live.portfolio import PortfolioManager
from market_predictor_ml.live.risk import LiveRiskMonitor
from market_predictor_ml.live.state_store import StateStore, create_state_store
from market_predictor_ml.live.signals import SignalGenerator
from market_predictor_ml.monitoring.logger import setup_structured_logging
from market_predictor_ml.monitoring.metrics import MetricsCollector
from market_predictor_ml.monitoring.alerting import AlertManager, AlertRule, AlertSeverity
from market_predictor_ml.data.providers import MarketDataLoader, create_data_provider
from market_predictor_ml.features.pipeline import (
    FeaturePipeline,
    MomentumTransformer,
    TechnicalIndicatorTransformer,
    TimeFeatureTransformer,
    VolatilityTransformer,
    VolumeTransformer,
    create_default_pipeline,
)
from market_predictor_ml.models.model_registry import InMemoryModelRegistry


logger = logging.getLogger(__name__)


class PaperTradingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self._shutdown_requested = False
        self.state_store = None
        self.data_loader = None
        self.feature_pipeline = None
        self.order_manager = None
        self.initial_cash = 0.0
        self.model_registry = None
        self.predictor = None
        self.signal_generator = None
        self.portfolio_manager = None
        self.risk_monitor = None
        self.broker = None
        self.trading_engine = None
        self.metrics_collector = None
        self.alert_manager = None
        self.session_id = None
    
    def _initialize_components(self):
        logger.info("Initializing trading system components...")
        
        log_config = self.config.get("monitoring", {})
        log_level = getattr(logging, log_config.get("log_level", "INFO"))
        log_file = log_config.get("log_file", "paper_trading.log")
        json_logging = log_config.get("json_logging", True)
        
        setup_structured_logging(level=log_level, log_file=log_file, json_format=json_logging)
        logger.info(f"Logging initialized (level={log_level}, file={log_file})")
        
        state_config = self.config.get("state_store", {})
        db_path = state_config.get("db_path", "trading_state.db")
        self.state_store = create_state_store(db_path=db_path)
        logger.info(f"State store initialized at {db_path}")
        
        self._load_previous_state()
        
        data_config = self.config.get("data", {})
        provider_type = data_config.get("provider", "yfinance")
        cache_dir = data_config.get("cache_dir", "./data_cache")
        lookback_days = data_config.get("lookback_days", 60)
        
        self.data_loader = MarketDataLoader(
            
            provider=create_data_provider(provider_type)
        )
        logger.info(f"Data loader initialized (provider={provider_type})")
        
        features_config = self.config.get("features", {}).get("pipeline_config", {})
        self.feature_pipeline = self._create_feature_pipeline(features_config)
        logger.info("Feature pipeline initialized")
        
        models_config = self.config.get("models", {})
        registry_path = models_config.get("registry_path", "./model_registry")
        self.model_registry = InMemoryModelRegistry(storage_path=registry_path)
        logger.info(f"Model registry initialized at {registry_path}")
        
        general_config = self.config.get("general", {})
        symbols = general_config.get("symbols", ["SPY"])
        model_mapping = models_config.get("symbol_model_mapping", {})
        
        if not model_mapping:
            default_model_id = models_config.get("default_model_id", "default_model")
            model_mapping = {symbol: default_model_id for symbol in symbols}
        
        self.predictor = LivePredictor(
            data_loader=self.data_loader,
            feature_pipeline=self.feature_pipeline,
            model_registry=self.model_registry,
            symbols=symbols,
            model_ids=model_mapping,
            min_confidence=models_config.get("min_confidence", 0.65),
            prediction_threshold=models_config.get("prediction_threshold", 0.02),
            lookback_days=lookback_days,
            check_market_hours=general_config.get("check_market_hours", True)
        )
        logger.info(f"Predictor initialized for {len(symbols)} symbols")
        
        self.signal_generator = SignalGenerator()
        logger.info("Signal generator initialized")
        
        portfolio_config = self.config.get("portfolio", {})
        initial_cash = portfolio_config.get("initial_cash", 100000.0)
        
        self.order_manager = OrderManager()
        self.portfolio_manager = PortfolioManager(
            order_manager=self.order_manager,
            max_position_weight=portfolio_config.get("max_position_pct", 0.25),
            cash_buffer=max(0.0, 1.0 - portfolio_config.get("max_total_exposure", 1.0)),
        )
        self.portfolio_manager.update_cash(initial_cash)
        logger.info(f"Portfolio manager initialized with ${initial_cash:,.2f}")
        
        risk_config = self.config.get("risk", {})
        self.risk_monitor = LiveRiskMonitor(
            max_drawdown=risk_config.get("max_drawdown_pct", 0.10),
            daily_loss_limit=risk_config.get("daily_loss_limit_pct", 0.03),
            max_position_size=risk_config.get("concentration_limit", 0.25),
            var_limit=risk_config.get("max_var_pct", 0.05),
        )
        if risk_config.get("halt_trading_on_breach", True) is False:
            logger.warning(
                "halt_trading_on_breach=false is not supported: LiveRiskMonitor "
                "always halts trading on a breach"
            )
        logger.info("Risk monitor initialized")
        
        broker_config = dict(self.config.get("broker", {}))
        adapter_type = broker_config.get("adapter", "paper")
        broker_config.setdefault("initial_cash", initial_cash)
        if self.config.get("general", {}).get("dry_run", False):
            if adapter_type != "paper":
                logger.warning(
                    f"dry_run=True: overriding broker adapter '{adapter_type}' with the "
                    "simulated PaperBroker (no orders reach a real venue)"
                )
            adapter_type = "paper"
        
        self.broker = create_broker(adapter_type=adapter_type, config=broker_config)
        logger.info(f"Broker initialized (adapter={adapter_type})")
        
        if self.config.get("monitoring", {}).get("metrics_enabled", True):
            self.metrics_collector = MetricsCollector()
            logger.info("Metrics collector initialized")
        
        if self.config.get("monitoring", {}).get("alerts_enabled", True):
            self.alert_manager = AlertManager()
            self._setup_alert_rules()
            logger.info("Alert manager initialized")
        
        self.trading_engine = TradingEngine(
            broker=self.broker,
            signal_generator=self.signal_generator,
            portfolio_manager=self.portfolio_manager,
            risk_monitor=self.risk_monitor,
            symbols=symbols,
            config=self.config,
        )
        logger.info("Trading engine initialized")
        logger.info("All components initialized successfully")
    
    def _create_feature_pipeline(self, config: Dict) -> FeaturePipeline:
        """Build a FeaturePipeline from the ``features.pipeline_config`` block.

        Each transformer group can be toggled with its ``enabled`` flag; the
        time features stay off by default because in live trading the timestamp
        is known but the label horizon is not.
        """
        transformers = []

        momentum = config.get("momentum", {})
        if momentum.get("enabled", True):
            transformers.append(MomentumTransformer(
                periods=momentum.get("periods", [5, 10, 21])
            ))

        volatility = config.get("volatility", {})
        if volatility.get("enabled", True):
            transformers.append(VolatilityTransformer(
                windows=volatility.get("periods", [10, 21])
            ))

        volume = config.get("volume", {})
        if volume.get("enabled", True):
            transformers.append(VolumeTransformer(
                windows=volume.get("periods", [5, 10])
            ))

        technical = config.get("technical_indicators", {})
        if technical.get("enabled", True):
            transformers.append(TechnicalIndicatorTransformer(
                rsi_period=technical.get("rsi_period", 14)
            ))

        time_features = config.get("time_features", {})
        if time_features.get("enabled", False):
            transformers.append(TimeFeatureTransformer())

        if not transformers:
            logger.warning("No feature transformers enabled; falling back to defaults")
            transformers = create_default_pipeline().transformers

        return FeaturePipeline(transformers=transformers, name="paper_trading_pipeline")
    
    def _load_previous_state(self):
        try:
            portfolio_state = self.state_store.load_portfolio_state()
            if portfolio_state:
                logger.info(f"Loaded previous portfolio state: cash=${portfolio_state.get('current_cash', 0):,.2f}")
            
            positions = self.state_store.load_positions()
            if positions:
                logger.info(f"Loaded {len(positions)} previous positions")
            
            risk_state = self.state_store.load_risk_state()
            if risk_state and risk_state.get("trading_halted"):
                logger.warning(f"Previous session ended with trading halted: {risk_state.get('halt_reason')}")
            
            active_session = self.state_store.get_active_session()
            if active_session:
                self.session_id = active_session["id"]
                logger.info(f"Continuing active session {self.session_id}")
        except Exception as e:
            logger.error(f"Error loading previous state: {e}")
    
    def _setup_alert_rules(self):
        if not self.alert_manager:
            return
        risk_config = self.config.get("risk", {})
        self.alert_manager.add_rule(AlertRule(
            name="max_drawdown_breach",
            metric_name="drawdown_pct",
            condition=AlertManager.threshold_condition(
                risk_config.get("max_drawdown_pct", 0.10), ">"
            ),
            severity=AlertSeverity.ERROR,
            cooldown_minutes=5,
        ))
    
    def _save_current_state(self):
        try:
            if self.portfolio_manager:
                portfolio_data = self.portfolio_manager.get_summary()
                self.state_store.save_portfolio_state(
                    initial_cash=self.initial_cash,
                    current_cash=portfolio_data.get("cash", 0.0),
                    total_equity=portfolio_data.get("portfolio_value", 0.0),
                    realized_pnl=0.0,
                    unrealized_pnl=portfolio_data.get("total_unrealized_pnl", 0.0),
                )
                positions = {
                    pos.symbol: {
                        "quantity": pos.quantity,
                        "avg_cost": pos.avg_cost,
                        "current_price": pos.current_price,
                    }
                    for pos in self.portfolio_manager.get_positions()
                }
                self.state_store.save_positions(positions)
            
            if self.risk_monitor:
                risk_status = self.risk_monitor.get_risk_status()
                halted = bool(risk_status.get("trading_halted", False))
                self.state_store.save_risk_state(
                    trading_halted=halted,
                    halt_reason="max_drawdown_breach" if halted else "",
                    max_drawdown_pct=risk_status.get("max_drawdown_limit", 0.0),
                    current_drawdown_pct=risk_status.get("drawdown", 0.0),
                    daily_pnl=risk_status.get("daily_return", 0.0),
                    peak_equity=risk_status.get("peak_value", 0.0),
                )
            logger.debug("State saved successfully")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _trading_loop(self):
        logger.info("Starting trading loop...")
        auto_save_interval = self.config.get("state_store", {}).get("auto_save_interval_seconds", 60)
        last_save_time = time.time()
        
        while self.running and not self._shutdown_requested:
            try:
                current_time = time.time()
                
                risk_status = self.risk_monitor.get_risk_status()
                if risk_status.get("trading_halted"):
                    logger.warning(
                        f"Trading halted: drawdown {risk_status.get('drawdown', 0.0):.2%} "
                        f">= limit {risk_status.get('max_drawdown_limit', 0.0):.2%}"
                    )
                    time.sleep(60)
                    continue
                
                predictions = self.predictor.get_latest_predictions()
                logger.info(f"Generated {len(predictions)} predictions")
                
                for symbol, pred in predictions.items():
                    price = pred.raw_data.get("close")
                    if price:
                        self.trading_engine.on_market_data(
                            symbol, float(price), datetime.now()
                        )
                    self.trading_engine.on_prediction(symbol, pred.prediction)
                
                if current_time - last_save_time > auto_save_interval:
                    self._save_current_state()
                    last_save_time = current_time
                
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(60)
        
        logger.info("Trading loop stopped")
    
    def start(self):
        logger.info("=" * 60)
        logger.info("Starting Paper Trading System")
        logger.info("=" * 60)
        
        self._initialize_components()
        
        portfolio_config = self.config.get("portfolio", {})
        initial_cash = portfolio_config.get("initial_cash", 100000.0)
        
        self.initial_cash = initial_cash

        if self.session_id is None:
            self.session_id = self.state_store.start_session(initial_cash)
            logger.info(f"New trading session started: ID={self.session_id}")
        
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if not self.trading_engine.start():
            raise RuntimeError(
                "Trading engine failed to start (check the broker configuration)"
            )
        
        try:
            self._trading_loop()
        finally:
            self.stop()
    
    def stop(self):
        logger.info("Stopping paper trading system...")
        self.running = False
        self._shutdown_requested = True
        self._save_current_state()

        if self.trading_engine:
            try:
                self.trading_engine.stop()
            except Exception as e:
                logger.error(f"Error stopping trading engine: {e}")
        
        if self.session_id and self.state_store:
            try:
                portfolio_data = self.portfolio_manager.get_summary() if self.portfolio_manager else {}
                total_pnl = portfolio_data.get("total_unrealized_pnl", 0.0)
                total_trades = len(self.state_store.get_trade_log())
                self.state_store.end_session(
                    session_id=self.session_id,
                    final_cash=portfolio_data.get("cash", 0.0),
                    total_trades=total_trades,
                    total_pnl=total_pnl
                )
                logger.info(f"Trading session {self.session_id} ended")
            except Exception as e:
                logger.error(f"Error ending session: {e}")
        
        if self.state_store:
            self.state_store.close()
        logger.info("Paper trading system stopped")
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_requested = True


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Market Predictor ML - Paper Trading System")
    parser.add_argument("--config", "-c", type=str, default="paper_trading.yaml", help="Path to configuration YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        if args.dry_run:
            config.setdefault("general", {})["dry_run"] = True
            logger.info("Running in dry-run mode")
        
        system = PaperTradingSystem(config)
        system.start()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Copy paper_trading.example.yaml to paper_trading.yaml and customize it")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
