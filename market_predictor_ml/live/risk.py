"""
Live Risk Monitor - Real-time risk monitoring and circuit breakers.
"""

from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass
class RiskMetric:
    """Represents a risk metric measurement."""
    name: str
    value: float
    threshold: float
    level: RiskLevel
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LiveRiskMonitor:
    """
    Real-time risk monitoring with circuit breakers.
    
    Features:
    - Drawdown monitoring
    - Position concentration limits
    - Daily loss limits
    - Volatility spikes
    - Circuit breaker triggers
    - Risk alerts
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        daily_loss_limit: float = 0.05,
        max_position_size: float = 0.25,
        max_portfolio_volatility: float = 0.30,
        var_limit: float = 0.10,
        max_consecutive_losses: Optional[int] = None,  # Accepted but not yet implemented
        **kwargs  # For future extensibility
    ):
        """
        Initialize risk monitor.
        
        Args:
            max_drawdown: Maximum allowed drawdown from peak
            daily_loss_limit: Maximum daily loss percentage
            max_position_size: Maximum single position weight
            max_portfolio_volatility: Maximum portfolio volatility
            var_limit: Maximum Value at Risk
            max_consecutive_losses: Max consecutive losses before halt (future)
        """
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.max_position_size = max_position_size
        self.max_portfolio_volatility = max_portfolio_volatility
        self.var_limit = var_limit
        self.max_consecutive_losses = max_consecutive_losses
        
        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._daily_start_value: float = 0.0
        self._positions: Dict[str, float] = {}
        self._returns_history: List[float] = []
        self._alerts: List[RiskMetric] = []
        self._callbacks: List[Callable[[RiskMetric], None]] = []
        self._trading_halted: bool = False
        self._last_check: Optional[datetime] = None
    
    def register_callback(self, callback: Callable[[RiskMetric], None]):
        """Register callback for risk alerts."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, metric: RiskMetric):
        """Notify callbacks of risk alert."""
        for callback in self._callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Risk callback error: {e}")
    
    def update_portfolio_value(self, value: float):
        """Update current portfolio value."""
        self._current_value = value
        
        # Update peak
        if value > self._peak_value:
            self._peak_value = value
        
        # Set daily start if not set
        if self._daily_start_value == 0:
            self._daily_start_value = value
    
    def reset_daily(self):
        """Reset daily metrics (call at market open)."""
        self._daily_start_value = self._current_value
        logger.info("Daily risk metrics reset")
    
    def update_positions(self, positions: Dict[str, float], portfolio_value: float):
        """Update position weights."""
        self._positions = {}
        for symbol, qty in positions.items():
            # Would need prices to calculate weights
            # Simplified for now
            pass
    
    def add_return(self, return_pct: float):
        """Add return to history for volatility calculation."""
        self._returns_history.append(return_pct)
        # Keep last 252 trading days
        if len(self._returns_history) > 252:
            self._returns_history = self._returns_history[-252:]
    
    def check_all_risks(self) -> List[RiskMetric]:
        """
        Check all risk metrics.
        
        Returns:
            List of risk metrics that breached thresholds
        """
        breaches = []
        self._last_check = datetime.now()
        
        # Check drawdown
        drawdown_metric = self.check_drawdown()
        if drawdown_metric and drawdown_metric.level != RiskLevel.NORMAL:
            breaches.append(drawdown_metric)
        
        # Check daily loss
        daily_metric = self.check_daily_loss()
        if daily_metric and daily_metric.level != RiskLevel.NORMAL:
            breaches.append(daily_metric)
        
        # Check position concentration
        position_metrics = self.check_position_concentration()
        breaches.extend(position_metrics)
        
        # Check volatility
        vol_metric = self.check_volatility()
        if vol_metric and vol_metric.level != RiskLevel.NORMAL:
            breaches.append(vol_metric)
        
        # Update trading halt status
        critical_breaches = [m for m in breaches if m.level == RiskLevel.HALT]
        if critical_breaches:
            self._trading_halted = True
            logger.critical(f"TRADING HALTED: {len(critical_breaches)} critical breaches")
        elif self._trading_halted and all(m.level != RiskLevel.HALT for m in breaches):
            self._trading_halted = False
            logger.info("Trading halt lifted")
        
        return breaches
    
    def check_drawdown(self) -> Optional[RiskMetric]:
        """Check current drawdown from peak."""
        if self._peak_value == 0:
            return None
        
        drawdown = (self._peak_value - self._current_value) / self._peak_value
        
        if drawdown >= self.max_drawdown:
            level = RiskLevel.HALT
        elif drawdown >= self.max_drawdown * 0.8:
            level = RiskLevel.CRITICAL
        elif drawdown >= self.max_drawdown * 0.5:
            level = RiskLevel.WARNING
        else:
            level = RiskLevel.NORMAL
        
        metric = RiskMetric(
            name="drawdown",
            value=drawdown,
            threshold=self.max_drawdown,
            level=level,
            timestamp=datetime.now(),
            metadata={"peak_value": self._peak_value, "current_value": self._current_value},
        )
        
        if level != RiskLevel.NORMAL:
            self._alerts.append(metric)
            self._notify_callbacks(metric)
            logger.warning(f"Drawdown alert: {drawdown:.2%} (limit: {self.max_drawdown:.2%})")
        
        return metric
    
    def check_daily_loss(self) -> Optional[RiskMetric]:
        """Check daily P&L."""
        if self._daily_start_value == 0:
            return None
        
        daily_return = (self._current_value - self._daily_start_value) / self._daily_start_value
        
        if daily_return <= -self.daily_loss_limit:
            level = RiskLevel.HALT
        elif daily_return <= -self.daily_loss_limit * 0.8:
            level = RiskLevel.CRITICAL
        elif daily_return <= -self.daily_loss_limit * 0.5:
            level = RiskLevel.WARNING
        else:
            level = RiskLevel.NORMAL
        
        metric = RiskMetric(
            name="daily_loss",
            value=daily_return,
            threshold=-self.daily_loss_limit,
            level=level,
            timestamp=datetime.now(),
            metadata={"start_value": self._daily_start_value},
        )
        
        if level != RiskLevel.NORMAL:
            self._alerts.append(metric)
            self._notify_callbacks(metric)
            logger.warning(f"Daily loss alert: {daily_return:.2%} (limit: {-self.daily_loss_limit:.2%})")
        
        return metric
    
    def check_position_concentration(self) -> List[RiskMetric]:
        """Check position concentration limits."""
        metrics = []
        # Simplified - would need actual position weights
        return metrics
    
    def check_volatility(self) -> Optional[RiskMetric]:
        """Check portfolio volatility."""
        if len(self._returns_history) < 20:
            return None
        
        import statistics
        daily_vol = statistics.stdev(self._returns_history)
        annualized_vol = daily_vol * (252 ** 0.5)
        
        if annualized_vol >= self.max_portfolio_volatility:
            level = RiskLevel.WARNING
        elif annualized_vol >= self.max_portfolio_volatility * 0.8:
            level = RiskLevel.WARNING
        else:
            level = RiskLevel.NORMAL
        
        metric = RiskMetric(
            name="volatility",
            value=annualized_vol,
            threshold=self.max_portfolio_volatility,
            level=level,
            timestamp=datetime.now(),
            metadata={"daily_vol": daily_vol, "observations": len(self._returns_history)},
        )
        
        if level != RiskLevel.NORMAL:
            self._alerts.append(metric)
            self._notify_callbacks(metric)
            logger.warning(f"Volatility alert: {annualized_vol:.2%} (limit: {self.max_portfolio_volatility:.2%})")
        
        return metric
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (not halted)."""
        return not self._trading_halted
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        drawdown = (self._peak_value - self._current_value) / self._peak_value if self._peak_value > 0 else 0.0
        daily_return = (self._current_value - self._daily_start_value) / self._daily_start_value if self._daily_start_value > 0 else 0.0
        
        return {
            "trading_halted": self._trading_halted,
            "current_value": self._current_value,
            "peak_value": self._peak_value,
            "drawdown": drawdown,
            "max_drawdown_limit": self.max_drawdown,
            "daily_return": daily_return,
            "daily_loss_limit": -self.daily_loss_limit,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "alerts_count": len(self._alerts),
            "recent_alerts": [
                {"name": a.name, "value": a.value, "level": a.level.value}
                for a in self._alerts[-10:]
            ],
        }
    
    def get_alerts(self, limit: int = 100) -> List[RiskMetric]:
        """Get recent risk alerts."""
        return self._alerts[-limit:]
    
    def clear_alerts(self):
        """Clear alert history."""
        self._alerts = []
        logger.info("Risk alerts cleared")
