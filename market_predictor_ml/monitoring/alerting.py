"""
Alerting Module

Provides alerting system for monitoring anomalies and threshold breaches.
Supports multiple alert channels (email, Slack, webhook) and severity levels.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
import json


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents an alert event."""
    name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class AlertRule:
    """
    Defines a rule for triggering alerts.
    
    Attributes:
        name: Unique rule identifier
        metric_name: Name of the metric to monitor
        condition: Callable that takes metric value and returns True if alert should fire
        severity: Alert severity level
        cooldown_minutes: Minimum time between alerts for this rule
        message_template: Template for alert message (can use {value}, {threshold}, etc.)
        enabled: Whether the rule is active
        threshold: Optional threshold value used in message templates
    """
    name: str
    metric_name: str
    condition: Callable[[float], bool]
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_minutes: int = 60
    message_template: str = "{metric_name} triggered alert: {value}"
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    threshold: Optional[float] = None
    
    def check_and_trigger(self, value: float, context: Optional[Dict[str, Any]] = None) -> Optional[Alert]:
        """
        Check if condition is met and trigger alert if appropriate.
        
        Args:
            value: Current metric value
            context: Additional context for the alert
        
        Returns:
            Alert object if triggered, None otherwise
        """
        if not self.enabled:
            return None
        
        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return None
        
        # Check condition
        if self.condition(value):
            self.last_triggered = datetime.now()
            
            # Format message
            message = self.message_template.format(
                metric_name=self.metric_name,
                value=value,
                threshold=self.threshold,
            )
            
            return Alert(
                name=self.name,
                message=message,
                severity=self.severity,
                context=context or {},
            )
        
        return None


class AlertManager:
    """
    Manages alert rules and dispatches alerts to configured channels.
    
    Features:
    - Multiple alert rules with different conditions
    - Severity-based routing
    - Cooldown periods to prevent alert fatigue
    - Multiple notification channels (console, webhook, email, Slack)
    - Alert history and acknowledgment
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize alert manager.
        
        Args:
            logger: Logger instance for alert logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.handlers: Dict[str, Callable[[Alert], None]] = {}
        
        # Add default console handler
        self.add_handler("console", self._console_handler)
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name."""
        if name in self.rules:
            del self.rules[name]
            self.logger.info(f"Removed alert rule: {name}")
    
    def enable_rule(self, name: str) -> None:
        """Enable an alert rule."""
        if name in self.rules:
            self.rules[name].enabled = True
    
    def disable_rule(self, name: str) -> None:
        """Disable an alert rule."""
        if name in self.rules:
            self.rules[name].enabled = False
    
    def add_handler(self, name: str, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler."""
        self.handlers[name] = handler
        self.logger.info(f"Added alert handler: {name}")
    
    def remove_handler(self, name: str) -> None:
        """Remove a notification handler."""
        if name in self.handlers:
            del self.handlers[name]
    
    def check_metric(self, metric_name: str, value: float, context: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """
        Check all rules for a given metric and trigger alerts if needed.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            context: Additional context
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for rule in self.rules.values():
            if rule.metric_name == metric_name:
                alert = rule.check_and_trigger(value, context)
                if alert:
                    triggered.append(alert)
                    self.alert_history.append(alert)
                    self._dispatch_alert(alert)
        
        return triggered
    
    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert to all registered handlers."""
        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        for name, handler in self.handlers.items():
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler {name} failed: {e}")
    
    def _console_handler(self, alert: Alert) -> None:
        """Default console handler for alerts."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }
        
        emoji = severity_emoji.get(alert.severity, "📢")
        print(f"{emoji} [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
    
    def create_webhook_handler(self, url: str) -> Callable[[Alert], None]:
        """
        Create a webhook handler for sending alerts to external services.
        
        Args:
            url: Webhook URL
        
        Returns:
            Handler function
        """
        def webhook_handler(alert: Alert) -> None:
            try:
                import urllib.request
                
                payload = json.dumps(alert.to_dict()).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={'Content-Type': 'application/json'},
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    self.logger.debug(f"Webhook alert sent: {response.status}")
            except Exception as e:
                self.logger.error(f"Failed to send webhook alert: {e}")
        
        return webhook_handler
    
    def create_slack_handler(self, webhook_url: str, channel: Optional[str] = None) -> Callable[[Alert], None]:
        """
        Create a Slack webhook handler.
        
        Args:
            webhook_url: Slack incoming webhook URL
            channel: Optional channel override
        
        Returns:
            Handler function
        """
        def slack_handler(alert: Alert) -> None:
            try:
                import urllib.request
                
                # Slack color based on severity
                colors = {
                    AlertSeverity.INFO: "#36a64f",
                    AlertSeverity.WARNING: "#ff9800",
                    AlertSeverity.ERROR: "#f44336",
                    AlertSeverity.CRITICAL: "#9c27b0",
                }
                
                payload = {
                    "text": f"*{alert.name}*",
                    "attachments": [
                        {
                            "color": colors.get(alert.severity, "#808080"),
                            "fields": [
                                {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                                {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                                {"title": "Message", "value": alert.message, "short": False},
                            ],
                        }
                    ],
                }
                
                if channel:
                    payload["channel"] = channel
                
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    webhook_url,
                    data=data,
                    headers={'Content-Type': 'application/json'},
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    self.logger.debug(f"Slack alert sent: {response.status}")
            except Exception as e:
                self.logger.error(f"Failed to send Slack alert: {e}")
        
        return slack_handler
    
    def acknowledge_alert(self, alert_name: str) -> bool:
        """Acknowledge an alert."""
        for alert in reversed(self.alert_history):
            if alert.name == alert_name and not alert.acknowledged:
                alert.acknowledged = True
                self.logger.info(f"Acknowledged alert: {alert_name}")
                return True
        return False
    
    def resolve_alert(self, alert_name: str) -> bool:
        """Mark an alert as resolved."""
        for alert in reversed(self.alert_history):
            if alert.name == alert_name and not alert.resolved:
                alert.resolved = True
                self.logger.info(f"Resolved alert: {alert_name}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged and unresolved alerts."""
        return [
            alert for alert in self.alert_history
            if not alert.acknowledged and not alert.resolved
        ]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    @staticmethod
    def threshold_condition(threshold: float, operator: str = ">") -> Callable[[float], bool]:
        """
        Create a threshold condition function.
        
        Args:
            threshold: Threshold value
            operator: Comparison operator (>, <, >=, <=, ==, !=)
        
        Returns:
            Condition function
        """
        operator = operator.strip()
        operators = {
            ">": lambda x: x > threshold,
            "<": lambda x: x < threshold,
            ">=": lambda x: x >= threshold,
            "<=": lambda x: x <= threshold,
            "==": lambda x: x == threshold,
            "!=": lambda x: x != threshold,
        }
        
        if operator not in operators:
            raise ValueError(f"Invalid operator: {operator!r}")
        
        return operators[operator]
    
    @staticmethod
    def range_condition(lower: float, upper: float, inside: bool = True) -> Callable[[float], bool]:
        """
        Create a range condition function.
        
        Args:
            lower: Lower bound
            upper: Upper bound
            inside: If True, trigger when value is inside range; otherwise outside
        
        Returns:
            Condition function
        """
        if inside:
            return lambda x: lower <= x <= upper
        else:
            return lambda x: x < lower or x > upper


# Pre-built alert rules for common scenarios
def create_sharpe_ratio_alert(min_sharpe: float = -1.0) -> AlertRule:
    """Create alert rule for low Sharpe ratio."""
    return AlertRule(
        name="low_sharpe_ratio",
        metric_name="sharpe_ratio",
        condition=AlertManager.threshold_condition(min_sharpe, "<"),
        severity=AlertSeverity.WARNING,
        message_template="Sharpe ratio dropped to {value} (below {threshold})",
        threshold=min_sharpe,
        cooldown_minutes=1440,  # 24 hours
    )


def create_drawdown_alert(max_drawdown: float = -0.1) -> AlertRule:
    """Create alert rule for excessive drawdown."""
    return AlertRule(
        name="max_drawdown_breach",
        metric_name="max_drawdown",
        condition=AlertManager.threshold_condition(max_drawdown, "<"),
        severity=AlertSeverity.CRITICAL,
        message_template="Drawdown reached {value} (breached {threshold})",
        threshold=max_drawdown,
        cooldown_minutes=60,
    )


def create_prediction_error_alert(max_error: float = 0.1) -> AlertRule:
    """Create alert rule for high prediction error."""
    return AlertRule(
        name="high_prediction_error",
        metric_name="prediction_error",
        condition=AlertManager.threshold_condition(max_error, ">"),
        severity=AlertSeverity.WARNING,
        message_template="Prediction error {value} exceeded threshold {threshold}",
        threshold=max_error,
        cooldown_minutes=30,
    )


def create_volume_spike_alert(multiplier: float = 3.0) -> AlertRule:
    """Create alert rule for unusual volume spikes."""
    return AlertRule(
        name="volume_spike",
        metric_name="volume_ratio",
        condition=AlertManager.threshold_condition(multiplier, ">"),
        severity=AlertSeverity.INFO,
        message_template="Volume spike detected: {value}x average",
        threshold=multiplier,
        cooldown_minutes=60,
    )
