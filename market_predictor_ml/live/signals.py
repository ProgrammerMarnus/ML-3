"""
Signal Generator - Converts model predictions into trading signals.
"""

from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Type of trading signal."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    INCREASE_LONG = "increase_long"
    DECREASE_LONG = "decrease_long"
    INCREASE_SHORT = "increase_short"
    DECREASE_SHORT = "decrease_short"


@dataclass
class TradingSignal:
    """
    Represents a trading signal generated from model predictions.
    
    Attributes:
        symbol: Ticker symbol
        signal_type: Type of signal (long, short, flat, etc.)
        strength: Signal confidence/strength (-1 to 1 or 0 to 1)
        target_quantity: Desired position size
        entry_price: Suggested entry price
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        metadata: Additional information
    """
    symbol: str
    signal_type: SignalType
    strength: float
    target_quantity: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal requires action."""
        return self.signal_type != SignalType.FLAT and abs(self.strength) > 0.1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "target_quantity": self.target_quantity,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


class SignalGenerator:
    """
    Generates trading signals from model predictions.
    
    Features:
    - Convert raw predictions to signals
    - Apply signal filtering and smoothing
    - Generate entry/exit levels
    - Track signal history
    """
    
    def __init__(
        self,
        long_threshold: float = 0.5,
        short_threshold: float = -0.5,
        min_strength: float = 0.1,
        signal_smoothing: int = 1,
    ):
        """
        Initialize signal generator.
        
        Args:
            long_threshold: Prediction threshold for long signals
            short_threshold: Prediction threshold for short signals
            min_strength: Minimum signal strength to act on
            signal_smoothing: Number of periods to smooth signals
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.min_strength = min_strength
        self.signal_smoothing = signal_smoothing
        self._signal_history: Dict[str, List[TradingSignal]] = {}
        self._callbacks: List[Callable[[TradingSignal], None]] = []
    
    def register_callback(self, callback: Callable[[TradingSignal], None]):
        """Register callback for new signals."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, signal: TradingSignal):
        """Notify callbacks of new signal."""
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
    
    def generate_signal(
        self,
        symbol: str,
        prediction: float,
        current_price: float,
        volatility: float = 0.02,
        current_position: float = 0.0,
        account_value: float = 100000.0,
    ) -> TradingSignal:
        """
        Generate trading signal from prediction.
        
        Args:
            symbol: Ticker symbol
            prediction: Model prediction (e.g., expected return)
            current_price: Current market price
            volatility: Asset volatility for position sizing
            current_position: Current position size
            account_value: Total account value
        
        Returns:
            TradingSignal object
        """
        # Determine signal type
        if prediction >= self.long_threshold:
            if current_position > 0:
                signal_type = SignalType.INCREASE_LONG
            elif current_position < 0:
                signal_type = SignalType.DECREASE_SHORT
            else:
                signal_type = SignalType.LONG
        elif prediction <= self.short_threshold:
            if current_position < 0:
                signal_type = SignalType.INCREASE_SHORT
            elif current_position > 0:
                signal_type = SignalType.DECREASE_LONG
            else:
                signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.FLAT
        
        # Calculate signal strength (normalized)
        strength = max(-1.0, min(1.0, prediction))
        
        # Skip weak signals
        if abs(strength) < self.min_strength:
            signal_type = SignalType.FLAT
            strength = 0.0
        
        # Calculate target quantity based on strength and volatility
        if signal_type != SignalType.FLAT:
            # Risk-based position sizing
            risk_per_trade = account_value * 0.02  # 2% risk
            dollar_volatility = current_price * volatility
            target_quantity = risk_per_trade / dollar_volatility * abs(strength)
            target_quantity = round(target_quantity, 0)
            
            # Adjust for existing position
            if current_position != 0:
                if (current_position > 0 and signal_type in [SignalType.LONG, SignalType.INCREASE_LONG]) or \
                   (current_position < 0 and signal_type in [SignalType.SHORT, SignalType.INCREASE_SHORT]):
                    target_quantity = abs(target_quantity - abs(current_position))
                else:
                    target_quantity = min(target_quantity, abs(current_position))
        else:
            target_quantity = 0.0
        
        # Calculate entry/exit levels
        entry_price = current_price
        if volatility > 0:
            stop_loss = current_price * (1 - 2 * volatility) if strength > 0 else current_price * (1 + 2 * volatility)
            take_profit = current_price * (1 + 3 * volatility) if strength > 0 else current_price * (1 - 3 * volatility)
        else:
            stop_loss = None
            take_profit = None
        
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            target_quantity=target_quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "prediction": prediction,
                "volatility": volatility,
                "current_position": current_position,
            },
        )
        
        # Store in history
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
        self._signal_history[symbol].append(signal)
        
        # Keep only recent history
        if len(self._signal_history[symbol]) > 1000:
            self._signal_history[symbol] = self._signal_history[symbol][-1000:]
        
        # Notify callbacks
        if signal.is_actionable:
            self._notify_callbacks(signal)
            logger.info(
                f"Signal: {signal_type.value} {symbol} (strength: {strength:.2f}, qty: {target_quantity})",
                extra={"symbol": symbol, "signal_type": signal_type.value}
            )
        
        return signal
    
    def get_signal_history(self, symbol: str, limit: int = 100) -> List[TradingSignal]:
        """Get recent signal history for symbol."""
        history = self._signal_history.get(symbol, [])
        return history[-limit:]
    
    def get_latest_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get most recent signal for symbol."""
        history = self._signal_history.get(symbol, [])
        return history[-1] if history else None
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Get all non-flat signals."""
        active = []
        for symbol, history in self._signal_history.items():
            if history and history[-1].is_actionable:
                active.append(history[-1])
        return active
