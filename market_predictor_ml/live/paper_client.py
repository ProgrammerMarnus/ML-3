"""Alpaca paper-trading client wrapper.

Falls back to a dry-run (log-only) mode when ``alpaca-py`` is not
installed or credentials are missing, so the paper-trader script
can still be exercised without live keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    _ALPACA_AVAILABLE = True
except Exception:  # pragma: no cover
    TradingClient = None  # type: ignore
    MarketOrderRequest = None  # type: ignore
    OrderSide = None  # type: ignore
    TimeInForce = None  # type: ignore
    StockHistoricalDataClient = None  # type: ignore
    StockLatestQuoteRequest = None  # type: ignore
    _ALPACA_AVAILABLE = False


def is_alpaca_available() -> bool:
    """Return True when the ``alpaca-py`` package is importable."""
    return _ALPACA_AVAILABLE


@dataclass
class AlpacaPaperClient:
    """Thin wrapper around Alpaca paper trading.

    Parameters
    ----------
    api_key, secret_key, base_url:
        Alpaca credentials. Defaults are read from the environment
        (``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` /
        ``APCA_API_BASE_URL``) which are loaded from ``.env``.
    paper:
        Always True - this client is paper-only by design.
    dry_run:
        When True (or when alpaca-py/creds are missing) orders are
        logged instead of submitted.
    """

    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: str = "https://paper-api.alpaca.markets"
    paper: bool = True
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("APCA_API_KEY_ID")
        self.secret_key = self.secret_key or os.getenv("APCA_API_SECRET_KEY")
        self.base_url = os.getenv("APCA_API_BASE_URL", self.base_url)
        if not _ALPACA_AVAILABLE or not self.api_key or not self.secret_key:
            self.dry_run = True
            self._trading = None
            self._data = None
        else:
            self._trading = TradingClient(
                self.api_key, self.secret_key, paper=True, url_override=self.base_url
            )
            self._data = StockHistoricalDataClient(self.api_key, self.secret_key)

    @property
    def live(self) -> bool:
        """True when orders will actually be submitted."""
        return not self.dry_run

    def get_account(self) -> Dict[str, Any]:
        """Return account snapshot (or dry-run placeholder)."""
        if self.dry_run or self._trading is None:
            return {
                "mode": "dry_run",
                "equity": None,
                "cash": None,
                "buying_power": None,
            }
        acct = self._trading.get_account()
        return {
            "mode": "paper",
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
        }

    def get_position_qty(self, symbol: str) -> float:
        """Return current position qty for ``symbol`` (0 when flat/dry-run)."""
        if self.dry_run or self._trading is None:
            return 0.0
        try:
            pos = self._trading.get_open_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return latest quote mid-price, or None when unavailable."""
        if self.dry_run or self._data is None:
            return None
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._data.get_stock_latest_quote(req)
            q = quotes[symbol]
            return float((float(q.ask_price) + float(q.bid_price)) / 2.0)
        except Exception:
            return None

    def submit_market_order(
        self, symbol: str, qty: float, side: str, price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Submit a market order (or log it in dry-run mode).

        Parameters
        ----------
        symbol: ticker, e.g. ``"AAPL"``.
        qty: positive share quantity (fractional allowed).
        side: ``"buy"`` or ``"sell"``.
        price: reference price. When given and ``qty`` is fractional the
            order is sent as a notional (dollar) order, which is what
            Alpaca expects for fractional market orders.

        Falls back to a whole-share order if the broker rejects the
        fractional/notional request.
        """
        qty = abs(float(qty))
        if qty <= 0:
            return {"status": "skipped", "reason": "qty<=0"}
        fractional = abs(qty - round(qty)) > 1e-6
        use_notional = bool(price) and fractional and qty * float(price) > 0
        if self.dry_run or self._trading is None:
            kind = f"notional=${qty * float(price):,.2f}" if use_notional else f"qty={qty:.4f}"
            print(f"[DRY-RUN] {side.upper()} {kind} {symbol} @ market")
            return {"status": "dry_run", "symbol": symbol, "qty": qty, "side": side}
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        if use_notional:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(qty * float(price), 2),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        try:
            order = self._trading.submit_order(req)
        except Exception as exc:
            whole = int(qty)  # floor to whole shares
            if not fractional or whole < 1:
                return {"status": "rejected", "symbol": symbol, "qty": qty,
                        "side": side, "reason": str(exc)}
            print(f"[WARN] Fractional order rejected ({exc}); retrying with "
                  f"{whole} whole share(s)")
            fallback = MarketOrderRequest(
                symbol=symbol,
                qty=whole,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self._trading.submit_order(fallback)
        return {
            "status": "submitted",
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_id": str(order.id),
        }
