"""
State Persistence & Recovery

Saves and loads trading state to prevent data loss on restarts.
Uses SQLite for robust storage of positions, orders, and P&L.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

logger = logging.getLogger(__name__)


class StateStore:
    """
    Persistent storage for trading state.
    
    Stores:
    - Portfolio state (cash, equity, P&L)
    - Current positions
    - Active and historical orders
    - Risk monitor state
    - Trading session metadata
    
    Uses SQLite for ACID compliance and concurrent access safety.
    """
    
    def __init__(self, db_path: str = "trading_state.db"):
        """
        Initialize the state store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_database()
        logger.info(f"StateStore initialized at {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Portfolio state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_cash REAL NOT NULL,
                current_cash REAL NOT NULL,
                total_equity REAL NOT NULL,
                realized_pnl REAL DEFAULT 0.0,
                unrealized_pnl REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL,
                current_price REAL DEFAULT 0.0,
                market_value REAL DEFAULT 0.0,
                unrealized_pnl REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                order_type TEXT NOT NULL,
                limit_price REAL,
                stop_price REAL,
                status TEXT NOT NULL,
                submitted_at TIMESTAMP,
                filled_at TIMESTAMP,
                avg_fill_price REAL,
                filled_quantity REAL DEFAULT 0.0,
                commission REAL DEFAULT 0.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                trading_halted INTEGER DEFAULT 0,
                halt_reason TEXT,
                max_drawdown_pct REAL DEFAULT 0.0,
                current_drawdown_pct REAL DEFAULT 0.0,
                daily_pnl REAL DEFAULT 0.0,
                peak_equity REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trade log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0.0,
                pnl REAL DEFAULT 0.0,
                order_id TEXT,
                notes TEXT
            )
        """)
        
        # Session metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                initial_cash REAL NOT NULL,
                final_cash REAL,
                total_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        conn.commit()
        logger.debug("Database tables initialized")
    
    def save_portfolio_state(
        self,
        initial_cash: float,
        current_cash: float,
        total_equity: float,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
    ):
        """Save portfolio state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO portfolio_state 
            (id, initial_cash, current_cash, total_equity, realized_pnl, unrealized_pnl, last_updated)
            VALUES (1, ?, ?, ?, ?, ?, ?)
        """, (initial_cash, current_cash, total_equity, realized_pnl, unrealized_pnl, datetime.now()))
        
        conn.commit()
        logger.debug("Portfolio state saved")
    
    def load_portfolio_state(self) -> Optional[Dict]:
        """Load portfolio state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM portfolio_state WHERE id = 1")
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return dict(row)
    
    def save_position(
        self,
        symbol: str,
        quantity: float,
        avg_cost: float,
        current_price: float = 0.0,
    ):
        """Save or update a position."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        market_value = quantity * current_price
        unrealized_pnl = (current_price - avg_cost) * quantity
        
        cursor.execute("""
            INSERT OR REPLACE INTO positions 
            (symbol, quantity, avg_cost, current_price, market_value, unrealized_pnl, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, quantity, avg_cost, current_price, market_value, unrealized_pnl, datetime.now()))
        
        conn.commit()
        logger.debug(f"Position saved for {symbol}: {quantity} @ {avg_cost:.2f}")
    
    def save_positions(self, positions: Dict[str, Dict]):
        """Save multiple positions."""
        for symbol, pos_data in positions.items():
            self.save_position(
                symbol=symbol,
                quantity=pos_data.get('quantity', 0.0),
                avg_cost=pos_data.get('avg_cost', 0.0),
                current_price=pos_data.get('current_price', 0.0),
            )
        logger.info(f"Saved {len(positions)} positions")
    
    def load_positions(self) -> Dict[str, Dict]:
        """Load all positions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM positions")
        rows = cursor.fetchall()
        
        positions = {}
        for row in rows:
            row_dict = dict(row)
            positions[row_dict['symbol']] = row_dict
        
        logger.info(f"Loaded {len(positions)} positions")
        return positions
    
    def clear_positions(self):
        """Clear all positions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions")
        conn.commit()
        logger.info("All positions cleared")
    
    def save_order(self, order_data: Dict):
        """Save or update an order."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Convert metadata dict to JSON string
        metadata_json = json.dumps(order_data.get('metadata', {})) if order_data.get('metadata') else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO orders 
            (order_id, symbol, side, quantity, order_type, limit_price, stop_price, 
             status, submitted_at, filled_at, avg_fill_price, filled_quantity, 
             commission, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_data.get('order_id'),
            order_data.get('symbol'),
            order_data.get('side'),
            order_data.get('quantity'),
            order_data.get('order_type'),
            order_data.get('limit_price'),
            order_data.get('stop_price'),
            order_data.get('status'),
            order_data.get('submitted_at'),
            order_data.get('filled_at'),
            order_data.get('avg_fill_price'),
            order_data.get('filled_quantity', 0.0),
            order_data.get('commission', 0.0),
            metadata_json,
            order_data.get('created_at', datetime.now()),
        ))
        
        conn.commit()
        logger.debug(f"Order saved: {order_data.get('order_id')}")
    
    def save_orders(self, orders: List[Dict]):
        """Save multiple orders."""
        for order in orders:
            self.save_order(order)
        logger.info(f"Saved {len(orders)} orders")
    
    def load_orders(self, status_filter: Optional[str] = None) -> List[Dict]:
        """Load orders, optionally filtered by status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status_filter:
            cursor.execute("SELECT * FROM orders WHERE status = ?", (status_filter,))
        else:
            cursor.execute("SELECT * FROM orders")
        
        rows = cursor.fetchall()
        orders = []
        for row in rows:
            row_dict = dict(row)
            # Parse metadata JSON
            if row_dict.get('metadata'):
                try:
                    row_dict['metadata'] = json.loads(row_dict['metadata'])
                except:
                    pass
            orders.append(row_dict)
        
        logger.info(f"Loaded {len(orders)} orders")
        return orders
    
    def save_risk_state(
        self,
        trading_halted: bool = False,
        halt_reason: str = "",
        max_drawdown_pct: float = 0.0,
        current_drawdown_pct: float = 0.0,
        daily_pnl: float = 0.0,
        peak_equity: float = 0.0,
    ):
        """Save risk monitor state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO risk_state 
            (id, trading_halted, halt_reason, max_drawdown_pct, current_drawdown_pct, 
             daily_pnl, peak_equity, last_updated)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        """, (
            1 if trading_halted else 0,
            halt_reason,
            max_drawdown_pct,
            current_drawdown_pct,
            daily_pnl,
            peak_equity,
            datetime.now(),
        ))
        
        conn.commit()
        logger.debug("Risk state saved")
    
    def load_risk_state(self) -> Optional[Dict]:
        """Load risk state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM risk_state WHERE id = 1")
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        row_dict = dict(row)
        # Convert trading_halted back to boolean
        row_dict['trading_halted'] = bool(row_dict['trading_halted'])
        
        return row_dict
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        pnl: float = 0.0,
        order_id: str = "",
        notes: str = "",
    ):
        """Log a trade execution."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trade_log 
            (symbol, side, quantity, price, commission, pnl, order_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, side, quantity, price, commission, pnl, order_id, notes))
        
        conn.commit()
        logger.info(f"Trade logged: {side} {quantity} {symbol} @ {price:.2f}")
    
    def get_trade_log(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trade log, optionally filtered by symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute(
                "SELECT * FROM trade_log WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM trade_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def start_session(self, initial_cash: float) -> int:
        """Start a new trading session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (start_time, initial_cash, status)
            VALUES (?, ?, 'active')
        """, (datetime.now(), initial_cash))
        
        session_id = cursor.lastrowid
        conn.commit()
        
        logger.info(f"Trading session {session_id} started with ${initial_cash:.2f}")
        return session_id
    
    def end_session(self, session_id: int, final_cash: float, total_trades: int, total_pnl: float):
        """End a trading session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET end_time = ?, final_cash = ?, total_trades = ?, total_pnl = ?, status = 'completed'
            WHERE id = ?
        """, (datetime.now(), final_cash, total_trades, total_pnl, session_id))
        
        conn.commit()
        logger.info(f"Trading session {session_id} ended: P&L=${total_pnl:.2f}, {total_trades} trades")
    
    def get_active_session(self) -> Optional[Dict]:
        """Get the currently active session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM sessions WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        return dict(row) if row else None
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_state_store(db_path: str = "trading_state.db") -> StateStore:
    """Factory function to create a StateStore."""
    return StateStore(db_path=db_path)
