"""
State Persistence Module

Provides lightweight state storage for:
- Portfolio positions and P&L
- Risk monitor state (circuit breaker status, drawdown levels)
- Trading engine state (last run time, event counts)
- Signal history

Uses SQLite for durability and JSON for simple serialization.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Snapshot of portfolio state."""
    timestamp: datetime
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    entry_prices: Dict[str, float]  # symbol -> avg entry price
    total_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class RiskState:
    """Snapshot of risk monitor state."""
    timestamp: datetime
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    circuit_breaker_active: bool
    consecutive_losses: int
    trading_halted: bool
    halt_reason: Optional[str]


@dataclass
class EngineState:
    """Snapshot of trading engine state."""
    timestamp: datetime
    last_signal_time: Optional[datetime]
    last_order_time: Optional[datetime]
    signals_generated: int
    orders_submitted: int
    orders_filled: int
    errors_count: int
    is_running: bool


class StateStore:
    """
    SQLite-based state persistence layer.
    
    Stores and retrieves state snapshots for portfolio, risk, and engine.
    Supports automatic cleanup of old states and atomic updates.
    """
    
    def __init__(self, db_path: str = "trading_state.db"):
        """
        Initialize the state store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"StateStore initialized at {db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cash REAL NOT NULL,
                positions TEXT NOT NULL,
                entry_prices TEXT NOT NULL,
                total_value REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Risk states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                current_drawdown REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                circuit_breaker_active INTEGER NOT NULL,
                consecutive_losses INTEGER NOT NULL,
                trading_halted INTEGER NOT NULL,
                halt_reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Engine states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engine_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                last_signal_time TEXT,
                last_order_time TEXT,
                signals_generated INTEGER NOT NULL,
                orders_submitted INTEGER NOT NULL,
                orders_filled INTEGER NOT NULL,
                errors_count INTEGER NOT NULL,
                is_running INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Signals history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL,
                strength REAL NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_states(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_timestamp ON risk_states(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_engine_timestamp ON engine_states(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals_history(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals_history(timestamp)')
        
        conn.commit()
        conn.close()
    
    def save_portfolio_state(self, state: PortfolioState):
        """
        Save portfolio state snapshot.
        
        Args:
            state: PortfolioState to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolio_states 
            (timestamp, cash, positions, entry_prices, total_value, unrealized_pnl, realized_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            state.timestamp.isoformat(),
            state.cash,
            json.dumps(state.positions),
            json.dumps(state.entry_prices),
            state.total_value,
            state.unrealized_pnl,
            state.realized_pnl
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Saved portfolio state: cash={state.cash:.2f}, value={state.total_value:.2f}")
    
    def get_latest_portfolio_state(self) -> Optional[PortfolioState]:
        """
        Retrieve the most recent portfolio state.
        
        Returns:
            Latest PortfolioState or None if no states exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, cash, positions, entry_prices, total_value, unrealized_pnl, realized_pnl
            FROM portfolio_states
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return PortfolioState(
            timestamp=datetime.fromisoformat(row[0]),
            cash=row[1],
            positions=json.loads(row[2]),
            entry_prices=json.loads(row[3]),
            total_value=row[4],
            unrealized_pnl=row[5],
            realized_pnl=row[6]
        )
    
    def save_risk_state(self, state: RiskState):
        """
        Save risk monitor state snapshot.
        
        Args:
            state: RiskState to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO risk_states 
            (timestamp, current_drawdown, max_drawdown, daily_pnl, 
             circuit_breaker_active, consecutive_losses, trading_halted, halt_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state.timestamp.isoformat(),
            state.current_drawdown,
            state.max_drawdown,
            state.daily_pnl,
            1 if state.circuit_breaker_active else 0,
            state.consecutive_losses,
            1 if state.trading_halted else 0,
            state.halt_reason
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Saved risk state: drawdown={state.current_drawdown:.2%}, halted={state.trading_halted}")
    
    def get_latest_risk_state(self) -> Optional[RiskState]:
        """
        Retrieve the most recent risk state.
        
        Returns:
            Latest RiskState or None if no states exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, current_drawdown, max_drawdown, daily_pnl,
                   circuit_breaker_active, consecutive_losses, trading_halted, halt_reason
            FROM risk_states
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return RiskState(
            timestamp=datetime.fromisoformat(row[0]),
            current_drawdown=row[1],
            max_drawdown=row[2],
            daily_pnl=row[3],
            circuit_breaker_active=bool(row[4]),
            consecutive_losses=row[5],
            trading_halted=bool(row[6]),
            halt_reason=row[7]
        )
    
    def save_engine_state(self, state: EngineState):
        """
        Save trading engine state snapshot.
        
        Args:
            state: EngineState to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO engine_states 
            (timestamp, last_signal_time, last_order_time, signals_generated,
             orders_submitted, orders_filled, errors_count, is_running)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state.timestamp.isoformat(),
            state.last_signal_time.isoformat() if state.last_signal_time else None,
            state.last_order_time.isoformat() if state.last_order_time else None,
            state.signals_generated,
            state.orders_submitted,
            state.orders_filled,
            state.errors_count,
            1 if state.is_running else 0
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Saved engine state: signals={state.signals_generated}, running={state.is_running}")
    
    def get_latest_engine_state(self) -> Optional[EngineState]:
        """
        Retrieve the most recent engine state.
        
        Returns:
            Latest EngineState or None if no states exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, last_signal_time, last_order_time, signals_generated,
                   orders_submitted, orders_filled, errors_count, is_running
            FROM engine_states
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return EngineState(
            timestamp=datetime.fromisoformat(row[0]),
            last_signal_time=datetime.fromisoformat(row[1]) if row[1] else None,
            last_order_time=datetime.fromisoformat(row[2]) if row[2] else None,
            signals_generated=row[3],
            orders_submitted=row[4],
            orders_filled=row[5],
            errors_count=row[6],
            is_running=bool(row[7])
        )
    
    def save_signal(self, signal_data: Dict[str, Any]):
        """
        Save a trading signal to history.
        
        Args:
            signal_data: Dictionary with signal attributes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals_history 
            (timestamp, symbol, direction, strength, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('timestamp', datetime.now().isoformat()),
            signal_data['symbol'],
            signal_data['direction'],
            signal_data['strength'],
            signal_data['confidence'],
            json.dumps(signal_data.get('metadata', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def get_signals_history(
        self, 
        symbol: Optional[str] = None, 
        start_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve signal history with optional filtering.
        
        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of signal dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT timestamp, symbol, direction, strength, confidence, metadata FROM signals_history'
        conditions = []
        params = []
        
        if symbol:
            conditions.append('symbol = ?')
            params.append(symbol)
        
        if start_date:
            conditions.append('timestamp >= ?')
            params.append(start_date.isoformat())
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        signals = []
        for row in rows:
            signals.append({
                'timestamp': datetime.fromisoformat(row[0]),
                'symbol': row[1],
                'direction': row[2],
                'strength': row[3],
                'confidence': row[4],
                'metadata': json.loads(row[5])
            })
        
        return signals
    
    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """
        Retrieve portfolio state history as DataFrame.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with portfolio history
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, cash, total_value, unrealized_pnl, realized_pnl
            FROM portfolio_states
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def cleanup_old_states(self, days_to_keep: int = 90):
        """
        Remove state snapshots older than specified days.
        
        Args:
            days_to_keep: Number of days to retain
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - 
                      pd.Timedelta(days=days_to_keep)).isoformat()
        
        tables = ['portfolio_states', 'risk_states', 'engine_states']
        for table in tables:
            cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff_date,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted} old state records older than {days_to_keep} days")
    
    def export_state_to_json(self, output_path: str):
        """
        Export all current states to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        portfolio = self.get_latest_portfolio_state()
        risk = self.get_latest_risk_state()
        engine = self.get_latest_engine_state()
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'portfolio': asdict(portfolio) if portfolio else None,
            'risk': asdict(risk) if risk else None,
            'engine': asdict(engine) if engine else None,
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported state to {output_path}")
    
    def import_state_from_json(self, input_path: str):
        """
        Import states from a JSON file.
        
        Args:
            input_path: Path to input JSON file
        """
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        if import_data.get('portfolio'):
            portfolio_dict = import_data['portfolio']
            portfolio_dict['timestamp'] = datetime.fromisoformat(portfolio_dict['timestamp'])
            state = PortfolioState(**portfolio_dict)
            self.save_portfolio_state(state)
        
        if import_data.get('risk'):
            risk_dict = import_data['risk']
            risk_dict['timestamp'] = datetime.fromisoformat(risk_dict['timestamp'])
            state = RiskState(**risk_dict)
            self.save_risk_state(state)
        
        if import_data.get('engine'):
            engine_dict = import_data['engine']
            engine_dict['timestamp'] = datetime.fromisoformat(engine_dict['timestamp'])
            if engine_dict.get('last_signal_time'):
                engine_dict['last_signal_time'] = datetime.fromisoformat(engine_dict['last_signal_time'])
            if engine_dict.get('last_order_time'):
                engine_dict['last_order_time'] = datetime.fromisoformat(engine_dict['last_order_time'])
            state = EngineState(**engine_dict)
            self.save_engine_state(state)
        
        logger.info(f"Imported state from {input_path}")
