import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from src.utils.logger import get_logger
from src.infra.telemetry import Telemetry
from src.core.risk_manager import RiskManager

logger = get_logger('trade_memory')

class TradeMemory:
    def __init__(self, db_path: str = "trades.db"):
        """
        Persistent trade logging and PnL tracking
        :param db_path: Path to SQLite database file
        """
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.telemetry = Telemetry()
        self.open_positions = {}
        self.pnl_history = []
        
    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                fee REAL NOT NULL,
                fee_currency TEXT NOT NULL,
                exchange TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_strength REAL,
                latency_ms REAL
            )
        ''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_percentage REAL,
                duration_sec REAL,
                status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed'))
        ''')
        
        # PnL history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pnl_history (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def record_trade(self, trade: Dict):
        """
        Record a completed trade
        :param trade: Dictionary with trade details
            Required keys: id, symbol, side, price, quantity, fee, fee_currency, 
                           exchange, strategy
            Optional keys: signal_strength, latency_ms
        """
        required_keys = ['id', 'symbol', 'side', 'price', 'quantity', 'fee', 
                         'fee_currency', 'exchange', 'strategy']
        
        if not all(key in trade for key in required_keys):
            missing = [key for key in required_keys if key not in trade]
            logger.error(f"Trade missing required keys: {missing}")
            return False
            
        # Prepare data for insertion
        data = (
            trade['id'],
            datetime.utcnow().isoformat(),
            trade['symbol'],
            trade['side'],
            trade['price'],
            trade['quantity'],
            trade['fee'],
            trade['fee_currency'],
            trade['exchange'],
            trade['strategy'],
            trade.get('signal_strength', None),
            trade.get('latency_ms', None)
        )
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO trades 
                (id, timestamp, symbol, side, price, quantity, fee, fee_currency, exchange, strategy, signal_strength, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            self.conn.commit()
            
            # Update position tracking
            self._update_position(trade)
            
            # Update PnL tracking
            self._update_pnl(trade['symbol'])
            
            logger.info(f"Recorded trade {trade['id']} for {trade['symbol']}")
            self.telemetry.incr('trades.recorded')
            return True
        except sqlite3.IntegrityError:
            logger.error(f"Trade {trade['id']} already exists in database")
            return False
    
    def _update_position(self, trade: Dict):
        """Update open positions based on trade activity"""
        symbol = trade['symbol']
        quantity = trade['quantity'] if trade['side'] == 'buy' else -trade['quantity']
        
        # Get current position
        current_position = self.open_positions.get(symbol, {
            'quantity': 0,
            'entry_price': 0,
            'entry_time': datetime.utcnow()
        })
        
        # Calculate new position
        if current_position['quantity'] == 0:
            # New position
            new_quantity = quantity
            new_entry_price = trade['price']
        else:
            # Existing position
            if (current_position['quantity'] > 0 and quantity > 0) or \
               (current_position['quantity'] < 0 and quantity < 0):
                # Adding to position
                total_cost = (current_position['quantity'] * current_position['entry_price'] +
                              quantity * trade['price'])
                new_quantity = current_position['quantity'] + quantity
                new_entry_price = total_cost / new_quantity
            else:
                # Reducing or closing position
                new_quantity = current_position['quantity'] + quantity
                new_entry_price = current_position['entry_price'] if new_quantity != 0 else 0
                
                # If position closed, record to database
                if new_quantity == 0:
                    self._close_position(symbol, trade['price'])
        
        # Update in-memory position
        self.open_positions[symbol] = {
            'quantity': new_quantity,
            'entry_price': new_entry_price,
            'entry_time': current_position['entry_time']
        }
    
    def _close_position(self, symbol: str, exit_price: float):
        """Record closed position in database"""
        if symbol not in self.open_positions:
            return
            
        position = self.open_positions[symbol]
        entry_time = position['entry_time']
        exit_time = datetime.utcnow()
        duration = (exit_time - entry_time).total_seconds()
        
        # Calculate PnL
        if position['quantity'] > 0:  # Long position
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:  # Short position
            pnl = (position['entry_price'] - exit_price) * abs(position['quantity'])
        
        pnl_percentage = (pnl / (position['entry_price'] * abs(position['quantity']))) * 100
        
        # Insert into positions table
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO positions 
            (symbol, entry_time, exit_time, entry_price, exit_price, quantity, pnl, pnl_percentage, duration_sec, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed')
        ''', (
            symbol,
            entry_time.isoformat(),
            exit_time.isoformat(),
            position['entry_price'],
            exit_price,
            position['quantity'],
            pnl,
            pnl_percentage,
            duration
        ))
        self.conn.commit()
        
        # Remove from open positions
        del self.open_positions[symbol]
        logger.info(f"Closed position for {symbol}. PnL: ${pnl:.2f} ({pnl_percentage:.2f}%)")
    
    def _update_pnl(self, symbol: str):
        """Update PnL history"""
        # Get current market price (would come from market feed)
        # For demo purposes, we'll use a placeholder
        current_price = 50000  # This should be replaced with real market data
        
        # Calculate unrealized PnL for open positions
        unrealized_pnl = 0
        for sym, position in self.open_positions.items():
            if position['quantity'] > 0:  # Long
                unrealized_pnl += (current_price - position['entry_price']) * position['quantity']
            else:  # Short
                unrealized_pnl += (position['entry_price'] - current_price) * abs(position['quantity'])
        
        # Calculate realized PnL from closed positions
        cursor = self.conn.cursor()
        cursor.execute('SELECT SUM(pnl) FROM positions WHERE status = "closed"')
        realized_pnl = cursor.fetchone()[0] or 0
        
        total_pnl = realized_pnl + unrealized_pnl
        
        # Record to database
        cursor.execute('''
            INSERT INTO pnl_history (timestamp, total_pnl, realized_pnl, unrealized_pnl)
            VALUES (?, ?, ?, ?)
        ''', (datetime.utcnow().isoformat(), total_pnl, realized_pnl, unrealized_pnl))
        self.conn.commit()
        
        # Update telemetry
        self.telemetry.gauge('pnl.total', total_pnl)
        self.telemetry.gauge('pnl.realized', realized_pnl)
        self.telemetry.gauge('pnl.unrealized', unrealized_pnl)
        
        # Add to history for quick access
        self.pnl_history.append({
            'timestamp': datetime.utcnow(),
            'total_pnl': total_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl
        })
    
    def get_open_positions(self) -> Dict[str, Dict]:
        """Get current open positions"""
        return self.open_positions
    
    def get_position_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get historical position data"""
        query = "SELECT * FROM positions"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
            
        query += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get historical trade data"""
        query = "SELECT * FROM trades"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_pnl_history(self, days: int = 30) -> pd.DataFrame:
        """Get PnL history as a DataFrame"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM pnl_history 
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp
        ''', (f'-{days} days',))
        
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_risk_metrics(self, risk_manager: RiskManager):
        """Calculate risk-adjusted performance metrics"""
        # Get recent trades
        trades = self.get_trade_history(limit=1000)
        if not trades:
            return {}
            
        # Calculate win rate
        winning_trades = [t for t in trades if 
                          (t['side'] == 'buy' and t['price'] > t['entry_price']) or 
                          (t['side'] == 'sell' and t['price'] < t['entry_price'])]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Calculate profit factor
        total_profit = sum(t['pnl'] for t in winning_trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t not in winning_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        pnl_history = self.get_pnl_history(days=30)['total_pnl'].pct_change().dropna()
        sharpe_ratio = pnl_history.mean() / pnl_history.std() * np.sqrt(365) if not pnl_history.empty else 0
        
        # Calculate maximum drawdown
        equity_curve = self.get_pnl_history(days=90)['total_pnl']
        if not equity_curve.empty:
            peak = equity_curve.cummax()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
            
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'risk_score': risk_manager.calculate_portfolio_risk()
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()