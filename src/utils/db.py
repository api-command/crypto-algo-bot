import sqlite3
import aiosqlite
import asyncio
import os
import contextlib
from typing import Any, AsyncGenerator, List, Dict, Optional, Tuple
from src.utils.logger import get_logger
from src.utils.config_loader import config_loader
from src.infra.telemetry import telemetry

logger = get_logger('db')

class DatabaseManager:
    def __init__(self, db_path: str = 'trading_bot.db'):
        """
        Database abstraction layer for synchronous and asynchronous operations
        :param db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database with required schema"""
        with self.sync_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables if not exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    parameters TEXT,
                    source TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    resolution TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    symbol TEXT,
                    title TEXT,
                    content TEXT,
                    raw_data TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT,
                    component TEXT,
                    message TEXT,
                    metadata TEXT
                )
            ''')
            
            # Indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_signals_symbol ON trade_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_events_symbol ON news_events(symbol)')
            
            conn.commit()
    
    @contextlib.contextmanager
    def sync_connection(self) -> sqlite3.Connection:
        """Context manager for synchronous database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    async def async_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Async context manager for database connection"""
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        try:
            yield conn
        finally:
            await conn.close()
    
    async def execute(self, query: str, params: tuple = (), commit: bool = False) -> int:
        """
        Execute a write operation asynchronously
        :return: Last row ID for INSERT, rowcount for others
        """
        start_time = time.perf_counter_ns()
        async with self.async_connection() as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute(query, params)
                if commit:
                    await conn.commit()
                
                # Return appropriate result
                if query.strip().lower().startswith('insert'):
                    return cursor.lastrowid
                return cursor.rowcount
            except aiosqlite.Error as e:
                logger.error(f"Database error: {e} - Query: {query}")
                telemetry.incr('db_errors')
                raise
            finally:
                latency = (time.perf_counter_ns() - start_time) / 1e6
                telemetry.latency('db_write', latency / 1000)  # seconds
    
    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict]:
        """Fetch a single row asynchronously"""
        start_time = time.perf_counter_ns()
        async with self.async_connection() as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute(query, params)
                row = await cursor.fetchone()
                return dict(row) if row else None
            except aiosqlite.Error as e:
                logger.error(f"Database error: {e} - Query: {query}")
                telemetry.incr('db_errors')
                return None
            finally:
                latency = (time.perf_counter_ns() - start_time) / 1e6
                telemetry.latency('db_read', latency / 1000)  # seconds
    
    async def fetch_all(self, query: str, params: tuple = ()) -> List[Dict]:
        """Fetch all rows asynchronously"""
        start_time = time.perf_counter_ns()
        async with self.async_connection() as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
            except aiosqlite.Error as e:
                logger.error(f"Database error: {e} - Query: {query}")
                telemetry.incr('db_errors')
                return []
            finally:
                latency = (time.perf_counter_ns() - start_time) / 1e6
                telemetry.latency('db_read', latency / 1000)  # seconds
    
    async def insert_trade_signal(self, signal: Dict) -> int:
        """Insert a trade signal record"""
        query = '''
            INSERT INTO trade_signals (
                timestamp, symbol, signal_type, score, confidence, parameters, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            signal.get('timestamp', datetime.utcnow().isoformat()),
            signal['symbol'],
            signal['signal_type'],
            signal['score'],
            signal['confidence'],
            json.dumps(signal.get('parameters', {})),
            signal.get('source', 'unknown')
        )
        return await self.execute(query, params, commit=True)
    
    async def insert_market_data(self, data: Dict) -> int:
        """Insert market data snapshot"""
        query = '''
            INSERT INTO market_data (
                timestamp, symbol, open, high, low, close, volume, resolution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            data.get('timestamp', datetime.utcnow().isoformat()),
            data['symbol'],
            data['open'],
            data['high'],
            data['low'],
            data['close'],
            data['volume'],
            data.get('resolution', '1m')
        )
        return await self.execute(query, params, commit=True)
    
    async def insert_news_event(self, news: Dict) -> int:
        """Insert news event with full content"""
        query = '''
            INSERT INTO news_events (
                timestamp, source, symbol, title, content, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?)
        '''
        params = (
            news.get('timestamp', datetime.utcnow().isoformat()),
            news['source'],
            news['symbol'],
            news.get('title', ''),
            news.get('content', ''),
            json.dumps(news)
        )
        return await self.execute(query, params, commit=True)
    
    async def insert_system_event(self, event: Dict) -> int:
        """Insert system event log"""
        query = '''
            INSERT INTO system_events (
                timestamp, event_type, component, message, metadata
            ) VALUES (?, ?, ?, ?, ?)
        '''
        params = (
            event.get('timestamp', datetime.utcnow().isoformat()),
            event['event_type'],
            event['component'],
            event['message'],
            json.dumps(event.get('metadata', {}))
        )
        return await self.execute(query, params, commit=True)
    
    async def get_latest_signals(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get latest trade signals for a symbol"""
        query = '''
            SELECT * FROM trade_signals
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        return await self.fetch_all(query, (symbol, limit))
    
    async def get_ohlcv_history(self, symbol: str, 
                               start: datetime, 
                               end: datetime, 
                               resolution: str = '1h') -> List[Dict]:
        """Get OHLCV data for a specific period"""
        query = '''
            SELECT * FROM market_data
            WHERE symbol = ? 
            AND resolution = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        '''
        return await self.fetch_all(query, (symbol, resolution, start.isoformat(), end.isoformat()))
    
    async def bulk_insert(self, table: str, data: List[Dict]):
        """Bulk insert records efficiently"""
        if not data:
            return
        
        # Generate placeholders
        columns = list(data[0].keys())
        placeholders = ', '.join(['?'] * len(columns))
        col_names = ', '.join(columns)
        
        query = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
        
        # Prepare parameters
        params = [tuple(item.values()) for item in data]
        
        async with self.async_connection() as conn:
            try:
                await conn.executemany(query, params)
                await conn.commit()
                logger.info(f"Bulk inserted {len(data)} records into {table}")
            except aiosqlite.Error as e:
                logger.error(f"Bulk insert failed: {e}")
                telemetry.incr('db_errors')
    
    def vacuum(self):
        """Optimize database (run periodically)"""
        logger.info("Running database VACUUM")
        with self.sync_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database optimization complete")

# Global database instance
db_manager = DatabaseManager()

# For high-frequency components
class HighFrequencyDB:
    """Optimized for low-latency write operations"""
    def __init__(self, db_path: str = 'hf_data.db'):
        self.db_path = db_path
        self.queue = asyncio.Queue(maxsize=10000)
        self.writer_task = asyncio.create_task(self._batch_writer())
        self.logger = get_perf_logger('hfdb')
        
    async def _batch_writer(self):
        """Background task to write batched data"""
        from aiosqlite import connect
        
        async with connect(self.db_path) as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS market_ticks (
                    timestamp DATETIME PRIMARY KEY,
                    symbol TEXT,
                    bid REAL,
                    ask REAL,
                    last REAL,
                    volume REAL
                ) WITHOUT ROWID
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS order_events (
                    timestamp DATETIME,
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    type TEXT,
                    side TEXT,
                    price REAL,
                    amount REAL,
                    status TEXT
                ) WITHOUT ROWID
            ''')
            
            while True:
                items = []
                while not self.queue.empty():
                    items.append(await self.queue.get())
                
                if items:
                    try:
                        # Group by table for efficient bulk insert
                        tables = {}
                        for item in items:
                            table = item['table']
                            if table not in tables:
                                tables[table] = []
                            tables[table].append(item['data'])
                        
                        # Process each table
                        for table, records in tables.items():
                            columns = list(records[0].keys())
                            placeholders = ', '.join(['?'] * len(columns))
                            col_names = ', '.join(columns)
                            query = f'''
                                INSERT OR REPLACE INTO {table} ({col_names})
                                VALUES ({placeholders})
                            '''
                            params = [tuple(record.values()) for record in records]
                            await conn.executemany(query, params)
                        
                        await conn.commit()
                        self.logger.info(f"Wrote {len(items)} events to HFDB")
                    except Exception as e:
                        self.logger.error(f"HFDB batch write failed: {e}")
                        telemetry.incr('hfdb_errors')
                
                await asyncio.sleep(1)  # Batch interval
    
    async def record_tick(self, symbol: str, bid: float, ask: float, last: float, volume: float):
        """Record market tick with nanosecond precision"""
        await self.queue.put({
            'table': 'market_ticks',
            'data': {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': last,
                'volume': volume
            }
        })
    
    async def record_order_event(self, order: Dict):
        """Record order status change"""
        await self.queue.put({
            'table': 'order_events',
            'data': {
                'timestamp': order['timestamp'],
                'order_id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'price': order['price'],
                'amount': order['amount'],
                'status': order['status']
            }
        })
    
    async def close(self):
        """Gracefully shutdown writer"""
        while not self.queue.empty():
            await asyncio.sleep(0.1)
        self.writer_task.cancel()

# Global HFDB instance
hf_db = HighFrequencyDB()