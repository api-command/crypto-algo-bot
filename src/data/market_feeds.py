import asyncio
import ccxt.pro as ccxt
import numpy as np
from collections import deque
from src.utils.logger import get_logger
from src.infra.microservices import AsyncEventBus
from src.core.latency_monitor import LatencyMonitor
from src.utils.config_loader import config_loader

logger = get_logger('market_feeds')

class MarketDataFeed:
    def __init__(self, exchange_id='coinbasepro', symbols=None, config=None):
        self.exchange_id = exchange_id
        self.symbols = symbols or ['BTC/USD', 'ETH/USD', 'SOL/USD']
        self.config = config or config_loader.load_toml(f'config/exchange_configs/{exchange_id}.toml')
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'rateLimit': self.config['api'].get('rate_limit', 1000),
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
        self.orderbooks = {symbol: None for symbol in self.symbols}
        self.trade_queues = {symbol: asyncio.Queue(maxsize=1000) for symbol in self.symbols}
        self.ohlcv = {symbol: deque(maxlen=500) for symbol in self.symbols}
        self.latency_monitor = LatencyMonitor()
        self.event_bus = AsyncEventBus()
        self.running = False

    async def start_orderbook_feed(self, depth=10):
        """Real-time order book streaming with nanosecond precision"""
        while self.running:
            try:
                for symbol in self.symbols:
                    self.latency_monitor.start_timer(f'ob_{symbol}')
                    orderbook = await self.exchange.watch_order_book(symbol, depth)
                    self.orderbooks[symbol] = orderbook
                    latency = self.latency_monitor.record_latency(f'ob_{symbol}')
                    
                    # Publish to event bus
                    await self.event_bus.publish('orderbook', {
                        'symbol': symbol,
                        'bids': orderbook['bids'][:5],  # Top 5 bids
                        'asks': orderbook['asks'][:5],  # Top 5 asks
                        'timestamp': orderbook['timestamp'],
                        'latency': latency
                    })
            except Exception as e:
                logger.error(f"Orderbook feed error: {e}")
                await asyncio.sleep(5)  # Reconnect delay

    async def start_trades_feed(self):
        """Real-time trade execution streaming"""
        while self.running:
            try:
                for symbol in self.symbols:
                    self.latency_monitor.start_timer(f'trades_{symbol}')
                    trades = await self.exchange.watch_trades(symbol)
                    self.latency_monitor.record_latency(f'trades_{symbol}')
                    
                    # Add trades to processing queue
                    for trade in trades:
                        await self.trade_queues[symbol].put(trade)
                    
                    # Publish to event bus
                    await self.event_bus.publish('trades', {
                        'symbol': symbol,
                        'trades': trades[-10:],  # Last 10 trades
                        'timestamp': trades[-1]['timestamp'] if trades else None
                    })
            except Exception as e:
                logger.error(f"Trade feed error: {e}")
                await asyncio.sleep(5)

    async def start_ohlcv_feed(self, timeframe='1m'):
        """OHLCV candle streaming with technical indicator support"""
        while self.running:
            try:
                for symbol in self.symbols:
                    self.latency_monitor.start_timer(f'ohlcv_{symbol}')
                    candles = await self.exchange.watch_ohlcv(symbol, timeframe)
                    self.ohlcv[symbol].extend(candles)
                    latency = self.latency_monitor.record_latency(f'ohlcv_{symbol}')
                    
                    # Calculate technical indicators
                    closes = np.array([c[4] for c in candles])
                    if len(closes) > 10:
                        sma = closes[-10:].mean()
                        rsi = self.calculate_rsi(closes)
                    else:
                        sma = rsi = None
                    
                    # Publish to event bus
                    await self.event_bus.publish('ohlcv', {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'candle': candles[-1] if candles else None,
                        'sma': sma,
                        'rsi': rsi,
                        'latency': latency
                    })
            except Exception as e:
                logger.error(f"OHLCV feed error: {e}")
                await asyncio.sleep(30)

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = 100. - (100. / (1. + rs))
        
        for i in range(period+1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi = 100. - (100. / (1. + rs))
            
        return rsi

    async def get_next_trade(self, symbol):
        """Get next trade from queue (async)"""
        return await self.trade_queues[symbol].get()

    def get_orderbook(self, symbol):
        """Get current order book state"""
        return self.orderbooks.get(symbol)

    def get_ohlcv(self, symbol):
        """Get OHLCV history"""
        return list(self.ohlcv.get(symbol, []))

    async def start(self):
        """Start all market data feeds"""
        self.running = True
        tasks = [
            asyncio.create_task(self.start_orderbook_feed()),
            asyncio.create_task(self.start_trades_feed()),
            asyncio.create_task(self.start_ohlcv_feed())
        ]
        return tasks

    async def stop(self):
        """Stop all market data feeds"""
        self.running = False
        await self.exchange.close()