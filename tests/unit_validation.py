import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.core.execution_engine import ExecutionEngine
from src.core.risk_manager import RiskManager
from src.core.trade_memory import TradeMemory
from src.ml.sentiment_nlp import SentimentAnalyzer
from src.ml.signal_orchestrator import SignalOrchestrator
from src.data.market_feeds import MarketDataFeed
from src.data.sentinel_agent import SentinelAgent
from src.infra.alert_manager import AlertManager
from src.infra.microservices import AsyncEventBus
from src.utils.db import DatabaseManager

logger = get_logger('unit_tests')

# Fixtures for reusable test objects
@pytest.fixture
def mock_exchange():
    """Mock cryptocurrency exchange API"""
    exchange = MagicMock()
    exchange.create_order = AsyncMock(return_value={
        'id': 'order-12345',
        'symbol': 'BTC/USD',
        'status': 'filled',
        'price': 50000.0,
        'amount': 0.1,
        'filled': 0.1,
        'fee': {'cost': 5.0, 'currency': 'USD'}
    })
    exchange.get_order_book = AsyncMock(return_value={
        'bids': [[49900, 1.5], [49800, 2.0]],
        'asks': [[50100, 1.2], [50200, 3.0]]
    })
    exchange.cancel_order = AsyncMock(return_value=True)
    return exchange

@pytest.fixture
def execution_engine(mock_exchange):
    """Execution engine with mocked exchange"""
    return ExecutionEngine(exchange_api=mock_exchange)

@pytest.fixture
def risk_manager():
    """Risk manager with test configuration"""
    manager = RiskManager()
    manager.params = {
        'max_daily_loss': -0.05,
        'max_position_risk': 0.1,
        'volatility_shutdown': 0.15
    }
    return manager

@pytest.fixture
def trade_memory():
    """Trade memory with in-memory database"""
    return TradeMemory(db_path=":memory:")

@pytest.fixture
def sentiment_analyzer():
    """Sentiment analyzer with test model"""
    analyzer = SentimentAnalyzer()
    analyzer.model = MagicMock()
    analyzer.model.return_value = [{'label': 'positive', 'score': 0.95}]
    return analyzer

@pytest.fixture
def alert_manager():
    """Alert manager with mocked notification channels"""
    manager = AlertManager()
    manager._send_slack = AsyncMock()
    manager._send_telegram = AsyncMock()
    manager._send_email = AsyncMock()
    return manager

@pytest.fixture
def event_bus():
    """Asynchronous event bus"""
    return AsyncEventBus()

# Test Execution Engine
class TestExecutionEngine:
    """Unit tests for order execution components"""
    
    @pytest.mark.asyncio
    async def test_market_order_execution(self, execution_engine):
        """Test successful market order execution"""
        order = {
            'symbol': 'BTC/USD',
            'type': 'market',
            'side': 'buy',
            'amount': 0.1,
            'price': None
        }
        
        result = await execution_engine.market_order(order['symbol'], order['amount'])
        
        # Validate results
        assert result['status'] == 'filled'
        assert result['filled'] == order['amount']
        assert result['price'] == 50000.0
        
        # Verify exchange called
        execution_engine.exchange.create_order.assert_awaited_once_with(
            order['symbol'], 'market', 'buy', order['amount'], None, {}
        )
    
    @pytest.mark.asyncio
    async def test_limit_order_execution(self, execution_engine):
        """Test limit order placement"""
        order = {
            'symbol': 'BTC/USD',
            'type': 'limit',
            'side': 'sell',
            'amount': 0.5,
            'price': 50500.0
        }
        
        result = await execution_engine.limit_order(
            order['symbol'], order['amount'], order['price']
        )
        
        # Validate results
        assert result['status'] == 'filled'
        assert result['price'] == order['price']
        
        # Verify exchange called
        execution_engine.exchange.create_order.assert_awaited_once_with(
            order['symbol'], 'limit', 'sell', order['amount'], order['price'], {}
        )
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, execution_engine):
        """Test order cancellation"""
        order_id = 'order-67890'
        result = await execution_engine.cancel_order(order_id)
        
        assert result is True
        execution_engine.exchange.cancel_order.assert_awaited_once_with(order_id)
    
    @pytest.mark.asyncio
    async def test_slippage_control(self, execution_engine):
        """Test slippage control mechanism"""
        # Setup
        execution_engine.exchange.create_order.return_value = {
            'id': 'order-23456',
            'symbol': 'BTC/USD',
            'status': 'filled',
            'price': 49950.0,  # Worse than requested
            'amount': 0.2,
            'filled': 0.2
        }
        
        # Place order with max slippage
        execution_engine.slippage_tolerance = 0.001  # 0.1%
        order = await execution_engine.limit_order('BTC/USD', 0.2, 50000.0)
        
        # Should be accepted
        assert order['status'] == 'filled'
        
        # Place order that exceeds slippage
        execution_engine.slippage_tolerance = 0.0001  # 0.01%
        with pytest.raises(ValueError, match="Slippage exceeded"):
            await execution_engine.limit_order('BTC/USD', 0.2, 50000.0)
    
    @pytest.mark.asyncio
    async def test_iceberg_order(self, execution_engine):
        """Test iceberg order splitting"""
        order = {
            'symbol': 'BTC/USD',
            'type': 'iceberg',
            'side': 'buy',
            'amount': 10.0,  # Large order
            'price': 50000.0
        }
        
        # Set market depth
        execution_engine.exchange.get_order_book.return_value = {
            'bids': [[49900, 1.5], [49800, 2.0]],
            'asks': [[50000, 0.5], [50100, 1.2], [50200, 3.0]]
        }
        
        # Execute iceberg order
        result = await execution_engine.iceberg_order(
            order['symbol'], order['amount'], order['price']
        )
        
        # Should split into multiple orders
        assert execution_engine.exchange.create_order.call_count > 1
        total_amount = sum(call.args[3] for call in execution_engine.exchange.create_order.call_args_list)
        assert total_amount == order['amount']

# Test Risk Manager
class TestRiskManager:
    """Unit tests for risk management components"""
    
    @pytest.mark.asyncio
    async def test_position_sizing(self, risk_manager):
        """Test risk-adjusted position sizing"""
        risk_manager.capital = 100000  # $100k capital
        volatility = 0.02  # 2% daily volatility
        
        size = risk_manager.calculate_position_size(volatility)
        
        # Should be 1% of capital / volatility
        expected = 100000 * 0.01 / 0.02
        assert size == expected
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit(self, risk_manager, trade_memory):
        """Test daily loss limit enforcement"""
        risk_manager.trade_memory = trade_memory
        
        # Simulate profitable day
        risk_manager.check_daily_loss_limit()
        assert not risk_manager.kill_switch_triggered
        
        # Simulate losing day
        trade_memory.pnl_history = [
            {'total_pnl': 5000}, 
            {'total_pnl': -3000},
            {'total_pnl': -8000}  # Total loss: -$6000 (6% of $100k)
        ]
        risk_manager.capital = 100000
        risk_manager.check_daily_loss_limit()
        
        # Should trigger kill switch
        assert risk_manager.kill_switch_triggered
    
    @pytest.mark.asyncio
    async def test_volatility_shutdown(self, risk_manager, market_feeds):
        """Test volatility-based shutdown"""
        # Mock market feeds
        market_feeds.get_volatility = AsyncMock(return_value=0.12)  # 12% volatility
        risk_manager.market_feeds = market_feeds
        
        # Below threshold
        risk_manager.check_volatility_shutdown()
        assert not risk_manager.kill_switch_triggered
        
        # Above threshold
        market_feeds.get_volatility.return_value = 0.18  # 18% volatility
        risk_manager.check_volatility_shutdown()
        assert risk_manager.kill_switch_triggered
    
    @pytest.mark.asyncio
    async def test_leverage_management(self, risk_manager):
        """Test leverage adjustments based on volatility"""
        # Low volatility
        risk_manager.current_volatility = 0.05
        leverage = risk_manager.calculate_max_leverage()
        assert leverage == 3.0  # Higher leverage allowed
        
        # High volatility
        risk_manager.current_volatility = 0.15
        leverage = risk_manager.calculate_max_leverage()
        assert leverage == 1.0  # Minimum leverage
    
    @pytest.mark.asyncio
    async def test_position_risk(self, risk_manager):
        """Test position risk calculation"""
        positions = [
            {'symbol': 'BTC/USD', 'amount': 0.5, 'entry_price': 50000, 'current_price': 52000},
            {'symbol': 'ETH/USD', 'amount': 10, 'entry_price': 3000, 'current_price': 2900}
        ]
        
        risk = risk_manager.calculate_portfolio_risk(positions)
        
        # BTC: +$1000, ETH: -$1000 → Net $0
        # Risk = max(0.5, 10) * max(50000, 3000) / capital
        # Assuming capital = 100000
        risk_manager.capital = 100000
        expected_risk = max(0.5*50000, 10*3000) / 100000  # 30,000 / 100,000 = 0.3
        assert risk == expected_risk

# Test Sentiment Analysis
class TestSentimentAnalysis:
    """Unit tests for NLP sentiment analysis"""
    
    def test_sentiment_scoring(self, sentiment_analyzer):
        """Test sentiment scoring accuracy"""
        text = "Bitcoin is showing incredible strength in today's market rally!"
        result = sentiment_analyzer.analyze_sentiment(text)
        
        assert result['score'] == 1  # Positive
        assert result['confidence'] > 0.9
    
    def test_crypto_specific_adjustments(self, sentiment_analyzer):
        """Test cryptocurrency-specific adjustments"""
        # Positive boost for crypto keywords
        text = "Bitcoin to the moon! #bullish"
        result = sentiment_analyzer.calculate_crypto_specific_score(text, "BTC")
        assert result['final_score'] > 0.9
        
        # Negative detection for scam alerts
        text = "This project looks like a rug pull, avoid!"
        result = sentiment_analyzer.calculate_crypto_specific_score(text, "SHIT")
        assert result['final_score'] < -0.8
    
    def test_sentiment_decay(self, sentiment_analyzer):
        """Test temporal decay of sentiment scores"""
        # Initial strong sentiment
        text1 = "Revolutionary blockchain upgrade announced!"
        result1 = sentiment_analyzer.calculate_crypto_specific_score(text1, "ETH")
        
        # After decay
        sentiment_analyzer.sentiment_history["ETH"] = result1['final_score']
        for _ in range(3):
            sentiment_analyzer.sentiment_history["ETH"] *= sentiment_analyzer.decay_factor
        
        # New event
        text2 = "Network congestion issues reported"
        result2 = sentiment_analyzer.calculate_crypto_specific_score(text2, "ETH")
        
        assert result2['historical_component'] < result1['final_score'] * 0.8
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, sentiment_analyzer):
        """Test batch sentiment analysis performance"""
        texts = ["Positive news"] * 20 + ["Negative development"] * 20
        
        # Mock batch processing
        sentiment_analyzer.model = MagicMock(return_value=[
            {'label': 'positive', 'score': 0.95} if "Positive" in t 
            else {'label': 'negative', 'score': 0.90} for t in texts
        ])
        
        results = sentiment_analyzer.analyze_batch(texts)
        
        assert len(results) == len(texts)
        assert results[0]['score'] == 1
        assert results[-1]['score'] == -1

# Test Signal Orchestration
class TestSignalOrchestration:
    """Unit tests for trading signal generation"""
    
    @pytest.fixture
    def orchestrator(self, sentiment_analyzer):
        """Signal orchestrator with mocked dependencies"""
        orchestrator = SignalOrchestrator()
        orchestrator.sentiment_analyzer = sentiment_analyzer
        orchestrator.market_feeds = MagicMock()
        orchestrator.market_feeds.get_ohlcv = MagicMock(return_value=[
            [None, None, None, None, 50000, 100], 
            [None, None, None, None, 51000, 150]
        ])
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, orchestrator):
        """Test end-to-end signal generation"""
        news_event = {
            'text': "Major institutional adoption of Bitcoin announced",
            'symbol': 'BTC',
            'source': 'coindesk',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        signal = await orchestrator.generate_trading_signal('BTC/USD', news_event)
        
        assert 'final_score' in signal
        assert 'components' in signal
        assert signal['confidence'] > 0.7
    
    @pytest.mark.asyncio
    async def test_technical_indicators(self, orchestrator):
        """Test technical indicator calculations"""
        # Mock OHLCV data: [timestamp, open, high, low, close, volume]
        ohlcv = [
            [None, None, None, None, 49000, 100],
            [None, None, None, None, 49500, 120],
            [None, None, None, None, 50500, 150],
            [None, None, None, None, 51000, 200],
            [None, None, None, None, 51500, 180]  # Latest
        ]
        orchestrator.market_feeds.get_ohlcv.return_value = ohlcv
        
        indicators = await orchestrator.get_technical_signals('BTC/USD')
        
        # Validate RSI
        assert 30 < indicators['rsi'] < 70
        
        # Validate MACD
        assert indicators['macd'] > 0  # Should be positive in uptrend
    
    def test_signal_fusion(self, orchestrator):
        """Test multi-source signal fusion"""
        normalized_signals = {
            'sentiment': 0.8,
            'rsi': -0.2,  # Overbought
            'macd': 0.1,
            'bollinger': 0.7,  # Near top of band
            'volume': 0.5,
            'nvt': 0.4,
            'netflow': -0.3,
            'mvr': 0.6,
            'hash_rate': 0.5
        }
        
        # Set weights
        orchestrator.signal_weights = {
            'sentiment': 0.4,
            'technical': 0.4,
            'on_chain': 0.2
        }
        
        fused = orchestrator.fuse_signals(normalized_signals)
        
        # Sentiment positive but technicals mixed → neutral
        assert -0.3 < fused['final_score'] < 0.3

# Test Infrastructure Components
class TestInfrastructure:
    """Unit tests for core infrastructure components"""
    
    @pytest.mark.asyncio
    async def test_alert_manager(self, alert_manager):
        """Test alert delivery mechanisms"""
        await alert_manager.send_alert(
            "TEST_ALERT", 
            "This is a test alert", 
            severity="CRITICAL"
        )
        
        # Verify delivery channels
        alert_manager._send_slack.assert_awaited()
        alert_manager._send_telegram.assert_awaited()
        alert_manager._send_email.assert_awaited()
    
    @pytest.mark.asyncio
    async def test_event_bus(self, event_bus):
        """Test event bus pub/sub functionality"""
        # Create mock subscriber
        subscriber = AsyncMock()
        event_bus.subscribe('test_event', subscriber)
        
        # Publish event
        test_data = {'value': 42}
        await event_bus.publish('test_event', test_data)
        
        # Verify delivery
        subscriber.assert_awaited_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_database_manager(self):
        """Test database CRUD operations"""
        db = DatabaseManager(db_path=":memory:")
        
        # Test trade signal insertion
        signal_id = await db.insert_trade_signal({
            'symbol': 'TEST',
            'signal_type': 'unit_test',
            'score': 0.95,
            'confidence': 0.92,
            'source': 'pytest'
        })
        assert signal_id > 0
        
        # Test retrieval
        signals = await db.get_latest_signals('TEST', limit=1)
        assert len(signals) == 1
        assert signals[0]['score'] == 0.95

# Test Data Components
class TestDataComponents:
    """Unit tests for data ingestion components"""
    
    @pytest.fixture
    def market_feed(self, mock_exchange):
        """Market data feed with mocked exchange"""
        feed = MarketDataFeed()
        feed.exchange = mock_exchange
        return feed
    
    @pytest.fixture
    def sentinel_agent(self):
        """Sentinel agent with mocked APIs"""
        agent = SentinelAgent()
        agent.fetch_alpha_vantage_news = AsyncMock(return_value=[])
        agent.monitor_twitter = AsyncMock(return_value=[])
        agent.fetch_coindesk_news = AsyncMock(return_value=[])
        return agent
    
    @pytest.mark.asyncio
    async def test_market_data_stream(self, market_feed, event_bus):
        """Test market data streaming"""
        # Register event handler
        handler = AsyncMock()
        event_bus.subscribe('orderbook', handler)
        market_feed.event_bus = event_bus
        
        # Start feed
        market_feed.running = True
        await market_feed.start_orderbook_feed()
        
        # Verify exchange called
        market_feed.exchange.watch_order_book.assert_awaited()
        
        # Simulate data
        test_data = {'symbol': 'BTC/USD', 'bids': [[50000, 1]], 'asks': [[50100, 1]]}
        await market_feed.event_bus.publish('orderbook', test_data)
        
        # Verify handler
        handler.assert_awaited_with(test_data)
    
    @pytest.mark.asyncio
    async def test_news_collection(self, sentinel_agent, event_bus):
        """Test news collection pipeline"""
        # Mock news data
        sentinel_agent.fetch_alpha_vantage_news.return_value = [{
            'title': 'Test News',
            'content': 'This is a test news article',
            'symbol': 'BTC'
        }]
        
        # Register handler
        handler = AsyncMock()
        event_bus.subscribe('raw_news', handler)
        sentinel_agent.event_bus = event_bus
        
        # Run collection
        sentinel_agent.running = True
        await sentinel_agent.start_news_collection()
        
        # Verify processing
        assert handler.await_count > 0

# Run all tests from command line
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", "-s", __file__]))