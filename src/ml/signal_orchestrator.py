import asyncio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger
from src.infra.telemetry import telemetry
from src.core.latency_monitor import LatencyMonitor
from src.utils.config_loader import config_loader
from src.ml.sentiment_nlp import sentiment_analyzer
from src.ml.alpha_signal import AlphaSignalGenerator
from src.data.market_feeds import MarketDataFeed

logger = get_logger('signal_orchestrator')

class SignalOrchestrator:
    def __init__(self):
        self.config = config_loader.load_toml('config/bot_params.toml')['signals']
        self.latency_monitor = LatencyMonitor()
        self.signal_weights = self._init_weights()
        self.alpha_generator = AlphaSignalGenerator()
        self.market_feed = MarketDataFeed()
        self.model = self._init_model()
        self.signal_history = []
        
    def _init_weights(self):
        """Initialize signal weights based on config"""
        return {
            'sentiment': self.config['weights']['sentiment'],
            'technical': self.config['weights']['technical'],
            'on_chain': self.config['weights']['on_chain']
        }
    
    def _init_model(self):
        """Initialize ML model for signal fusion"""
        # In production, this would be a trained model
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def update_weights_based_on_volatility(self, volatility):
        """Adjust signal weights based on market conditions"""
        # High volatility: More weight to technicals
        # Low volatility: More weight to fundamentals/sentiment
        if volatility > 0.8:  # High volatility
            self.signal_weights = {
                'sentiment': 0.2,
                'technical': 0.6,
                'on_chain': 0.2
            }
        elif volatility > 0.5:  # Medium volatility
            self.signal_weights = {
                'sentiment': 0.4,
                'technical': 0.4,
                'on_chain': 0.2
            }
        else:  # Low volatility
            self.signal_weights = {
                'sentiment': 0.5,
                'technical': 0.3,
                'on_chain': 0.2
            }
    
    async def get_technical_signals(self, symbol):
        """Generate technical indicators from market data"""
        self.latency_monitor.start_timer('technical_signals')
        
        # Get OHLCV data
        ohlcv = self.market_feed.get_ohlcv(symbol)
        if not ohlcv or len(ohlcv) < 50:
            return {}
        
        closes = np.array([c[4] for c in ohlcv])
        volumes = np.array([c[5] for c in ohlcv])
        
        # Calculate indicators
        signals = {}
        
        # RSI
        delta = closes[-14:] - np.roll(closes[-14:], 1)
        gain = np.where(delta > 0, delta, 0).sum()
        loss = np.where(delta < 0, -delta, 0).sum()
        signals['rsi'] = 100 - (100 / (1 + (gain / loss))) if loss != 0 else 100
        
        # MACD
        ema12 = closes[-26:].mean()
        ema26 = closes[-26:].mean()
        signals['macd'] = ema12 - ema26
        signals['macd_signal'] = closes[-9:].mean()
        
        # Bollinger Bands
        sma20 = closes[-20:].mean()
        std20 = closes[-20:].std()
        signals['bollinger_upper'] = sma20 + 2 * std20
        signals['bollinger_lower'] = sma20 - 2 * std20
        signals['bollinger_percent'] = (closes[-1] - signals['bollinger_lower']) / \
                                      (signals['bollinger_upper'] - signals['bollinger_lower'])
        
        # Volume analysis
        signals['volume_ma'] = volumes[-10:].mean()
        signals['volume_spike'] = 1 if volumes[-1] > 2 * signals['volume_ma'] else 0
        
        latency = self.latency_monitor.record_latency('technical_signals')
        telemetry.latency('technical_signals', latency/1000)
        return signals
    
    async def get_on_chain_signals(self, symbol):
        """Fetch on-chain metrics (simplified)"""
        # In production, integrate with Glassnode, Santiment, etc.
        self.latency_monitor.start_timer('on_chain_signals')
        
        # Placeholder values
        signals = {
            'nvt': np.random.uniform(0.7, 1.3),  # Network Value to Transaction
            'exchange_netflow': np.random.uniform(-0.1, 0.1),
            'mvr': np.random.uniform(0.8, 1.2),   # Market Value to Realized Value
            'hash_rate': np.random.uniform(0.9, 1.1)
        }
        
        latency = self.latency_monitor.record_latency('on_chain_signals')
        telemetry.latency('on_chain_signals', latency/1000)
        return signals
    
    def normalize_signals(self, signals):
        """Normalize all signals to [-1, 1] range"""
        normalized = {}
        
        # Sentiment score is already normalized
        normalized['sentiment'] = signals.get('sentiment', 0)
        
        # Technical indicators
        tech = signals.get('technical', {})
        normalized['rsi'] = (tech.get('rsi', 50) - 50) / 50
        normalized['macd'] = tech.get('macd', 0) / 0.05  # Normalize MACD difference
        normalized['bollinger'] = (tech.get('bollinger_percent', 0.5) - 0.5) * 4
        normalized['volume'] = tech.get('volume_spike', 0) * 0.5
        
        # On-chain metrics
        chain = signals.get('on_chain', {})
        normalized['nvt'] = (chain.get('nvt', 1.0) - 1.0) * 2
        normalized['netflow'] = chain.get('exchange_netflow', 0) * 10
        normalized['mvr'] = (chain.get('mvr', 1.0) - 1.0) * 2
        normalized['hash_rate'] = (chain.get('hash_rate', 1.0) - 1.0) * 2
        
        return normalized
    
    def fuse_signals(self, normalized_signals):
        """
        Fuse multiple signals into a single trading decision
        Uses weighted average + ML model ensemble
        """
        # Weighted average method (simple baseline)
        weighted_score = 0
        weights_sum = 0
        
        # Sentiment component
        sentiment_weight = self.signal_weights['sentiment']
        weighted_score += normalized_signals['sentiment'] * sentiment_weight
        weights_sum += sentiment_weight
        
        # Technical component (average of technical indicators)
        tech_weight = self.signal_weights['technical']
        tech_signals = [
            normalized_signals['rsi'],
            normalized_signals['macd'],
            normalized_signals['bollinger'],
            normalized_signals['volume']
        ]
        tech_score = np.mean(tech_signals)
        weighted_score += tech_score * tech_weight
        weights_sum += tech_weight
        
        # On-chain component (average of on-chain metrics)
        chain_weight = self.signal_weights['on_chain']
        chain_signals = [
            normalized_signals['nvt'],
            normalized_signals['netflow'],
            normalized_signals['mvr'],
            normalized_signals['hash_rate']
        ]
        chain_score = np.mean(chain_signals)
        weighted_score += chain_score * chain_weight
        weights_sum += chain_weight
        
        weighted_avg = weighted_score / weights_sum
        
        # ML model prediction (would require training data in production)
        # features = list(normalized_signals.values())
        # ml_score = self.model.predict([features])[0]  # In practice
        
        # For now, return weighted average
        return {
            'final_score': weighted_avg,
            'components': {
                'sentiment': normalized_signals['sentiment'],
                'technical': tech_score,
                'on_chain': chain_score
            },
            'weights': self.signal_weights
        }
    
    async def generate_trading_signal(self, symbol, news_event=None):
        """
        Generate comprehensive trading signal from all sources
        :param symbol: Trading symbol (e.g., BTC/USD)
        :param news_event: Optional news event to trigger sentiment analysis
        """
        self.latency_monitor.start_timer('full_signal')
        
        # Get signals from all sources
        signal_sources = {}
        
        # Sentiment analysis (if news provided)
        if news_event:
            sentiment_signal = await self.alpha_generator.generate_signal(news_event)
            signal_sources['sentiment'] = sentiment_signal['fused_score']
        
        # Technical indicators
        signal_sources['technical'] = await self.get_technical_signals(symbol)
        
        # On-chain metrics
        signal_sources['on_chain'] = await self.get_on_chain_signals(symbol)
        
        # Normalize all signals
        normalized = self.normalize_signals(signal_sources)
        
        # Fuse signals
        fused_signal = self.fuse_signals(normalized)
        
        # Add metadata
        latency = self.latency_monitor.record_latency('full_signal')
        fused_signal.update({
            'symbol': symbol,
            'latency_ms': latency,
            'timestamp': news_event['timestamp'] if news_event else time.time(),
            'source': 'ensemble'
        })
        
        # Store for historical analysis
        self.signal_history.append(fused_signal)
        if len(self.signal_history) > 1000:
            self.signal_history.pop(0)
        
        telemetry.latency('full_signal_generation', latency/1000)
        return fused_signal
    
    def get_signal_strength(self, signal):
        """Convert score to actionable strength"""
        score = signal['final_score']
        if score > 0.8:
            return 'strong_buy'
        elif score > 0.3:
            return 'buy'
        elif score < -0.8:
            return 'strong_sell'
        elif score < -0.3:
            return 'sell'
        return 'neutral'
    
    def generate_trade_recommendation(self, signal, current_position):
        """
        Generate trade recommendation based on signal and current position
        :return: dict with action, size, and confidence
        """
        action = self.get_signal_strength(signal)
        confidence = min(1.0, abs(signal['final_score']) * 0.8 + 0.2)  # 0.2-1.0 range
        
        # Position-aware recommendations
        if action == 'strong_buy' and current_position <= 0:
            return {'action': 'buy', 'size': 'large', 'confidence': confidence}
        elif action == 'buy' and current_position <= 0:
            return {'action': 'buy', 'size': 'medium', 'confidence': confidence}
        elif action == 'strong_sell' and current_position >= 0:
            return {'action': 'sell', 'size': 'large', 'confidence': confidence}
        elif action == 'sell' and current_position >= 0:
            return {'action': 'sell', 'size': 'medium', 'confidence': confidence}
        elif action == 'strong_buy' and current_position > 0:
            return {'action': 'hold', 'size': None, 'confidence': 0.7}
        elif action == 'strong_sell' and current_position < 0:
            return {'action': 'hold', 'size': None, 'confidence': 0.7}
        
        return {'action': 'hold', 'size': None, 'confidence': 0.5}