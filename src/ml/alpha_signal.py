import aiohttp
import asyncio
import numpy as np
from src.utils.logger import get_logger
from src.infra.telemetry import telemetry
from src.core.latency_monitor import LatencyMonitor
from src.utils.config_loader import config_loader
from src.ml.sentiment_nlp import sentiment_analyzer

logger = get_logger('alpha_signal')

class AlphaSignalGenerator:
    def __init__(self):
        self.config = config_loader.load_toml('config/bot_params.toml')['signals']
        self.api_keys = config_loader.load_secure_yml('config/api_keys.yml')
        self.latency_monitor = LatencyMonitor()
        self.sentiment_weight = self.config['weights']['sentiment']
        self.fundamental_weight = 1.0 - self.sentiment_weight
    
    async def get_alpha_vantage_data(self, symbol):
        """Fetch fundamental data from Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_keys['alpha_vantage']['api_key']
        }
        
        self.latency_monitor.start_timer('alpha_vantage')
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    latency = self.latency_monitor.record_latency('alpha_vantage')
                    telemetry.latency('alpha_vantage', latency/1000)
                    return data
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return {}
    
    def calculate_fundamental_score(self, data):
        """Convert fundamental data to numeric score (0-1)"""
        if not data:
            return 0.5  # Neutral if no data
        
        score = 0.5  # Base neutral score
        
        try:
            # Earnings quality
            if 'PERatio' in data:
                pe = float(data['PERatio'])
                # Normalize PE (lower is better for value)
                pe_score = 1.0 - min(1.0, max(0.0, pe / 50))
                score += 0.15 * pe_score
            
            # Growth potential
            if 'EPS' in data and 'QuarterlyEarningsGrowthYOY' in data:
                eps_growth = float(data['QuarterlyEarningsGrowthYOY'])
                growth_score = min(1.0, max(0.0, eps_growth / 100))
                score += 0.25 * growth_score
            
            # Profitability
            if 'ProfitMargin' in data:
                margin = float(data['ProfitMargin'])
                margin_score = min(1.0, max(0.0, margin / 50))
                score += 0.15 * margin_score
            
            # Market sentiment
            if 'AnalystTargetPrice' in data and 'Price' in data:
                target = float(data['AnalystTargetPrice'])
                price = float(data['Price'])
                upside = (target - price) / price
                upside_score = min(1.0, max(0.0, (upside + 0.5) / 1.0))
                score += 0.20 * upside_score
            
            # Volatility
            if 'Beta' in data:
                beta = float(data['Beta'])
                # Lower beta is better for risk-adjusted returns
                beta_score = 1.0 - min(1.0, max(0.0, beta / 2))
                score += 0.10 * beta_score
                
            # Dividend (less relevant for crypto but included for completeness)
            if 'DividendYield' in data:
                yield_val = float(data['DividendYield'])
                yield_score = min(1.0, max(0.0, yield_val / 10))
                score += 0.15 * yield_score
                
            # Normalize to 0-1 range
            score = max(0.0, min(1.0, score))
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing fundamental data: {e}")
            score = 0.5
        
        return score
    
    async def generate_signal(self, news_event):
        """
        Generate trading signal by fusing sentiment and fundamental analysis
        :param news_event: Dictionary with keys 'text', 'symbol', 'source'
        :return: Signal dictionary with score, confidence, and metadata
        """
        symbol = news_event.get('symbol', 'BTC')
        text = news_event.get('text', '')
        
        # Run sentiment analysis
        sentiment_result = sentiment_analyzer.calculate_crypto_specific_score(text, symbol)
        
        # Get fundamental data
        fundamental_data = await self.get_alpha_vantage_data(symbol)
        fundamental_score = self.calculate_fundamental_score(fundamental_data)
        
        # Fuse scores
        sentiment_component = sentiment_result['final_score'] * self.sentiment_weight
        fundamental_component = (fundamental_score * 2 - 1) * self.fundamental_weight
        fused_score = sentiment_component + fundamental_component
        
        # Confidence as weighted average
        sentiment_conf = sentiment_result['confidence']
        fundamental_conf = min(1.0, len(fundamental_data)/10)  # Confidence based on data completeness
        confidence = (sentiment_conf * self.sentiment_weight + 
                     fundamental_conf * self.fundamental_weight)
        
        return {
            'symbol': symbol,
            'fused_score': fused_score,
            'confidence': confidence,
            'sentiment_score': sentiment_result['final_score'],
            'fundamental_score': fundamental_score,
            'sentiment_confidence': sentiment_conf,
            'fundamental_confidence': fundamental_conf,
            'timestamp': news_event.get('timestamp'),
            'source': news_event.get('source')
        }
    
    def interpret_signal(self, signal):
        """Convert signal score to trading action"""
        if signal['confidence'] < self.config.get('min_confidence', 0.65):
            return 'hold'
        
        score = signal['fused_score']
        if score > self.config['positive_threshold']:
            return 'strong_buy'
        elif score > self.config['weak_positive_threshold']:
            return 'buy'
        elif score < -self.config['negative_threshold']:
            return 'strong_sell'
        elif score < -self.config['weak_negative_threshold']:
            return 'sell'
        return 'hold'