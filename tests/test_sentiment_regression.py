import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.ml.sentiment_nlp import SentimentAnalyzer
from src.utils.db import db_manager
from src.utils.logger import get_logger

logger = get_logger('sentiment_test')

class TestSentimentAnalysis:
    """Comprehensive regression tests for sentiment analysis pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.analyzer = SentimentAnalyzer()
        self.test_samples = self._load_test_cases()
        self.historical_data = None
        
    def _load_test_cases(self):
        """Load labeled test cases for sentiment analysis"""
        return [
            {
                "text": "Bitcoin is skyrocketing to new all-time highs!",
                "expected_sentiment": "positive",
                "symbol": "BTC",
                "min_confidence": 0.7
            },
            {
                "text": "The SEC is cracking down on crypto exchanges, causing panic selling",
                "expected_sentiment": "negative",
                "symbol": "BTC",
                "min_confidence": 0.65
            },
            {
                "text": "Ethereum network fees remain stable around 20 gwei",
                "expected_sentiment": "neutral",
                "symbol": "ETH",
                "min_confidence": 0.6
            },
            {
                "text": "🚀🚀 SOL to the moon! 10x incoming! #bullish",
                "expected_sentiment": "positive",
                "symbol": "SOL",
                "min_confidence": 0.75
            },
            {
                "text": "Rug pull alert: Dev team dumped all tokens, price crashed 95%",
                "expected_sentiment": "negative",
                "symbol": "SHITCOIN",
                "min_confidence": 0.8
            }
        ]

    async def _load_historical_data(self, days=30):
        """Load historical sentiment data from database"""
        if self.historical_data is None:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Query news events and price changes
            news_events = await db_manager.fetch_all('''
                SELECT symbol, content, timestamp 
                FROM news_events 
                WHERE timestamp BETWEEN ? AND ?
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            price_data = await db_manager.fetch_all('''
                SELECT symbol, close, timestamp 
                FROM market_data 
                WHERE resolution = '1h' 
                AND timestamp BETWEEN ? AND ?
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            # Create DataFrame and calculate hourly returns
            df_news = pd.DataFrame(news_events)
            df_prices = pd.DataFrame(price_data)
            
            if not df_prices.empty:
                df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])
                df_prices = df_prices.sort_values(['symbol', 'timestamp'])
                df_prices['price_change'] = df_prices.groupby('symbol')['close'].pct_change()
            
            self.historical_data = {
                'news': df_news,
                'prices': df_prices
            }
        
        return self.historical_data

    @pytest.mark.parametrize("test_case", _load_test_cases(None))
    def test_sentiment_accuracy(self, test_case):
        """Test basic sentiment classification accuracy"""
        result = self.analyzer.analyze_sentiment(test_case['text'])
        
        assert result['label'].lower() == test_case['expected_sentiment'], \
            f"Sentiment mismatch for: {test_case['text']}"
            
        assert result['score'] >= test_case['min_confidence'], \
            f"Confidence too low ({result['score']}) for: {test_case['text']}"

    @pytest.mark.parametrize("symbol", ["BTC", "ETH", "SOL"])
    def test_crypto_specific_rules(self, symbol):
        """Test cryptocurrency-specific sentiment adjustments"""
        # Positive keywords test
        positive_text = f"{symbol} is the future of finance! Adoption growing exponentially."
        pos_result = self.analyzer.calculate_crypto_specific_score(positive_text, symbol)
        assert pos_result['final_score'] > 0.7, f"Positive boost failed for {symbol}"
        
        # Negative keywords test
        negative_text = f"Avoid {symbol} - major security vulnerability discovered!"
        neg_result = self.analyzer.calculate_crypto_specific_score(negative_text, symbol)
        assert neg_result['final_score'] < -0.7, f"Negative detection failed for {symbol}"
        
        # Neutral test
        neutral_text = f"{symbol} price remains unchanged this week."
        neutral_result = self.analyzer.calculate_crypto_specific_score(neutral_text, symbol)
        assert -0.3 < neutral_result['final_score'] < 0.3, f"Neutral detection failed for {symbol}"

    @pytest.mark.asyncio
    async def test_sentiment_decay(self):
        """Test temporal decay of sentiment scores"""
        symbol = "BTC"
        text1 = "Bitcoin ETF approved! Institutional floodgates opening!"
        text2 = "Market correction expected after recent rally"
        
        # Initial strong positive sentiment
        result1 = self.analyzer.calculate_crypto_specific_score(text1, symbol)
        assert result1['final_score'] > 0.8
        
        # Check decay after simulated time
        original_decay = self.analyzer.decay_factor
        self.analyzer.decay_factor = 0.9  # Faster decay for test
        
        # Should decay significantly after 5 iterations
        for _ in range(5):
            self.analyzer.calculate_crypto_specific_score("", symbol)  # Force decay
        
        result2 = self.analyzer.calculate_crypto_specific_score(text2, symbol)
        assert result2['historical_component'] < result1['final_score'] * 0.5, \
            "Sentiment decay not working properly"
        
        self.analyzer.decay_factor = original_decay  # Reset

    @pytest.mark.asyncio
    async def test_sentiment_price_correlation(self):
        """Test correlation between sentiment and price movements"""
        data = await self._load_historical_data(days=30)
        if data['news'].empty or data['prices'].empty:
            pytest.skip("Insufficient historical data for correlation test")
        
        # Analyze sentiment for all news
        sentiments = []
        for _, row in data['news'].iterrows():
            result = self.analyzer.calculate_crypto_specific_score(row['content'], row['symbol'])
            sentiments.append({
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'sentiment': result['final_score']
            })
        
        df_sentiment = pd.DataFrame(sentiments)
        df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'])
        
        # Merge with price data and calculate hourly correlations
        merged = pd.merge_asof(
            df_sentiment.sort_values('timestamp'),
            data['prices'].sort_values('timestamp'),
            on='timestamp',
            by='symbol',
            direction='forward'
        )
        
        # Calculate correlations
        correlations = merged.groupby('symbol')[['sentiment', 'price_change']].corr().iloc[0::2, -1]
        correlations = correlations.reset_index().drop(columns='level_1')
        correlations.columns = ['symbol', 'correlation']
        
        logger.info(f"Sentiment-Price Correlations:\n{correlations}")
        
        # Assert positive correlation for major coins
        for symbol in ['BTC', 'ETH']:
            if symbol in correlations['symbol'].values:
                corr = correlations[correlations['symbol'] == symbol]['correlation'].values[0]
                assert corr > 0.1, f"Low correlation ({corr:.2f}) for {symbol}"
            else:
                pytest.skip(f"No data for {symbol}")

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch sentiment analysis performance"""
        texts = [case['text'] for case in self.test_samples] * 10  # 50 samples
        batch_size = 16  # Match model's batch size
        
        # Time single-threaded processing
        start_single = time.perf_counter()
        single_results = []
        for text in texts:
            single_results.append(self.analyzer.analyze_sentiment(text))
        single_time = time.perf_counter() - start_single
        
        # Time batch processing
        start_batch = time.perf_counter()
        batch_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results.extend(self.analyzer.analyze_batch(batch))
        batch_time = time.perf_counter() - start_batch
        
        # Verify results consistency
        for single, batch in zip(single_results[:5], batch_results[:5]):  # Check subset
            assert single['label'] == batch['label'], "Batch result mismatch"
            assert abs(single['score'] - batch['score']) < 0.1, "Score deviation too high"
        
        # Verify performance gain
        speedup = single_time / batch_time
        logger.info(f"Batch processing speedup: {speedup:.1f}x")
        assert speedup > 1.5, f"Insufficient batch processing speedup ({speedup:.1f}x)"

    def test_special_characters_handling(self):
        """Test handling of social media special characters"""
        test_cases = [
            {
                "text": "BTC 🔥🔥🔥 #ToTheMoon",
                "expected": "positive"
            },
            {
                "text": "Warning! ETH hack alert! 🚨🚨 #Scam",
                "expected": "negative"
            },
            {
                "text": "SOL update v1.2.3 released ⚙️",
                "expected": "neutral"
            }
        ]
        
        for case in test_cases:
            result = self.analyzer.analyze_sentiment(case['text'])
            assert result['label'].lower() == case['expected'], \
                f"Failed on text: {case['text']}"

    @pytest.mark.asyncio
    async def test_model_consistency(self):
        """Test model produces consistent results over time"""
        test_text = "Bitcoin volatility increases as macroeconomic uncertainty grows"
        
        # Run analysis multiple times
        results = []
        for _ in range(10):
            results.append(self.analyzer.analyze_sentiment(test_text))
        
        # Check consistency
        first_result = results[0]
        for result in results[1:]:
            assert result['label'] == first_result['label'], "Label inconsistency"
            assert abs(result['score'] - first_result['score']) < 0.05, "Score variance too high"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test robustness against malformed inputs"""
        test_cases = [
            None,
            "",
            "   ",
            12345,
            "X" * 1000  # Very long string
        ]
        
        for case in test_cases:
            try:
                result = self.analyzer.analyze_sentiment(case)
                assert isinstance(result, dict), "Invalid return type"
                assert 'label' in result and 'score' in result, "Missing keys"
            except Exception as e:
                pytest.fail(f"Failed on input {case} with error: {str(e)}")

# Performance benchmark (not a regular test)
def test_sentiment_benchmark(benchmark):
    """Performance benchmark for sentiment analysis"""
    analyzer = SentimentAnalyzer()
    test_text = "The cryptocurrency market is experiencing unprecedented growth this quarter"
    
    def analyze():
        return analyzer.analyze_sentiment(test_text)
    
    result = benchmark(analyze)
    
    assert result['score'] > 0.5  # Sanity check
    logger.info(f"Benchmark result: {result}")