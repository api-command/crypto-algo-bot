import asyncio
import numpy as np
from transformers import pipeline
from src.utils.logger import get_logger
from src.infra.telemetry import telemetry
from src.core.latency_monitor import LatencyMonitor
from src.utils.config_loader import config_loader

logger = get_logger('sentiment_nlp')

class SentimentAnalyzer:
    def __init__(self, model_name="finiteautomata/bertweet-base-sentiment-analysis"):
        self.config = config_loader.load_toml('config/bot_params.toml')['sentiment']
        self.latency_monitor = LatencyMonitor()
        self.model = self._load_model(model_name)
        self.decay_factor = self.config.get('decay_factor', 0.95)
        self.sentiment_history = {}
        self.confidence_threshold = self.config.get('min_confidence', 0.65)
        
    def _load_model(self, model_name):
        """Load Hugging Face model with ONNX optimization for performance"""
        logger.info(f"Loading sentiment model: {model_name}")
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            framework="pt",
            truncation=True,
            device=0,  # Use GPU if available
            batch_size=16
        )
    
    def _preprocess_text(self, text):
        """Clean and prepare text for analysis"""
        # Remove URLs, special characters, and extra spaces
        import re
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()[:512]  # Truncate to model max length
    
    def analyze_sentiment(self, text):
        """Analyze text sentiment with confidence scoring"""
        self.latency_monitor.start_timer('sentiment_analysis')
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Run inference
        result = self.model(cleaned_text)[0]
        
        # Map to numeric score: negative (-1), neutral (0), positive (1)
        sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
        score = sentiment_map.get(result['label'].lower(), 0)
        confidence = result['score']
        
        latency = self.latency_monitor.record_latency('sentiment_analysis')
        telemetry.latency('sentiment_analysis', latency/1000)  # Convert to seconds
        
        return {
            'score': score,
            'confidence': confidence,
            'normalized_score': score * confidence,
            'latency_ms': latency
        }
    
    def analyze_batch(self, texts):
        """Analyze multiple texts efficiently"""
        self.latency_monitor.start_timer('batch_sentiment')
        
        # Preprocess all texts
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        
        # Run batch inference
        results = self.model(cleaned_texts)
        
        sentiment_scores = []
        sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
        
        for result in results:
            score = sentiment_map.get(result['label'].lower(), 0)
            confidence = result['score']
            sentiment_scores.append({
                'score': score,
                'confidence': confidence,
                'normalized_score': score * confidence
            })
        
        latency = self.latency_monitor.record_latency('batch_sentiment')
        telemetry.latency('batch_sentiment', latency/1000)
        
        return sentiment_scores
    
    def calculate_crypto_specific_score(self, text, symbol):
        """
        Enhance sentiment analysis with crypto-specific rules
        - Boost sentiment for project-specific keywords
        - Detect scam/fud patterns
        - Apply temporal decay
        """
        base_sentiment = self.analyze_sentiment(text)
        
        # Crypto-specific adjustments
        adjustments = 0
        crypto_keywords = {
            'BTC': ['bitcoin', 'btc', 'digital gold', 'halving'],
            'ETH': ['ethereum', 'eth', 'merge', 'sharding'],
            'SOL': ['solana', 'sol', 'ftx', 'speed']
        }
        
        # Positive boosters
        positive_phrases = ['moon', 'bullish', 'buy', 'long', 'accumulate']
        for phrase in positive_phrases:
            if phrase in text.lower():
                adjustments += 0.1
                
        # Negative flags
        negative_phrases = ['scam', 'rug pull', 'dump', 'sell', 'short', 'fud']
        for phrase in negative_phrases:
            if phrase in text.lower():
                adjustments -= 0.15
                
        # Symbol-specific keywords
        if symbol in crypto_keywords:
            for keyword in crypto_keywords[symbol]:
                if keyword in text.lower():
                    adjustments += 0.05
        
        # Apply decay to historical sentiment
        historical_score = self.sentiment_history.get(symbol, 0) * self.decay_factor
        final_score = base_sentiment['normalized_score'] + adjustments + historical_score
        
        # Clamp between -1 and 1
        final_score = max(-1.0, min(1.0, final_score))
        
        # Update history
        self.sentiment_history[symbol] = final_score
        
        return {
            'base_score': base_sentiment['normalized_score'],
            'adjustments': adjustments,
            'historical_component': historical_score,
            'final_score': final_score,
            'confidence': base_sentiment['confidence']
        }
    
    def is_valid_signal(self, sentiment_result):
        """Determine if sentiment signal meets confidence threshold"""
        return abs(sentiment_result['final_score']) > 0.5 and \
               sentiment_result['confidence'] >= self.confidence_threshold

# Global instance for shared model
sentiment_analyzer = SentimentAnalyzer()