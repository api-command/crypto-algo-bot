import aiohttp
import asyncio
import json
import time
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.infra.microservices import AsyncEventBus
from src.core.latency_monitor import LatencyMonitor
from src.utils.config_loader import config_loader

logger = get_logger('sentinel_agent')

class SentinelAgent:
    def __init__(self, sources=None):
        self.sources = sources or ['alpha_vantage', 'twitter', 'coindesk']
        self.api_keys = config_loader.load_secure_yml('config/api_keys.yml')
        self.event_bus = AsyncEventBus()
        self.latency_monitor = LatencyMonitor()
        self.session = None
        self.news_buffer = []
        self.last_fetch_time = {}
        self.running = False

    async def init_session(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    async def fetch_alpha_vantage_news(self, symbols=None):
        """Fetch news from Alpha Vantage API"""
        if not symbols:
            symbols = ['BTC', 'ETH', 'SOL']
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ",".join(symbols),
            'apikey': self.api_keys['alpha_vantage']['api_key'],
            'limit': 50
        }
        
        try:
            self.latency_monitor.start_timer('av_news')
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                self.latency_monitor.record_latency('av_news')
                
                articles = data.get('feed', [])
                logger.info(f"Received {len(articles)} news articles from Alpha Vantage")
                return articles
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return []

    async def monitor_twitter(self, keywords=None, limit=100):
        """Monitor Twitter for crypto keywords using filtered stream"""
        if not keywords:
            keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain']
        
        bearer_token = self.api_keys['twitter']['bearer_token']
        rules = [{"value": f"{kw} lang:en", "tag": kw} for kw in keywords]
        url = "https://api.twitter.com/2/tweets/search/stream"
        
        # Set up rules
        async with self.session.post(
            "https://api.twitter.com/2/tweets/search/stream/rules",
            headers={"Authorization": f"Bearer {bearer_token}"},
            json={"add": rules}
        ) as response:
            if response.status != 201:
                logger.error(f"Twitter rules error: {await response.text()}")
                return []
        
        # Stream tweets
        tweets = []
        try:
            self.latency_monitor.start_timer('twitter_stream')
            async with self.session.get(
                url,
                headers={"Authorization": f"Bearer {bearer_token}"},
                timeout=30
            ) as response:
                async for line in response.content:
                    if not self.running:
                        break
                    if line:
                        tweet = json.loads(line)
                        tweets.append({
                            'id': tweet['data']['id'],
                            'text': tweet['data']['text'],
                            'created_at': datetime.utcnow().isoformat(),
                            'source': 'twitter'
                        })
                        if len(tweets) >= limit:
                            break
            self.latency_monitor.record_latency('twitter_stream')
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"Twitter stream error: {e}")
        
        return tweets

    async def fetch_coindesk_news(self):
        """Fetch latest news from CoinDesk"""
        url = "https://www.coindesk.com/wp-json/v1/news/listing?page=1&size=10"
        
        try:
            self.latency_monitor.start_timer('coindesk')
            async with self.session.get(url) as response:
                data = await response.json()
                self.latency_monitor.record_latency('coindesk')
                
                articles = []
                for item in data.get('results', []):
                    articles.append({
                        'title': item['title'],
                        'description': item['excerpt'],
                        'url': item['url'],
                        'published_at': item['date'],
                        'source': 'coindesk'
                    })
                logger.info(f"Received {len(articles)} news articles from CoinDesk")
                return articles
        except Exception as e:
            logger.error(f"CoinDesk error: {e}")
            return []

    async def process_news(self, articles, source):
        """Process and publish news articles"""
        for article in articles:
            # Add metadata
            article['source'] = source
            article['received_at'] = datetime.utcnow().isoformat()
            
            # Publish raw news event
            await self.event_bus.publish('raw_news', article)
            
            # Add to buffer for batching
            self.news_buffer.append(article)
            
            logger.debug(f"Received news: {article.get('title', article.get('text', 'Untitled'))[:50]}...")

    async def start_news_collection(self):
        """Main news collection loop"""
        await self.init_session()
        
        while self.running:
            # Fetch from each source with appropriate throttling
            sources_to_fetch = []
            
            # Alpha Vantage: 1 request per minute
            if 'alpha_vantage' in self.sources:
                last_fetch = self.last_fetch_time.get('alpha_vantage', datetime.min)
                if datetime.utcnow() - last_fetch > timedelta(minutes=1):
                    sources_to_fetch.append(('alpha_vantage', self.fetch_alpha_vantage_news))
            
            # Twitter: Continuous stream
            if 'twitter' in self.sources:
                sources_to_fetch.append(('twitter', lambda: self.monitor_twitter(limit=50)))
            
            # CoinDesk: 1 request per 2 minutes
            if 'coindesk' in self.sources:
                last_fetch = self.last_fetch_time.get('coindesk', datetime.min)
                if datetime.utcnow() - last_fetch > timedelta(minutes=2):
                    sources_to_fetch.append(('coindesk', self.fetch_coindesk_news))
            
            # Run fetches concurrently
            results = await asyncio.gather(
                *(fetch_func() for _, fetch_func in sources_to_fetch),
                return_exceptions=True
            )
            
            # Process results
            for i, result in enumerate(results):
                source_name = sources_to_fetch[i][0]
                if isinstance(result, Exception):
                    logger.error(f"Error fetching from {source_name}: {result}")
                else:
                    await self.process_news(result, source_name)
                    self.last_fetch_time[source_name] = datetime.utcnow()
            
            # Publish batch every 30 seconds
            if self.news_buffer:
                await self.event_bus.publish('news_batch', self.news_buffer.copy())
                self.news_buffer.clear()
            
            # Throttle if no sources were fetched
            if not sources_to_fetch:
                await asyncio.sleep(5)

    async def start(self):
        """Start the news collection service"""
        self.running = True
        await self.start_news_collection()

    async def stop(self):
        """Stop the news collection service"""
        self.running = False
        await self.close_session()