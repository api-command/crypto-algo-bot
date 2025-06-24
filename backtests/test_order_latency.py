import time
import numpy as np
from src.core.execution_engine import ExecutionEngine
from src.infra.microservices import AsyncOrderGateway
from src.utils.logger import get_logger

logger = get_logger('latency_tests')

class LatencyBenchmark:
    def __init__(self, exchange_api, symbol='BTC/USD'):
        self.engine = ExecutionEngine(exchange_api)
        self.async_gateway = AsyncOrderGateway(exchange_api)
        self.symbol = symbol
        
    def test_market_order(self, iterations=1000):
        """Measure end-to-end market order latency"""
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            order = self.engine.market_order(self.symbol, 0.001)
            end = time.perf_counter_ns()
            latencies.append(end - start)
            
            # Cancel immediately if filled partially
            if order['filled'] < order['amount']:
                self.engine.cancel_order(order['id'])
        
        self._analyze_results("Market Order", latencies)
    
    def test_limit_order(self, iterations=500):
        """Measure limit order placement latency"""
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            order = self.engine.limit_order(
                self.symbol, 
                0.001, 
                self.engine.get_bbo(self.symbol)['bid'] * 0.99
            )
            end = time.perf_counter_ns()
            latencies.append(end - start)
            self.engine.cancel_order(order['id'])
        
        self._analyze_results("Limit Order", latencies)
    
    def test_async_throughput(self, iterations=10000):
        """Stress test async order processing"""
        start = time.perf_counter_ns()
        for i in range(iterations):
            self.async_gateway.queue_order({
                'symbol': self.symbol,
                'type': 'limit',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'amount': 0.001,
                'price': self.engine.get_bbo(self.symbol)['bid'] * 0.99
            })
        
        # Wait for queue to drain
        while self.async_gateway.queue_size() > 0:
            time.sleep(0.001)
            
        duration = (time.perf_counter_ns() - start) / 1e9
        logger.info(f"Processed {iterations} orders in {duration:.4f}s")
        logger.info(f"Throughput: {iterations/duration:.2f} orders/sec")
    
    def _analyze_results(self, test_name, latencies):
        latencies_ms = np.array(latencies) / 1e6  # Convert to ms
        
        logger.info(f"\n{test_name} Latency Report:")
        logger.info(f"Mean: {np.mean(latencies_ms):.4f}ms")
        logger.info(f"Median: {np.median(latencies_ms):.4f}ms")
        logger.info(f"99th %ile: {np.percentile(latencies_ms, 99):.4f}ms")
        logger.info(f"Max: {np.max(latencies_ms):.4f}ms")
        logger.info(f"Min: {np.min(latencies_ms):.4f}ms")
        
        # Generate histogram
        plt.hist(latencies_ms, bins=50)
        plt.title(f"{test_name} Latency Distribution")
        plt.savefig(f"{test_name.replace(' ', '_')}_latency.png")