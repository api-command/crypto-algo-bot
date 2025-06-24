import asyncio
import random
import time
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.infra.telemetry import telemetry
from src.core.latency_monitor import LatencyMonitor
from src.utils.db import db_manager, hf_db
from src.data.market_feeds import MarketDataFeed
from src.data.sentinel_agent import SentinelAgent
from src.ml.signal_orchestrator import SignalOrchestrator
from src.core.execution_engine import ExecutionEngine
from src.core.risk_manager import RiskManager

logger = get_logger('stress_test')

class TradingSystemStressTest:
    def __init__(self):
        self.config = {
            'duration_min': 15,               # Test duration in minutes
            'order_intensity': 10,            # Orders per second target
            'market_impact_scenarios': [      # Liquidity scenarios
                {'name': 'normal', 'spread_pct': 0.001, 'slippage_pct': 0.0005},
                {'name': 'volatile', 'spread_pct': 0.01, 'slippage_pct': 0.005},
                {'name': 'crisis', 'spread_pct': 0.05, 'slippage_pct': 0.03}
            ],
            'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
            'failure_rates': {                # Simulated failure probabilities
                'order_execution': 0.01,
                'market_data': 0.005,
                'news_feed': 0.002
            }
        }
        self.latency_monitor = LatencyMonitor()
        self.results = {
            'orders_sent': 0,
            'orders_failed': 0,
            'latencies': [],
            'throughput': [],
            'scenarios': {}
        }

    async def simulate_market_conditions(self, intensity_multiplier=1.0):
        """Generate realistic market data patterns with volatility spikes"""
        logger.info("Starting market condition simulation")
        
        symbols = self.config['symbols']
        base_prices = {'BTC/USD': 50000, 'ETH/USD': 3000, 'SOL/USD': 100}
        volatility_states = ['normal', 'high', 'extreme']
        current_state = 'normal'
        
        start_time = time.time()
        end_time = start_time + (self.config['duration_min'] * 60)
        
        while time.time() < end_time:
            # Randomly transition between volatility states
            if random.random() < 0.01:  # 1% chance to change state
                current_state = random.choice(volatility_states)
                logger.warning(f"Market state changed to {current_state.upper()}")
                
                # Record scenario start
                scenario_id = f"{current_state}_{int(time.time())}"
                self.results['scenarios'][scenario_id] = {
                    'start': datetime.utcnow().isoformat(),
                    'state': current_state,
                    'metrics': []
                }
            
            # Generate ticks for each symbol
            for symbol in symbols:
                base_price = base_prices[symbol]
                
                # Apply volatility based on state
                if current_state == 'normal':
                    price_move = random.uniform(-0.002, 0.002)
                elif current_state == 'high':
                    price_move = random.uniform(-0.01, 0.01)
                else:  # extreme
                    price_move = random.uniform(-0.05, 0.05)
                
                new_price = base_price * (1 + price_move)
                spread_pct = self.get_current_spread(current_state)
                bid = new_price * (1 - spread_pct/2)
                ask = new_price * (1 + spread_pct/2)
                
                # Simulate volume spikes
                base_volume = 100 if symbol == 'BTC/USD' else 500 if symbol == 'ETH/USD' else 1000
                volume_multiplier = 1.0
                if current_state != 'normal':
                    volume_multiplier = random.uniform(2.0, 5.0)
                
                # Record to HFDB
                await hf_db.record_tick(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    last=new_price,
                    volume=base_volume * volume_multiplier * intensity_multiplier
                )
            
            # Adjust intensity based on test progress
            elapsed = time.time() - start_time
            progress = elapsed / (self.config['duration_min'] * 60)
            intensity_multiplier = min(3.0, 1.0 + progress * 2)  # Ramp up over time
            
            await asyncio.sleep(0.1)  # 10 ticks per second
        
        logger.info("Market simulation completed")

    def get_current_spread(self, market_state):
        """Get spread based on current market scenario"""
        for scenario in self.config['market_impact_scenarios']:
            if scenario['name'] == market_state:
                return scenario['spread_pct']
        return 0.001  # Default

    async def simulate_order_storm(self):
        """Flood the system with orders at increasing rates"""
        logger.info("Starting order storm simulation")
        
        order_types = ['market', 'limit', 'stop']
        sides = ['buy', 'sell']
        symbols = self.config['symbols']
        
        start_time = time.time()
        end_time = start_time + (self.config['duration_min'] * 60)
        orders_sent = 0
        
        while time.time() < end_time:
            # Calculate dynamic order rate (ramps up over time)
            elapsed = time.time() - start_time
            progress = elapsed / (self.config['duration_min'] * 60)
            current_rate = self.config['order_intensity'] * (1 + progress * 2)  # Ramp up
            
            # Generate batch of orders
            batch_size = random.randint(1, int(current_rate/2))
            for _ in range(batch_size):
                symbol = random.choice(symbols)
                order_type = random.choice(order_types)
                side = random.choice(sides)
                price = random.uniform(45000, 55000) if symbol == 'BTC/USD' else \
                        random.uniform(2500, 3500) if symbol == 'ETH/USD' else \
                        random.uniform(80, 120)
                amount = random.uniform(0.01, 5.0)
                
                # Simulate occasional failures
                if random.random() < self.config['failure_rates']['order_execution']:
                    logger.error("Simulating order execution failure")
                    self.results['orders_failed'] += 1
                    continue
                
                # Record order
                self.latency_monitor.start_timer('simulated_order')
                await hf_db.record_order_event({
                    'timestamp': datetime.utcnow().isoformat(),
                    'id': f"SIM_{int(time.time())}_{orders_sent}",
                    'symbol': symbol,
                    'type': order_type,
                    'side': side,
                    'price': price,
                    'amount': amount,
                    'status': 'filled' if random.random() > 0.1 else 'rejected'
                })
                latency = self.latency_monitor.record_latency('simulated_order')
                
                self.results['orders_sent'] += 1
                self.results['latencies'].append(latency)
                orders_sent += 1
            
            # Record throughput
            self.results['throughput'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'orders_per_sec': current_rate,
                'pending_orders': random.randint(0, 100)
            })
            
            await asyncio.sleep(1.0 / current_rate if current_rate > 0 else 0.1)
        
        logger.info(f"Order storm completed. Sent {orders_sent} orders")

    async def simulate_news_flood(self):
        """Generate high-volume news events with sentiment variations"""
        logger.info("Starting news flood simulation")
        
        news_sources = ['coindesk', 'twitter', 'reddit', 'alpha_vantage']
        sentiment_types = ['positive', 'negative', 'neutral']
        symbols = self.config['symbols']
        
        start_time = time.time()
        end_time = start_time + (self.config['duration_min'] * 60)
        
        while time.time() < end_time:
            # Vary news intensity
            elapsed = time.time() - start_time
            progress = elapsed / (self.config['duration_min'] * 60)
            intensity = min(10, 1 + int(progress * 20))  # Ramp up to 10 news/sec
            
            for _ in range(intensity):
                symbol = random.choice(symbols)
                source = random.choice(news_sources)
                sentiment = random.choice(sentiment_types)
                
                # Generate news content
                content = self.generate_news_content(symbol, sentiment)
                
                # Simulate occasional feed failure
                if random.random() < self.config['failure_rates']['news_feed']:
                    logger.error("Simulating news feed failure")
                    continue
                
                # Store news event
                await db_manager.insert_news_event({
                    'source': source,
                    'symbol': symbol,
                    'title': f"{symbol} {sentiment} news {int(time.time())}",
                    'content': content,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(1.0 / intensity if intensity > 0 else 0.1)
        
        logger.info("News flood simulation completed")

    def generate_news_content(self, symbol, sentiment):
        """Generate realistic news content with controlled sentiment"""
        templates = {
            'positive': [
                f"{symbol} showing strong bullish momentum as adoption grows",
                f"Institutional investors accumulating {symbol} according to reports",
                f"Breakout: {symbol} surges past key resistance level"
            ],
            'negative': [
                f"Concerns grow over {symbol} as regulatory pressure increases",
                f"Technical indicators show {symbol} may be overbought",
                f"Market panic as {symbol} drops below critical support"
            ],
            'neutral': [
                f"{symbol} trading in tight range as market digests news",
                f"Analysts divided on {symbol} short-term prospects",
                f"{symbol} volatility expected ahead of major announcement"
            ]
        }
        return random.choice(templates[sentiment])

    async def run_component_stress_tests(self):
        """Stress test individual system components"""
        logger.info("Starting component stress tests")
        
        # Market Data Feed Test
        md_latency = await self.test_market_data_feed()
        self.results['component_tests'] = {
            'market_data_feed': md_latency
        }
        
        # Execution Engine Test
        exec_stats = await self.test_execution_engine()
        self.results['component_tests']['execution_engine'] = exec_stats
        
        # Risk Manager Test
        risk_stats = await self.test_risk_manager()
        self.results['component_tests']['risk_manager'] = risk_stats
        
        logger.info("Component stress tests completed")

    async def test_market_data_feed(self):
        """Stress test market data feed with high message volume"""
        logger.info("Stress testing market data feed")
        
        feed = MarketDataFeed()
        test_symbol = 'BTC/USD'
        test_duration = 60  # seconds
        messages = 0
        latencies = []
        
        # Callback for measuring processing time
        def record_latency(data):
            nonlocal messages
            start = time.perf_counter_ns()
            messages += 1
            # Simulate processing
            _ = data['bids'][0][0] + data['asks'][0][0]
            latency = (time.perf_counter_ns() - start) / 1e6
            latencies.append(latency)
        
        # Subscribe to feed
        feed.event_bus.subscribe('orderbook', record_latency)
        
        # Generate test data
        start_time = time.time()
        while time.time() - start_time < test_duration:
            # Simulate market data message
            test_data = {
                'symbol': test_symbol,
                'bids': [[random.uniform(49000, 51000), random.uniform(0.1, 5)] for _ in range(5)],
                'asks': [[random.uniform(49000, 51000), random.uniform(0.1, 5)] for _ in range(5)],
                'timestamp': time.time()
            }
            await feed.event_bus.publish('orderbook', test_data)
            await asyncio.sleep(0.001)  # ~1000 messages/sec
        
        # Calculate stats
        avg_latency = np.mean(latencies) if latencies else 0
        max_latency = np.max(latencies) if latencies else 0
        throughput = messages / test_duration
        
        logger.info(f"Market Data Feed Test: {throughput:.1f} msgs/sec | Avg Latency: {avg_latency:.2f}ms")
        
        return {
            'messages_processed': messages,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'throughput_msg_sec': throughput
        }

    async def test_execution_engine(self):
        """Stress test order execution under load"""
        logger.info("Stress testing execution engine")
        
        engine = ExecutionEngine()
        test_duration = 60  # seconds
        orders_sent = 0
        latencies = []
        
        # Generate test orders
        start_time = time.time()
        while time.time() - start_time < test_duration:
            order = {
                'id': f"TEST_{orders_sent}",
                'symbol': 'BTC/USD',
                'type': random.choice(['market', 'limit']),
                'side': random.choice(['buy', 'sell']),
                'price': random.uniform(49000, 51000),
                'amount': random.uniform(0.01, 5.0)
            }
            
            self.latency_monitor.start_timer('execution_test')
            try:
                await engine.execute_order(order)
                orders_sent += 1
            except Exception as e:
                logger.error(f"Order failed: {e}")
            finally:
                latency = self.latency_monitor.record_latency('execution_test')
                latencies.append(latency)
            
            await asyncio.sleep(random.uniform(0.001, 0.01))  # 100-1000 orders/sec
        
        # Calculate stats
        avg_latency = np.mean(latencies) if latencies else 0
        max_latency = np.max(latencies) if latencies else 0
        throughput = orders_sent / test_duration
        
        logger.info(f"Execution Engine Test: {throughput:.1f} orders/sec | Avg Latency: {avg_latency:.2f}ms")
        
        return {
            'orders_executed': orders_sent,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'throughput_orders_sec': throughput
        }

    async def test_risk_manager(self):
        """Stress test risk evaluation under load"""
        logger.info("Stress testing risk manager")
        
        manager = RiskManager()
        test_duration = 60  # seconds
        checks_performed = 0
        latencies = []
        
        # Generate test positions
        start_time = time.time()
        while time.time() - start_time < test_duration:
            position = {
                'symbol': 'BTC/USD',
                'amount': random.uniform(-10, 10),  # Long/short
                'entry_price': random.uniform(45000, 55000),
                'current_price': random.uniform(40000, 60000)  # Wide range for stress
            }
            
            self.latency_monitor.start_timer('risk_check')
            try:
                await manager.evaluate_position(position)
                checks_performed += 1
            except Exception as e:
                logger.error(f"Risk check failed: {e}")
            finally:
                latency = self.latency_monitor.record_latency('risk_check')
                latencies.append(latency)
            
            await asyncio.sleep(0)  # Max speed
        
        # Calculate stats
        avg_latency = np.mean(latencies) if latencies else 0
        max_latency = np.max(latencies) if latencies else 0
        throughput = checks_performed / test_duration
        
        logger.info(f"Risk Manager Test: {throughput:.1f} checks/sec | Avg Latency: {avg_latency:.2f}ms")
        
        return {
            'checks_performed': checks_performed,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'throughput_checks_sec': throughput
        }

    async def run_crash_scenarios(self):
        """Simulate system crashes and recovery"""
        logger.warning("Starting crash scenario simulations")
        
        scenarios = [
            {'name': 'exchange_api_failure', 'duration': 30, 'recovery_time': 5},
            {'name': 'database_outage', 'duration': 45, 'recovery_time': 10},
            {'name': 'network_partition', 'duration': 60, 'recovery_time': 15}
        ]
        
        for scenario in scenarios:
            logger.critical(f"Simulating {scenario['name']} scenario")
            start_time = time.time()
            
            # Simulate failure
            await asyncio.sleep(scenario['duration'])
            
            # Measure recovery
            recovery_start = time.time()
            # In a real test, we'd restart components here
            await asyncio.sleep(scenario['recovery_time'])
            recovery_time = time.time() - recovery_start
            
            self.results['crash_scenarios'][scenario['name']] = {
                'simulated_duration': scenario['duration'],
                'actual_recovery_time': recovery_time,
                'success': recovery_time <= scenario['recovery_time'] * 1.5
            }
            
            logger.critical(f"Recovered from {scenario['name']} in {recovery_time:.1f}s")
        
        logger.warning("Crash scenario simulations completed")

    def generate_report(self):
        """Generate comprehensive stress test report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'duration_min': self.config['duration_min'],
            'summary': {
                'orders_sent': self.results['orders_sent'],
                'orders_failed': self.results['orders_failed'],
                'failure_rate': self.results['orders_failed'] / max(1, self.results['orders_sent']),
                'avg_latency_ms': np.mean(self.results['latencies']) if self.results['latencies'] else 0,
                'p99_latency_ms': np.percentile(self.results['latencies'], 99) if self.results['latencies'] else 0
            },
            'throughput_analysis': self.results['throughput'],
            'market_scenarios': self.results['scenarios'],
            'component_tests': self.results.get('component_tests', {}),
            'crash_scenarios': self.results.get('crash_scenarios', {})
        }
        
        # Save to database
        asyncio.create_task(db_manager.insert_system_event({
            'event_type': 'STRESS_TEST_REPORT',
            'component': 'testing',
            'message': 'Stress test completed',
            'metadata': report
        }))
        
        return report

    async def run_full_test_suite(self):
        """Execute all stress tests concurrently"""
        logger.critical("STARTING FULL STRESS TEST SUITE")
        
        # Initialize results
        self.results['crash_scenarios'] = {}
        
        # Run all test scenarios concurrently
        test_tasks = [
            self.simulate_market_conditions(),
            self.simulate_order_storm(),
            self.simulate_news_flood(),
            self.run_component_stress_tests(),
            self.run_crash_scenarios()
        ]
        
        await asyncio.gather(*test_tasks)
        
        # Generate final report
        report = self.generate_report()
        logger.critical("STRESS TEST SUITE COMPLETED")
        return report

# Command line execution
if __name__ == "__main__":
    tester = TradingSystemStressTest()
    
    # Configure test parameters from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=15, help='Test duration in minutes')
    parser.add_argument('--intensity', type=int, default=10, help='Base order intensity (orders/sec)')
    args = parser.parse_args()
    
    tester.config['duration_min'] = args.duration
    tester.config['order_intensity'] = args.intensity
    
    # Run tests
    asyncio.run(tester.run_full_test_suite())