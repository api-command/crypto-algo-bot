import time
import psutil
import asyncio
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter, Summary, Histogram
from src.utils.logger import get_logger
from src.utils.config_loader import config_loader
from src.infra.alert_manager import alert_manager

logger = get_logger('telemetry')

# Prometheus Metrics
LATENCY = Histogram(
    'trading_latency_seconds', 
    'Latency of trading operations', 
    ['process'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
CPU_USAGE = Gauge('system_cpu_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_percent', 'Memory usage percentage')
NETWORK_LATENCY = Gauge('system_network_latency_ms', 'Network latency in ms')
ORDERS_SENT = Counter('orders_sent_total', 'Total orders sent', ['exchange', 'symbol', 'type'])
ALERTS_TRIGGERED = Counter('alerts_triggered_total', 'Total alerts triggered', ['severity'])
PNL = Gauge('trading_pnl', 'Current Profit and Loss')
QUEUE_DEPTH = Gauge('event_queue_depth', 'Event queue depth', ['queue'])

class Telemetry:
    def __init__(self, prometheus_port=8000):
        self.config = config_loader.load_toml('config/bot_params.toml').get('telemetry', {})
        self.prometheus_enabled = self.config.get('prometheus', True)
        self.prometheus_port = prometheus_port
        self.in_memory_metrics = {
            'latency': {},
            'resource_usage': {}
        }
        self.start_time = time.time()
        
        # Start Prometheus server if enabled
        if self.prometheus_enabled:
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
    
    def gauge(self, name, value, labels=None):
        """Record a gauge metric"""
        # Store in memory
        if name not in self.in_memory_metrics:
            self.in_memory_metrics[name] = {}
        self.in_memory_metrics[name][datetime.utcnow()] = value
        
        # Log if not using Prometheus
        if not self.prometheus_enabled:
            logger.debug(f"Metric [{name}]: {value}")
    
    def incr(self, name, count=1, tags=None):
        """Increment a counter"""
        # Log if not using Prometheus
        if not self.prometheus_enabled:
            logger.debug(f"Counter [{name}] incremented by {count}")
    
    def latency(self, process_name, latency_seconds):
        """Record latency for a process"""
        LATENCY.labels(process=process_name).observe(latency_seconds)
        self.gauge(f'latency.{process_name}', latency_seconds)
    
    def record_order(self, exchange, symbol, order_type):
        """Record order submission"""
        ORDERS_SENT.labels(exchange=exchange, symbol=symbol, type=order_type).inc()
        self.incr(f'orders.{exchange}.{symbol}.{order_type}')
    
    def record_alert(self, severity):
        """Record alert trigger"""
        ALERTS_TRIGGERED.labels(severity=severity).inc()
        self.incr(f'alerts.{severity}')
    
    def record_pnl(self, pnl_value):
        """Record current PnL"""
        PNL.set(pnl_value)
        self.gauge('pnl', pnl_value)
    
    def record_queue_depth(self, queue_name, depth):
        """Record event queue depth"""
        QUEUE_DEPTH.labels(queue=queue_name).set(depth)
        self.gauge(f'queue.{queue_name}.depth', depth)
    
    async def monitor_resources(self, interval=5):
        """Continuously monitor system resources"""
        while True:
            # CPU
            cpu_percent = psutil.cpu_percent()
            CPU_USAGE.set(cpu_percent)
            self.gauge('system.cpu', cpu_percent)
            
            # Memory
            mem_percent = psutil.virtual_memory().percent
            MEMORY_USAGE.set(mem_percent)
            self.gauge('system.memory', mem_percent)
            
            # Network (simplified ping to Google DNS)
            start = time.time()
            try:
                reader, writer = await asyncio.open_connection('8.8.8.8', 53)
                writer.close()
                await writer.wait_closed()
                net_latency = (time.time() - start) * 1000  # ms
            except:
                net_latency = 9999  # Timeout value
            
            NETWORK_LATENCY.set(net_latency)
            self.gauge('system.network', net_latency)
            
            # Check resource thresholds
            if cpu_percent > 90:
                await alert_manager.send_alert(
                    "HIGH_CPU", 
                    f"CPU usage at {cpu_percent}%", 
                    severity="WARNING"
                )
                
            if mem_percent > 90:
                await alert_manager.send_alert(
                    "HIGH_MEMORY", 
                    f"Memory usage at {mem_percent}%", 
                    severity="WARNING"
                )
            
            await asyncio.sleep(interval)
    
    def get_uptime(self):
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def get_metrics(self):
        """Get in-memory metrics for API exposure"""
        return self.in_memory_metrics
    
    def health_check(self):
        """Generate health status report"""
        return {
            "status": "running",
            "uptime": self.get_uptime(),
            "services": list(microservice_manager.services.keys()),
            "resource_usage": {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent
            }
        }

# Global telemetry instance
telemetry = Telemetry()