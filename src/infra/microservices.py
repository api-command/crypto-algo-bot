import asyncio
import uvloop
import time
import signal
from concurrent.futures import ThreadPoolExecutor
from src.utils.logger import get_logger
from src.core.latency_monitor import LatencyMonitor

logger = get_logger('microservices')

class AsyncEventBus:
    """Lightweight event bus for inter-service communication"""
    def __init__(self):
        self.subscribers = {}
        self.latency_monitor = LatencyMonitor()
    
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type, callback):
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
    
    async def publish(self, event_type, data):
        if event_type not in self.subscribers:
            return
        
        self.latency_monitor.start_timer(f'event_{event_type}')
        tasks = []
        for callback in self.subscribers[event_type]:
            tasks.append(asyncio.create_task(callback(data)))
        
        await asyncio.gather(*tasks)
        self.latency_monitor.record_latency(f'event_{event_type}')

class MicroserviceManager:
    """Orchestrates async services with low-latency optimizations"""
    def __init__(self):
        # Use uvloop for faster I/O
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        self.loop = asyncio.get_event_loop()
        self.loop.set_debug(False)
        
        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Service registry
        self.services = {}
        self.event_bus = AsyncEventBus()
        self.latency_monitor = LatencyMonitor()
        self.running = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"Received shutdown signal {signum}")
        self.loop.create_task(self.stop_all())
    
    def register_service(self, name, service_instance):
        """Register a microservice"""
        self.services[name] = service_instance
    
    async def start_service(self, name):
        """Start a single service"""
        if name in self.services and hasattr(self.services[name], 'start'):
            logger.info(f"Starting service: {name}")
            await self.services[name].start()
        else:
            logger.error(f"Service {name} not found or missing start method")
    
    async def stop_service(self, name):
        """Stop a single service"""
        if name in self.services and hasattr(self.services[name], 'stop'):
            logger.info(f"Stopping service: {name}")
            await self.services[name].stop()
        else:
            logger.error(f"Service {name} not found or missing stop method")
    
    async def start_all(self):
        """Start all registered services concurrently"""
        self.running = True
        tasks = [asyncio.create_task(self.start_service(name)) for name in self.services]
        await asyncio.gather(*tasks)
        logger.info("All services started")
    
    async def stop_all(self):
        """Stop all registered services gracefully"""
        if not self.running:
            return
            
        self.running = False
        tasks = [asyncio.create_task(self.stop_service(name)) for name in self.services]
        await asyncio.gather(*tasks)
        logger.info("All services stopped")
    
    def run_blocking(self, func, *args):
        """Run blocking function in thread pool"""
        return self.loop.run_in_executor(self.thread_pool, func, *args)
    
    def run(self):
        """Run the event loop indefinitely"""
        try:
            self.loop.run_until_complete(self.start_all())
            self.loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self.loop.run_until_complete(self.stop_all())
            self.loop.close()
            logger.info("Microservices shutdown complete")

# Global instance for easy access
microservice_manager = MicroserviceManager()