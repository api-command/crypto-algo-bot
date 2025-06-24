import time
import numpy as np
import psutil
from collections import deque
from datetime import datetime
from src.utils.logger import get_logger
from src.infra.telemetry import Telemetry
from src.infra.alert_manager import AlertManager

logger = get_logger('latency_monitor')

class LatencyMonitor:
    def __init__(self, max_samples=1000, alert_threshold=100):
        """
        Tracks system and trading operation latencies
        :param max_samples: Number of samples to keep for rolling stats
        :param alert_threshold: Milliseconds threshold for latency alerts
        """
        self.max_samples = max_samples
        self.alert_threshold = alert_threshold
        self.latency_history = {
            'event_ingestion': deque(maxlen=max_samples),
            'signal_generation': deque(maxlen=max_samples),
            'order_execution': deque(maxlen=max_samples),
            'total_loop': deque(maxlen=max_samples)
        }
        self.system_metrics = {
            'cpu': deque(maxlen=max_samples),
            'memory': deque(maxlen=max_samples),
            'network': deque(maxlen=max_samples)
        }
        self.telemetry = Telemetry()
        self.alert_manager = AlertManager()
        self.start_times = {}
        
    def start_timer(self, process_name: str):
        """Mark the start of a process with nanosecond precision"""
        self.start_times[process_name] = time.perf_counter_ns()
        
    def record_latency(self, process_name: str):
        """Record the latency for a process in milliseconds"""
        if process_name not in self.start_times:
            logger.warning(f"No start time recorded for {process_name}")
            return None
            
        end_time = time.perf_counter_ns()
        latency_ms = (end_time - self.start_times[process_name]) / 1e6  # Convert to ms
        
        # Record latency
        if process_name in self.latency_history:
            self.latency_history[process_name].append(latency_ms)
            
            # Check alert threshold
            if latency_ms > self.alert_threshold:
                self.trigger_alert(process_name, latency_ms)
        else:
            logger.error(f"Unknown process name: {process_name}")
            
        # Send to telemetry
        self.telemetry.gauge(f'latency.{process_name}', latency_ms)
        return latency_ms
    
    def trigger_alert(self, process_name: str, latency: float):
        """Generate latency alert"""
        message = (f"High latency alert! {process_name} took {latency:.2f}ms "
                   f"(threshold: {self.alert_threshold}ms)")
        logger.critical(message)
        self.alert_manager.send_alert(
            "LATENCY_ALERT", 
            message,
            severity="CRITICAL"
        )
        
    def record_system_metrics(self):
        """Capture current system resource utilization"""
        # CPU utilization (percentage)
        cpu_percent = psutil.cpu_percent()
        self.system_metrics['cpu'].append(cpu_percent)
        
        # Memory usage (percentage)
        mem_percent = psutil.virtual_memory().percent
        self.system_metrics['memory'].append(mem_percent)
        
        # Network latency (ms)
        # Note: This is a placeholder - real implementation would ping a reliable host
        net_latency = np.random.uniform(0.5, 2.0) * 1000  # Simulated latency
        self.system_metrics['network'].append(net_latency)
        
        # Send to telemetry
        self.telemetry.gauge('system.cpu', cpu_percent)
        self.telemetry.gauge('system.memory', mem_percent)
        self.telemetry.gauge('system.network', net_latency)
    
    def get_latency_stats(self, process_name: str) -> dict:
        """Get statistics for a latency metric"""
        if process_name not in self.latency_history:
            return {}
            
        data = list(self.latency_history[process_name])
        if not data:
            return {}
            
        return {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            '99th': np.percentile(data, 99),
            'max': max(data),
            'min': min(data),
            'last': data[-1],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all latency metrics"""
        return {
            process: self.get_latency_stats(process)
            for process in self.latency_history.keys()
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable latency report"""
        report = "Latency Performance Report\n"
        report += "=" * 40 + "\n"
        
        for process, stats in self.get_all_stats().items():
            if not stats:
                continue
                
            report += (f"{process.replace('_', ' ').title()}:\n"
                       f"  Last: {stats['last']:.2f}ms\n"
                       f"  Avg: {stats['mean']:.2f}ms\n"
                       f"  99th %ile: {stats['99th']:.2f}ms\n"
                       f"  Max: {stats['max']:.2f}ms\n\n")
                       
        # Add system metrics
        report += "System Metrics:\n"
        report += f"  CPU: {self.system_metrics['cpu'][-1] if self.system_metrics['cpu'] else 'N/A'}%\n"
        report += f"  Memory: {self.system_metrics['memory'][-1] if self.system_metrics['memory'] else 'N/A'}%\n"
        report += f"  Network: {self.system_metrics['network'][-1] if self.system_metrics['network'] else 'N/A'}ms\n"
        
        return report

    def reset(self):
        """Reset all latency tracking"""
        for key in self.latency_history:
            self.latency_history[key].clear()
        for key in self.system_metrics:
            self.system_metrics[key].clear()
        self.start_times = {}