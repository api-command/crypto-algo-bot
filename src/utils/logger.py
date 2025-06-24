import logging
import logging.handlers
import os
import sys
import time
import json
from datetime import datetime
from logging import Logger
from src.utils.config_loader import config_loader

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record):
        log_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'process': record.processName,
            'thread': record.threadName,
            'location': f"{record.filename}:{record.lineno}",
            'message': record.getMessage(),
        }
        
        # Add exception info if available
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)

def get_logger(name: str) -> Logger:
    """
    Get a configured logger instance with unified settings
    :param name: Logger name (usually __name__)
    :return: Configured Logger instance
    """
    # Load logging configuration
    config = config_loader.load_toml('config/bot_params.toml').get('logging', {})
    log_level = config.get('level', 'INFO').upper()
    log_dir = config.get('directory', 'logs')
    max_size = config.get('max_size_mb', 100) * 1024 * 1024  # Convert MB to bytes
    backup_count = config.get('backup_count', 7)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Return existing logger if already configured
    if logger.handlers:
        return logger
    
    # Create log directory if needed
    os.makedirs(log_dir, exist_ok=True)
    
    # Create handlers
    handlers = []
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)-20s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler (JSON format)
    log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())
    handlers.append(file_handler)
    
    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    # Add latency tracking for critical operations
    original_methods = {}
    critical_methods = ['info', 'warning', 'error', 'critical', 'exception']
    
    def make_timed_method(level):
        def timed_method(msg, *args, **kwargs):
            start = time.perf_counter_ns()
            original = original_methods[level]
            result = original(msg, *args, **kwargs)
            latency = (time.perf_counter_ns() - start) / 1e6  # ms
            if latency > 10:  # Only log if significant
                logger.debug(f"Logging latency for {level}: {latency:.4f}ms")
            return result
        return timed_method
    
    # Wrap critical methods with latency tracking
    for level in critical_methods:
        original_methods[level] = getattr(logger, level)
        setattr(logger, level, make_timed_method(level))
    
    return logger

# Specialized loggers for high-frequency components
def get_perf_logger(name: str) -> Logger:
    """Get a performance-optimized logger for low-latency components"""
    logger = logging.getLogger(f"perf.{name}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    # Disable propagation
    logger.propagate = False
    
    # Create memory handler that flushes to file periodically
    from logging.handlers import MemoryHandler
    
    # File handler (binary format for speed)
    file_handler = logging.FileHandler(f"logs/perf_{name}.log", mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Memory buffer that flushes every 100 records or 1 second
    mem_handler = MemoryHandler(
        capacity=100,
        flushLevel=logging.INFO,
        target=file_handler,
        flushOnClose=True
    )
    logger.addHandler(mem_handler)
    
    return logger

# Initialize main logger immediately for config errors
try:
    config_loader  # Test if available
except NameError:
    # Fallback basic config if config loader not available
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)-8s %(name)-20s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )