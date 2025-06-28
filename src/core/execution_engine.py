import os
import ccxt
import toml
from src.utils.config_loader import config_loader
from src.utils.logger import get_logger

logger = get_logger('execution_engine')

def decrypt_key(exchange_name, key_type):
    """Get decrypted API key from environment"""
    env_map = {
        'coinbase_pro': {
            'api_key': 'COINBASE_PRO_API_KEY',
            'api_secret': 'COINBASE_PRO_API_SECRET',
            'passphrase': 'COINBASE_PRO_PASSPHRASE'
        },
        'binance': {
            'api_key': 'BINANCE_API_KEY',
            'api_secret': 'BINANCE_API_SECRET'
        }
    }
    
    if exchange_name in env_map and key_type in env_map[exchange_name]:
        return os.getenv(env_map[exchange_name][key_type], '')
    return ''

def load_exchange_config(exchange_name):
    config_path = f"config/exchange_configs/{exchange_name}.toml"
    return config_loader.load_toml(config_path)

class ExecutionEngine:
    def __init__(self, exchange_name):
        self.config = load_exchange_config(exchange_name)
        self.exchange = ccxt.pro(exchange_name)({
            'apiKey': decrypt_key(exchange_name, 'api_key'),
            'secret': decrypt_key(exchange_name, 'api_secret'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Apply exchange-specific settings
        self.exchange.load_markets()
        self.min_order_sizes = self.config['order_params']['min_order_size']
        self.price_precision = self.config['order_params']['price_precision']
        
    def create_order(self, symbol, order_type, amount, price=None):
        # Enforce exchange-specific limits
        min_size = self.min_order_sizes.get(symbol, 0.001)
        if amount < min_size:
            self.logger.warning(f"Order size {amount} below min {min_size} for {symbol}")
            return None
            
        # Apply price precision
        if price:
            precision = self.price_precision.get(symbol, 2)
            price = round(price, precision)
        
        # Binance batch order optimization
        if self.config['latency_optimization']['use_batch_orders']:
            return self._batch_order([{
                'symbol': symbol,
                'type': order_type,
                'amount': amount,
                'price': price
            }])
        
        return self.exchange.create_order(symbol, order_type, 'buy', amount, price)