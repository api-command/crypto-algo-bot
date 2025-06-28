import os
import ccxt
import time
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
        self.logger = logger
        self.config = load_exchange_config(exchange_name)
        
        # Use sandbox/testnet for safe testing
        sandbox = os.getenv('COINBASE_SANDBOX', 'true').lower() == 'true'
        testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        if exchange_name == 'coinbase_pro':
            try:
                import cbpro
                if sandbox:
                    self.exchange = cbpro.AuthenticatedClient(
                        decrypt_key(exchange_name, 'api_key'),
                        decrypt_key(exchange_name, 'api_secret'),
                        decrypt_key(exchange_name, 'passphrase'),
                        sandbox=True
                    )
                else:
                    self.exchange = cbpro.AuthenticatedClient(
                        decrypt_key(exchange_name, 'api_key'),
                        decrypt_key(exchange_name, 'api_secret'),
                        decrypt_key(exchange_name, 'passphrase')
                    )
            except ImportError:
                logger.warning("cbpro not installed, using ccxt fallback")
                self._init_ccxt_exchange(exchange_name, sandbox)
        else:
            self._init_ccxt_exchange(exchange_name, testnet)
        
        # Initialize order parameters
        self.min_order_sizes = self.config.get('order_params', {}).get('min_order_size', {})
        self.price_precision = self.config.get('order_params', {}).get('price_precision', {})
    
    def _init_ccxt_exchange(self, exchange_name, sandbox_mode=True):
        """Initialize CCXT exchange with API keys"""
        exchange_class = getattr(ccxt, exchange_name.replace('_pro', ''))
        
        config = {
            'apiKey': decrypt_key(exchange_name, 'api_key'),
            'secret': decrypt_key(exchange_name, 'api_secret'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        }
        
        if exchange_name == 'coinbase_pro' and sandbox_mode:
            config['sandbox'] = True
        elif exchange_name == 'binance' and sandbox_mode:
            config['options']['defaultType'] = 'future'  # Use testnet
        
        self.exchange = exchange_class(config)
        
        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.warning(f"Could not load markets for {exchange_name}: {e}")
        
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
        
        # For demo/paper trading, just log the order
        if os.getenv('PAPER_TRADING', 'true').lower() == 'true':
            self.logger.info(f"PAPER TRADE: {order_type} {amount} {symbol} at {price}")
            return {
                'id': f'paper_{int(time.time())}',
                'symbol': symbol,
                'type': order_type,
                'amount': amount,
                'price': price,
                'status': 'closed',
                'filled': amount
            }
        
        # Binance batch order optimization
        if self.config.get('latency_optimization', {}).get('use_batch_orders'):
            return self._batch_order([{
                'symbol': symbol,
                'type': order_type,
                'amount': amount,
                'price': price
            }])
        
        try:
            return self.exchange.create_order(symbol, order_type, 'buy', amount, price)
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return None