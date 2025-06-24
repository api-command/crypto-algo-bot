def load_exchange_config(exchange_name):
    config_path = f"config/exchange_configs/{exchange_name}.toml"
    return toml.load(config_path)

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