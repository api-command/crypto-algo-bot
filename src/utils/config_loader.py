"""
Configuration loader utility
"""
import toml
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

class ConfigLoader:
    def __init__(self):
        pass
    
    def load_toml(self, path: str) -> dict:
        """Load TOML configuration file"""
        try:
            with open(path, 'r') as f:
                return toml.load(f)
        except FileNotFoundError:
            # Return default configuration if file not found
            return self._get_default_config(path)
    
    def load_secure_yml(self, path: str) -> dict:
        """Load YAML configuration file (for compatibility)"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return environment variables as fallback
            return self._get_env_config()
    
    def _get_default_config(self, path: str) -> dict:
        """Return default configuration based on file type"""
        if 'bot_params' in path:
            return {
                'general': {
                    'log_level': 'INFO',
                    'base_currency': 'USD',
                    'assets': ['BTC', 'ETH', 'SOL'],
                    'heartbeat_interval': 5000
                },
                'execution': {
                    'default_exchange': 'coinbase_pro',
                    'slippage_tolerance': 0.0015,
                    'latency_cutoff': 100
                },
                'sentiment': {
                    'sources': ['alpha_vantage', 'coindesk'],
                    'positive_threshold': 0.75
                },
                'signals': {
                    'weights': {
                        'sentiment': 0.6,
                        'technical': 0.3,
                        'on_chain': 0.1
                    },
                    'rebalance_frequency': '1h'
                },
                'circuit_breakers': {
                    'max_daily_loss': -0.05,
                    'max_position_risk': 0.1,
                    'volatility_shutdown': 0.15
                },
                'logging': {
                    'level': 'INFO',
                    'directory': 'logs',
                    'max_size_mb': 100,
                    'backup_count': 7
                }
            }
        return {}
    
    def _get_env_config(self) -> dict:
        """Get configuration from environment variables"""
        return {
            'alpha_vantage': {
                'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
            },
            'hugging_face': {
                'api_key': os.getenv('HUGGING_FACE_API_KEY', 'hf_demo')
            },
            'twitter': {
                'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', '')
            },
            'coinbase_pro': {
                'api_key': os.getenv('COINBASE_PRO_API_KEY', ''),
                'api_secret': os.getenv('COINBASE_PRO_API_SECRET', ''),
                'passphrase': os.getenv('COINBASE_PRO_PASSPHRASE', ''),
                'sandbox': os.getenv('COINBASE_SANDBOX', 'true').lower() == 'true'
            },
            'binance': {
                'api_key': os.getenv('BINANCE_API_KEY', ''),
                'api_secret': os.getenv('BINANCE_API_SECRET', ''),
                'testnet': os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            }
        }

# Global instance
config_loader = ConfigLoader()

# Configuration class for easier access
class Config:
    def __init__(self):
        self.bot_params = config_loader.load_toml('config/bot_params.toml')
        self.api_keys = config_loader.load_secure_yml('config/api_keys.yml')
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.bot_params
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self, service: str, key: str = 'api_key'):
        """Get API key for a service"""
        if service in self.api_keys:
            return self.api_keys[service].get(key)
        return None