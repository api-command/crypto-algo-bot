# src/utils/config_loader.py
import toml
import yaml
import os
from .crypto_manager import CryptoVault

class ConfigManager:
    def __init__(self):
        self.vault = CryptoVault(os.environ["MASTER_KEY"])
    
    def load_toml(self, path: str) -> dict:
        with open(path, 'r') as f:
            return toml.load(f)
    
    def load_secure_yml(self, path: str) -> dict:
        with open(path, 'r') as f:
            encrypted_config = yaml.safe_load(f)
            return self._decrypt_nested(encrypted_config)
    
    def _decrypt_nested(self, config: dict) -> dict:
        decrypted = {}
        for key, value in config.items():
            if isinstance(value, dict):
                decrypted[key] = self._decrypt_nested(value)
            elif key.endswith('_encrypted'):
                decrypted[key.replace('_encrypted', '')] = self.vault.decrypt(value)
            else:
                decrypted[key] = value
        return decrypted

# Initialization
config = ConfigManager()
bot_params = config.load_toml('config/bot_params.toml')
api_keys = config.load_secure_yml('config/api_keys.yml')