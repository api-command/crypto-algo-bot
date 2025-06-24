#!/usr/bin/env python3
"""
Key Rotation Script - Runs weekly via cron
Securely rotates API credentials and master encryption key
"""
import os
import yaml
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from src.utils.crypto_manager import CryptoVault

def derive_key(password: str, salt: bytes) -> bytes:
    """Derive 256-bit key from password"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000
    )
    return kdf.derive(password.encode())

def rotate_keys():
    # Load current config
    with open('config/api_keys.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get master password from secure input
    password = os.environ.get('MASTER_PASSWORD')
    if not password:
        raise EnvironmentError("MASTER_PASSWORD environment variable not set")
    
    # Decrypt current credentials with old key
    old_vault = CryptoVault(derive_key(password, b'static_salt_placeholder'))
    decrypted_config = old_vault.decrypt_nested(config['services'])
    
    # Generate new master key
    new_key = os.urandom(32)
    new_vault = CryptoVault(new_key)
    
    # Re-encrypt all credentials with new key
    new_services = {}
    for service, creds in decrypted_config.items():
        new_services[service] = {}
        for key, value in creds.items():
            new_services[service][key] = new_vault.encrypt(value)
    
    # Update encryption metadata
    new_key_id = f"key_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config['encryption'] = {
        'current_key_id': new_key_id,
        'keys': {new_key_id: new_vault.encrypt(new_key.hex())},
        'last_rotated': datetime.now().isoformat(),
        'next_rotation': (datetime.now() + timedelta(days=7)).isoformat()
    }
    config['services'] = new_services
    
    # Write updated config
    with open('config/api_keys.yml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    
    print(f"Successfully rotated keys. New key ID: {new_key_id}")

if __name__ == "__main__":
    rotate_keys()