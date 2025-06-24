# src/utils/crypto_manager.py
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import os

class CryptoVault:
    def __init__(self, master_key: bytes):
        self.master_key = master_key
    
    def encrypt(self, plaintext: str) -> str:
        iv = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(self.master_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        return base64.b64encode(iv + encryptor.tag + ciphertext).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        data = base64.b64decode(ciphertext)
        iv, tag, ciphertext = data[:12], data[12:28], data[28:]
        cipher = Cipher(
            algorithms.AES(self.master_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize().decode()

# Usage
vault = CryptoVault(os.environ["MASTER_KEY"])
decrypted_secret = vault.decrypt(encrypted_api_key)


from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import os

class CryptoVault:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        self.key = key
    
    def encrypt(self, plaintext: str) -> str:
        iv = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        return base64.b64encode(iv + encryptor.tag + ciphertext).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        data = base64.b64decode(ciphertext)
        iv, tag, ciphertext = data[:12], data[12:28], data[28:]
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return (decryptor.update(ciphertext) + decryptor.finalize()).decode()
    
    def decrypt_nested(self, config: dict) -> dict:
        """Recursively decrypt nested configuration"""
        decrypted = {}
        for key, value in config.items():
            if isinstance(value, dict):
                decrypted[key] = self.decrypt_nested(value)
            elif isinstance(value, str):
                try:
                    decrypted[key] = self.decrypt(value)
                except:
                    decrypted[key] = value  # Not encrypted value
            else:
                decrypted[key] = value
        return decrypted