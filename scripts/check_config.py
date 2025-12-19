"""Check resolved API keys from config loader and mask secrets for display.
Run from project root:
    python scripts\check_config.py
"""
import importlib.util
import os

spec = importlib.util.spec_from_file_location("cfg", "src/utils/config_loader.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
conf = cfg.Config()

api_keys = conf.api_keys or {}

def mask(val: str) -> str:
    if not val:
        return "(empty)"
    s = str(val)
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "*" * (len(s) - 8) + s[-4:]

print("Resolved API keys (masked):")
for svc, data in api_keys.items():
    print(f"- {svc}:")
    if not isinstance(data, dict):
        print(f"    value: {mask(data)}")
        continue
    found = False
    for key in ["api_key", "api_secret", "passphrase", "api_key_encrypted", "api_secret_encrypted"]:
        if key in data and data[key]:
            print(f"    {key}: {mask(data[key])}")
            found = True
    if not found:
        print("    (no keys in file; environment variables will be used if set)")

# Also show which env vars are set (non-empty)
print('\nEnvironment variables (present):')
interesting = [
    'COINBASE_PRO_API_KEY','COINBASE_PRO_API_SECRET','COINBASE_PRO_PASSPHRASE',
    'ALPHA_VANTAGE_API_KEY','HUGGING_FACE_API_KEY','BINANCE_API_KEY','BINANCE_API_SECRET'
]
for v in interesting:
    val = os.getenv(v)
    print(f"- {v}: {'SET' if val else 'NOT SET'}")
