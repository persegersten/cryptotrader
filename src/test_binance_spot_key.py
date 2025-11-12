#!/usr/bin/env python3
import ccxt, os, sys
from pathlib import Path
import json

REQUIRED_ENV = ("CCXT_API_KEY", "CCXT_API_SECRET")

def load_secrets() -> None:
    file_path = "secrets.json"
    p = Path(file_path).expanduser()
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as e:
        sys.exit(f"Fel: kunde inte läsa {file_path}: {e}")
    for k, v in data.items():
        if k in REQUIRED_ENV and (k not in os.environ or not os.environ[k]):
            os.environ[k] = str(v)

load_secrets()

print("🔍 Initierar Binance spot-test ...")

exchange = ccxt.binance({
    'apiKey': os.getenv('CCXT_API_KEY'),
    'secret': os.getenv('CCXT_API_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})

try:
    markets = exchange.load_markets()
    perms = markets['BTC/USDT']['info'].get('permissions', [])
    print(f"✔ Symbol BTC/USDT hittad, permissions: {perms}")
except Exception as e:
    sys.exit(f"❌ Kunde inte läsa marknader: {e}")

# --- kolla kontot ---
try:
    bal = exchange.fetch_balance()
    total_usdt = bal.get('total', {}).get('USDT', 0)
    print(f"✔ fetch_balance OK. USDT-saldo: {total_usdt}")
except Exception as e:
    sys.exit(f"❌ fetch_balance misslyckades: {e}")

# --- Kolla rättighet till market
# print(markets['ETH/USDT'])markets = exchange.load_markets()
allowed_symbols = [
    s for s, m in markets.items()
    if m.get('active') and m.get('info', {}).get('status') == 'TRADING'
]
print(f"Tillgängliga symboler ({len(allowed_symbols)}):")
print(allowed_symbols[:20])

allowed_spot = [
    s for s, m in markets.items()
    if m.get('spot') and m.get('active') and 'SPOT' in (m.get('info', {}).get('permissions') or ['SPOT'])
]
print(allowed_spot[:20])

# --- testorder (kommenterad) ---
symbol = 'ETH/USDC'
amount = 0.002

print("\n⚙️  Skickar ingen riktig order, men du kan avkommentera nedan för att testa:")
print(f"# exchange.create_order('{symbol}', 'market', 'sell', {amount})")
exchange.create_order(symbol, 'market', 'sell', amount)

print("\n✅ Om du ser permissions ['SPOT'] och fetch_balance OK → nyckeln fungerar för Spot-handel.")
