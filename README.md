# cryptotrader
*************
Crypto trader
*************
Läser din CSV med OHLCV-data
Räknar signal (BUY / SELL / HOLD)
Synkar portfölj från börsen (eller simulerar i dry-run)
Utför köp/sälj enligt 0–100%-strategin
Loggar till trades_log.csv
Sparar uppdaterad portfolio.json
Med --dry-run: inga ordrar skickas, allt bara skrivs ut

# initiera tom portfölj med kontanter (exempel)
echo '{"cash": 10000, "positions": {}}' > portfolio.json

# kör agenten för BTC med din CSV
python3 ta_signal_agent.py --csv path/to/your_btc.csv --asset BTC --portfolio portfolio.json

# exempel med parametrar:
Dry-run för att bara se vad agenten skulle göra:

python3 ta_signal_agent_live_binary.py \
  --csv crypto_bitcoin_usd_20251015_143022.csv \
  --symbol BTC/USDT \
  --exchange binance \
  --portfolio portfolio.json \
  --dry-run

Live (på riktigt, med API-nycklar):
export CCXT_API_KEY="din_api_key"
export CCXT_API_SECRET="ditt_api_secret"

python3 ta_signal_agent_live_binary.py \
  --csv crypto_bitcoin_usd_20251015_143022.csv \
  --symbol BTC/USDT \
  --exchange binance \
  --portfolio portfolio.json \
  --buy-alloc 1.0 \
  --sell-alloc 0.0

