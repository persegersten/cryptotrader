# cryptotrader
*************
Crypto trader
*************
# initiera tom portfölj med kontanter (exempel)
echo '{"cash": 10000, "positions": {}}' > portfolio.json

# kör agenten för BTC med din CSV
python3 ta_signal_agent.py --csv path/to/your_btc.csv --asset BTC --portfolio portfolio.json

# exempel med parametrar:
python3 ta_signal_agent.py \
  --csv crypto_bitcoin_usd_20251015_143022.csv \
  --asset BTC \
  --portfolio portfolio.json \
  --buy-alloc 0.20 \
  --sell-alloc 0.0 \
  --min-trade 10 \
  --fee-bps 10
