#!/bin/sh

echo "Rebalance portfolio"

mv ./kursdata/* ./history/

python download_ohlcv.py

# Hämta första filen i ./kursdata
FILE=$(find ./kursdata -maxdepth 1 -type f | head -n 1)

if [ -z "$FILE" ]; then
  echo "Ingen fil hittades i ./kursdata"
  exit 1
fi

# Sätt miljövariabeln
export IN_DATA="$FILE"

python download_portfolio.py

python ta_signal_agent_live_binary.py \
  --csv ./history/crypto_bitcoin_usd_20251020_171228.csv \
  --symbol BTC/USDT \
  --exchange binance \
  --portfolio portfolio.json
  --portfolio portfolio.json

  echo "Done"