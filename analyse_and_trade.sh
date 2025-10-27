#!/bin/sh

echo "========================"
echo "CryptoHunk start trading"
echo "========================"

python schedule_gate.py --grace-minutes 5 --at 0 4 8 12 16 20 --tz Europe/Stockholm || exit 0

echo "Rebalance portfolio"

mv ./kursdata/* ./history/

python download_ohlcv.py
cat portfolio.json | grep ETH

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
  --symbol ETH/USDT \
  --exchange binance \
  --portfolio portfolio.json

  echo "Done"