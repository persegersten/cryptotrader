#!/bin/sh


python schedule_gate.py --at 0 4 8 12 16 20 --tz Europe/Stockholm || exit 0

echo "Rebalance portfolio"

mv ./kursdata/* ./history/

python download_ohlcv.py

# Hämta första filen i ./kursdata
FILE=$(find ./kursdata -maxdepth 1 -type f | head -n 1)

if [ -z "$FILE" ]; then
  echo "Ingen fil hittades i ./kursdata"
  exit 1
fi

python download_portfolio.py --sandbox

# Sätt miljövariabeln
export IN_DATA="$FILE"

python ta_signal_agent_live_binary.py \
  --csv $IN_DATA \
  --symbol BTC/USDT \
  --exchange binance \
  --portfolio portfolio.json \
  --sandbox

echo "Done"