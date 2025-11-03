#!/bin/sh

echo "========================"
echo "CryptoHunk start trading"
echo "========================"

python schedule_gate.py --grace-minutes 5 --at 0 4 8 12 16 20 21 --tz Europe/Stockholm || exit 0

echo "Rebalance portfolio"

python download_portfolio.py
# cat portfolio.json | grep ETH
# mv ./kursdata/* ./history/

python download_ohlcv.py --coin-id 'binancecoin' --data-folder 'bnb_data'
python download_ohlcv.py --coin-id 'bitcoin' --data-folder 'bitcoin_data'
python download_ohlcv.py --coin-id 'ethereum' --data-folder 'ethereum_data'

# Hämta första filen i ./kursdata
FILE_BNB=$(find ./bnb_data -maxdepth 1 -type f | head -n 1)
FILE_BITCOIN=$(find ./bitcoin_data -maxdepth 1 -type f | head -n 1)
FILE_ETHEREUM=$(find ./ethereum_data -maxdepth 1 -type f | head -n 1)

if [ -z "$FILE_BNB" ]; then
  echo "Ingen fil hittades i ./bnb_data"
  exit 1
fi
if [ -z "$FILE_BITCOIN" ]; then
  echo "Ingen fil hittades i ./bitcoin_data"
  exit 1
fi
if [ -z "$FILE_ETHEREUM" ]; then
  echo "Ingen fil hittades i ./ethereum_data"
  exit 1
fi

# Sätt miljövariabeln
export IN_DATA_BNB="$FILE_BNB"
export IN_DATA_BITCOIN="$FILE_BITCOIN"
export IN_DATA_ETHEREUM="$FILE_ETHEREUM"

#python ta_signal_agent_live_three_assets.py \
#  --csv $IN_DATA \
#  --symbol ETH/USDT \
#  --exchange binance \
#  --portfolio portfolio.json

 python ta_signal_agent_live_three_assets.py \
  --csvA $IN_DATA_BNB \
  --csvB $IN_DATA_BITCOIN \
  --csvC $IN_DATA_ETHEREUM \
  --symbols BNB/USDT,BTC/USDT,ETH/USDT \
  --exchange binance \
  --tolerance 0.10

   echo "Done"