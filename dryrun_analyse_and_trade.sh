#!/bin/sh

echo "========================"
echo "CryptoHunk start trading"
echo "========================"

# TODO Enabld before move to cloud
# python src/schedule_gate.py --grace-minutes 5 --at 0 4 7 8 12 16 20 21 --tz Europe/Stockholm || exit 0

echo "Rebalance portfolio"

python src/download_portfolio.py
# cat portfolio.json | grep ETH
mv ./bnb_data/* ./history/
mv ./ethereum_data/* ./history/
mv ./solana_data/* ./history/

python src/download_binance_ohlcv.py --symbol 'BNBUSDT' --data-folder 'bnb_data'
python src/download_binance_ohlcv.py --symbol 'ETHUSDT' --data-folder 'ethereum_data'
python src/download_binance_ohlcv.py --symbol 'SOLUSDT' --data-folder 'solana_data'


# Hämta första filen i ./kursdata
FILE_BNB=$(find ./bnb_data -maxdepth 1 -type f | head -n 1)
# FILE_BITCOIN=$(find ./bitcoin_data -maxdepth 1 -type f | head -n 1)
FILE_ETHEREUM=$(find ./ethereum_data -maxdepth 1 -type f | head -n 1)
FILE_SOLANA=$(find ./solana_data -maxdepth 1 -type f | head -n 1)

if [ -z "$FILE_BNB" ]; then
  echo "Ingen fil hittades i ./bnb_data"
  exit 1
fi
#if [ -z "$FILE_BITCOIN" ]; then
#  echo "Ingen fil hittades i ./bitcoin_data"
#  exit 1
#fi
if [ -z "$FILE_ETHEREUM" ]; then
  echo "Ingen fil hittades i ./ethereum_data"
  exit 1
fi
if [ -z "$FILE_SOLANA" ]; then
  echo "Ingen fil hittades i ./solana_data"
  exit 1
fi

# Sätt miljövariabeln
export IN_DATA_BNB="$FILE_BNB"
# export IN_DATA_BITCOIN="$FILE_BITCOIN"
export IN_DATA_ETHEREUM="$FILE_ETHEREUM"
export IN_DATA_SOLANA="$FILE_SOLANA"

#python ta_signal_agent_live_three_assets.py \
#  --csv $IN_DATA \
#  --symbol ETH/USDT \
#  --exchange binance \
#  --portfolio portfolio.json

 python src/ta_signal_agent_live_three_assets.py \
  --csvA $IN_DATA_BNB \
  --csvB $IN_DATA_ETHEREUM \
  --csvC $IN_DATA_SOLANA \
  --symbols BNB/USDC,ETH/USDC,SOL/USDC \
  --dry-run

   echo "Done"