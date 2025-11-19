#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hämtar OHLCV från Binance klines-API och sparar till CSV.

- Parametrar:
  --symbol (t.ex. BTCUSDT)
  --interval (t.ex. 1h, 4h, 1d) [default: 4h]
  --days (antal dagar bakåt) [default: 30]
  --limit (max candles per request, Binance max 1000) [default: 1000]
  --data-folder (ut-mapp) [default: ./kursdata]

- Kolumner i CSV: timestamp(ISO UTC), open, high, low, close, volume (base asset volume)
- Filnamn taggas med YYYYMMDD_hhmmss i Europe/Stockholm.

Exempel:
  python3 binance_ohlcv.py --symbol BTCUSDT --interval 1h --days 30 --limit 1000
"""

import csv
import time
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Binance-intervall → millisekunder
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,  # approx
}

def ms_now_utc() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    """
    Hämtar en sida klines. Returnerar listan Binance skickar:
    [ openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, ... ]
    """
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": max(1, min(limit, 1000)),
    }
    r = requests.get(BINANCE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"Oväntat svar: {data}")
    return data

def collect_klines(symbol: str, interval: str, days: int, limit: int = 1000, pause_s: float = 0.25):
    """
    Hämtar alla klines för angiven period genom att stega över tiden i fönster av storlek (interval*limit).
    Returnerar lista av tuples: (openTime_ms, open, high, low, close, volume)
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Ogiltigt interval '{interval}'. Tillåtna: {', '.join(INTERVAL_MS.keys())}")
    interval_ms = INTERVAL_MS[interval]

    end_ms = ms_now_utc()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    all_rows = []
    seen = set()  # dedup på openTime
    cursor = start_ms

    # Stegstorlek så att varje anrop max returnerar 'limit' candles
    step_ms = interval_ms * max(1, min(limit, 1000))

    while cursor < end_ms:
        window_end = min(cursor + step_ms - 1, end_ms)
        tries = 2
        while True:
            try:
                page = fetch_klines(symbol, interval, cursor, window_end, limit=limit)
                break
            except requests.HTTPError as e:
                # Liten retry på 429/418/5xx
                status = e.response.status_code if e.response is not None else None
                if status in (418, 429, 500, 502, 503, 504) and tries > 0:
                    tries -= 1
                    time.sleep(1.5)
                    continue
                raise

        if not page:
            # Ingen data i detta fönster — hoppa framåt
            cursor = window_end + 1
            continue

        for row in page:
            # row: [0]openTime, [1]open, [2]high, [3]low, [4]close, [5]volume, [6]closeTime, ...
            ot = int(row[0])
            if ot in seen:
                continue
            seen.add(ot)
            o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4]); v = float(row[5])
            all_rows.append((ot, o, h, l, c, v))

        # Nästa fönster börjar efter sista closeTime/openTime vi fick
        last_close_time = int(page[-1][6])  # closeTime
        cursor = last_close_time + 1

        # Varsam paus för att undvika rate limits
        time.sleep(pause_s)

    # Sortera kronologiskt
    all_rows.sort(key=lambda x: x[0])
    return all_rows

def ms_to_iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","open","high","low","close","volume"])
        writer.writeheader()
        for (t_ms, o, h, l, c, v) in rows:
            writer.writerow({
                "timestamp": ms_to_iso_utc(t_ms),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })

def parse_args():
    p = argparse.ArgumentParser(description="Download OHLCV data from Binance klines API")
    p.add_argument('--symbol', default="BTCUSDT", help='Trading pair symbol (default: BTCUSDT)')
    p.add_argument('--interval', default="4h", help='Kline interval (e.g. 1h, 4h, 1d). Default: 4h')
    p.add_argument('--days', type=int, default=30, help='Number of days back from now (default: 30)')
    p.add_argument('--limit', type=int, default=1000, help='Max candles per request (default: 1000, Binance max)')
    p.add_argument('--data-folder', default="./kursdata", help='Folder for downloaded price data (default: ./kursdata)')
    return p.parse_args()

def run(symbol :str='BTCUSDT', interval :str='4h', days :str=30, limit :int=1000, data_folder :str='./kursdata'):
    data_folder = Path(data_folder)

    # Hämta data
    rows = collect_klines(symbol, interval, days, limit=limit)

    # Filnamn med Stockholm-stämpel
    now_se = datetime.now(ZoneInfo("Europe/Stockholm"))
    tag = now_se.strftime("%Y%m%d_%H%M%S")
    fname = f"binance_{symbol}_{interval}_{days}d_{tag}.csv"
    out = (data_folder / fname).resolve()

    write_csv(rows, out)
    print(f"✅ Sparat {len(rows)} rader till: {out}")

if __name__ == "__main__":
    print('Starting a dowload from main')
    run()
